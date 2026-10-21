"""Text safety-neuron detection with iterative ablation (paper Sec. 3).

Data: the 900 harmful + 900 safe text queries of data/text/text_detection.jsonl, split 80/20
(stratified) into the detection set and an in-distribution validation set that detection
never sees.

Each round k runs on f_theta^(k), the model with every neuron confirmed in rounds < k
clamped to its safe mean:
  1. Behavioral labels: response (max_new_tokens 512) -> 17-template keyword pass -> GPT-4o
     re-judges keyword-COMPLY responses (--no-judge: keyword labels only).
     Positives = harmful refused, negatives = safe complied.
  2. Stage 1: behavioral probing over mlp_gate, mlp_up and self_attn -> candidates (|z| > 3.0,
     earlier confirmed neurons excluded).
  3. Stage 2: mean-activation patching of the top-500 candidates against 300 control neurons,
     BH-FDR at 0.05 -> confirmed set B_k, clamped from round k+1 on.
The loop stops after 3 rounds or when a round has fewer than 20 behavioral positives.

The responses of a round are saved before they are judged. If a run stops in round k,
rerunning the same command with --start-round k reuses them when the model, the queries and
the clamps are unchanged.

Usage:
    python src/run_iterative_text.py --model qwen25vl-7b --output-dir results/text/qwen25vl-7b
"""

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import MODEL_REGISTRY, REPO_ROOT, TEXT_COMPONENTS, DetectionConfig, load_pipeline_config
from judge import JUDGE_DEPLOYMENT, behavioral_labels, judge_stats
from neuron_detection_vlm import (
    VLMSafetyNeuronDetector, check_resume, clamp_list, install_permanent_hooks, load_neurons,
    probe_split, read_jsonl, repo_path, save_iterative_summary, save_neurons, write_jsonl,
)
from refusal import final_answer, kw_label, thinking_setting
from text_manifest import load_text_manifest
from text_models import load_model, load_processor

DEFAULT_MANIFEST = REPO_ROOT / "data" / "text" / "text_detection.jsonl"


def generation_records(extractor, queries, max_new_tokens, desc="Generate"):
    """Greedy responses and keyword labels for a list of queries."""
    thinking = bool(extractor.thinking_mode)
    tok = extractor.tokenizer
    records = []
    for q in tqdm(queries, desc=desc):
        gen_ids = extractor.generate(q, max_new_tokens=max_new_tokens)
        answer = final_answer(gen_ids, tok, thinking, max_answer_tokens=max_new_tokens)
        records.append({
            "query_text": q.text,
            "label": q.label,
            "source": (q.metadata or {}).get("source", "unknown"),
            "response": tok.decode(gen_ids, skip_special_tokens=False),
            "answer": answer or "",
            # no final answer: unlabeled
            "kw_label": "NO_ANSWER" if answer is None else kw_label(answer),
            "failed": answer is None,
        })
    return records


def save_generations(round_dir, fingerprint, records):
    """per_sample/generate.jsonl, then generation_done.json with the fingerprint it was made under."""
    (round_dir / "generation_done.json").unlink(missing_ok=True)
    write_jsonl(round_dir / "per_sample" / "generate.jsonl", records)
    with open(round_dir / "generation_done.json", "w") as f:
        json.dump(fingerprint, f)


def load_generations(round_dir, fingerprint):
    """The saved records of a round if they were generated under the same fingerprint, else None."""
    marker, path = round_dir / "generation_done.json", round_dir / "per_sample" / "generate.jsonl"
    if not (marker.exists() and path.exists()):
        return None
    with open(marker) as f:
        return read_jsonl(path) if json.load(f) == fingerprint else None


def main():
    parser = argparse.ArgumentParser(description="Iterative text safety-neuron detection")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY),
                        help="Model key from configs/model_registry.yaml")
    parser.add_argument("--n-rounds", type=int, default=None, help="Number of rounds (default: 3, from configs/pipeline.yaml)")
    parser.add_argument("--data-manifest", type=str, default=str(DEFAULT_MANIFEST),
                        help="Text query manifest (default: data/text/text_detection.jsonl)")
    parser.add_argument("--iid-ratio", type=float, default=0.2,
                        help="Fraction held out as the in-distribution validation split (default: 0.2)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Data-parallel generation over this many GPUs (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/text/<model>)")
    parser.add_argument("--start-round", type=int, default=1,
                        help="Resume from this round (earlier rounds are read from --output-dir)")
    parser.add_argument("--no-judge", dest="use_judge", action="store_false",
                        help="Keyword labels only (default: GPT-4o re-judges keyword-COMPLY responses)")
    parser.add_argument("--judge-deployment", type=str, default=JUDGE_DEPLOYMENT,
                        help="Azure OpenAI deployment name of GPT-4o")
    args = parser.parse_args()
    if args.use_judge:
        from judge import preflight_judge
        preflight_judge(args.judge_deployment)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    hf_repo = MODEL_REGISTRY[args.model]
    config = DetectionConfig.from_pipeline_yaml(load_pipeline_config())
    config.seed, config.components = args.seed, TEXT_COMPONENTS
    if args.n_rounds:
        config.n_rounds = args.n_rounds
    n_rounds = config.n_rounds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.output_dir) if args.output_dir
                  else REPO_ROOT / "results" / "text" / args.model)

    settings = {"model_key": args.model, "hf_repo": hf_repo, "seed": args.seed,
                "data_manifest": repo_path(args.data_manifest), "iid_ratio": args.iid_ratio,
                "use_judge": args.use_judge}
    if args.start_round > 1:
        check_resume(output_dir, args.start_round, settings, "stage2_confirmed.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: shuffle, then a stratified 80/20 detection / IID split ----
    harmful, safe = load_text_manifest(args.data_manifest)
    rng = random.Random(args.seed)
    rng.shuffle(harmful)
    rng.shuffle(safe)
    queries = harmful + safe
    distinct = list(dict.fromkeys((q.text, q.label) for q in queries))
    _, iid_distinct = probe_split(np.array([int(lab == "harmful") for _, lab in distinct]),
                                  args.iid_ratio, args.seed)
    iid_keys = {distinct[i] for i in iid_distinct}
    det_idx = [i for i, q in enumerate(queries) if (q.text, q.label) not in iid_keys]
    iid_idx = [i for i, q in enumerate(queries) if (q.text, q.label) in iid_keys]
    y = np.array([1] * len(harmful) + [0] * len(safe))
    det_harmful = [queries[i] for i in det_idx if y[i] == 1]
    det_safe = [queries[i] for i in det_idx if y[i] == 0]
    split = {"detection": [{"query_text": queries[i].text, "label": queries[i].label} for i in det_idx],
             "iid_validation": [{"query_text": queries[i].text, "label": queries[i].label} for i in iid_idx]}
    split_path = output_dir / "data_split.json"
    if args.start_round > 1 and split_path.exists():
        with open(split_path) as f:
            if json.load(f) != split:
                sys.exit(f"--start-round {args.start_round}: the queries differ from {split_path}")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=1, ensure_ascii=False)
    print(f"Detection set: {len(det_harmful)} harmful + {len(det_safe)} safe; "
          f"IID validation: {len(iid_idx)} queries")

    # ---- model ----
    thinking = thinking_setting(args.model)
    processor = load_processor(args.model)
    model = load_model(args.model, processor)
    detector = VLMSafetyNeuronDetector(model, processor, config, thinking_mode=thinking)
    print(f"Model {args.model} ({hf_repo}): {detector.extractor.num_layers} layers, "
          f"components {detector.components}, enable_thinking={thinking}")

    run_config = {
        **settings, "n_rounds": n_rounds, "enable_thinking": thinking,
        "n_layers": detector.extractor.num_layers, "components": detector.components,
        "detection": vars(config), "n_detection_harmful": len(det_harmful),
        "n_detection_safe": len(det_safe), "timestamp": timestamp,
    }
    if args.start_round == 1:  # a resumed run keeps the config of the run it continues
        with open(output_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    all_confirmed = {}
    round_summaries = []
    if args.start_round > 1:
        for prev in range(1, args.start_round):
            prev_neurons = load_neurons(output_dir / f"round_{prev}" / "stage2_confirmed.json")
            all_confirmed.update(prev_neurons)
        install_permanent_hooks(detector.extractor, all_confirmed)
        print(f"Resumed: {len(all_confirmed)} neurons clamped")
        summary_path = output_dir / "iterative_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                round_summaries = [r for r in json.load(f).get("rounds", []) if r["round"] < args.start_round]

    for rnd in range(args.start_round, n_rounds + 1):
        print(f"\n{'=' * 60}\nROUND {rnd}/{n_rounds}: {len(all_confirmed)} neurons clamped\n{'=' * 60}")
        round_dir = output_dir / f"round_{rnd}"
        round_dir.mkdir(exist_ok=True)
        det_queries = det_harmful + det_safe

        # 1. generation under the current clamps (responses are saved before judging)
        t0 = time.time()
        fingerprint = {"model_key": args.model, "hf_repo": hf_repo, "enable_thinking": thinking,
                       "max_new_tokens": config.max_new_tokens, "thinking_budget": config.thinking_budget,
                       "clamps": clamp_list(all_confirmed),
                       "queries": [[q.text, q.label] for q in det_queries]}
        records = load_generations(round_dir, fingerprint) if rnd == args.start_round else None
        if records is not None:
            print(f"  Reusing the responses in {round_dir} (delete generation_done.json to generate again)")
        elif args.n_gpus > 1:
            from parallel_generation import parallel_generation_records
            clamps = [{"layer": l, "component": c, "index": i, "patch_value": n.metadata["patch_value"]}
                      for (l, c, i), n in all_confirmed.items()]
            del detector, model
            gc.collect()
            torch.cuda.empty_cache()
            records = parallel_generation_records(args.model, det_queries, clamps,
                                                  args.n_gpus, config.max_new_tokens,
                                                  config.thinking_budget)
            model = load_model(args.model, processor)
            detector = VLMSafetyNeuronDetector(model, processor, config, thinking_mode=thinking)
            install_permanent_hooks(detector.extractor, all_confirmed)
            save_generations(round_dir, fingerprint, records)
        else:
            records = generation_records(detector.extractor, det_queries, config.max_new_tokens,
                                         desc=f"R{rnd} generate")
            save_generations(round_dir, fingerprint, records)
        generate_time = time.time() - t0

        # 2. behavioral labels
        labels = behavioral_labels(records, args.use_judge, args.judge_deployment, desc=f"R{rnd}")
        stats = None
        if args.use_judge:
            write_jsonl(round_dir / "per_sample_judged.jsonl", records)
            stats = judge_stats(records, labels)
            with open(round_dir / "label_comparison.json", "w") as f:
                json.dump(stats, f, indent=2)
        positives = [q for q, l in zip(det_queries, labels) if l == 1]
        negatives = [q for q, l in zip(det_queries, labels) if l == 0]
        print(f"  Positives (harmful refused): {len(positives)}/{len(det_harmful)}; "
              f"negatives (safe complied): {len(negatives)}/{len(det_safe)}")
        if len(positives) < config.min_positives:
            print(f"  Fewer than {config.min_positives} behavioral positives: stopping.")
            round_summaries.append({"round": rnd, "status": "stopped", "n_positives": len(positives)})
            break
        min_negatives = int(np.ceil(2 / config.probe_holdout_ratio))
        if len(negatives) < min_negatives:
            sys.exit(f"  Fewer than {min_negatives} behavioral negatives (safe queries answered): "
                             "the probe's validation split would hold fewer than 2 of them")

        # 3. Stage 1: behavioral probing
        t1 = time.time()
        candidates = detector.detect_linear_probe(positives, negatives, exclude=all_confirmed)
        probe_time = time.time() - t1
        save_neurons(candidates, round_dir / "stage1_candidates.json")

        # 4. Stage 2: activation patching
        t2 = time.time()
        confirmed = detector.detect_activation_patch(
            candidates, positives, detector.heldout_safe_queries,
            n_queries=config.patch_n_queries, exclude=all_confirmed)
        patch_time = time.time() - t2
        save_neurons(confirmed, round_dir / "stage2_confirmed.json")
        with open(round_dir / "patching_results.json", "w") as f:
            json.dump(detector.last_patch_results, f, indent=1)

        all_confirmed.update(confirmed)
        install_permanent_hooks(detector.extractor, confirmed)

        summary = {
            "round": rnd, "status": "completed",
            "n_positives": len(positives), "n_negatives": len(negatives),
            "n_heldout_safe": len(detector.heldout_safe_queries),
            "stage1_candidates": len(candidates), "stage2_confirmed": len(confirmed),
            "total_confirmed": len(all_confirmed), "label_method": "kw+judge" if args.use_judge else "kw",
            "generate_time_s": round(generate_time, 1), "probe_time_s": round(probe_time, 1),
            "patch_time_s": round(patch_time, 1),
        }
        if stats:
            summary["judge"] = stats
        round_summaries.append(summary)
        with open(round_dir / "round_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        save_iterative_summary(output_dir, all_confirmed, round_summaries, args.model, hf_repo)

    save_iterative_summary(output_dir, all_confirmed, round_summaries, args.model, hf_repo)
    print(f"\nDONE: {len(all_confirmed)} neurons over {len(round_summaries)} rounds -> {output_dir}")


if __name__ == "__main__":
    main()
