"""Visual safety-neuron detection with iterative ablation (paper Sec. 3, ViSafe-Detect).

Every ViSafe-Detect image is paired with the neutral procedural query. Round k runs on
f_theta^(k), the model with all neurons confirmed in rounds < k clamped to their safe mean:
  1. Prefill (activations at the last input token) and generation (max_new_tokens 512)
     under the clamps; keyword labels on the response
  2. GPT-4o re-judges keyword-COMPLY responses (--no-judge: keyword labels only)
  3. Stage 1: behavioral probes over mlp_gate, mlp_up, mlp_out and self_attn -> candidates
     (|z| > 3.0, earlier confirmed neurons excluded)
  4. Stage 2: mean-activation patching (top-500 candidates, 300 control neurons, BH-FDR)
     with safe means taken over the held-out safe images of the probe split
The loop stops after 3 rounds or when a round has fewer than 20 behavioral positives.

The activations and responses of a round are saved before they are judged. If a run stops in
round k, rerunning the same command with --start-round k reuses them when the model, the
images and the clamps are unchanged.

Usage:
    python src/run_iterative_visual.py --model qwen3vl-8b --output-dir results/visual/qwen3vl-8b
    # reuse the round-1 activations and generations of an earlier run:
    python src/run_iterative_visual.py --model qwen3vl-8b \\
        --reuse-round1 <earlier output dir>/round_1 --output-dir results/visual/qwen3vl-8b
"""

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from config import MODEL_REGISTRY, DetectionConfig, load_pipeline_config
from judge import JUDGE_DEPLOYMENT, behavioral_labels, judge_stats
from neuron_detection_vlm import (
    VLMSafetyNeuronDetector, check_resume, clamp_list, install_permanent_hooks, load_neurons,
    read_jsonl, repo_path, save_iterative_summary, save_neurons, write_jsonl,
)
from neuron_types import SafetyNeuron
from refusal import thinking_setting
from run_stage1_screening import (
    DEFAULT_LABEL_FILE,
    build_valid_keys,
    filter_harmful_labels,
    filter_safe_labels,
    generate_and_extract,
    get_layer_list,
    load_images_for_labels,
    load_labels,
    load_model,
    load_registry,
    load_safe_images_for_screening,
    run_probes_from_disk,
)


def judge_round(round_dir: Path, deployment: str) -> dict:
    """GPT-4o re-judging of keyword-COMPLY responses; writes labels_kwjudge.npy."""
    records = read_jsonl(round_dir / "per_sample" / "generate.jsonl")
    labels = behavioral_labels(records, True, deployment, desc=round_dir.name)
    np.save(round_dir / "labels_kwjudge.npy", np.array(labels, dtype=np.int8))
    write_jsonl(round_dir / "per_sample_judged.jsonl", records)
    stats = judge_stats(records, labels)
    with open(round_dir / "label_comparison.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def saved_generation_matches(round_dir: Path, fingerprint: dict) -> bool:
    """True if round_dir holds a complete generation pass made under the same fingerprint."""
    marker = round_dir / "generation_done.json"
    if not (marker.exists() and (round_dir / "activations").is_dir()
            and (round_dir / "per_sample" / "generate.jsonl").exists()):
        return False
    with open(marker) as f:
        return json.load(f) == fingerprint


def reuse_round1_data(source_dir: Path, target_dir: Path):
    """Link the activations and copy the generations of an earlier round-1 run, replacing
    whatever an earlier attempt left in target_dir."""
    if target_dir.exists() and source_dir.resolve() == target_dir.resolve():
        return  # reusing this output's own round 1: the files are already in place
    target_dir.mkdir(parents=True, exist_ok=True)
    act_dst = target_dir / "activations"
    if act_dst.is_symlink():
        act_dst.unlink()
    elif act_dst.exists():
        shutil.rmtree(act_dst)
    os.symlink(str((source_dir / "activations").resolve()), str(act_dst))
    if (target_dir / "per_sample").exists():
        shutil.rmtree(target_dir / "per_sample")
    shutil.copytree(str(source_dir / "per_sample"), str(target_dir / "per_sample"))
    for fname in ["labels.npy", "group_labels.npy", "generation_stats.json"]:
        if (source_dir / fname).exists():
            shutil.copy2(str(source_dir / fname), str(target_dir / fname))


def main():
    parser = argparse.ArgumentParser(description="Iterative visual safety-neuron detection")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY),
                        help="Model key from configs/model_registry.yaml")
    parser.add_argument("--n-rounds", type=int, default=None, help="Number of rounds (default: 3, from configs/pipeline.yaml)")
    parser.add_argument("--label-file", type=str, default=None,
                        help="ViSafe-Detect label JSONL (default: data/visafe_detect/labels.jsonl)")
    parser.add_argument("--prefill-bs", dest="gen_bs", type=int, default=4,
                        help="Prefill batch size, at most 8 (default: 4; generation runs one sample at a time)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/visual/<model>)")
    parser.add_argument("--start-round", type=int, default=1,
                        help="Resume from this round (earlier rounds are read from --output-dir)")
    parser.add_argument("--n-harmful", type=int, default=None, help="Cap on harmful images (default: all)")
    parser.add_argument("--n-safe", type=int, default=None, help="Cap on safe images (default: all)")
    parser.add_argument("--no-judge", dest="use_judge", action="store_false",
                        help="Keyword labels only (default: GPT-4o re-judges keyword-COMPLY responses)")
    parser.add_argument("--judge-deployment", type=str, default=JUDGE_DEPLOYMENT,
                        help="Azure OpenAI deployment name of GPT-4o")
    parser.add_argument("--reuse-round1", type=str, default=None,
                        help="round_1 directory of an earlier run whose activations and generations "
                             "are reused (the judge and both stages are re-run)")
    args = parser.parse_args()
    if args.use_judge:
        from judge import preflight_judge
        preflight_judge(args.judge_deployment)

    registry = load_registry()
    pipeline_cfg = load_pipeline_config()
    model_cfg = registry["models"][args.model]
    hf_repo = model_cfg["hf_repo"]
    query_template = pipeline_cfg["data"]["harmful_image"]["query_template"]
    max_side = pipeline_cfg["image"]["max_side"]
    config = DetectionConfig.from_pipeline_yaml(pipeline_cfg)
    config.seed = args.seed
    if args.n_rounds:
        config.n_rounds = args.n_rounds
    n_rounds = config.n_rounds
    thinking = thinking_setting(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.output_dir) if args.output_dir
                  else REPO_ROOT / "results" / "visual" / args.model)

    reuse_path = None
    if args.reuse_round1:
        reuse_path = Path(args.reuse_round1)
        for rf in [reuse_path / "activations", reuse_path / "per_sample" / "generate.jsonl"]:
            if not rf.exists():
                sys.exit(f"--reuse-round1: missing {rf}")
        src_cfg = reuse_path.parent / "config.json"
        src = {}
        if src_cfg.exists():
            with open(src_cfg) as f:
                src = json.load(f)
        if src.get("hf_repo") not in (None, hf_repo) or src.get("model_key") not in (None, args.model):
            sys.exit(f"--reuse-round1: {reuse_path} was run with a different model key")

    # ---- data ----
    label_path = Path(args.label_file) if args.label_file else DEFAULT_LABEL_FILE
    if not label_path.exists():
        sys.exit(f"{label_path} not found: run data/build/fetch_visafe_detect.py first")
    all_labels = load_labels(label_path)
    harmful_labels = filter_harmful_labels(all_labels)
    safe_labels = filter_safe_labels(all_labels)
    rng = random.Random(args.seed)
    rng.shuffle(harmful_labels)
    rng.shuffle(safe_labels)
    harmful_labels = harmful_labels[:args.n_harmful or len(harmful_labels)]
    safe_labels = safe_labels[:args.n_safe or len(safe_labels)]
    harmful_queries = load_images_for_labels(harmful_labels, query_template, max_side)
    safe_queries = load_safe_images_for_screening(safe_labels, query_template, max_side)
    n_harmful = len(harmful_queries)
    print(f"ViSafe-Detect: {n_harmful} harmful, {len(safe_queries)} safe images")
    if not harmful_queries or not safe_queries:
        sys.exit("ViSafe-Detect needs both harmful and safe images; "
                         "run data/build/fetch_visafe_detect.py first")
    del all_labels, harmful_labels, safe_labels
    gc.collect()

    settings = {"model_key": args.model, "hf_repo": hf_repo, "seed": args.seed,
                "label_file": repo_path(label_path), "n_harmful_images": n_harmful,
                "n_safe_images": len(safe_queries), "use_judge": args.use_judge}
    if args.start_round > 1:
        check_resume(output_dir, args.start_round, settings, "stage2_confirmed.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    model, processor = load_model(hf_repo, args.model)
    detector = VLMSafetyNeuronDetector(model, processor, config, thinking_mode=thinking)
    extractor = detector.extractor
    components = detector.components
    valid_keys = build_valid_keys(model)
    print(f"Model {args.model} ({hf_repo}): {extractor.num_layers} layers, components {components}, "
          f"enable_thinking={thinking}")

    run_config = {
        **settings, "n_rounds": n_rounds, "enable_thinking": thinking,
        "n_layers": extractor.num_layers, "components": components, "detection": vars(config),
        "reuse_round1": str(reuse_path) if reuse_path else None, "timestamp": timestamp,
    }
    if args.start_round == 1:  # a resumed run keeps the config of the run it continues
        with open(output_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    all_confirmed: Dict[Tuple, SafetyNeuron] = {}
    round_summaries = []
    if args.start_round > 1:
        for prev in range(1, args.start_round):
            prev_neurons = load_neurons(output_dir / f"round_{prev}" / "stage2_confirmed.json")
            all_confirmed.update(prev_neurons)
        install_permanent_hooks(extractor, all_confirmed)
        print(f"Resumed: {len(all_confirmed)} neurons clamped")
        summary_path = output_dir / "iterative_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                round_summaries = [r for r in json.load(f).get("rounds", []) if r["round"] < args.start_round]

    for rnd in range(args.start_round, n_rounds + 1):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"\n{'=' * 60}\nROUND {rnd}/{n_rounds}: {len(all_confirmed)} neurons clamped\n{'=' * 60}")
        round_dir = output_dir / f"round_{rnd}"
        round_dir.mkdir(exist_ok=True)

        # 1. prefill + generation under the current clamps
        t0 = time.time()
        samples = [[(q.metadata or {}).get("source"), (q.metadata or {}).get("source_index")]
                   for q in harmful_queries + safe_queries]
        fingerprint = {"model_key": args.model, "hf_repo": hf_repo, "enable_thinking": thinking,
                       "max_new_tokens": config.max_new_tokens, "thinking_budget": config.thinking_budget,
                       "clamps": clamp_list(all_confirmed), "samples": samples}
        if rnd == 1 and reuse_path is not None:
            reuse_round1_data(reuse_path, round_dir)
            reused = read_jsonl(round_dir / "per_sample" / "generate.jsonl")
            current = [((q.metadata or {}).get("source"), (q.metadata or {}).get("source_index"))
                       for q in harmful_queries + safe_queries]
            if [(r["source"], r["source_index"]) for r in reused] != current:
                sys.exit(f"--reuse-round1: {reuse_path} was run on a different sample list")
        elif rnd == args.start_round and saved_generation_matches(round_dir, fingerprint):
            print(f"  Reusing the activations and responses in {round_dir} "
                  "(delete generation_done.json to generate again)")
        else:
            (round_dir / "generation_done.json").unlink(missing_ok=True)
            generate_and_extract(model, processor, extractor, harmful_queries, safe_queries,
                                 components, round_dir, max_new_tokens=config.max_new_tokens,
                                 valid_keys=valid_keys, gen_batch_size=args.gen_bs)
            with open(round_dir / "generation_done.json", "w") as f:
                json.dump(fingerprint, f)
        generate_time = time.time() - t0

        # 2. behavioral labels
        stats = None
        if args.use_judge:
            stats = judge_round(round_dir, args.judge_deployment)
            labels = np.load(round_dir / "labels_kwjudge.npy")
        else:
            labels = np.load(round_dir / "labels.npy")
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        print(f"  Positives (harmful refused): {n_pos}/{n_harmful}; "
              f"negatives (safe complied): {n_neg}/{len(safe_queries)}")
        if n_pos < config.min_positives:
            print(f"  Fewer than {config.min_positives} behavioral positives: stopping.")
            round_summaries.append({"round": rnd, "status": "stopped", "n_positives": n_pos})
            break
        min_negatives = int(np.ceil(2 / config.probe_holdout_ratio))
        if n_neg < min_negatives:
            sys.exit(f"  Fewer than {min_negatives} behavioral negatives (safe images answered): "
                     "the probe's validation split would hold fewer than 2 of them")

        # 3. Stage 1: behavioral probing
        valid_mask = labels >= 0
        t1 = time.time()
        candidates, diag = run_probes_from_disk(
            act_dirs=[round_dir / "activations"], labels=labels[valid_mask].astype(np.float64),
            layer_list=get_layer_list(round_dir / "activations"),
            z_threshold=config.probe_z_threshold, holdout_ratio=config.probe_holdout_ratio,
            probe_C=config.probe_C, max_iter=config.probe_max_iter, seed=args.seed,
            valid_mask=valid_mask)
        candidates = {k: v for k, v in candidates.items() if k not in all_confirmed}
        probe_time = time.time() - t1
        save_neurons(candidates, round_dir / "stage1_candidates.json")
        with open(round_dir / "probe_diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)

        # 4. Stage 2: activation patching
        positives = [harmful_queries[i] for i in range(n_harmful) if labels[i] == 1]
        heldout_safe = [safe_queries[r - n_harmful] for r in diag["val_rows"]
                        if r >= n_harmful and labels[r] == 0]
        t2 = time.time()
        confirmed = detector.detect_activation_patch(
            candidates, positives, heldout_safe, n_queries=config.patch_n_queries, exclude=all_confirmed)
        patch_time = time.time() - t2
        save_neurons(confirmed, round_dir / "stage2_confirmed.json")
        with open(round_dir / "patching_results.json", "w") as f:
            json.dump(detector.last_patch_results, f, indent=1)

        all_confirmed.update(confirmed)
        install_permanent_hooks(extractor, confirmed)

        summary = {
            "round": rnd, "status": "completed",
            "n_positives": n_pos, "n_negatives": n_neg, "n_heldout_safe": len(heldout_safe),
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
    by_comp = defaultdict(int)
    for (_, comp, _) in all_confirmed:
        by_comp[comp] += 1
    print(f"\nDONE: {len(all_confirmed)} neurons over {len(round_summaries)} rounds "
          f"({dict(by_comp)}) -> {output_dir}")


if __name__ == "__main__":
    main()
