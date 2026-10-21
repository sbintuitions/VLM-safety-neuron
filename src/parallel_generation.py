"""Data-parallel generation for the text pipeline (one model copy per GPU).

Each worker loads the model on its own GPU, installs the clamps of the neurons confirmed in
earlier rounds and generates the responses for its share of the queries. Records come back
in the original query order, in the format of run_iterative_text.generation_records.
"""

import json
import shutil
import tempfile
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp

from data_utils import SafetyQuery


def _worker(rank, model_key, query_data, indices, clamps, max_new_tokens, thinking_budget,
            result_dir):
    torch.cuda.set_device(rank)
    try:
        from neuron_detection_vlm import VLMActivationExtractor
        from refusal import thinking_setting
        from run_iterative_text import generation_records
        from text_models import load_model, load_processor

        processor = load_processor(model_key)
        model = load_model(model_key, processor, device_map={"": rank})
        extractor = VLMActivationExtractor(model, processor, thinking_mode=thinking_setting(model_key),
                                           thinking_budget=thinking_budget)
        for c in clamps:
            extractor.add_permanent_hook(c["layer"], c["component"], c["index"], c["patch_value"])
        queries = [SafetyQuery(text=q["text"], image=None, label=q["label"], metadata=q["metadata"])
                   for q in query_data]
        records = generation_records(extractor, queries, max_new_tokens, desc=f"GPU {rank}")
        with open(Path(result_dir) / f"worker_{rank}.json", "w") as f:
            json.dump({"indices": indices, "records": records}, f, ensure_ascii=False)
    except Exception as e:
        (Path(result_dir) / f"worker_{rank}_error.txt").write_text(f"{e}\n{traceback.format_exc()}")
        raise


def parallel_generation_records(model_key, queries, clamps, n_gpus, max_new_tokens,
                                thinking_budget=1024):
    """Round-robin the queries over n_gpus worker processes and merge their records."""
    chunks = [[] for _ in range(n_gpus)]
    indices = [[] for _ in range(n_gpus)]
    for i, q in enumerate(queries):
        chunks[i % n_gpus].append({"text": q.text, "label": q.label, "metadata": q.metadata})
        indices[i % n_gpus].append(i)

    result_dir = tempfile.mkdtemp(prefix="parallel_gen_")
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(n_gpus):
        p = mp.Process(target=_worker, args=(rank, model_key, chunks[rank], indices[rank],
                                             clamps, max_new_tokens, thinking_budget,
                                             result_dir))
        p.start()
        procs.append(p)
    errors = []
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            err = Path(result_dir) / f"worker_{rank}_error.txt"
            errors.append(f"worker {rank}: {err.read_text() if err.exists() else p.exitcode}")
    if errors:
        raise RuntimeError("Parallel generation failed:\n" + "\n".join(errors))

    merged = [None] * len(queries)
    for rank in range(n_gpus):
        with open(Path(result_dir) / f"worker_{rank}.json") as f:
            data = json.load(f)
        for i, rec in zip(data["indices"], data["records"]):
            merged[i] = rec
    shutil.rmtree(result_dir, ignore_errors=True)
    assert all(r is not None for r in merged)
    return merged
