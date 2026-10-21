"""Load the released text-detection query set (data/text/text_detection.jsonl).

Each manifest row names a row of a public Hugging Face dataset. The query text is
rebuilt from that row (HarmBench contextual prompts carry their context, Alpaca
instructions their input), and the stored SHA-1 prefix is checked so that a changed
upstream dataset is caught.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

from data_utils import SafetyQuery


def _text(repo: str, config, row: dict) -> str:
    if repo == "tatsu-lab/alpaca":
        return f"{row['instruction']}\n{row['input']}" if row.get("input") else row["instruction"]
    if repo == "walledai/HarmBench" and config == "contextual" and row.get("context"):
        return f"Context: {row['context']}\n\n{row['prompt']}"
    return row["prompt"]


def load_text_manifest(path: str) -> Tuple[List[SafetyQuery], List[SafetyQuery]]:
    with open(Path(path)) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    cache = {}
    harmful, safe, mismatched = [], [], 0
    for r in rows:
        key = (r["hf_repo"], r["hf_config"], r["hf_split"])
        if key not in cache:
            repo, cfg, split = key
            cache[key] = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
        text = _text(r["hf_repo"], r["hf_config"], cache[key][r["hf_row"]])
        if hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] != r["text_sha1_16"]:
            mismatched += 1
        q = SafetyQuery(text=text, label=r["label"], metadata={"source": r["source"]})
        (harmful if r["label"] == "harmful" else safe).append(q)
    if mismatched:
        print(f"WARNING: {mismatched}/{len(rows)} text queries differ from the upstream text")
    print(f"Loaded text manifest: {len(harmful)} harmful, {len(safe)} safe")
    return harmful, safe
