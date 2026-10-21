"""Materialize ViSafe-Eval from the released manifest.

Reads data/visafe_eval/{intents,visafe_eval}.jsonl and writes
data/visafe_eval/metadata.jsonl plus data/visafe_eval/images/ (one row per sample with
its query, image path relative to the repository root, intent, category and condition).

Sources:
  intent text       MM-SafetyBench Text_only / HarmBench standard rows
  safe queries      databricks-dolly-15k rows
  natural_harmful   MM-SafetyBench SD rows
  rendered*, natural_harmful for HarmBench intents (FLUX.1-schnell)
                    our generated images (visafe_eval_generated_images.zip on Hugging Face,
                    checked against generated_images.sha256)
  blank, noise      generated here (a white image and Gaussian noise with a fixed seed)
  safe              natural photos streamed from the Hugging Face Hub

Usage:
  python data/build/fetch_visafe_eval.py                              # everything
  python data/build/fetch_visafe_eval.py --generated-zip <path|URL>   # use a local copy of the zip

HarmBench and ImageNet (ILSVRC/imagenet-1k) are gated and need an approved Hugging Face token
(HF_TOKEN or `hf auth login`).
"""

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from data.image_utils import generate_blank, generate_gaussian_noise, resize_max_side  # noqa: E402

DATA = REPO_ROOT / "data" / "visafe_eval"
IMAGES = DATA / "images"

with open(REPO_ROOT / "configs" / "pipeline.yaml") as _f:
    PROCEDURAL = yaml.safe_load(_f)["data"]["harmful_image"]["query_template"]  # the ViSafe-Detect query
GENERATED_ZIP_URL = ("https://huggingface.co/datasets/sbintuitions/ViSafe-Eval-generated-internal/resolve/main/"
                     "visafe_eval_generated_images.zip")
GENERATED_SHA = DATA / "generated_images.sha256"


def sha(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def save_max1024(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_jpeg(resize_max_side(img.convert("RGB"), 1024), path)


def save_jpeg(img, path):
    """JPEG quality 95, renamed into place when complete so an interrupted run leaves no file."""
    tmp = path.with_name(path.name + ".part")
    img.save(tmp, "JPEG", quality=95)
    os.replace(tmp, path)


def load_intent_texts(intents):
    from datasets import load_dataset
    cache, texts, mismatched = {}, {}, 0
    for it in intents:
        key = (it["hf_repo"], it["hf_config"], it["hf_split"])
        if key not in cache:
            cache[key] = load_dataset(it["hf_repo"], it["hf_config"], split=it["hf_split"])
        text = cache[key][it["hf_row"]][it["field"]].strip()
        mismatched += sha(text) != it["text_sha1_16"]
        texts[it["intent_id"]] = text
    if mismatched:
        print(f"WARNING: {mismatched}/{len(intents)} intents differ from the upstream text")
    return texts, cache


def fetch_natural(intents, cache):
    """MM-SafetyBench SD images; the FLUX images for HarmBench intents come with the zip."""
    from datasets import load_dataset
    out_dir = IMAGES / "natural_harmful"
    n = 0
    for it in intents:
        spec = it["natural_harmful_image"]
        path = out_dir / f"{it['intent_id']}.jpg"
        if spec["kind"] != "hf" or path.exists():
            continue
        key = (spec["hf_repo"], spec["hf_config"], spec["hf_split"])
        if key not in cache:
            cache[key] = load_dataset(spec["hf_repo"], spec["hf_config"], split=spec["hf_split"])
        save_max1024(cache[key][spec["hf_row"]]["image"], path)
        n += 1
    if n:
        print(f"  natural_harmful: {n} images")


def fetch_generated(src):
    """Unpack our generated images and verify every checksum,
    including those of files already on disk (a file that does not match is unpacked again)."""
    want = dict(reversed(l.split("  ", 1)) for l in GENERATED_SHA.read_text().splitlines())
    todo = {rel: h for rel, h in want.items()
            if not (DATA / rel).exists() or hashlib.sha256((DATA / rel).read_bytes()).hexdigest() != h}
    if not todo:
        return
    if src.startswith(("http://", "https://")):
        print(f"  downloading {src}")
        req = urllib.request.Request(src)
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()  # `hf auth login`
            except ImportError:
                pass
        if token:  # not sent on to the storage host the Hub redirects to
            req.add_unredirected_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as r:
            src = io.BytesIO(r.read())
    with zipfile.ZipFile(src) as z:
        for rel, h in todo.items():
            data = z.read(rel)
            if hashlib.sha256(data).hexdigest() != h:
                sys.exit(f"checksum mismatch: {rel}")
            path = DATA / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
    print(f"  generated images: {len(todo)} files")


def make_blank_noise():
    """The blank and noise images (they are not in the zip)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, make in (("blank", generate_blank), ("noise", lambda: generate_gaussian_noise(seed=42))):
        path = IMAGES / f"{name}.jpg"
        if not path.exists():
            save_jpeg(make(), path)


def fetch_safe(rows):
    from datasets import load_dataset
    want = {}
    for r in rows:
        if r["image_type"] == "safe" and r["image"]["kind"] == "hf":
            im = r["image"]
            want.setdefault((im["hf_repo"], im["hf_split"]), {})[im["hf_row"]] = r["intent_id"]
    for (repo, split), by_row in want.items():
        todo = {i: iid for i, iid in by_row.items() if not (IMAGES / "safe" / f"{iid}.jpg").exists()}
        if not todo:
            continue
        last = max(todo)
        ds = load_dataset(repo, split=split, streaming=True)
        for i, row in enumerate(ds):
            if i in todo:
                save_max1024(row["image"], IMAGES / "safe" / f"{todo[i]}.jpg")
            if i >= last:
                break
        print(f"  safe: {len(todo)} images from {repo}")


def write_metadata(rows, texts):
    from datasets import load_dataset
    cache = {}
    path_for = {
        "blank": lambda r: IMAGES / "blank.jpg",
        "noise": lambda r: IMAGES / "noise.jpg",
        "natural_harmful": lambda r: IMAGES / "natural_harmful" / f"{r['intent_id']}.jpg",
        "safe": lambda r: IMAGES / "safe" / f"{r['intent_id']}.jpg",
    }
    missing = mismatched = n_dolly = 0
    with open(DATA / "metadata.jsonl", "w") as f:
        for r in rows:
            intent = texts[r["intent_id"]]
            q = r["query"]
            if q["kind"] == "procedural_template":
                query = PROCEDURAL
            elif q["kind"] == "intent_text":
                query = intent  # the source dataset's harmful text, verbatim
            else:  # a Dolly instruction
                key = (q["hf_repo"], q["hf_config"], q["hf_split"])
                if key not in cache:
                    cache[key] = load_dataset(q["hf_repo"], q["hf_config"], split=q["hf_split"])
                query = cache[key][q["hf_row"]][q["field"]]
                mismatched += sha(query) != q["text_sha1_16"]
                n_dolly += 1
            t = r["image_type"]
            if t == "text_only":
                image_path = ""
            else:
                p = path_for.get(t, lambda r: IMAGES / t / f"{r['intent_id']}.jpg")(r)
                missing += not p.exists()
                image_path = str(p.relative_to(REPO_ROOT))  # relative to the repository root
            f.write(json.dumps({
                "intent_id": r["intent_id"], "intent_text": intent, "harm_category": r["harm_category"],
                "image_type": t, "query_condition": r["query_condition"], "query_text": query,
                "image_path": image_path,
            }, ensure_ascii=False) + "\n")
    if mismatched:
        print(f"WARNING: {mismatched}/{n_dolly} Dolly queries differ from the upstream text")
    print(f"Wrote {DATA / 'metadata.jsonl'} ({len(rows)} rows, {missing} image files missing)")
    if missing:
        print("WARNING: some images are missing; run the script again to fetch them")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated-zip", default=GENERATED_ZIP_URL,
                    help="local path or URL of visafe_eval_generated_images.zip")
    args = ap.parse_args()
    with open(DATA / "intents.jsonl") as f:
        intents = [json.loads(l) for l in f if l.strip()]
    with open(DATA / "visafe_eval.jsonl") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    texts, cache = load_intent_texts(intents)
    fetch_generated(args.generated_zip)
    make_blank_noise()
    fetch_natural(intents, cache)
    fetch_safe(rows)
    write_metadata(rows, texts)


if __name__ == "__main__":
    main()
