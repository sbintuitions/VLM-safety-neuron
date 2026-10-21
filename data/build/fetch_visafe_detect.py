"""Materialize ViSafe-Detect from the released manifest.

Reads data/visafe_detect/visafe_detect.jsonl and writes
  data/visafe_detect/images/<source>/<source_index:05d>.jpg
  data/visafe_detect/labels.jsonl                (label file read by the visual pipeline)
which is the layout src/run_stage1_screening.py and src/run_iterative_visual.py expect.

Every row points to a row of a public dataset (for VLGuard, a file in its archive).
HoliSafe, UnsafeBench and VLGuard are gated and need an approved Hugging Face token
(HF_TOKEN or `hf auth login`).
The output is pixel-identical to the images we used (tested with Pillow 12.1).

Usage:
  python data/build/fetch_visafe_detect.py
  python data/build/fetch_visafe_detect.py --sources hades,unsafebench   # some sources only
The label file lists every manifest row whose image is on disk.
"""

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data" / "visafe_detect"
MANIFEST = DATA / "visafe_detect.jsonl"
IMAGE_DIR = DATA / "images"
LABEL_FILE = DATA / "labels.jsonl"


MAX_SIDE = 512


def save(img, path, crop_square_top=False):
    img = img.convert("RGB")
    if crop_square_top:
        w, h = img.size
        if h > w:
            img = img.crop((0, 0, w, w))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)  # downscale only, keeps aspect ratio
    tmp = path.with_name(path.name + ".part")  # renamed when complete: an interrupted run leaves no file
    img.save(tmp, "JPEG", quality=70)
    os.replace(tmp, path)


def fetch_hf_rows(rows):
    from datasets import load_dataset
    groups = defaultdict(list)
    for r in rows:
        im = r["image"]
        groups[(im["hf_repo"], im["hf_config"], im["hf_split"])].append(r)
    for (repo, cfg, split), rs in sorted(groups.items(), key=lambda x: str(x[0])):
        todo = [r for r in rs if not r["_path"].exists()]
        if not todo:
            continue
        if todo[0]["image"].get("streaming"):  # ImageNet: stream up to the last needed row
            by_row = defaultdict(list)
            for r in todo:
                by_row[r["image"]["hf_row"]].append(r)
            last = max(by_row)
            for i, row in enumerate(load_dataset(repo, split=split, streaming=True)):
                for r in by_row.get(i, []):
                    save(row["image"], r["_path"])
                if i >= last:
                    break
        else:
            ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
            for r in todo:
                save(ds[r["image"]["hf_row"]]["image"], r["_path"], r["image"].get("crop_square_top", False))
        print(f"  {' '.join(x for x in (repo, cfg, split) if x)}: {len(todo)} images")


def fetch_hf_zip(rows):
    """Files inside a zip archive of a Hugging Face dataset (VLGuard)."""
    from huggingface_hub import hf_hub_download
    groups = defaultdict(list)
    for r in rows:
        if not r["_path"].exists():
            groups[(r["image"]["hf_repo"], r["image"]["zip_file"])].append(r)
    for (repo, zip_file), todo in sorted(groups.items()):
        zf = zipfile.ZipFile(hf_hub_download(repo, zip_file, repo_type="dataset"))
        for r in todo:
            save(Image.open(io.BytesIO(zf.read(r["image"]["zip_member"]))), r["_path"])
        print(f"  {repo} {zip_file}: {len(todo)} images")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=None,
                    help="comma-separated subset of: beavertails_v, hades, holisafe, safe, unsafebench, vlguard")
    args = ap.parse_args()
    with open(MANIFEST) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    wanted = set(args.sources.split(",")) if args.sources else None
    if wanted and wanted - {r["source"] for r in rows}:
        ap.error(f"unknown source(s): {', '.join(sorted(wanted - {r['source'] for r in rows}))}")
    seen = set()
    for r in rows:
        r["_path"] = IMAGE_DIR / r["source"] / f"{r['source_index']:05d}.jpg"
    unique = []
    for r in rows:
        key = (r["source"], r["source_index"])
        if key in seen or (wanted and r["source"] not in wanted):
            continue
        seen.add(key)
        unique.append(r)
    fetch_hf_rows([r for r in unique if r["image"]["kind"] == "hf"])
    fetch_hf_zip([r for r in unique if r["image"]["kind"] == "hf_zip"])

    keep = ("source", "source_index", "label", "source_category", "harm_category")
    # every manifest row whose image is on disk (so fetching sources one at a time accumulates)
    label_rows = [r for r in rows if r["_path"].exists()]
    with open(LABEL_FILE, "w") as f:
        for r in label_rows:
            f.write(json.dumps({k: r[k] for k in keep}) + "\n")
    print(f"Wrote {LABEL_FILE} ({len(label_rows)} of {len(rows)} rows; "
          f"{len(rows) - len(label_rows)} without a local image are left out)")
    if len(label_rows) < len(rows):
        print("WARNING: the label file is incomplete; fetch the remaining sources before running "
              "the full visual detection")


if __name__ == "__main__":
    main()
