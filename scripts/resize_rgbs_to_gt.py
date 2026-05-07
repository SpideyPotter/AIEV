"""Resize all leftImg8bit RGBs to match the panoptic GT resolution.

Background: the panoptic GT we generated is at 1280x720 (AutoNUE leaderboard
canonical size), but the RGBs transferred from the source IDD release are at
mixed native sizes — 1920x1080, 1280x964, 1280x600, 1296x1032, 1280x720.
Detectron2's check_image_size raises if RGB dims don't match the JSON-declared
dims (which are the GT dims for our generation pipeline).

This script overwrites RGBs in-place, resizing to the per-image dims declared
in the panoptic JSON. Backs up nothing (data is on hostPath, re-rsync from
laptop if needed).

Run inside the pod:
    source /workspace/.local/activate.sh
    python3 scripts/resize_rgbs_to_gt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

DATA = Path("/workspace/data/idd_full")


def resolve_rgb(panoptic_filename: str, image_root: Path) -> Path | None:
    stem = panoptic_filename.replace("_gtFine_panopticlevel3Ids.png", "")
    scene, _, sid = stem.partition("_")
    base = image_root / scene / f"{sid}_leftImg8bit"
    for ext in (".jpg", ".png"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def process(split: str) -> tuple[int, int, int]:
    j = json.loads((DATA / "gtFine" / f"{split}_panoptic.json").read_text())
    image_root = DATA / "leftImg8bit" / split

    n_total = len(j["images"])
    n_ok = n_resized = n_missing = 0

    for i, im in enumerate(j["images"]):
        if i % 500 == 0:
            print(f"  [{split}] {i}/{n_total}  ok={n_ok} resized={n_resized} missing={n_missing}",
                  flush=True)
        rgb = resolve_rgb(im["file_name"], image_root)
        if rgb is None:
            n_missing += 1
            continue
        target = (int(im["width"]), int(im["height"]))
        with Image.open(rgb) as img:
            if img.size == target:
                n_ok += 1
                continue
            resized = img.convert("RGB").resize(target, Image.BILINEAR)
        # overwrite in place, keep original extension
        resized.save(rgb, quality=92 if rgb.suffix == ".jpg" else None)
        n_resized += 1

    print(f"  [{split}] DONE  total={n_total}  ok={n_ok}  resized={n_resized}  missing={n_missing}")
    return n_ok, n_resized, n_missing


def main() -> int:
    total_missing = 0
    for split in ("train", "val"):
        print(f"\n=== {split} ===")
        _, _, miss = process(split)
        total_missing += miss
    return 0 if total_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
