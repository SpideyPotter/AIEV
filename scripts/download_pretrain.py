#!/usr/bin/env python3
"""Download Mask2Former pretrained checkpoints.

Default target: the Cityscapes-panoptic Swin-L checkpoint we fine-tune from.

Usage:
    python scripts/download_pretrain.py
    python scripts/download_pretrain.py --name swinl_cityscapes_panoptic --output-dir pretrain/
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    # Mask2Former Swin-L IN21k 384, Cityscapes panoptic, 90k iters, bs=16
    "swinl_cityscapes_panoptic": (
        "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/"
        "maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl"
    ),
    # COCO panoptic version (alternative warm-start; different domain so usually worse)
    "swinl_coco_panoptic": (
        "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/"
        "maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl"
    ),
}


def download(url: str, dest: Path) -> None:
    print(f"  GET  {url}")
    print(f"  →    {dest}")
    last_pct = -1

    def _hook(blocks: int, blocksize: int, total: int) -> None:
        nonlocal last_pct
        if total <= 0:
            return
        pct = min(100, int(blocks * blocksize * 100 / total))
        if pct != last_pct and pct % 5 == 0:
            print(f"  ... {pct}%", flush=True)
            last_pct = pct

    urllib.request.urlretrieve(url, dest, reporthook=_hook)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="swinl_cityscapes_panoptic",
                   choices=list(CHECKPOINTS))
    p.add_argument("--output-dir", default="/workspace/FinalTry/pretrain")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.name}.pkl"

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"already exists: {dest} ({size_mb:.1f} MB)")
        return 0

    try:
        download(CHECKPOINTS[args.name], dest)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return 1
    print(f"OK ({dest.stat().st_size / 1e6:.1f} MB) → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
