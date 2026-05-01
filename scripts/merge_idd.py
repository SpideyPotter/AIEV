#!/usr/bin/env python3
"""Merge IDD Segmentation Part I and Part II into a single dataset root.

Both parts share the same on-disk layout:

    <part>/
        gtFine/{train,val,test}/<scene_id>/<*.png|*.json>
        leftImg8bit/{train,val,test}/<scene_id>/<*.png>

Scene IDs are disjoint across parts, so we can place files from both parts
side-by-side under one root that `createLabels.py --datadir` can consume.

Hardlinks (default) keep the merged tree free of extra disk usage. Pass
``--copy`` to force regular copies.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


def iter_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def link_one(src: Path, dst: Path, use_copy: bool) -> str:
    if dst.exists():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not use_copy:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_parts = [
        repo_root / "data" / "IDD_Segmentation",
        repo_root / "data" / "idd20kII",
    ]
    default_dest = repo_root / "data" / "idd_full"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", nargs="+", type=Path, default=default_parts,
                   help="Source dataset parts (default: IDD_Segmentation + idd20kII)")
    p.add_argument("--dest", type=Path, default=default_dest,
                   help="Merged dataset root (default: data/idd_full)")
    p.add_argument("--copy", action="store_true",
                   help="Always copy (default: hardlink, fall back to copy)")
    args = p.parse_args()

    for part in args.parts:
        if not part.exists():
            sys.exit(f"Source missing: {part}")

    args.dest.mkdir(parents=True, exist_ok=True)
    counts = {"linked": 0, "copied": 0, "skipped": 0}

    for part in args.parts:
        files = list(iter_files(part))
        for src in tqdm(files, desc=f"Merging {part.name}"):
            rel = src.relative_to(part)
            dst = args.dest / rel
            counts[link_one(src, dst, args.copy)] += 1

    print(
        f"Done: {counts['linked']} linked, {counts['copied']} copied, "
        f"{counts['skipped']} skipped (already present). Output: {args.dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
