#!/usr/bin/env python3
"""Run *only* stage 2 of the panoptic GT pipeline (the panoptic converter).

Stage 1 (semantic + instance PNG generation) is handled by `createLabels.py`
and writes per-scene `_instancelevel3Ids.png` files next to the polygon JSONs.
Once stage 1 has completed at least once, those PNGs persist on disk, so we
can iterate on stage 2 — calling `panoptic_converter` directly — without
paying the ~7 minute stage 1 cost again.

Use this when:
- stage 1 already ran successfully and we're iterating on a stage 2 fix, or
- we just want to regenerate panoptic JSON / PNGs without re-rasterising.

Run inside the autonue env:
    conda activate autonue
    python scripts/run_panoptic_only.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUB  = REPO / "external" / "public-code"
DATA = REPO / "data" / "idd_full"

sys.path.insert(0, str(PUB / "helpers"))
sys.path.insert(0, str(PUB / "preperation"))

from cityscape_panoptic_gt import panoptic_converter  # noqa: E402


def main(num_workers: int = 4) -> int:
    folder_name = DATA / "gtFine"
    if not folder_name.is_dir():
        sys.exit(f"Missing: {folder_name}. Run prepare_panoptic.sh first.")

    for split in ("train", "val"):
        in_folder  = folder_name / split
        out_folder = folder_name / f"{split}_panoptic"
        out_file   = folder_name / f"{split}_panoptic.json"
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f"=== {split} ===")
        panoptic_converter(num_workers, str(in_folder), str(out_folder), str(out_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
