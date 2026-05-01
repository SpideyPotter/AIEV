"""Smoke-test: confirm IDD-panoptic dataset registration loads correctly.

Runs without GPU. Useful before paying for compute time to ensure:
  - the panoptic JSONs / PNGs exist where the registration expects them,
  - RGB image paths derive correctly from panoptic filenames,
  - metadata (categories, thing/stuff split, contiguous ID maps) looks sane.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402
register_idd_panoptic(REPO)

from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402


def check(name: str) -> bool:
    print(f"\n=== {name} ===")
    recs = DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    print(f"  records:        {len(recs)}")
    print(f"  thing classes:  {len(meta.thing_classes)} → {meta.thing_classes}")
    print(f"  stuff classes:  {len(meta.stuff_classes)}")
    print(f"  thing ID map:   {meta.thing_dataset_id_to_contiguous_id}")
    print(f"  ignore_label:   {meta.ignore_label}")
    print(f"  evaluator:      {meta.evaluator_type}")

    r = recs[0]
    print(f"  sample[0]:")
    print(f"    file_name:         {r['file_name']}")
    print(f"    pan_seg_file_name: {r['pan_seg_file_name']}")
    print(f"    {len(r['segments_info'])} segs, "
          f"first={r['segments_info'][0] if r['segments_info'] else '(empty)'}")

    ok = True
    for k in ("file_name", "pan_seg_file_name"):
        if not Path(r[k]).exists():
            print(f"  MISSING FILE: {r[k]}")
            ok = False
    print(f"  files exist: {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok_t = check("idd_panoptic_train")
    ok_v = check("idd_panoptic_val")
    print()
    print("PASS" if ok_t and ok_v else "FAIL")
    return 0 if ok_t and ok_v else 1


if __name__ == "__main__":
    raise SystemExit(main())
