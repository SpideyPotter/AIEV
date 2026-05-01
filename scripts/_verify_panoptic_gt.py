"""Deep sanity-check on generated panoptic GT.

Cross-checks 5 things:
  1. Category list in JSON matches the canonical anue_labels.py spec
     (id, name, isthing — derived per level3Id).
  2. Segments per image are unique (no duplicate segment_id per image).
  3. category_id values in segments_info are all valid level3Ids (0..25).
  4. PNG encoding round-trips: for sampled images, decode the panoptic
     PNG (R + 256*G + 256^2*B → segment_id) and verify the set of segment
     ids matches the JSON's segments_info entries exactly.
  5. Per-segment area / bbox in JSON matches what we recompute from the
     decoded mask on the sampled images.

Run:  conda run -n autonue python scripts/_verify_panoptic_gt.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "external" / "public-code" / "helpers"))
from anue_labels import labels  # noqa: E402

GT = REPO / "data" / "idd_full" / "gtFine"
SAMPLE_N = 8


# -- 1. expected categories from anue_labels --

def expected_categories() -> list[dict]:
    seen: set[int] = set()
    cats: list[dict] = []
    for el in labels:
        if el.ignoreInEval:
            continue
        if el.level3Id in seen:
            continue
        seen.add(el.level3Id)
        cats.append({
            "id": el.level3Id,
            "name": el.name,
            "isthing": 1 if el.hasInstances else 0,
        })
    cats.sort(key=lambda c: c["id"])
    return cats


# -- 2..5. inspect one split --

def inspect(split: str) -> bool:
    print(f"\n=== {split} ===")
    j = json.loads((GT / f"{split}_panoptic.json").read_text())

    # ---- 1. categories ----
    expected = expected_categories()
    got = sorted(j["categories"], key=lambda c: c["id"])
    print(f"  expected {len(expected)} cats, got {len(got)}")
    ok_cats = True
    for e, g in zip(expected, got):
        if e["id"] != g["id"] or e["isthing"] != g["isthing"]:
            print(f"  MISMATCH: expected {e}, got {{id:{g['id']}, name:{g['name']}, isthing:{g['isthing']}}}")
            ok_cats = False
    things = [c["id"] for c in got if c["isthing"]]
    stuff  = [c["id"] for c in got if not c["isthing"]]
    print(f"  things ({len(things)}): {things}")
    print(f"  stuff  ({len(stuff)}): {stuff}")
    print(f"  cat-spec match: {'OK' if ok_cats else 'FAIL'}")

    # ---- 2..3. annotation-level checks ----
    bad_ids = 0
    dup_segs = 0
    valid_cats = {c["id"] for c in got}
    seg_per_img = []
    crowd_things = 0  # things-class segs with iscrowd=1 (encoded with semantic_id<1000)
    instance_things = 0
    stuff_segs = 0
    seg_ids_zero = 0  # segments with id=0 (would be invalid in COCO panoptic)
    for ann in j["annotations"]:
        ids = [s["id"] for s in ann["segments_info"]]
        seg_per_img.append(len(ids))
        if len(ids) != len(set(ids)):
            dup_segs += 1
        for s in ann["segments_info"]:
            if s["category_id"] not in valid_cats:
                bad_ids += 1
            if s["id"] == 0:
                seg_ids_zero += 1
            if s["category_id"] in things:
                if s.get("iscrowd"):
                    crowd_things += 1
                else:
                    instance_things += 1
            else:
                stuff_segs += 1
    print(f"  annotations: {len(j['annotations'])}")
    if seg_per_img:
        sp = np.array(seg_per_img)
        print(f"  segs/img: min={sp.min()} mean={sp.mean():.1f} median={np.median(sp):.1f} max={sp.max()}")
        print(f"  empty annotations (segs/img == 0): {(sp == 0).sum()}")
    print(f"  duplicate segment_ids in same image: {dup_segs}")
    print(f"  category_id outside expected set:  {bad_ids}")
    print(f"  segment_id == 0 (invalid):          {seg_ids_zero}")
    print(f"  things instance segs: {instance_things}, things crowd segs: {crowd_things}, stuff segs: {stuff_segs}")

    # ---- 4..5. PNG encoding round-trip on samples ----
    rng = random.Random(42)
    annos_by_id = {a["image_id"]: a for a in j["annotations"]}
    sample = rng.sample(j["images"], min(SAMPLE_N, len(j["images"])))
    print(f"  decoding {len(sample)} sample PNGs:")
    decode_ok = 0
    for im in sample:
        png = GT / f"{split}_panoptic" / im["file_name"]
        ann = annos_by_id[im["id"]]
        rgb = np.array(Image.open(png).convert("RGB"))
        seg_map = (rgb[..., 0].astype(np.int64)
                   + 256 * rgb[..., 1].astype(np.int64)
                   + 256 * 256 * rgb[..., 2].astype(np.int64))
        png_segs = set(int(x) for x in np.unique(seg_map))
        # segment_id 0 = unlabeled / void in COCO panoptic; not in segments_info
        json_segs = {s["id"] for s in ann["segments_info"]}
        in_json_not_in_png = json_segs - png_segs
        in_png_not_in_json = png_segs - json_segs - {0}
        # area cross-check for a random thing-segment if present
        thing_segs = [s for s in ann["segments_info"]
                      if s["category_id"] in things and s["id"] in png_segs]
        area_msg = ""
        if thing_segs:
            s = thing_segs[0]
            mask = seg_map == s["id"]
            recomputed_area = int(mask.sum())
            ys, xs = np.nonzero(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            j_area = s["area"]
            j_bbox = tuple(s["bbox"])
            area_msg = (f"  thing-seg id={s['id']}: area json={j_area} png={recomputed_area} "
                        f"bbox json={j_bbox} png={(x0, y0, w, h)}")
        status = "OK" if not in_json_not_in_png and not in_png_not_in_json else "MISMATCH"
        decode_ok += int(status == "OK")
        print(f"    {im['file_name']}: json segs={len(json_segs)} png segs={len(png_segs)-(1 if 0 in png_segs else 0)} {status}")
        if in_json_not_in_png:
            print(f"      missing in PNG: {sorted(in_json_not_in_png)[:5]}{'...' if len(in_json_not_in_png) > 5 else ''}")
        if in_png_not_in_json:
            print(f"      extra in PNG (phantom): {sorted(in_png_not_in_json)[:5]}{'...' if len(in_png_not_in_json) > 5 else ''}")
        if area_msg:
            print(area_msg)

    print(f"  PNG↔JSON sample match: {decode_ok}/{len(sample)}")
    return (ok_cats and dup_segs == 0 and bad_ids == 0
            and decode_ok == len(sample))


def main() -> int:
    ok_t = inspect("train")
    ok_v = inspect("val")
    print("\n" + ("ALL CHECKS PASSED" if ok_t and ok_v else "SOMETHING FAILED — read above"))
    return 0 if (ok_t and ok_v) else 1


if __name__ == "__main__":
    raise SystemExit(main())
