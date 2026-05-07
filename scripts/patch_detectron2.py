#!/usr/bin/env python3
"""Patch detectron2's COCOPanopticEvaluator to write spec-compliant outputs.

Problem:
    detectron2/evaluation/panoptic_evaluation.py builds segments_info from
    np.unique(panoptic_img) BEFORE the postprocessing argmax/threshold has
    run. As a result, the prediction PNG and prediction JSON can disagree:

      - JSON entry exists for a segment whose pixels got clobbered by a
        higher-confidence overlapping segment (panopticapi raises
        "segment IDs [X] are presented in JSON and not presented in PNG").
      - PNG contains pixels for a segment that wasn't in segments_info
        (panopticapi raises "segment with ID X is presented in PNG and not
        presented in JSON").

    panopticapi's spec REQUIRES PNG ↔ JSON consistency: every non-zero
    pixel value must have a segments_info entry, and every entry must
    have at least one pixel.

Fix:
    Reconcile the two before writing. Walk np.unique(panoptic_img) again
    AFTER segments_info is finalized:
      1. Drop JSON entries with zero pixels in the PNG (no IoU possible
         against any GT, so this is mathematically equivalent to "the
         model didn't predict that segment").
      2. Force orphan PNG pixels (no JSON entry) to VOID (id=0). Tiny
         effect — these are typically <0.1% of pixels, on segments the
         postprocessor would have removed anyway.

    The reconciled output is spec-compliant and PQ is computed on exactly
    what the model predicted at the pixel level.

Idempotent: running on already-patched code is a no-op.

Run inside the pod after `setup_dgx.sh`:
    python3 scripts/patch_detectron2.py
"""
from __future__ import annotations

import sys
from pathlib import Path


CANDIDATES = [
    "/workspace/.local/local/lib/python3.10/dist-packages/detectron2/evaluation/panoptic_evaluation.py",
    "/workspace/.local/lib/python3.10/site-packages/detectron2/evaluation/panoptic_evaluation.py",
    "/usr/local/lib/python3.10/dist-packages/detectron2/evaluation/panoptic_evaluation.py",
]


# We replace the trailing block of the `process` method — from the
# `file_name = os.path.basename(...)` line through the `self._predictions.append(...)`
# call — with one that reconciles segments_info against panoptic_img.
OLD = """            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )"""

NEW = """            # --- BEGIN PATCH ---
            # Bug 1: detectron2 derives file_name_png from os.path.basename(rgb_path).
            # IDD's RGBs live in scene subdirs (val/<scene>/<id>_leftImg8bit.jpg), so
            # basename strips the scene and predictions for different scenes that
            # happen to share an image id collide on disk (last writer wins).
            # Fix: use the panoptic GT file_name from input (which encodes scene),
            # falling back to the RGB basename for COCO-style flat datasets.
            if "pan_seg_file_name" in input:
                file_name_png = os.path.basename(input["pan_seg_file_name"])
            else:
                file_name = os.path.basename(input["file_name"])
                file_name_png = os.path.splitext(file_name)[0] + ".png"

            # Bug 2: panopticapi requires every non-zero pixel in the PNG to
            # appear in segments_info, and every segments_info entry to have
            # >=1 pixel in the PNG. Mask2Former's postprocess can violate this
            # on segments clobbered by higher-confidence overlapping masks.
            present_in_png = set(int(x) for x in np.unique(panoptic_img).tolist())
            present_in_png.discard(0)  # 0 is VOID
            seg_ids_in_json = {s["id"] for s in segments_info}
            segments_info = [s for s in segments_info if s["id"] in present_in_png]
            orphan = present_in_png - seg_ids_in_json
            if orphan:
                panoptic_img[np.isin(panoptic_img, list(orphan))] = 0
            # --- END PATCH ---
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )"""


def main() -> int:
    target = next((Path(p) for p in CANDIDATES if Path(p).exists()), None)
    if target is None:
        print("ERROR: detectron2/evaluation/panoptic_evaluation.py not found in any of:",
              file=sys.stderr)
        for p in CANDIDATES:
            print(f"  {p}", file=sys.stderr)
        return 1

    text = target.read_text(encoding="utf-8")
    if "BEGIN PATCH ---" in text and "Bug 1: detectron2 derives" in text:
        print(f"already patched (current version): {target}")
        return 0
    # If an older patch is in place, reinstall detectron2 to get vanilla source first.
    if "BEGIN PATCH" in text:
        print(f"ERROR: an older version of this patch is in place. Reinstall detectron2 first:",
              file=sys.stderr)
        print(f"  python3 -m pip install --prefix=/workspace/.local --no-build-isolation \\\\",
              file=sys.stderr)
        print(f"      --force-reinstall --no-deps 'git+https://github.com/facebookresearch/detectron2.git'",
              file=sys.stderr)
        return 1
    if OLD not in text:
        print(f"NO MATCH (detectron2 source has changed; manual review needed): {target}",
              file=sys.stderr)
        return 1
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
