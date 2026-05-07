"""Reproduce the panopticapi crash on a small subset of failing images.

This bypasses detectron2's COCOPanopticEvaluator entirely. Steps:
  1. Inference on 4 known-failing images via Trainer's eval data path.
  2. Reconcile PNG <-> JSON (same logic as the patch).
  3. Convert category_id from contiguous to dataset_id (same as
     _convert_category_id in detectron2's evaluator).
  4. Save prediction PNGs named after the GT panoptic file.
  5. Build a tiny GT-subset JSON + a predictions JSON with only the 4
     matching annotations.
  6. Call panopticapi.pq_compute directly.
  7. Print PQ result OR the exact crash with full traceback.

If this crashes -> we have a clean repro to debug.
If this succeeds -> the bug is in how detectron2 hands data to panopticapi
                    (file naming, JSON structure, etc.).

Outputs in /workspace/outputs/eval_subset/ (preserved across runs).

Run:
    source /workspace/.local/activate.sh
    python3 scripts/eval_subset_panoptic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from panopticapi.utils import id2rgb  # noqa: E402
from train_net import Trainer  # noqa: E402


CONFIG = REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"
WEIGHTS = "/workspace/outputs/m2f_swinl_idd_panoptic/model_final.pth"
OUT = Path("/workspace/outputs/eval_subset")

TARGETS = (
    ("234", "frame2944"),
    ("291", "frame2976"),
    ("341", "frame0004"),
    ("420", "0007867"),
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pred_pngs = OUT / "pngs"
    pred_pngs.mkdir(exist_ok=True)

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(CONFIG))
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    cfg.freeze()

    loader = Trainer.build_test_loader(cfg, "idd_panoptic_val")
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    meta = MetadataCatalog.get("idd_panoptic_val")
    things = set(meta.thing_dataset_id_to_contiguous_id.values())
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}
    print(f"thing contig->dataset map: {thing_c2d}")
    print(f"stuff contig->dataset map: {stuff_c2d}")

    pred_anns: list[dict] = []
    found: set[tuple] = set()
    target_set = set(TARGETS)
    target_image_ids: dict[tuple, int] = {}

    for batch in loader:
        if found == target_set:
            break
        inp = batch[0]
        fn = inp["file_name"]
        match = next(
            ((sc, sid) for sc, sid in TARGETS
             if f"/{sc}/{sid}_leftImg8bit" in fn and (sc, sid) not in found),
            None,
        )
        if match is None:
            continue

        with torch.no_grad():
            out = model([inp])[0]
        pan_seg, segs = out["panoptic_seg"]
        panoptic_img = pan_seg.cpu().numpy()

        # Reconcile PNG <-> segments_info
        present_in_png = set(int(x) for x in np.unique(panoptic_img).tolist()) - {0}
        json_ids = {s["id"] for s in segs}
        segs = [s for s in segs if s["id"] in present_in_png]
        orphan = present_in_png - json_ids
        if orphan:
            panoptic_img[np.isin(panoptic_img, list(orphan))] = 0

        # Convert contiguous -> dataset category_id (matches detectron2 _convert_category_id)
        segments_info = []
        for s in segs:
            isthing = s.get("isthing", s["category_id"] in things)
            cat_contig = s["category_id"]
            cat_dataset = (thing_c2d if isthing else stuff_c2d)[cat_contig]
            segments_info.append({
                "id": int(s["id"]),
                "category_id": int(cat_dataset),
                "isthing": bool(isthing),
            })

        # Write PNG with the GT panoptic filename — that's what panopticapi expects
        sc, sid = match
        gt_png_name = f"{sc}_{sid}_gtFine_panopticlevel3Ids.png"
        out_png = pred_pngs / gt_png_name
        Image.fromarray(id2rgb(panoptic_img)).save(out_png, format="PNG")

        rgb = np.array(Image.open(out_png).convert("RGB"))
        seg = (rgb[..., 0].astype(np.int64)
               + 256 * rgb[..., 1].astype(np.int64)
               + 256 * 256 * rgb[..., 2].astype(np.int64))
        ids_in_png = sorted(int(x) for x in np.unique(seg).tolist())
        ids_in_json = sorted(s["id"] for s in segments_info)

        print(f"\n--- {sc}/{sid} ---")
        print(f"  PNG ids:        {ids_in_png}")
        print(f"  JSON ids:       {ids_in_json}")
        print(f"  segments_info:  {segments_info}")
        ok = (set(ids_in_png) - {0}) == set(ids_in_json)
        print(f"  consistent:     {ok}")

        pred_anns.append({
            "image_id": inp["image_id"],
            "file_name": gt_png_name,
            "segments_info": segments_info,
        })
        found.add(match)
        target_image_ids[match] = inp["image_id"]

    # Build subset GT JSON
    gt_full = json.loads(
        (REPO / "data" / "idd_full" / "gtFine" / "val_panoptic.json").read_text()
    )
    keep_image_ids = set(target_image_ids.values())
    gt_subset = {
        "categories": gt_full["categories"],
        "images": [im for im in gt_full["images"] if im["id"] in keep_image_ids],
        "annotations": [ann for ann in gt_full["annotations"] if ann["image_id"] in keep_image_ids],
    }
    gt_subset_path = OUT / "gt_subset.json"
    gt_subset_path.write_text(json.dumps(gt_subset))

    pred_json = {**gt_subset, "annotations": pred_anns}
    pred_json_path = OUT / "predictions.json"
    pred_json_path.write_text(json.dumps(pred_json))

    print(f"\nWrote GT subset:  {gt_subset_path}")
    print(f"Wrote pred JSON:  {pred_json_path}")
    print(f"PNGs:             {pred_pngs}")

    print("\n=== Running pq_compute ===")
    from panopticapi.evaluation import pq_compute
    try:
        res = pq_compute(
            str(gt_subset_path),
            str(pred_json_path),
            gt_folder="/workspace/data/idd_full/gtFine/val_panoptic",
            pred_folder=str(pred_pngs),
        )
        print("\n=== RESULT ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
