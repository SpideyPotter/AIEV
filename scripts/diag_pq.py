"""Diagnose why PQ is ~0 despite training-loss convergence.

For a few val images, compares per-category pixel coverage between GT and
prediction. If the model is producing the right kinds of pixels but in
wrong category bins, we'll see a category-id mapping bug. If the model
just doesn't predict things at all, we'll see no thing-class coverage.

Run:
    source /workspace/.local/activate.sh
    python3 scripts/diag_pq.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import src._silence_warnings  # noqa: E402,F401
from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from train_net import Trainer  # noqa: E402


CONFIG = REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"
WEIGHTS = os.environ.get(
    "DIAG_WEIGHTS",
    "/workspace/outputs/m2f_swinl_idd_panoptic/model_final.pth",
)
GT_JSON = REPO / "data" / "idd_full" / "gtFine" / "val_panoptic.json"
GT_DIR = REPO / "data" / "idd_full" / "gtFine" / "val_panoptic"

# Pick 4 val images that should have lots of things
TARGETS = ("119_035471", "119_038937", "147_510418", "167_002389")


def decode_panoptic_png(png_path: Path) -> np.ndarray:
    rgb = np.array(Image.open(png_path).convert("RGB"))
    return (rgb[..., 0].astype(np.int64)
            + 256 * rgb[..., 1].astype(np.int64)
            + 256 * 256 * rgb[..., 2].astype(np.int64))


def main() -> int:
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
    cat_names = {c["id"]: c["name"] for c in json.loads(GT_JSON.read_text())["categories"]}
    things_dataset_ids = set(meta.thing_dataset_id_to_contiguous_id.keys())
    print(f"Thing dataset_ids: {sorted(things_dataset_ids)}")
    print(f"thing_d2c: {meta.thing_dataset_id_to_contiguous_id}")
    print(f"stuff_d2c: {meta.stuff_dataset_id_to_contiguous_id}")
    print()

    gt_full = json.loads(GT_JSON.read_text())
    gt_by_imgid = {ann["image_id"]: ann for ann in gt_full["annotations"]}
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    found: set[str] = set()
    for batch in loader:
        if found == set(TARGETS):
            break
        inp = batch[0]
        match = None
        for t in TARGETS:
            sc, sid = t.split("_", 1)
            if f"/{sc}/{sid}_leftImg8bit" in inp["file_name"] and t not in found:
                match = t
                break
        if match is None:
            continue
        found.add(match)

        # GT
        gt_png_name = f"{match}_gtFine_panopticlevel3Ids.png"
        gt_seg = decode_panoptic_png(GT_DIR / gt_png_name)
        gt_ann = gt_by_imgid[gt_png_name]
        gt_segs = {s["id"]: s for s in gt_ann["segments_info"]}
        gt_cat_pixels: dict[int, int] = {}
        for sid_val in np.unique(gt_seg):
            sid_val = int(sid_val)
            if sid_val == 0 or sid_val not in gt_segs:
                continue
            cat = gt_segs[sid_val]["category_id"]
            gt_cat_pixels[cat] = gt_cat_pixels.get(cat, 0) + int((gt_seg == sid_val).sum())

        # Prediction
        with torch.no_grad():
            out = model([inp])[0]
        pan_seg, segs = out["panoptic_seg"]
        pred = pan_seg.cpu().numpy()
        pred_cat_pixels: dict[int, int] = {}
        for s in segs or []:
            isthing = s.get("isthing", False)
            cat_contig = s["category_id"]
            cat_dataset = (thing_c2d if isthing else stuff_c2d).get(cat_contig)
            if cat_dataset is None:
                print(f"  WARN: cat_contig={cat_contig} isthing={isthing} -> no dataset mapping")
                continue
            pred_cat_pixels[cat_dataset] = pred_cat_pixels.get(cat_dataset, 0) + int((pred == s["id"]).sum())

        print(f"=== {match} ===  shapes: gt={gt_seg.shape} pred={pred.shape}")
        print(f"  GT category coverage (top 8):")
        total = gt_seg.size
        for cat, pix in sorted(gt_cat_pixels.items(), key=lambda kv: -kv[1])[:8]:
            kind = "thing" if cat in things_dataset_ids else "stuff"
            print(f"    cat={cat:3d} {cat_names.get(cat,'?'):22s} {kind}  {pix:>8d} ({100*pix/total:5.1f}%)")
        print(f"  Pred category coverage (top 8):")
        for cat, pix in sorted(pred_cat_pixels.items(), key=lambda kv: -kv[1])[:8]:
            kind = "thing" if cat in things_dataset_ids else "stuff"
            print(f"    cat={cat:3d} {cat_names.get(cat,'?'):22s} {kind}  {pix:>8d} ({100*pix/total:5.1f}%)")
        print(f"  Raw pred segments_info ({len(segs or [])}):")
        for s in (segs or [])[:8]:
            print(f"    {s}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
