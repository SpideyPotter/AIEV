"""Render side-by-side RGB | GT | Pred for one val image.

Output: a single PNG at /workspace/outputs/viz/<stem>.png
Each panel: 720x1280, stacked horizontally → final image is 720 x 3840.

Run:
    DIAG_WEIGHTS=/workspace/outputs/m2f_swinl_idd_panoptic/model_0004999.pth \
    VIZ_TARGET=167_002389 \
    python3 scripts/viz_one.py
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
TARGET = os.environ.get("VIZ_TARGET", "167_002389")
GT_DIR = REPO / "data" / "idd_full" / "gtFine" / "val_panoptic"
GT_JSON = REPO / "data" / "idd_full" / "gtFine" / "val_panoptic.json"
OUT = Path("/workspace/outputs/viz")


def decode_panoptic_png(png_path: Path) -> np.ndarray:
    rgb = np.array(Image.open(png_path).convert("RGB"))
    return (rgb[..., 0].astype(np.int64)
            + 256 * rgb[..., 1].astype(np.int64)
            + 256 * 256 * rgb[..., 2].astype(np.int64))


def colorize_by_segments(seg: np.ndarray, segments_info: list[dict],
                         cat_colors: dict[int, tuple]) -> np.ndarray:
    """Render a panoptic map by category color (drop instance distinctness)."""
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    sid_to_cat = {s["id"]: s["category_id"] for s in segments_info}
    for sid in np.unique(seg):
        sid = int(sid)
        if sid == 0 or sid not in sid_to_cat:
            continue
        cat = sid_to_cat[sid]
        if cat in cat_colors:
            out[seg == sid] = cat_colors[cat]
    return out


def colorize_pred(panoptic_img: np.ndarray, segs: list[dict],
                  cat_colors_contig: dict[int, tuple]) -> np.ndarray:
    """Pred segments_info uses contiguous category_id."""
    out = np.zeros((*panoptic_img.shape, 3), dtype=np.uint8)
    sid_to_cat = {s["id"]: s["category_id"] for s in (segs or [])}
    for sid in np.unique(panoptic_img):
        sid = int(sid)
        if sid == 0 or sid not in sid_to_cat:
            continue
        cat = sid_to_cat[sid]
        if cat in cat_colors_contig:
            out[panoptic_img == sid] = cat_colors_contig[cat]
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

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

    gt_full = json.loads(GT_JSON.read_text())
    cat_colors_dataset = {c["id"]: tuple(c["color"]) for c in gt_full["categories"]}

    meta = MetadataCatalog.get("idd_panoptic_val")
    # Pred segments use contiguous IDs. Map contig -> dataset_id -> color.
    contig_to_dataset = {**{v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()},
                         **{v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}}
    cat_colors_contig = {c: cat_colors_dataset[d] for c, d in contig_to_dataset.items()}

    # Locate target in val loader
    sc, sid = TARGET.split("_", 1)
    target_substr = f"/{sc}/{sid}_leftImg8bit"
    inp = None
    for batch in loader:
        if target_substr in batch[0]["file_name"]:
            inp = batch[0]
            break
    if inp is None:
        print(f"target {TARGET} not found in val loader")
        return 1

    print(f"running inference on: {inp['file_name']}")
    with torch.no_grad():
        out = model([inp])[0]
    pan_seg, segs = out["panoptic_seg"]
    pred = pan_seg.cpu().numpy()

    # GT
    gt_png_name = f"{TARGET}_gtFine_panopticlevel3Ids.png"
    gt_seg = decode_panoptic_png(GT_DIR / gt_png_name)
    gt_ann = next(a for a in gt_full["annotations"] if a["image_id"] == gt_png_name)
    gt_color = colorize_by_segments(gt_seg, gt_ann["segments_info"], cat_colors_dataset)

    # RGB
    rgb = np.array(Image.open(inp["file_name"]).convert("RGB"))
    if rgb.shape[:2] != (720, 1280):
        rgb = np.array(Image.fromarray(rgb).resize((1280, 720), Image.BILINEAR))

    # Pred
    pred_color = colorize_pred(pred, segs or [], cat_colors_contig)
    if pred_color.shape[:2] != (720, 1280):
        pred_color = np.array(Image.fromarray(pred_color).resize((1280, 720), Image.NEAREST))

    # Stack horizontally with thin white separators
    sep = np.full((720, 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([rgb, sep, gt_color, sep, pred_color], axis=1)

    # 50% blend overlays as bonus panels
    rgb_gt = (rgb.astype(np.uint16) * 1 + gt_color.astype(np.uint16) * 1) // 2
    rgb_pred = (rgb.astype(np.uint16) * 1 + pred_color.astype(np.uint16) * 1) // 2
    overlay = np.concatenate([rgb_gt.astype(np.uint8), sep, rgb_pred.astype(np.uint8)], axis=1)

    out_path = OUT / f"{TARGET}_triptych.png"
    Image.fromarray(panel).save(out_path)
    overlay_path = OUT / f"{TARGET}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"wrote: {out_path}  (RGB | GT | Pred)")
    print(f"wrote: {overlay_path}  (RGB+GT | RGB+Pred blends)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
