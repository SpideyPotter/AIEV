#!/usr/bin/env python3
"""Run Mask2Former inference on an IDD split and write predictions in
panopticapi format compatible with the AutoNUE leaderboard upload.

For test split:
    python scripts/predict.py \
        --config-file configs/m2f_swinl_idd_panoptic.yaml \
        --weights outputs/m2f_swinl_idd_panoptic/model_final.pth \
        --split test --zip

Output:
    <output-dir>/<split>_pred_panoptic/<scene>_<id>_gtFine_panopticlevel3Ids.png
    <output-dir>/<split>_pred_panoptic.json
    <output-dir>/<split>_submission.zip   (with --zip)

Notes:
    - Resizes RGBs to 1280x720 (AutoNUE canonical) before inference, since
      IDD test images come at native 1920x1080 / 1280x964 / etc.
    - Picks up both .jpg and .png RGB extensions.
    - Reconciles panoptic PNG <-> segments_info per panopticapi spec
      (same logic as our detectron2 patch).
    - Maps category_id from contiguous (model output) -> dataset_id
      (AutoNUE level3Id, 0..25) before writing JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

import src._silence_warnings  # noqa: E402,F401
from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from panopticapi.utils import id2rgb  # noqa: E402


CANONICAL_WH = (1280, 720)  # AutoNUE leaderboard target (W, H)


def setup(config_file: str, weights: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.INPUT.MIN_SIZE_TEST = CANONICAL_WH[1]
    cfg.INPUT.MAX_SIZE_TEST = CANONICAL_WH[0]
    cfg.freeze()
    return cfg


def categories_from_gt():
    """Use the train-split JSON's categories list as the reference."""
    train_json = REPO / "data" / "idd_full" / "gtFine" / "train_panoptic.json"
    with open(train_json) as f:
        return json.load(f)["categories"]


def list_split_images(image_root: Path) -> list[Path]:
    """Find all *_leftImg8bit.{jpg,png} under <split>/<scene>/."""
    files: list[Path] = []
    for scene_dir in sorted(image_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for f in sorted(scene_dir.iterdir()):
            if f.is_file() and f.stem.endswith("_leftImg8bit"):
                if f.suffix.lower() in (".jpg", ".png"):
                    files.append(f)
    return files


def reconcile(panoptic_img: np.ndarray, segs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Drop JSON entries with no pixels; force orphan PNG pixels to VOID.
    Same logic as scripts/patch_detectron2.py — required for panopticapi spec."""
    present_in_png = set(int(x) for x in np.unique(panoptic_img).tolist()) - {0}
    json_ids = {s["id"] for s in segs}
    segs = [s for s in segs if s["id"] in present_in_png]
    orphan = present_in_png - json_ids
    if orphan:
        panoptic_img = panoptic_img.copy()
        panoptic_img[np.isin(panoptic_img, list(orphan))] = 0
    return panoptic_img, segs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None,
                   help="Defaults to outputs/<split>_pred_panoptic/")
    p.add_argument("--zip", action="store_true",
                   help="Bundle PNGs + JSON into a submission .zip")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit to first N images (for smoke testing). 0 = all.")
    args = p.parse_args()

    cfg = setup(args.config_file, args.weights)
    predictor = DefaultPredictor(cfg)

    image_root = REPO / "data" / "idd_full" / "leftImg8bit" / args.split
    image_files = list_split_images(image_root)
    if args.limit:
        image_files = image_files[: args.limit]
    print(f"[predict] {len(image_files)} images in split={args.split}")

    out_dir = Path(args.output_dir) if args.output_dir else \
              REPO / "outputs" / f"{args.split}_pred_panoptic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir.parent / f"{args.split}_pred_panoptic.json"

    # Map contiguous (model) category_id → dataset (level3Id) category_id
    meta_name = f"idd_panoptic_{'val' if args.split == 'test' else args.split}"
    meta = MetadataCatalog.get(meta_name)
    things_contig = set(meta.thing_dataset_id_to_contiguous_id.values())
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    images_out: list[dict] = []
    annos_out: list[dict] = []
    t0 = time.time()

    for img_path in tqdm(image_files, desc=f"{args.split}"):
        scene = img_path.parent.name
        image_id_short = img_path.stem.replace("_leftImg8bit", "")
        out_name = f"{scene}_{image_id_short}_gtFine_panopticlevel3Ids.png"

        # Resize to canonical 1280x720 for the leaderboard
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        if (img_rgb.shape[1], img_rgb.shape[0]) != CANONICAL_WH:
            img_rgb = np.array(Image.fromarray(img_rgb).resize(CANONICAL_WH, Image.BILINEAR))
        img_bgr = img_rgb[:, :, ::-1].copy()  # detectron2 wants BGR

        with torch.no_grad():
            outputs = predictor(img_bgr)
        pan_seg, segments_info = outputs["panoptic_seg"]
        pan_arr = pan_seg.cpu().numpy().astype(np.int64)
        pan_arr, segments_info = reconcile(pan_arr, segments_info or [])

        # Encode panoptic_id -> RGB (R + 256*G + 65536*B = id), save PNG
        Image.fromarray(id2rgb(pan_arr).astype(np.uint8)).save(out_dir / out_name)

        # Build segments_info in dataset_id space
        seg_out = []
        for s in segments_info:
            cat_contig = int(s["category_id"])
            isthing = bool(s.get("isthing", cat_contig in things_contig))
            cat_dataset = (thing_c2d if isthing else stuff_c2d).get(cat_contig)
            if cat_dataset is None:
                continue  # shouldn't happen with non-overlapping contig space
            seg_out.append({
                "id": int(s["id"]),
                "category_id": int(cat_dataset),
                "area": int((pan_arr == s["id"]).sum()),
                "iscrowd": 0,
            })

        h, w = pan_arr.shape
        images_out.append({
            "id": out_name,
            "file_name": out_name,
            "width": int(w),
            "height": int(h),
        })
        annos_out.append({
            "image_id": out_name,
            "file_name": out_name,
            "segments_info": seg_out,
        })

    out = {
        "images": images_out,
        "annotations": annos_out,
        "categories": categories_from_gt(),
    }
    with open(out_json_path, "w") as f:
        json.dump(out, f)
    dt = time.time() - t0
    print(f"[predict] wrote {out_json_path} ({out_json_path.stat().st_size/1e6:.1f} MB)")
    print(f"[predict] wrote {len(image_files)} PNGs to {out_dir}/")
    print(f"[predict] total: {dt/60:.1f} min  ({len(image_files)/max(dt,1):.2f} img/s)")

    if args.zip:
        zip_path = out_dir.parent / f"{args.split}_submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(out_json_path, arcname=out_json_path.name)
            for png in sorted(out_dir.iterdir()):
                zf.write(png, arcname=f"{out_dir.name}/{png.name}")
        print(f"[predict] wrote {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
