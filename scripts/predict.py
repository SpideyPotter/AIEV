#!/usr/bin/env python3
"""Run Mask2Former inference on an IDD split and write predictions in
COCO-panoptic format (compatible with our scripts/evaluate.sh and the
AutoNUE leaderboard upload format).

Usage:
    python scripts/predict.py \
        --config-file configs/m2f_swinl_idd_panoptic.yaml \
        --weights outputs/m2f_swinl_idd_panoptic/model_final.pth \
        --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from panopticapi.utils import id2rgb  # noqa: E402


def setup(config_file: str, weights: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.freeze()
    return cfg


def categories_from_gt():
    """Use the train-split JSON's categories list as the reference categories."""
    train_json = REPO / "data" / "idd_full" / "gtFine" / "train_panoptic.json"
    with open(train_json) as f:
        return json.load(f)["categories"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None,
                   help="Defaults to outputs/<split>_pred_panoptic/")
    args = p.parse_args()

    cfg = setup(args.config_file, args.weights)
    predictor = DefaultPredictor(cfg)

    image_root = REPO / "data" / "idd_full" / "leftImg8bit" / args.split
    image_files = sorted(image_root.glob("*/*_leftImg8bit.png"))
    print(f"[predict] {len(image_files)} images in split={args.split}")

    out_dir = Path(args.output_dir) if args.output_dir else \
              REPO / "outputs" / f"{args.split}_pred_panoptic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir.parent / f"{args.split}_pred_panoptic.json"

    # Map contiguous (training) class id → dataset (level3Id) class id
    if args.split in ("train", "val"):
        meta = MetadataCatalog.get(f"idd_panoptic_{args.split}")
    else:
        # test split has no metadata registered; reuse train metadata mapping
        meta = MetadataCatalog.get("idd_panoptic_train")
    contig_to_dataset = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    images_out = []
    annos_out = []

    for img_path in tqdm(image_files, desc=f"{args.split}"):
        scene = img_path.parent.name
        image_id_short = img_path.stem.replace("_leftImg8bit", "")
        out_name = f"{scene}_{image_id_short}_gtFine_panopticlevel3Ids.png"

        img_bgr = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1].copy()
        outputs = predictor(img_bgr)
        pan_seg, segments_info = outputs["panoptic_seg"]
        # pan_seg: (H, W) tensor of segment ids; 0 = void / unassigned.
        pan_arr = pan_seg.cpu().numpy().astype(np.int64)

        # Encode per-pixel segment id as RGB (R + 256*G + 65536*B = id).
        rgb = id2rgb(pan_arr).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / out_name)

        seg_out = []
        for s in segments_info:
            cat_contig = int(s["category_id"])
            if cat_contig not in contig_to_dataset:
                continue  # safety: skip if model emitted an unknown class
            seg_out.append({
                "id": int(s["id"]),
                "category_id": int(contig_to_dataset[cat_contig]),
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
    print(f"[predict] wrote {out_json_path}")
    print(f"[predict] wrote {len(image_files)} PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
