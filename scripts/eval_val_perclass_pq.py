#!/usr/bin/env python3
"""Run val inference with a Mask2Former checkpoint and report per-class PQ.

Iterates the GT val JSON (not the image folder), so the prediction set and
image ids line up with panopticapi exactly. IDD image ids are filename
strings (``<scene>_<id>_gtFine_panopticlevel3Ids.png``), not integers.

Predictions are written to local disk (default ``/tmp``) because writing
2,036 PNGs to the NFS-mounted ``/workspace`` from a worker pod can stall.

Usage (inside the dev pod):
    source /workspace/.local/activate.sh
    python3 scripts/eval_val_perclass_pq.py \
        --config-file configs/m2f_r50_idd_scratch.yaml \
        --weights outputs/m2f_r50_idd_scratch/model_final.pth \
        --report outputs/perclass_pq_r50_scratch.txt \
        --metrics-json outputs/perclass_pq_r50_scratch.json \
        MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.6

Trailing ``KEY VALUE`` pairs are merged into the config (A4 sweeps).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import src._silence_warnings  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mask2former import add_maskformer2_config  # noqa: E402
from src.datasets.idd_panoptic import _resolve_rgb_path, register_idd_panoptic  # noqa: E402
from src.config import add_idd_config  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from panopticapi.evaluation import pq_compute  # noqa: E402
from panopticapi.utils import id2rgb  # noqa: E402

GT_ROOT = REPO / "data" / "idd_full" / "gtFine"
IMG_ROOT = REPO / "data" / "idd_full" / "leftImg8bit" / "val"


def setup(config_file: str, weights: str, opts: list[str]):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_idd_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.freeze()
    return cfg


def reconcile(pan: np.ndarray, segs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Make PNG and segments_info consistent, as panopticapi requires."""
    present = set(int(x) for x in np.unique(pan).tolist()) - {0}
    segs = [s for s in segs if s["id"] in present]
    orphan = present - {s["id"] for s in segs}
    if orphan:
        pan = pan.copy()
        pan[np.isin(pan, list(orphan))] = 0
    return pan, segs


def run_inference(cfg, gt: dict, out_dir: Path, max_images: int) -> dict:
    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get("idd_panoptic_val")
    things_contig = set(meta.thing_dataset_id_to_contiguous_id.values())
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    images = gt["images"][:max_images] if max_images else gt["images"]
    out_dir.mkdir(parents=True, exist_ok=True)
    annos = []
    t0 = time.time()
    for i, im in enumerate(images, 1):
        rgb = np.array(Image.open(_resolve_rgb_path(im["file_name"], IMG_ROOT)).convert("RGB"))
        wh = (im["width"], im["height"])
        if (rgb.shape[1], rgb.shape[0]) != wh:
            rgb = np.array(Image.fromarray(rgb).resize(wh, Image.BILINEAR))
        with torch.no_grad():
            pan_seg, segs = predictor(rgb[:, :, ::-1].copy())["panoptic_seg"]
        pan = pan_seg.cpu().numpy().astype(np.int64)
        pan, segs = reconcile(pan, segs or [])
        Image.fromarray(id2rgb(pan).astype(np.uint8)).save(out_dir / im["file_name"])

        seg_out = []
        for s in segs:
            c = int(s["category_id"])
            isthing = bool(s.get("isthing", c in things_contig))
            d = (thing_c2d if isthing else stuff_c2d).get(c)
            if d is not None:
                seg_out.append({"id": int(s["id"]), "category_id": int(d),
                                "area": int((pan == s["id"]).sum()), "iscrowd": 0})
        annos.append({"image_id": im["id"], "file_name": im["file_name"],
                      "segments_info": seg_out})
        if i % 100 == 0:
            print(f"[perclass-pq] {i}/{len(images)} images  "
                  f"({i / (time.time() - t0):.2f} img/s)", flush=True)
    return {"images": images, "annotations": annos, "categories": gt["categories"]}


def image_frequency(gt: dict, image_ids: set) -> Counter:
    freq: Counter = Counter()
    for a in gt["annotations"]:
        if a["image_id"] in image_ids:
            for c in {s["category_id"] for s in a["segments_info"]}:
                freq[c] += 1
    return freq


def format_report(results: dict, gt: dict, n_images: int, image_ids: set) -> str:
    cats = {c["id"]: c for c in gt["categories"]}
    freq = image_frequency(gt, image_ids)
    lines = ["", "=== Per-class PQ (sorted ascending) ==="]
    per = results["per_class"]
    for cid in sorted(per, key=lambda k: per[k]["pq"]):
        r, c = per[cid], cats[cid]
        lines.append(
            f"{c['name']:25s} id={cid:2d} thing={c['isthing']} "
            f"PQ={100 * r['pq']:6.2f} SQ={100 * r['sq']:6.2f} RQ={100 * r['rq']:6.2f} "
            f"imgs={freq[cid]:4d} ({100 * freq[cid] / max(n_images, 1):5.1f}%)")
    lines += ["", "=== Aggregate ==="]
    for name in ("All", "Things", "Stuff"):
        r = results[name]
        lines.append(f"{name:6s}: PQ={100 * r['pq']:.2f} SQ={100 * r['sq']:.2f} "
                     f"RQ={100 * r['rq']:.2f} N={r['n']}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", default="/tmp/idd_val_pred_panoptic")
    p.add_argument("--max-images", type=int, default=0, help="0 = full val split")
    p.add_argument("--report", default=None, help="Write the text report here")
    p.add_argument("--metrics-json", default=None, help="Write aggregate + per-class JSON here")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides KEY VALUE ...")
    args = p.parse_args()

    cfg = setup(args.config_file, args.weights, args.opts)
    print(f"[perclass-pq] weights={args.weights}")
    print(f"[perclass-pq] overrides={args.opts}")

    gt_json = GT_ROOT / "val_panoptic.json"
    gt = json.loads(gt_json.read_text())
    out_dir = Path(args.out_dir)
    pred = run_inference(cfg, gt, out_dir, args.max_images)
    image_ids = {im["id"] for im in pred["images"]}

    gt_eval_json = gt_json
    if args.max_images:
        subset = dict(gt, images=pred["images"],
                      annotations=[a for a in gt["annotations"] if a["image_id"] in image_ids])
        gt_eval_json = out_dir.parent / "idd_val_gt_subset.json"
        gt_eval_json.write_text(json.dumps(subset))
    pred_json = out_dir.parent / "idd_val_predictions.json"
    pred_json.write_text(json.dumps(pred))

    print("\n=== Running panopticapi PQ (val) ===")
    with contextlib.redirect_stdout(io.StringIO()):
        results = pq_compute(str(gt_eval_json), str(pred_json),
                             gt_folder=str(GT_ROOT / "val_panoptic"),
                             pred_folder=str(out_dir))

    report = format_report(results, gt, len(image_ids), image_ids)
    print(report)
    header = (f"[perclass-pq] weights={args.weights}\n"
              f"[perclass-pq] config={args.config_file}\n"
              f"[perclass-pq] overrides={args.opts}\n"
              f"[perclass-pq] images={len(image_ids)}\n")
    if args.report:
        Path(args.report).write_text(header + report + "\n")
    if args.metrics_json:
        cats = {c["id"]: c["name"] for c in gt["categories"]}
        Path(args.metrics_json).write_text(json.dumps({
            "weights": args.weights, "config": args.config_file, "overrides": args.opts,
            "images": len(image_ids),
            "aggregate": {k: results[k] for k in ("All", "Things", "Stuff")},
            "per_class": {cats[k]: v for k, v in results["per_class"].items()},
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
