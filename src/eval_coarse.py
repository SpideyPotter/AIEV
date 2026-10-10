"""Panoptic quality at IDD level2 / level1.

GT and predictions (level3, COCO-panoptic format) are relabelled to the
coarse level; stuff segments that fall into the same coarse class are
merged into one segment per image, thing segments keep their instance ids.
"""
from __future__ import annotations

import contextlib
import io
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from panopticapi.evaluation import pq_compute
from panopticapi.utils import id2rgb, rgb2id
from PIL import Image

from src.datasets.hierarchy import LEVELS, coarse_names


def _categories(fine_cats: list[dict], level: int) -> tuple[dict[int, int], list[dict]]:
    fine_to_coarse = {c["id"]: LEVELS[c["name"]][2 - level] for c in fine_cats}
    isthing = {fine_to_coarse[c["id"]]: c["isthing"] for c in fine_cats}
    names = coarse_names(level)
    cats = [{"id": g, "name": names[g], "isthing": isthing[g], "color": [0, 0, 0]}
            for g in sorted(isthing)]
    return fine_to_coarse, cats


def _relabel(ann: dict, src_dir: Path, dst_dir: Path, fine_to_coarse: dict, isthing: dict) -> dict:
    pan = rgb2id(np.array(Image.open(src_dir / ann["file_name"]), dtype=np.uint32))
    segs, stuff_rep, remap = [], {}, {}
    for s in ann["segments_info"]:
        g = fine_to_coarse[s["category_id"]]
        if not isthing[g] and g in stuff_rep:
            rep = stuff_rep[g]
            remap[s["id"]] = rep["id"]
            rep["area"] += s["area"]
            rep["iscrowd"] = min(rep["iscrowd"], s.get("iscrowd", 0))
            continue
        seg = {"id": s["id"], "category_id": g, "area": s["area"], "iscrowd": s.get("iscrowd", 0)}
        if not isthing[g]:
            stuff_rep[g] = seg
        segs.append(seg)
    if remap:
        for old, new in remap.items():
            pan[pan == old] = new
    Image.fromarray(id2rgb(pan).astype(np.uint8)).save(dst_dir / ann["file_name"])
    return dict(ann, segments_info=segs)


def _relabel_split(data: dict, src_dir: Path, dst_dir: Path, level: int, workers: int) -> dict:
    fine_to_coarse, cats = _categories(data["categories"], level)
    isthing = {c["id"]: c["isthing"] for c in cats}
    dst_dir.mkdir(parents=True, exist_ok=True)
    fn = partial(_relabel, src_dir=src_dir, dst_dir=dst_dir,
                 fine_to_coarse=fine_to_coarse, isthing=isthing)
    with Pool(workers) as pool:
        annos = pool.map(fn, data["annotations"], chunksize=16)
    return dict(data, annotations=annos, categories=cats)


def coarse_pq(gt_json: Path, gt_dir: Path, pred_json: Path, pred_dir: Path,
              level: int, work_dir: Path, workers: int = 8) -> tuple[dict, dict]:
    """Returns (pq_compute results, coarse GT dict)."""
    out = {}
    for tag, js, src in (("gt", gt_json, gt_dir), ("pred", pred_json, pred_dir)):
        data = _relabel_split(json.loads(Path(js).read_text()), Path(src),
                              work_dir / tag, level, workers)
        (work_dir / f"{tag}.json").write_text(json.dumps(data))
        out[tag] = data
    with contextlib.redirect_stdout(io.StringIO()):
        results = pq_compute(str(work_dir / "gt.json"), str(work_dir / "pred.json"),
                             gt_folder=str(work_dir / "gt"), pred_folder=str(work_dir / "pred"))
    return results, out["gt"]
