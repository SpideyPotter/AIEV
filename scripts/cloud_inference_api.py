#!/usr/bin/env python3
"""FastAPI service for panoptic inference with overlay output.

Request:
    POST /segment (multipart/form-data with field name: image)

Response:
    {
      "shape": {"height": H, "width": W},
      "overlay_png_base64": "...",
      "segments": [...],
      "latency_ms": ...
    }
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import src._silence_warnings  # noqa: E402,F401
from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

import torch  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402


CONFIG = Path(
    os.environ.get(
        "MODEL_CONFIG",
        str(REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"),
    )
)
WEIGHTS = Path(
    os.environ.get(
        "MODEL_WEIGHTS",
        str(REPO / "outputs" / "m2f_swinl_idd_panoptic" / "model_final.pth"),
    )
)

app = FastAPI(title="Panoptic Cloud Inference")


class SegmentItem(BaseModel):
    id: int
    category_id: int
    category_name: str
    area: int
    iscrowd: int
    color_rgb: list[int]


def _setup_predictor() -> tuple[DefaultPredictor, dict[int, int], dict[int, int], dict[int, dict]]:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(CONFIG))
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get("idd_panoptic_val")
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    # Pull human-readable names/colors from gt categories file.
    gt_json = REPO / "data" / "idd_full" / "gtFine" / "train_panoptic.json"
    categories = {}
    if gt_json.exists():
        import json

        with open(gt_json, "r", encoding="utf-8") as f:
            cat_list = json.load(f).get("categories", [])
        categories = {int(c["id"]): c for c in cat_list}
    return predictor, thing_c2d, stuff_c2d, categories


def _colorize_panoptic(
    panoptic_img: np.ndarray,
    segments_info: list[dict],
    thing_c2d: dict[int, int],
    stuff_c2d: dict[int, int],
    categories: dict[int, dict],
) -> tuple[np.ndarray, list[dict]]:
    h, w = panoptic_img.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    out_segments: list[dict] = []

    for seg in (segments_info or []):
        seg_id = int(seg["id"])
        cat_contig = int(seg["category_id"])
        isthing = bool(seg.get("isthing", cat_contig in thing_c2d))
        cat_dataset = (thing_c2d if isthing else stuff_c2d).get(cat_contig, -1)

        cat_info = categories.get(cat_dataset, {})
        color = cat_info.get("color", [255, 255, 255])
        color = [int(color[0]), int(color[1]), int(color[2])]

        mask = panoptic_img == seg_id
        area = int(mask.sum())
        if area == 0:
            continue

        color_map[mask] = color
        out_segments.append(
            {
                "id": seg_id,
                "category_id": int(cat_dataset),
                "category_name": str(cat_info.get("name", f"cat_{cat_dataset}")),
                "area": area,
                "iscrowd": int(seg.get("iscrowd", 0)),
                "color_rgb": color,
            }
        )
    return color_map, out_segments


def _encode_png_base64(img_array: np.ndarray, mode: str) -> str:
    """Encode numpy image array as PNG and return base64 string."""
    buf = io.BytesIO()
    Image.fromarray(img_array, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _road_category_ids(categories: dict[int, dict]) -> set[int]:
    """Resolve road-like category ids from category names."""
    road_ids: set[int] = set()
    for cid, info in categories.items():
        name = str(info.get("name", "")).lower()
        if ("road" in name) or ("drivable" in name):
            road_ids.add(int(cid))
    return road_ids


@app.on_event("startup")
def _startup() -> None:
    if not CONFIG.exists():
        raise RuntimeError(f"Missing config file: {CONFIG}")
    if not WEIGHTS.exists():
        raise RuntimeError(f"Missing model weights: {WEIGHTS}")
    if not torch.cuda.is_available():
        print("[warn] CUDA not available; inference will run on CPU.")

    predictor, thing_c2d, stuff_c2d, categories = _setup_predictor()
    app.state.predictor = predictor
    app.state.thing_c2d = thing_c2d
    app.state.stuff_c2d = stuff_c2d
    app.state.categories = categories
    app.state.road_ids = _road_category_ids(categories)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/segment")
async def segment(image: UploadFile = File(...)) -> JSONResponse:
    started = time.perf_counter()
    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty image payload.")

        rgb = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()

        with torch.no_grad():
            outputs = app.state.predictor(bgr)
        pan_seg, segments_info = outputs["panoptic_seg"]
        pan_arr = pan_seg.cpu().numpy().astype(np.int64)

        color_map, segments = _colorize_panoptic(
            pan_arr,
            segments_info or [],
            app.state.thing_c2d,
            app.state.stuff_c2d,
            app.state.categories,
        )

        # Build per-pixel semantic-id map from panoptic ids.
        sid_to_dataset_cat = {int(s["id"]): int(s["category_id"]) for s in segments}
        semantic_id = np.zeros_like(pan_arr, dtype=np.uint16)
        for sid, cat in sid_to_dataset_cat.items():
            semantic_id[pan_arr == sid] = np.uint16(cat)

        # Binary drivable mask (255 where road-like categories are present).
        road_mask = np.zeros_like(pan_arr, dtype=np.uint8)
        for road_cat in app.state.road_ids:
            road_mask[semantic_id == np.uint16(road_cat)] = np.uint8(255)

        # 50-50 alpha blend for GTAV-side quick draw (server-rendered overlay).
        overlay = ((rgb.astype(np.uint16) + color_map.astype(np.uint16)) // 2).astype(np.uint8)
        overlay_b64 = _encode_png_base64(overlay, mode="RGB")
        semantic_b64 = _encode_png_base64(semantic_id, mode="I;16")
        road_b64 = _encode_png_base64(road_mask, mode="L")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = {
            "shape": {"height": int(overlay.shape[0]), "width": int(overlay.shape[1])},
            "overlay_png_base64": overlay_b64,
            "semantic_id_png_base64": semantic_b64,
            "road_mask_png_base64": road_b64,
            "road_category_ids": sorted(int(x) for x in app.state.road_ids),
            "segments": segments,
            "latency_ms": elapsed_ms,
        }
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
