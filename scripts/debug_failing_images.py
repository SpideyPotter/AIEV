"""Run inference on the specific images that crash panopticapi during eval.

Goal: capture exactly what Mask2Former returns for these images — the raw
panoptic_img tensor, the segments_info list, and the rendered prediction
PNG — so we can diagnose the PNG↔JSON desync without waiting for a full
eval cycle and a panopticapi crash to happen.

Outputs go to /workspace/outputs/pred_debug/<stem>/
  - rgb.jpg                    source image
  - panoptic_img.npy           raw int label map from Mask2Former
  - panoptic_img.png           rendered via id2rgb (what gets written for eval)
  - panoptic_img_roundtrip.npy decoded back from the PNG
  - segments_info.json         mask2former's segments_info, plus pre-/post-patch
                               reconciliation diagnostics
  - summary.txt                human-readable summary

Run:
    source /workspace/.local/activate.sh
    python3 scripts/debug_failing_images.py
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
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from panopticapi.utils import id2rgb  # noqa: E402


CONFIG = REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"
WEIGHTS = "/workspace/outputs/m2f_swinl_idd_panoptic/model_final.pth"
OUT = Path("/workspace/outputs/pred_debug")

# Images that crashed eval. The error reported the GT panoptic-PNG name,
# but the loader keys off the RGB; convert by stripping the GT suffix and
# resolving via the dataset record.
TARGETS = [
    "234_frame2944", "291_frame2976", "291_frame0119", "274_frame1919",
    "167_020459", "291_frame0119",
    "420_0007867", "341_frame0004", "289_frame0014", "279_frame0014",
    "360_frame0014", "378_frame6749", "349_frame0059", "346_frame0661",
    "330_frame0959", "205_frame1898", "363_frame0119", "279_frame1529",
    "435_frame2459",
]


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(CONFIG))
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    cfg.freeze()
    return cfg


def find_rgb(stem: str) -> Path | None:
    """Find the RGB file for a given <scene>_<id> stem."""
    scene, _, sid = stem.partition("_")
    base = REPO / "data" / "idd_full" / "leftImg8bit" / "val" / scene / f"{sid}_leftImg8bit"
    for ext in (".jpg", ".png"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def reconcile(panoptic_img: np.ndarray, segments_info: list[dict]) -> dict:
    """Mirror what scripts/patch_detectron2.py does, plus diagnostics."""
    present_in_png_pre = set(int(x) for x in np.unique(panoptic_img).tolist())
    present_in_png_pre.discard(0)
    seg_ids_in_json_pre = {s["id"] for s in segments_info}

    kept = [s for s in segments_info if s["id"] in present_in_png_pre]
    orphan = present_in_png_pre - seg_ids_in_json_pre

    panoptic_img_clean = panoptic_img.copy()
    if orphan:
        m = np.isin(panoptic_img_clean, list(orphan))
        panoptic_img_clean[m] = 0

    return {
        "ids_in_panoptic_img_pre": sorted(present_in_png_pre),
        "json_ids_pre": sorted(seg_ids_in_json_pre),
        "json_dropped": sorted(seg_ids_in_json_pre - present_in_png_pre),
        "orphan_pixels_zeroed": sorted(orphan),
        "kept_segments_info": kept,
        "panoptic_img_clean": panoptic_img_clean,
    }


def round_trip(panoptic_img: np.ndarray, png_path: Path) -> np.ndarray:
    Image.fromarray(id2rgb(panoptic_img)).save(png_path, format="PNG")
    rgb = np.array(Image.open(png_path).convert("RGB"))
    return (rgb[..., 0].astype(np.int64)
            + 256 * rgb[..., 1].astype(np.int64)
            + 256 * 256 * rgb[..., 2].astype(np.int64))


def process_one(predictor, stem: str) -> None:
    rgb_path = find_rgb(stem)
    if rgb_path is None:
        print(f"  [{stem}] RGB not found, skipping")
        return

    out = OUT / stem
    out.mkdir(parents=True, exist_ok=True)
    Image.open(rgb_path).save(out / "rgb.jpg", quality=92)

    img_bgr = np.array(Image.open(rgb_path).convert("RGB"))[..., ::-1].copy()
    with torch.no_grad():
        prediction = predictor(img_bgr)
    pan_seg, segs_info = prediction["panoptic_seg"]
    panoptic_img = pan_seg.cpu().numpy()
    segments_info = [dict(s) for s in segs_info] if segs_info else []

    np.save(out / "panoptic_img.npy", panoptic_img)
    rec = reconcile(panoptic_img, segments_info)
    rt = round_trip(rec["panoptic_img_clean"], out / "panoptic_img.png")
    np.save(out / "panoptic_img_roundtrip.npy", rt)
    ids_in_rt = sorted(int(x) for x in np.unique(rt).tolist())

    with open(out / "segments_info.json", "w") as f:
        json.dump({
            "raw_segments_info_from_mask2former": segments_info,
            **{k: v for k, v in rec.items() if k != "panoptic_img_clean"},
            "ids_in_png_after_round_trip": ids_in_rt,
        }, f, indent=2, default=int)

    summary = []
    summary.append(f"image:                {stem}")
    summary.append(f"panoptic_img dtype:   {panoptic_img.dtype}")
    summary.append(f"panoptic_img shape:   {panoptic_img.shape}")
    summary.append(f"raw segments count:   {len(segments_info)}")
    summary.append(f"raw segment ids:      {sorted(int(s['id']) for s in segments_info)}")
    summary.append(f"unique pre patch:     {rec['ids_in_panoptic_img_pre']}")
    summary.append(f"json_dropped:         {rec['json_dropped']}")
    summary.append(f"orphan zeroed:        {rec['orphan_pixels_zeroed']}")
    summary.append(f"kept JSON ids:        {sorted(int(s['id']) for s in rec['kept_segments_info'])}")
    summary.append(f"png round-trip ids:   {ids_in_rt}")
    consistent = (set(ids_in_rt) - {0}) == {int(s['id']) for s in rec['kept_segments_info']}
    summary.append(f"PNG↔JSON consistent:  {consistent}")
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    print(f"loaded weights from {WEIGHTS}")
    print(f"output dir: {OUT}\n")

    seen = set()
    for stem in TARGETS:
        if stem in seen:
            continue
        seen.add(stem)
        try:
            process_one(predictor, stem)
        except Exception as e:
            print(f"  [{stem}] FAILED: {e}")
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
