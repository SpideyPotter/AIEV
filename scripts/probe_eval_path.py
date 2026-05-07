"""Run the EXACT Trainer.test inference path on one failing image.

Compare its output to scripts/debug_failing_images.py (which uses
DefaultPredictor). If outputs differ, the eval mapper / model wrapper is
doing something the predictor isn't, and that explains the PNG↔JSON
desync we see in panopticapi.

Run:
    source /workspace/.local/activate.sh
    python3 scripts/probe_eval_path.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from train_net import Trainer  # noqa: E402


CONFIG = REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"
WEIGHTS = "/workspace/outputs/m2f_swinl_idd_panoptic/model_final.pth"
# Match against the path component <scene>/<id>_leftImg8bit since the loader
# returns paths like /workspace/data/.../val/234/frame2944_leftImg8bit.png
TARGETS = (
    "/234/frame2944_leftImg8bit",
    "/291/frame2976_leftImg8bit",
    "/420/0007867_leftImg8bit",
    "/341/frame0004_leftImg8bit",
)


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

    found: set[str] = set()
    seen_files: list[str] = []
    for batch_idx, batch in enumerate(loader):
        if len(found) == len(TARGETS):
            break
        inp = batch[0]
        fn = inp.get("file_name", "")
        if batch_idx < 3:
            seen_files.append(f"  example file_name [{batch_idx}]: {fn!r}")
            seen_files.append(f"  example input keys [{batch_idx}]: {sorted(inp.keys())}")
        match = next((t for t in TARGETS if t in fn and t not in found), None)
        if match is None:
            continue
        found.add(match)

        with torch.no_grad():
            out = model([inp])[0]
        pan_seg, segs = out["panoptic_seg"]
        panoptic_img = pan_seg.cpu().numpy()
        uniq, cnt = np.unique(panoptic_img, return_counts=True)

        print(f"\n=== {match} (eval-path) ===")
        print(f"  file_name           : {fn}")
        print(f"  input keys          : {sorted(inp.keys())}")
        print(f"  panoptic_img dtype  : {panoptic_img.dtype}")
        print(f"  panoptic_img shape  : {panoptic_img.shape}")
        print(f"  unique pixel ids    : {dict(zip(uniq.tolist(), cnt.tolist()))}")
        print(f"  segments_info type  : {type(segs).__name__}")
        if segs is None:
            print(f"  segments_info       : None")
        else:
            print(f"  segments_info count : {len(segs)}")
            for s in segs[:5]:
                print(f"    {s}")
            if len(segs) > 5:
                print(f"    ... and {len(segs)-5} more")

    if not found:
        print("WARNING: no target images encountered in val loader")
        print("\nFirst few file_names from the loader:")
        for line in seen_files:
            print(line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
