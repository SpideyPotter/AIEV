"""Run the standard detectron2 eval but PRESERVE the predictions.json + PNGs
so we can inspect exactly what was written when panopticapi crashes.

The default flow uses tempfile.TemporaryDirectory which auto-cleans on exit.
Here we monkey-patch tempfile to use a persistent dir under /workspace/outputs.

Run:
    source /workspace/.local/activate.sh
    python3 scripts/preserve_eval_output.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

PRESERVE = Path("/workspace/outputs/eval_preserve")
PRESERVE.mkdir(parents=True, exist_ok=True)


# Monkey-patch tempfile.TemporaryDirectory so the dir survives.
class PersistentTempDir:
    def __init__(self, prefix="tmp", **kwargs):
        self.name = str(PRESERVE / f"{prefix}{os.getpid()}")
        os.makedirs(self.name, exist_ok=True)
        # also write a sentinel so we can find it later
        with open(os.path.join(self.name, ".sentinel"), "w") as f:
            f.write(self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, *_):
        # Don't delete!
        print(f"\n[preserve] kept artifacts in: {self.name}")
        return False

    def cleanup(self):
        pass


tempfile.TemporaryDirectory = PersistentTempDir  # type: ignore[misc,assignment]


# Now do a normal eval-only run
from mask2former import add_maskformer2_config  # noqa: E402,F401
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from train_net import Trainer  # noqa: E402


def main() -> int:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"))
    cfg.MODEL.WEIGHTS = "/workspace/outputs/m2f_swinl_idd_panoptic/model_final.pth"
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 200
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    try:
        Trainer.test(cfg, model)
    except Exception as e:
        print(f"\n[preserve] eval failed (expected): {type(e).__name__}: {e}")
        # find the latest preserved dir
        dirs = sorted(PRESERVE.glob("panoptic_eval*"), key=lambda p: p.stat().st_mtime)
        if dirs:
            print(f"[preserve] latest panoptic_eval dir: {dirs[-1]}")
            print(f"[preserve] contents: {len(list(dirs[-1].iterdir()))} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
