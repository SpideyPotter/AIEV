#!/usr/bin/env python3
"""Mask2Former training entry point for IDD panoptic.

Sets up sys.path so we can import from external/Mask2Former, registers our
IDD-panoptic dataset with Detectron2's catalog, then calls Mask2Former's
own Trainer.

Usage:
    python scripts/train.py --config-file configs/m2f_swinl_idd_panoptic.yaml
    python scripts/train.py ... --eval-only MODEL.WEIGHTS path/to/model.pth
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import warnings


# Silence deprecation/future warnings from torch + timm + Mask2Former CUDA op
# before any heavy imports run. These fire on every iter and bury real errors.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning,ignore::FutureWarning")
# torch.cuda.amp deprecation spam
os.environ.setdefault("TORCH_WARN_ONCE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

# Mask2Former side-effect imports — registers config nodes + data mappers
from mask2former import (  # noqa: E402,F401
    add_maskformer2_config,
    COCOInstanceNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
)

from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import default_argument_parser, default_setup, launch  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

# Mask2Former's repo defines its own Trainer with the right data mappers / evaluators.
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))
from train_net import Trainer  # noqa: E402


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
