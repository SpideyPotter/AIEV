"""Mask2Former Trainer with the IDD ablation hooks (``cfg.IDD.*``).

Requires ``external/Mask2Former`` on ``sys.path`` (``train_net`` is imported
from there).
"""
from __future__ import annotations

import logging

from detectron2.data import build_detection_train_loader
from train_net import Trainer

logger = logging.getLogger("idd.trainer")


class IDDTrainer(Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME != "mask_former_panoptic":
            return super().build_train_loader(cfg)

        from mask2former import MaskFormerPanopticDatasetMapper

        mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
