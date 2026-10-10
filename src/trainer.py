"""Mask2Former Trainer with the IDD ablation hooks (``cfg.IDD.*``).

Requires ``external/Mask2Former`` on ``sys.path`` (``train_net`` is imported
from there).
"""
from __future__ import annotations

import logging

import torch
from detectron2.data import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.data.samplers import RepeatFactorTrainingSampler
from train_net import Trainer

from src.datasets.class_stats import class_loss_weights, repeat_factors

logger = logging.getLogger("idd.trainer")


class IDDTrainer(Trainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        cw = cfg.IDD.CLASS_WEIGHT
        if cw.MODE != "none":
            # SetCriterion uses empty_weight as the CE class-weight vector; the
            # last entry is the no-object weight and is left untouched.
            k = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=False)
            weights = class_loss_weights(dataset, k, cw.MODE, cw.POWER, cw.MAX)
            model.criterion.empty_weight[:k] = torch.tensor(weights)
            logger.info("Class-weighted CE (%s): %s", cw.MODE,
                        [round(w, 2) for w in model.criterion.empty_weight.tolist()])
        h = cfg.IDD.HIERARCHY
        if h.ENABLED:
            from detectron2.data import MetadataCatalog

            from src.datasets.hierarchy import coarse_ids
            from src.hierarchy_loss import make_hierarchical

            classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
            groups = coarse_ids(classes, h.LEVEL)
            make_hierarchical(model.criterion, groups, h.LOSS_WEIGHT)
            logger.info("Hierarchical CE: level%d, %d groups, weight %.2f: %s", h.LEVEL,
                        max(groups) + 1, h.LOSS_WEIGHT, dict(zip(classes, groups)))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME != "mask_former_panoptic":
            return super().build_train_loader(cfg)

        from mask2former import MaskFormerPanopticDatasetMapper

        dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=False)
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        sampler = None
        threshold = cfg.IDD.REPEAT_FACTOR_THRESHOLD
        if threshold > 0:
            rf = torch.tensor(repeat_factors(dataset, num_classes, threshold))
            logger.info("Repeat-factor sampling t=%.3f: epoch x%.3f, %d/%d images repeated",
                        threshold, rf.mean().item(), int((rf > 1).sum()), len(rf))
            sampler = RepeatFactorTrainingSampler(rf, seed=cfg.SEED if cfg.SEED >= 0 else None)

        if cfg.IDD.COPY_PASTE.ENABLED:
            from src.datasets.copy_paste import CopyPastePanopticDatasetMapper

            mapper = CopyPastePanopticDatasetMapper(cfg, True, dataset)
        else:
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset, sampler=sampler)
