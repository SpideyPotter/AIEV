"""Project-specific config keys, all under ``cfg.IDD``.

Every switch defaults to "off" so the stock Mask2Former recipe is
reproduced unless a config turns an ablation on.
"""
from __future__ import annotations

from detectron2.config import CfgNode as CN


def add_idd_config(cfg) -> None:
    cfg.IDD = CN()

    # Exp A1: LVIS-style repeat-factor sampling over panoptic segments_info.
    # r(c) = max(1, sqrt(t / f(c))), f(c) = fraction of train images containing c.
    # 0.0 disables it (uniform TrainingSampler).
    cfg.IDD.REPEAT_FACTOR_THRESHOLD = 0.0

    # Exp A2: copy-paste of rare-class segments from other train images.
    cfg.IDD.COPY_PASTE = CN()
    cfg.IDD.COPY_PASTE.ENABLED = False
    cfg.IDD.COPY_PASTE.PROB = 0.5
    cfg.IDD.COPY_PASTE.MAX_SEGMENTS = 3
    # Segments smaller than this (pixels, at source resolution) are skipped.
    cfg.IDD.COPY_PASTE.MIN_AREA = 64
    cfg.IDD.COPY_PASTE.CLASSES = ["bicycle", "traffic light", "traffic sign", "vehicle fallback"]

    # Exp A3: per-class weights on the mask-classification cross-entropy.
    # MODE: "none" | "img_freq" | "seg_freq"; w(c) = (n_max / n(c)) ** POWER,
    # clamped to MAX (see src/datasets/class_stats.py).
    cfg.IDD.CLASS_WEIGHT = CN()
    cfg.IDD.CLASS_WEIGHT.MODE = "none"
    cfg.IDD.CLASS_WEIGHT.POWER = 0.5
    cfg.IDD.CLASS_WEIGHT.MAX = 5.0

    # Path D: auxiliary coarse-level (IDD level2Id) classification loss.
    cfg.IDD.HIERARCHY = CN()
    cfg.IDD.HIERARCHY.ENABLED = False
    cfg.IDD.HIERARCHY.LOSS_WEIGHT = 1.0
