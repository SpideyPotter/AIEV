"""Exp A2: copy-paste of rare-class segments for Mask2Former panoptic training.

Segments are pasted at their original pixel location (every IDD frame is
1280x720), optionally mirrored, *before* the usual resize/crop/flip pipeline,
so pasted objects keep a plausible position in the scene and receive the same
scale jitter as the target image. The class to paste is drawn uniformly from
``IDD.COPY_PASTE.CLASSES`` and then a segment of that class uniformly, so
the rarest class is pasted as often as the others.

A pasted thing becomes a new instance. A pasted stuff segment is merged into
the target's existing segment of that class (panoptic stuff has one segment
per class per image). Segments fully covered by pastes are dropped.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from mask2former import MaskFormerPanopticDatasetMapper
from panopticapi.utils import id2rgb, rgb2id

logger = logging.getLogger("idd.copy_paste")


class CopyPastePanopticDatasetMapper(MaskFormerPanopticDatasetMapper):
    def __init__(self, cfg, is_train, dataset_dicts):
        super().__init__(cfg, is_train)
        cp = cfg.IDD.COPY_PASTE
        self.prob = cp.PROB
        self.max_segments = cp.MAX_SEGMENTS

        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        name_to_id = {n: i for i, n in enumerate(meta.stuff_classes)}
        unknown = [n for n in cp.CLASSES if n not in name_to_id]
        if unknown:
            raise ValueError(f"IDD.COPY_PASTE.CLASSES has unknown names: {unknown}")
        self.num_things = len(meta.thing_classes)

        wanted = {name_to_id[n] for n in cp.CLASSES}
        self.pool: dict[int, list[tuple[str, str, int]]] = {c: [] for c in wanted}
        for r in dataset_dicts:
            for s in r["segments_info"]:
                if s["category_id"] in wanted and not s["iscrowd"] and s.get("area", 0) >= cp.MIN_AREA:
                    self.pool[s["category_id"]].append(
                        (r["file_name"], r["pan_seg_file_name"], s["id"]))
        self.classes = [c for c in sorted(self.pool) if self.pool[c]]
        logger.info("Copy-paste pool: %s",
                    {meta.stuff_classes[c]: len(self.pool[c]) for c in self.classes})

    def _paste(self, image, pan, segments_info):
        segs = [dict(s) for s in segments_info]
        stuff_seg = {s["category_id"]: s["id"] for s in segs if s["category_id"] >= self.num_things}
        # IDD segment ids are RGB-encoded colours up to ~2**24, so max()+1 can
        # overflow the PNG encoding; take the smallest unused ids instead.
        used = set(np.unique(pan).tolist()) | {s["id"] for s in segs} | {0}
        free_ids = (i for i in range(1, 2**24) if i not in used)

        for _ in range(np.random.randint(1, self.max_segments + 1)):
            c = self.classes[np.random.randint(len(self.classes))]
            file_name, pan_file, seg_id = self.pool[c][np.random.randint(len(self.pool[c]))]
            src_pan = rgb2id(utils.read_image(pan_file, "RGB"))
            if src_pan.shape != pan.shape:
                continue
            mask = src_pan == seg_id
            src_img = utils.read_image(file_name, format=self.img_format)
            if np.random.rand() < 0.5:
                mask, src_img = mask[:, ::-1], src_img[:, ::-1]
            if not mask.any():
                continue

            image[mask] = src_img[mask]
            if c in stuff_seg:
                new_id = stuff_seg[c]
            else:
                new_id = next(free_ids)
                segs.append({"id": new_id, "category_id": c, "iscrowd": 0})
                if c >= self.num_things:
                    stuff_seg[c] = new_id
            pan[mask] = new_id

        present = set(np.unique(pan).tolist())
        return image, pan, [s for s in segs if s["id"] in present]

    def __call__(self, dataset_dict):
        assert self.is_train, "CopyPastePanopticDatasetMapper is training-only"
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        pan_seg_gt = rgb2id(utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB"))
        segments_info = dataset_dict["segments_info"]

        if np.random.rand() < self.prob:
            image, pan_seg_gt, segments_info = self._paste(image.copy(), pan_seg_gt, segments_info)

        # Below mirrors MaskFormerPanopticDatasetMapper.__call__ (no sem_seg in IDD).
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        pan_seg_gt = rgb2id(transforms.apply_segmentation(id2rgb(pan_seg_gt).astype(np.uint8)))

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
        if self.size_divisibility > 0:
            h, w = image.shape[-2:]
            padding = [0, self.size_divisibility - w, 0, self.size_divisibility - h]
            image = torch.nn.functional.pad(image, padding, value=128).contiguous()
            pan_seg_gt = torch.nn.functional.pad(pan_seg_gt, padding, value=0).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])
        dataset_dict["image"] = image

        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes, masks = [], []
        for s in segments_info:
            if not s["iscrowd"]:
                classes.append(s["category_id"])
                masks.append(pan_seg_gt == s["id"])
        instances.gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)
        if masks:
            instances.gt_masks = BitMasks(torch.stack(
                [torch.from_numpy(np.ascontiguousarray(m)) for m in masks])).tensor
        else:
            instances.gt_masks = torch.zeros((0, *pan_seg_gt.shape[-2:]))
        dataset_dict["instances"] = instances
        return dataset_dict
