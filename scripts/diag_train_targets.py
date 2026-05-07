"""Print what training targets the model actually sees.

The model trains on Mask2Former's MaskFormerPanopticDatasetMapper output,
not the raw GT JSON. If that mapper is producing wrong labels for things,
training-loss converges happily but the model learns the wrong objective
(e.g. all things become 'no object').

For 4 train images that have thing instances, prints:
  - GT segments from JSON (with category_id in dataset_id space)
  - Mapper output: instances tensor with mapped contiguous category_ids
  - Mismatches if any (e.g. thing segments dropped by the mapper)

Run:
    python3 scripts/diag_train_targets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "Mask2Former"))

import src._silence_warnings  # noqa: E402,F401
from mask2former import add_maskformer2_config  # noqa: E402,F401
from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import (  # noqa: E402
    MaskFormerPanopticDatasetMapper,
)
from src.datasets.idd_panoptic import register_idd_panoptic  # noqa: E402

register_idd_panoptic(REPO)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402


CONFIG = REPO / "configs" / "m2f_swinl_idd_panoptic.yaml"


def main() -> int:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(CONFIG))
    cfg.freeze()

    meta = MetadataCatalog.get("idd_panoptic_train")
    things_dataset = set(meta.thing_dataset_id_to_contiguous_id.keys())
    thing_d2c = meta.thing_dataset_id_to_contiguous_id
    stuff_d2c = meta.stuff_dataset_id_to_contiguous_id
    cat_names = {i: n for i, n in enumerate(meta.stuff_classes)}

    mapper = MaskFormerPanopticDatasetMapper(cfg, is_train=True)

    # find a few training records with thing instances
    train_recs = DatasetCatalog.get("idd_panoptic_train")
    target_recs = []
    for rec in train_recs:
        thing_segs = [s for s in rec["segments_info"] if s["category_id"] in things_dataset]
        if 2 <= len(thing_segs) <= 6:
            target_recs.append(rec)
        if len(target_recs) >= 4:
            break

    for rec in target_recs:
        print(f"\n=== {Path(rec['file_name']).name} ===")
        print(f"  GT segments_info ({len(rec['segments_info'])}):")
        for s in rec["segments_info"]:
            cat = s["category_id"]
            kind = "thing" if cat in things_dataset else "stuff"
            cat_in_contig = (thing_d2c if kind == "thing" else stuff_d2c).get(cat, "??")
            print(f"    cat_dataset={cat:3d} {kind} -> contig={cat_in_contig}  iscrowd={s.get('iscrowd',0)}")

        # Run the mapper
        mapped = mapper(dict(rec))  # mapper mutates so pass a copy
        instances = mapped.get("instances")
        sem_seg_gt = mapped.get("sem_seg_gt")
        print(f"  Mapper produced:")
        print(f"    image shape:        {tuple(mapped['image'].shape)}")
        if instances is not None:
            print(f"    instances count:    {len(instances)}")
            classes = instances.gt_classes.tolist() if hasattr(instances, "gt_classes") else "?"
            print(f"    instance classes (contig): {classes}")
            if hasattr(instances, "gt_masks"):
                # gt_masks is BitMasks; tensor shape (N, H, W)
                masks = instances.gt_masks.tensor if hasattr(instances.gt_masks, "tensor") else instances.gt_masks
                print(f"    instance mask shape: {tuple(masks.shape)}, sum per mask: {masks.sum(dim=(1,2)).tolist()}")
        else:
            print(f"    NO INSTANCES (instances=None or empty)")
        if sem_seg_gt is not None:
            uniq = sorted(int(x) for x in np.unique(sem_seg_gt).tolist())
            print(f"    sem_seg_gt unique labels: {uniq}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
