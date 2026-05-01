"""Register IDD-panoptic with Detectron2's catalog.

Side-effect: calling ``register_idd_panoptic(repo_root)`` registers the
splits ``idd_panoptic_train`` and ``idd_panoptic_val`` with
``DatasetCatalog`` and ``MetadataCatalog``.

The panoptic JSONs we generated have ``images.file_name`` pointing at the
panoptic GT PNG (``<scene>_<id>_gtFine_panopticlevel3Ids.png``). The
matching RGB image lives at::

    <image_root>/<scene>/<id>_leftImg8bit.png

This is *not* the standard COCO-panoptic flat layout, so we build the
records manually rather than using ``register_coco_panoptic``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from detectron2.data import DatasetCatalog, MetadataCatalog

_IDD_PANOPTIC_REGISTERED = False


def _resolve_rgb_path(panoptic_filename: str, image_root: Path) -> Path:
    """``<scene>_<id>_gtFine_panopticlevel3Ids.png`` → RGB path."""
    stem = panoptic_filename.replace("_gtFine_panopticlevel3Ids.png", "")
    scene, _, image_id_short = stem.partition("_")
    return image_root / scene / f"{image_id_short}_leftImg8bit.png"


def _load_records(json_path: Path, panoptic_root: Path, image_root: Path) -> list[dict[str, Any]]:
    with open(json_path) as f:
        data = json.load(f)
    images = {im["id"]: im for im in data["images"]}

    records: list[dict[str, Any]] = []
    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        rgb = _resolve_rgb_path(img["file_name"], image_root)
        pan = panoptic_root / img["file_name"]
        records.append({
            "file_name": str(rgb),
            "image_id": img["id"],
            "height": int(img["height"]),
            "width": int(img["width"]),
            "pan_seg_file_name": str(pan),
            "segments_info": [
                {
                    "id": int(s["id"]),
                    "category_id": int(s["category_id"]),
                    "iscrowd": int(s.get("iscrowd", 0)),
                }
                for s in ann["segments_info"]
            ],
        })
    return records


def _build_metadata(json_path: Path) -> dict[str, Any]:
    """Build the MetadataCatalog payload from our JSON's category list.

    Conventions Mask2Former / Detectron2 expect for panoptic datasets:
      - ``stuff_classes``: every class (things + stuff), in contiguous order.
      - ``thing_classes``: subset that are things, in their own contiguous order.
      - ``stuff_dataset_id_to_contiguous_id``: dataset (level3Id) → contiguous idx
      - ``thing_dataset_id_to_contiguous_id``: dataset (level3Id) → contiguous idx
        within ``thing_classes``.
    """
    with open(json_path) as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])

    stuff_classes = [c["name"] for c in cats]
    stuff_colors = [tuple(c["color"]) for c in cats]
    stuff_id_map = {c["id"]: i for i, c in enumerate(cats)}

    thing_cats = [c for c in cats if c["isthing"]]
    thing_classes = [c["name"] for c in thing_cats]
    thing_colors = [tuple(c["color"]) for c in thing_cats]
    thing_id_map = {c["id"]: i for i, c in enumerate(thing_cats)}

    return {
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "thing_dataset_id_to_contiguous_id": thing_id_map,
        "stuff_dataset_id_to_contiguous_id": stuff_id_map,
        "ignore_label": 255,
        "label_divisor": 1000,
    }


def register_idd_panoptic(repo_root: str | Path) -> None:
    global _IDD_PANOPTIC_REGISTERED
    if _IDD_PANOPTIC_REGISTERED:
        return

    repo_root = Path(repo_root).resolve()
    data_root = repo_root / "data" / "idd_full"

    for split in ("train", "val"):
        name = f"idd_panoptic_{split}"
        json_path = data_root / "gtFine" / f"{split}_panoptic.json"
        panoptic_root = data_root / "gtFine" / f"{split}_panoptic"
        image_root = data_root / "leftImg8bit" / split

        if not json_path.exists():
            print(f"[idd_panoptic] WARN: {json_path} not found — skip {name}",
                  file=sys.stderr)
            continue

        DatasetCatalog.register(
            name,
            lambda jp=json_path, pr=panoptic_root, ir=image_root:
                _load_records(jp, pr, ir),
        )
        MetadataCatalog.get(name).set(
            json_file=str(json_path),
            panoptic_root=str(panoptic_root),
            image_root=str(image_root),
            evaluator_type="coco_panoptic_seg",
            **_build_metadata(json_path),
        )

    _IDD_PANOPTIC_REGISTERED = True


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent.parent
    register_idd_panoptic(here)
    for n in ("idd_panoptic_train", "idd_panoptic_val"):
        recs = DatasetCatalog.get(n)
        print(f"{n}: {len(recs)} records, first: {recs[0]['file_name']}")
