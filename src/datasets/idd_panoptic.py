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
    """``<scene>_<id>_gtFine_panopticlevel3Ids.png`` → RGB path.

    IDD-20k ships RGB frames as JPEG; some converted releases use PNG.
    Try .jpg first (the common case), fall back to .png.
    """
    stem = panoptic_filename.replace("_gtFine_panopticlevel3Ids.png", "")
    scene, _, image_id_short = stem.partition("_")
    base = image_root / scene / f"{image_id_short}_leftImg8bit"
    jpg = base.with_suffix(".jpg")
    if jpg.exists():
        return jpg
    return base.with_suffix(".png")


def _load_records(
    json_path: Path,
    panoptic_root: Path,
    image_root: Path,
    dataset_to_contig: dict[int, int],
) -> list[dict[str, Any]]:
    """Build dataset records for detectron2.

    Crucially, ``segments_info[*].category_id`` is mapped from dataset_id
    (level3Id, 0..25) to the model's contiguous output index using the
    metadata's combined thing+stuff map. The Mask2Former panoptic mapper
    passes ``segment_info["category_id"]`` directly into ``Instances.gt_classes``
    without applying any further mapping, so we MUST do the mapping here
    or the model trains on the wrong class targets.
    """
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
                    "category_id": dataset_to_contig[int(s["category_id"])],
                    "iscrowd": int(s.get("iscrowd", 0)),
                }
                for s in ann["segments_info"]
            ],
        })
    return records


def _build_metadata(json_path: Path) -> dict[str, Any]:
    """Build the MetadataCatalog payload from our JSON's category list.

    Mask2Former panoptic convention (matches detectron2's COCO-panoptic
    registration in builtin_meta.py):
      - ``thing_classes``: just the thing names, indexed [0..N_th-1].
      - ``stuff_classes``: thing names FIRST, then stuff names, indexed
        [0..N_th-1] for things, [N_th..N_th+N_st-1] for stuff. These are
        the model's output channels (NUM_CLASSES = N_th + N_st).
      - ``thing_dataset_id_to_contiguous_id``: dataset_id → [0..N_th-1].
      - ``stuff_dataset_id_to_contiguous_id``: dataset_id → [N_th..N_th+N_st-1].

    The previous version had thing and stuff contig spaces both starting
    at 0, so contig_id=4 ambiguously meant "autorickshaw (thing)" and
    "drivable fallback (stuff)". The model couldn't tell things from
    stuff at the class head, collapsing all queries onto sky/vegetation.
    """
    with open(json_path) as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])

    thing_cats = [c for c in cats if c["isthing"]]
    stuff_only_cats = [c for c in cats if not c["isthing"]]

    thing_classes = [c["name"] for c in thing_cats]
    thing_colors = [tuple(c["color"]) for c in thing_cats]
    thing_id_map = {c["id"]: i for i, c in enumerate(thing_cats)}

    n_th = len(thing_cats)
    stuff_classes = thing_classes + [c["name"] for c in stuff_only_cats]
    stuff_colors = thing_colors + [tuple(c["color"]) for c in stuff_only_cats]
    stuff_id_map = {c["id"]: n_th + i for i, c in enumerate(stuff_only_cats)}
    # Things must also be looked-up-able through stuff_dataset_id_to_contiguous_id
    # because the panoptic mapper uses it as the global contiguous space.
    stuff_id_map.update(thing_id_map)

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

        meta_kwargs = _build_metadata(json_path)
        # The panoptic mapper feeds segments_info[*].category_id straight into
        # gt_classes — no remap. Apply the dataset_id -> contiguous mapping at
        # registration time using the combined stuff map (which includes things).
        d2c = dict(meta_kwargs["stuff_dataset_id_to_contiguous_id"])
        DatasetCatalog.register(
            name,
            lambda jp=json_path, pr=panoptic_root, ir=image_root, m=d2c:
                _load_records(jp, pr, ir, m),
        )
        MetadataCatalog.get(name).set(
            json_file=str(json_path),
            # detectron2's COCOPanopticEvaluator reads panoptic_json (not json_file)
            # and panoptic_root for the GT side of the eval.
            panoptic_json=str(json_path),
            panoptic_root=str(panoptic_root),
            image_root=str(image_root),
            evaluator_type="coco_panoptic_seg",
            **meta_kwargs,
        )

    _IDD_PANOPTIC_REGISTERED = True


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent.parent
    register_idd_panoptic(here)
    for n in ("idd_panoptic_train", "idd_panoptic_val"):
        recs = DatasetCatalog.get(n)
        sample_classes = sorted({s["category_id"] for s in recs[0]["segments_info"]})
        print(f"{n}: {len(recs)} records, first sample contig classes: {sample_classes}")
