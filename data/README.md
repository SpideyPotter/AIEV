# Dataset

This directory holds the **IDD 20k** dataset for the AutoNUE '19 panoptic
segmentation challenge. The dataset is **not** committed to git.

## What you should already have

After downloading both parts from <http://idd.insaan.iiit.ac.in/> and
extracting them here, you should see:

```
data/
├── IDD_Segmentation/      # Part I  (released 2018)
│   ├── gtFine/{train,val,test}/<scene_id>/...
│   └── leftImg8bit/{train,val,test}/<scene_id>/...
└── idd20kII/              # Part II (released 2019)
    ├── gtFine/{train,val,test}/<scene_id>/...
    └── leftImg8bit/{train,val,test}/<scene_id>/...
```

Both parts use disjoint scene IDs, so they can be merged into a single root.

## Step 1 — Merge Part I + Part II

`createLabels.py` expects one `--datadir` containing both `gtFine/` and
`leftImg8bit/`. Run:

```bash
python scripts/merge_idd.py
```

This creates `data/idd_full/` with the combined `gtFine/` and `leftImg8bit/`
trees. By default it uses **hardlinks** on Windows/NTFS (no extra disk usage),
falling back to copies if hardlinking fails. Pass `--copy` to force copies.

## Step 2 — Generate panoptic ground truth

```bash
bash scripts/prepare_panoptic.sh
# or with explicit args:
# bash scripts/prepare_panoptic.sh data/idd_full 4
```

This runs `external/public-code/preperation/createLabels.py` with
`--id-type level3Id --panoptic True` and produces, inside
`data/idd_full/gtFine/`:

- `train_panoptic/` and `val_panoptic/` — per-image panoptic PNGs
- `train_panoptic.json` and `val_panoptic.json` — COCO-style panoptic JSON

These are the inputs the panoptic API will compare predictions against.

## Notes

- The leaderboard benchmarks at **1280×720**. The official prep script writes
  GT at that resolution.
- Things classes are level3Ids 4–12 (vehicles + living things); everything
  else is stuff.
- The test split has no ground-truth labels — predictions are uploaded to
  the leaderboard for evaluation.
