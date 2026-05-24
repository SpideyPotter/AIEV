# AutoNUE '19 — Panoptic Segmentation

VI-sem PRJ. Goal: build a panoptic segmentation model for the **IDD 20k** dataset
(India Driving Dataset) and submit to the [AutoNUE '19 leaderboard](http://idd.insaan.iiit.ac.in/evaluation/submission/submit/).

In panoptic segmentation, every pixel gets a class label, and every instance of
a *thing* class (vehicles + living things, level3Ids 4–12) gets its own ID.

## Repo layout

```
FinalTry/
├── data/                     # Dataset (gitignored)
│   ├── IDD_Segmentation/     # Part I (you placed this)
│   ├── idd20kII/             # Part II (you placed this)
│   └── idd_full/             # Merged root for createLabels.py (generated)
├── external/                 # Reference repos from AutoNUE (gitignored)
│   ├── public-code/          # Ground-truth prep (createLabels.py)
│   └── panopticapi/          # Official evaluation
├── src/
│   ├── datasets/             # IDD panoptic dataset loader (TODO)
│   └── models/               # Model architecture (model TBD)
├── scripts/
│   ├── merge_idd.py          # Merge Part I + Part II into idd_full/
│   ├── prepare_panoptic.sh   # Run createLabels.py to generate panoptic GT
│   └── evaluate.sh           # Run panopticapi evaluation on predictions
├── configs/                  # Per-experiment YAML configs
├── outputs/                  # Checkpoints + predictions (gitignored)
├── notebooks/                # Exploration
├── requirements.txt
└── README.md
```

## Setup

1. **Python environment (conda, Python 3.10).** System Python is 3.14, too new
   for PyTorch / Detectron2 / MMDet wheels — so we use a conda env on 3.10
   that will host both data prep *and* the eventual model training.

   ```bash
   conda create -n autonue python=3.10 -y
   conda activate autonue
   pip install -r requirements.txt
   ```

2. **External repos.** Already cloned to `external/`. To re-clone:
   ```bash
   git clone https://github.com/AutoNUE/public-code.git external/public-code
   git clone https://github.com/AutoNUE/panopticapi.git external/panopticapi
   python scripts/patch_external.py    # apply local compat patches (idempotent)
   ```
   The patch script handles modern-Pillow compatibility and Windows
   multiprocessing in `public-code`. See [EXPERIMENTS.md](EXPERIMENTS.md)
   for what each patch does and why.

3. **Dataset.** See [data/README.md](data/README.md) for the merge + prep steps.

## Pipeline

```
[IDD Part I + Part II]
          │
          ▼  scripts/merge_idd.py
    [data/idd_full/]
          │
          ▼  scripts/prepare_panoptic.sh
[gtFine/{train,val}_panoptic/* + *.json]   ← official panoptic ground truth
          │
          ▼  (model — TBD)
   [predictions/]
          │
          ▼  scripts/evaluate.sh
        [PQ / SQ / RQ metrics]
```


## References

- Kirillov et al. *Panoptic Segmentation.* CVPR 2019. [paper](https://arxiv.org/abs/1801.00868)
- AutoNUE / IDD: <http://idd.insaan.iiit.ac.in/>
- Public code: <https://github.com/AutoNUE/public-code>
- Panoptic eval: <https://github.com/AutoNUE/panopticapi>
- COCO panoptic format: <https://research.mapillary.com/eccv18/#panoptic>
