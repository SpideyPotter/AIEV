# Experiment Log

Chronological record of decisions, runs, and results for the AutoNUE '19
panoptic segmentation project. Append new entries at the bottom. Every
non-trivial decision, command, or run goes here — even the failed ones.

Format per entry:
```
## YYYY-MM-DD — short title
**What:** what we did
**Why:** the motivation
**Result:** outcome / metrics / errors
**Next:** what this unlocks or surfaces
```

---

## 2026-04-29 — Repo scaffolding

**What:**
- Started fresh in `d:\college\VI-sem\PRJ\FinalTry` (empty directory).
- Initialised git on `main` branch.
- Created skeleton: `data/`, `external/`, `src/{datasets,models}/`, `scripts/`,
  `configs/`, `notebooks/`, `outputs/`.
- Cloned reference repos shallow (`--depth 1`) into `external/`:
  - `external/public-code/` → AutoNUE/public-code (GT preparation)
  - `external/panopticapi/` → AutoNUE/panopticapi (official evaluation)
- Authored `README.md`, `data/README.md`, `requirements.txt`,
  `.gitignore`, `.gitattributes`.
- Authored helper scripts:
  - `scripts/merge_idd.py` — merges Part I + Part II into `data/idd_full/`
    via NTFS hardlinks (no extra disk usage); `--copy` to force copies.
  - `scripts/prepare_panoptic.sh` — wraps `createLabels.py` with right
    `PYTHONPATH` and defaults.
  - `scripts/evaluate.sh` — wraps `panopticapi.evaluation` with split-aware
    default paths.

**Why:** Project is starting from zero. We need a clean structure that mirrors
the official pipeline (merge → GT prep → train → predict → eval → submit) so
each phase can be exercised independently.

**Result:** 13 files staged in git, none committed yet (deferred to user).
Tree:
```
.gitattributes  .gitignore  README.md  requirements.txt
configs/   data/README.md   notebooks/   outputs/
scripts/{merge_idd.py, prepare_panoptic.sh, evaluate.sh}
src/{__init__.py, datasets/__init__.py, models/__init__.py}
```

**Next:**
- Set up Python environment (next entry).
- Run `merge_idd.py` and `prepare_panoptic.sh` once env is ready.

---

## 2026-04-29 — Decisions register

Recording non-obvious choices made during scaffolding:

| Decision | Choice | Why |
|---|---|---|
| External repos: submodule vs clone+gitignore | clone + `.gitignore external/` | Simpler for solo college project; re-clone if wiped. |
| Merge strategy for Part I + Part II | `os.link` hardlinks (NTFS) → fallback to `shutil.copy2` | 20k+ files; hardlinks are instant and use no extra disk. |
| Pin `pandas==1.2.1` / `scipy==1.1.0` from public-code reqs? | No, leave unpinned | Old pins fail on modern Python; pandas only used in unsup_da/semisup_da paths we don't run. |
| Where to install `panopticapi` from | `git+...` URL in requirements.txt | Matches task brief; local `pip install -e ./external/panopticapi` is a documented fallback. |
| Init `src/models/` with anything? | Empty `__init__.py` only | "Decide on the model later" — no premature scaffolding. |
| Line endings | `.gitattributes` forces LF on `.sh`/`.py` | Bash chokes on `\r` in shebangs on Windows. |

---

## 2026-04-29 — Conda env: `autonue` (Python 3.10)

**What:**
- System Python on this machine is 3.14, too new for PyTorch / Detectron2 /
  MMDet wheels (currently target 3.10–3.12).
- Created conda env `autonue` with Python 3.10:
  ```
  conda create -n autonue python=3.10 -y
  conda activate autonue
  pip install -r requirements.txt
  ```
- Smoke-tested by importing every package; all clean.

**Why:** Need one env that works for both data prep *and* the eventual model
training, so we don't context-switch between two interpreters. 3.10 is the
sweet spot — broad framework support, still actively patched.

**Existing related envs on this machine** (left untouched):
- `idd-infer` — likely past inference work on IDD
- `idd_seg` — likely past IDD segmentation work
  Worth probing later if we need to crib pip-freeze data, but `autonue` is
  intentionally fresh.

**Result:**
- `autonue` created at `C:\Users\KARTHIK\miniconda3\envs\autonue` with
  **Python 3.10.20**.
- Installed packages (versions pinned by what pip resolved on 2026-04-29):
  - numpy 2.2.6, Pillow 12.2.0, tqdm 4.67.3, imageio 2.37.3
  - numpngw 0.1.4, pandas 2.3.3, opencv-python 4.13.0.92
  - panopticapi 0.1 (built from git+AutoNUE/panopticapi)
  - transitive: colorama, python-dateutil 2.9, pytz 2026.1, six 1.17, tzdata 2026.2
- Smoke test:
  ```
  import sys, numpy, PIL, imageio, numpngw, pandas, cv2, panopticapi
  → python 3.10.20, numpy 2.2.6, panopticapi loads from autonue env
  ```

**Next:**
- Run `python scripts/merge_idd.py` → produces `data/idd_full/`.
- Run `bash scripts/prepare_panoptic.sh` → produces panoptic GT.
- Both must run with `conda activate autonue` first.

---

---

## 2026-04-29 — Merge Part I + Part II

**What:** Ran `python scripts/merge_idd.py` inside `autonue` env. Merges
`data/IDD_Segmentation/` and `data/idd20kII/` into `data/idd_full/` via
hardlinks.

**Why:** `createLabels.py` needs a single `--datadir` containing `gtFine/` +
`leftImg8bit/`. Both parts use disjoint scene IDs, so they can sit side-by-
side under one root.

**Result:**
- IDD_Segmentation: 25,951 files merged (~17 s)
- idd20kII: 18,187 files merged (~12 s)
- **Total: 44,138 linked, 0 copied, 0 skipped** — no name collisions; all
  hardlinks succeeded on NTFS.
- Output structure (folder-level, images not enumerated):
  ```
  data/idd_full/
  ├── gtFine/
  │   ├── train/   369 scenes
  │   └── val/      72 scenes
  └── leftImg8bit/
      ├── train/   369 scenes
      ├── val/      72 scenes
      └── test/    111 scenes
  ```
- Note: `gtFine/test` is absent by design — IDD test split has no public
  ground truth; predictions go to the leaderboard.

**Next:** Run `bash scripts/prepare_panoptic.sh` to generate panoptic GT for
train + val (createLabels.py doesn't process test since there's no GT).

---

## 2026-04-29 — Bugfix: PYTHONPATH not reaching native Python on git-bash

**What:** First run of `prepare_panoptic.sh` failed instantly with
`ModuleNotFoundError: No module named 'anue_labels'` from inside
`json2labelImg.py`, despite the script doing
`export PYTHONPATH="$ROOT/external/public-code/helpers"`.

**Root cause:** Under git-bash on Windows, `pwd` returns MSYS-form paths
like `/d/college/VI-sem/PRJ/FinalTry`. Bash internally understands `/d/`,
but native Windows binaries (the conda env's `python.exe`) do not — they
need either `D:\...` or `D:/...`. Bash *does* auto-translate command-line
arguments via MSYS, but **not environment variables**. So `PYTHONPATH`
was set to `/d/.../helpers`, which `python.exe` couldn't resolve.

`bash -x` trace confirmed the export line:
```
+ export PYTHONPATH=/d/college/VI-sem/PRJ/FinalTry/external/public-code/helpers:
```

**Fix:** Convert ROOT with `cygpath -m` (mixed form: `D:/college/...`)
when `cygpath` is available, and use `;` as PYTHONPATH separator on
Windows instead of `:`. Applied to both `scripts/prepare_panoptic.sh`
and `scripts/evaluate.sh`. Verified: `python -c "import anue_labels"`
now succeeds with the converted PYTHONPATH.

**Why this is worth remembering:** Any future bash script that exports
file paths via env vars to native Windows binaries will hit the same
issue. Default to cygpath-translation in script preambles if the same
pattern shows up.

---

## 2026-04-29 — Bugfix: public-code is incompatible with modern Pillow

**What:** After fixing PYTHONPATH, the prep script crashed with:
```
Please install the module 'Pillow' for image processing, e.g.
pip install pillow
```
even though Pillow 12.2 is installed.

**Root cause:** `preperation/json2labelImg.py` and
`preperation/json2instanceImg.py` both do `from PIL import PILLOW_VERSION`,
which was **removed in Pillow 7.0** (early 2020) in favour of
`PIL.__version__`. The bare `except:` swallowed the ImportError and
incorrectly claimed Pillow was missing.

**Fix:** Patched both files to fall back to `__version__`:
```python
try:
    from PIL import PILLOW_VERSION
except ImportError:
    try:
        from PIL import __version__ as PILLOW_VERSION
    except ImportError:
        # original error path
        ...
```

**Reproducibility:** All patches to `external/` are codified in
`scripts/patch_external.py` (idempotent). After re-cloning public-code,
run it once.

---

## 2026-04-29 — Bugfix: public-code multiprocessing breaks on Windows

**What:** With Pillow patched, prep got past imports and started processing
16,063 polygon files, but every worker died instantly with:
```
AttributeError: 'NoneType' object has no attribute 'id_type'
  File "...createLabels.py", line 32, in process_folder
    dst = fn.replace("_polygons.json", "_label{}s.png".format(args.id_type))
```

**Root cause:** `createLabels.py` declares `args = None` at module scope and
sets it inside `if __name__ == "__main__":`. `process_folder` (run in worker
processes) reads it via `global args`. On Linux, `multiprocessing.Pool`
uses **fork**, so children inherit the parent's `args`. On Windows, it uses
**spawn** — children re-import the module fresh, so `args` is back to
`None`, and every worker call crashes on `args.id_type`.

**Fix:** Added `_init_worker(args_)` that sets the module-level `args` in
each worker, and changed
```python
pool = Pool(args.num_workers)
```
to
```python
pool = Pool(args.num_workers, initializer=_init_worker, initargs=(args,))
```

The argparse `Namespace` is picklable so it travels to workers cleanly.

**Reproducibility:** Codified in `scripts/patch_external.py`.

**Why this is worth remembering:** Any cross-platform Python project that
shares state with `multiprocessing.Pool` workers via module globals will
break on Windows. Always pass state via `initializer`/`initargs` or
function arguments.

**Result of run after this patch:** Stage 1 (semantic + instance) completed
cleanly — 16,063 polygon files in 6 min 31 s at ~41 it/s with 4 workers.
Stage 2 (panoptic conversion) hit the *same* multiprocessing pattern in a
*different* file → next entry.

---

## 2026-04-29 — Bugfix: panoptic-stage hits the same multiprocessing bug

**What:** Right after stage 1 finished, stage 2 (`panoptic_converter` in
`cityscape_panoptic_gt.py`) failed identically:
```
NameError: name 'file_list' is not defined
  File "...cityscape_panoptic_gt.py", line 34, in process_image
    f = file_list[working_idx]
```

**Root cause:** Same pattern as createLabels.py — `process_image` reads
three globals (`file_list`, `categories_dict`, `output_folder`) that
`panoptic_converter` sets locally and exposes via `global` declarations.
Spawn workers don't see them.

**Fix:** Added `_init_panoptic_worker(file_list, categories_dict, output_folder)`
and changed `pool = Pool(num_workers)` to pass it as `initializer`. Same
shape of fix as createLabels.

**Reproducibility:** Added to `scripts/patch_external.py`.

---

## 2026-04-29 — Bugfix: Windows path separators break filename construction

**What:** With multiprocessing fixed, stage 2 workers crashed with
`FileNotFoundError`. The output filename came out as e.g.
`data_idd_full\gtFine\train\0\005506_gtFine_panopticlevel3Ids.png`
— a relative path embedded *as the filename*, leading to a
non-existent target directory.

**Root cause:** `cityscape_panoptic_gt.py` builds the output filename via
`f.split('/')[-2]` and `f.split('/')[-1]`. On Windows, `glob.glob` returns
paths with backslashes, so `split('/')` returns the whole string (no `/`
to split on), and the code constructs a malformed filename with embedded
path separators.

**Fix:** Use `os.path.basename` and `os.path.dirname` instead — they
handle either separator portably:
```python
file_name = os.path.basename(f)
image_id = file_name.rsplit('_', 2)[0]
image_filename = '{}_{}_gtFine_panopticlevel3Ids.png'.format(
    os.path.basename(os.path.dirname(f)), image_id)
```

**Reproducibility:** Added to `scripts/patch_external.py`.

---

## 2026-04-29 — Bugfix: Pillow 7+ resize default makes panoptic GT 60× slower AND wrong

**What:** After the Windows-path fix, stage 2 finally produced files —
but at ~0.3 it/s, which projected to a **21-hour** run. Each
`process_image` call was taking 5–10 seconds.

**Root cause:** `cityscape_panoptic_gt.py` does
`img = img.resize((1280, 720))` without specifying a resample method.
In Pillow 7+, the default for `Image.resize` became BICUBIC for *all*
modes (older Pillow used mode-dependent defaults like NEAREST for I;16).

The instance label PNGs are I;16 mode — pixel values are
`semantic_id * 1000 + instance_id` integer codes. BICUBIC interpolates
between codes, producing **fake intermediate IDs** that don't correspond
to any segment. The downstream loop iterates over `np.unique(arr)` and
runs mask + bbox computation per ID, so phantom IDs blow up wall time
**and** corrupt the panoptic GT.

Microbenchmark on one image (`scripts/_debug_inspect.py`):
| resize | unique IDs | per-image loop |
|---|---|---|
| default (BICUBIC) | 1,811 | **7.99 s** |
| `Image.NEAREST`   | 30 | **0.13 s** |

That's a 60× speed-up *and* the correct number of segments.

**Fix:** Pass `Image.NEAREST` explicitly to `img.resize`. One-line patch.

**Reproducibility:** Added to `scripts/patch_external.py` — total of 8
patches now across 3 files.

---

## 2026-04-29 — Panoptic GT generation: success

**What:** Re-ran stage 2 only via `scripts/run_panoptic_only.py`
(skipping the now-stale 7-min stage 1 since semantic + instance PNGs
are already on disk).

**Result:**
| split | images | annotations | PNGs   | JSON size | wall time |
|-------|-------:|------------:|-------:|----------:|----------:|
| train | 14,027 | 14,027      | 14,027 | 47.0 MB   | ~26 min   |
| val   |  2,036 |  2,036      |  2,036 |  7.0 MB   |  4:07     |

Verified by `scripts/_verify_panoptic_gt.py`:
- 26 categories — **9 things (level3Ids 4–12)** + 17 stuff. Matches the
  AutoNUE spec exactly.
- All `image_id` ↔ `images.id` sets match.
- segments_info entries have required keys (id, category_id, area,
  bbox, iscrowd).
- segments per image: avg 33.7 (train) / 34.9 (val), max 226 (train).

Throughput drifted from ~15 it/s early to ~5 it/s late on train —
likely Windows directory-metadata pressure with 14k files in one
folder. Could be partly mitigated by sharding output or writing to
`data/` on a faster disk if it ever needs to be redone.

**Outputs (the inputs to model training):**
```
data/idd_full/gtFine/
├── train_panoptic/        14,027 PNGs (1280×720, 3-channel, color = id_generator output)
├── train_panoptic.json    47.0 MB COCO-panoptic format
├── val_panoptic/           2,036 PNGs
└── val_panoptic.json       7.0 MB
```

**Total bug surface for stage 2 on Windows / modern Pillow:**
1. PYTHONPATH stripped of MSYS path semantics → fixed via `cygpath -m`.
2. Pillow 7+ removed `PILLOW_VERSION` → fallback to `__version__`.
3. `multiprocessing.Pool` workers re-import the module on Windows
   (spawn) → use `initializer`/`initargs` (createLabels.py *and*
   cityscape_panoptic_gt.py).
4. `f.split('/')` breaks on Windows backslashed glob results → use
   `os.path.basename` / `os.path.dirname`.
5. Pillow 7+ defaults `Image.resize` to BICUBIC, which interpolates
   instance IDs and produces hundreds of phantom segments → pass
   `Image.NEAREST` explicitly. (60× speed-up *and* correctness.)

All five fixes are codified in `scripts/patch_external.py` (idempotent).

**Next:** Pick a model. Run `bash scripts/evaluate.sh val` once we have
val predictions to verify the eval pipeline end-to-end before training.

---

## 2026-04-29 — Model survey + recommendation

We need a panoptic-segmentation architecture that:

1. Has a pretrained Cityscapes checkpoint we can fine-tune from (IDD is
   the same domain — driving scenes — so transfer should be strong).
2. Is implemented in a maintained codebase (Detectron2 / MMDet) so we
   aren't fighting bit-rot.
3. Fits a single consumer GPU at the leaderboard's 1280×720 resolution.

### Comparison

| Model | Year | Cityscapes PQ (R50/equiv) | VRAM (train, R50, bs=2-4) | Codebase | Notes |
|---|---|---|---|---|---|
| **Panoptic FPN** | 2019 | ~58 | ~8 GB | Detectron2 | Two-branch baseline. Easiest to train. Lowest PQ. |
| **UPSNet** | 2019 | ~60 | ~10 GB | own (old) | Original AutoNUE-era pick, but PyTorch-1.x codebase rotted. **Avoid.** |
| **Panoptic-DeepLab** | 2020 | ~62–65 | ~10 GB | Detectron2 | Bottom-up single-stage, no NMS. Real-time inference. Strong on driving scenes. |
| **MaskFormer** | 2021 | ~58–63 | ~12 GB | Detectron2, MMDet | Mask-classification reframing. Superseded by Mask2Former — no reason to pick over it. |
| **Mask2Former** | 2022 | ~62 (R50), ~67–69 (Swin-B) | ~12–16 GB | Detectron2, MMDet | Current de-facto SOTA. Universal head (panoptic / instance / semantic from one arch). |
| **kMaX-DeepLab** | 2022 | ~64–68 | ~14 GB | own (TF/PyTorch ports) | Strong, but smaller community → harder debugging. |
| **OneFormer** | 2023 | ~67–70 (Swin-L) | 16 GB+ | own (built on Detectron2) | Multi-task, heavier. Diminishing returns vs Mask2Former for our goal. |

PQ numbers above are Cityscapes val with ResNet-50 backbone unless noted; treat them as ±2 since exact numbers vary across reproductions.

### Decision tree

- **VRAM ≤ 8 GB / Colab free / no dedicated GPU.** Start with **Panoptic FPN** (R50). Smallest training footprint, fastest to a working baseline. Expect modest PQ.
- **VRAM 10–12 GB (RTX 3060 / Kaggle T4 / Colab L4).** **Panoptic-DeepLab** is the sweet spot — driving-scene heritage, single-stage, well-supported in Detectron2. R50 backbone fits comfortably.
- **VRAM 16 GB+ (RTX 3090/4080/4090, Kaggle P100, A100).** **Mask2Former** with R50 or Swin-T. Highest PQ ceiling per training hour, current standard.
- **VRAM 24 GB+ and time to spare.** Mask2Former with Swin-B/L. Push for leaderboard.

### Recommendation

**Default plan: Panoptic-DeepLab (Detectron2, R50 backbone).**

Reasoning:
1. **Domain fit.** Designed and validated for driving scenes (Cityscapes,
   Mapillary). IDD is the same domain — Cityscapes-pretrained weights
   transfer directly.
2. **Compute fit.** Trains in ~10 GB VRAM at our target resolution.
   Single-stage means no proposal/NMS bottleneck — fast iteration.
3. **Codebase fit.** First-class Detectron2 support; the
   `projects/Panoptic-DeepLab` directory has working configs for
   Cityscapes that we can adapt to IDD with minimal changes (label
   mapping, dataset registration).
4. **Clear upgrade path.** If we have spare GPU time and want to push
   PQ, swap in Mask2Former later — same Detectron2 framework, same
   dataset registration, same eval pipeline. Sunk cost is small.

**Backup plan: Mask2Former** if user has ≥16 GB VRAM and the project
timeline permits ~2× longer training.

**Off the table:** UPSNet (rotted codebase), MaskFormer (just use M2F),
OneFormer (overkill for our goal).

### What we still need from the user

- **GPU spec** — confirms the tier above.
- **Compute time budget** — affects whether we go straight to
  Mask2Former or scaffold with Panoptic-DeepLab first.

Once chosen, the next entries cover:
- adding `torch` + `detectron2` (and the right CUDA wheel) to the env,
- writing an IDD-panoptic dataset registration for Detectron2,
- pulling the Cityscapes-panoptic checkpoint and adapting the head for
  26 categories instead of Cityscapes' 19.

---

## 2026-04-29 — Plan change: training moves to uni DGX (k8s, free)

User has access to a university DGX cluster with kubernetes-managed pod
allocation. Spec for our session:

- Image: `bmu-headnode:9443/nvcr.io/nvidia/pytorch:24.10-py3`
  (NGC PyTorch 24.10 = PyTorch 2.5.0a, CUDA 12.6, Python 3.10).
- MIG slice: `nvidia.com/mig-2g.35gb` — **35 GB VRAM** (~A100-class
  partition of an H200).
- 16 CPU cores, 32 GB system RAM (we use the full allotment in our pod).
- Persistent host volume: `/home/dgx-s-bmu-cse-230480/clearsar` mounted
  at `/workspace` inside the pod.
- Access: SSH to headnode → `kubectl apply` / `kubectl exec`.
  **On-campus WiFi only** — long jobs run inside the pod (survive SSH
  disconnect) or as a `kubectl Job` (fully detached).

**Cost replanning:** all paid-cloud cost discussion is moot; GPU compute
is free. Only constraint is fairness to other users (don't hog the
slice) and our own iteration time.

### Layout inside the volume (host path → pod path)

```
/home/dgx-s-bmu-cse-230480/clearsar/        →  /workspace/
├── FinalTry/                                   our repo (uploaded)
│   ├── data/idd_full/                          dataset (uploaded)
│   ├── external/Mask2Former/                   cloned by setup_dgx.sh
│   ├── pretrain/swinl_cityscapes_panoptic.pkl  fetched by download_pretrain.py
│   └── outputs/                                checkpoints + predictions
└── .venv/                                      python env (created by setup_dgx.sh)
```

### Files added this iteration

- `k8s/pod_dev.yaml`     — long-running interactive dev pod (matches the
  template the cluster already uses, scaled to the full 16 CPU / 32 Gi
  / 1× 2g.35gb slice; adds a `dshm` emptyDir for Detectron2's dataloader).
- `k8s/job_train.yaml`   — batch Job for unattended training; same
  resources, runs `scripts/train.sh` and tees output to
  `outputs/train.log`.
- `scripts/setup_dgx.sh` — one-time install inside the dev pod: venv at
  `/workspace/.venv` with `--system-site-packages`, then pip-installs
  detectron2 from source, clones Meta's Mask2Former repo, and compiles
  the `MultiScaleDeformableAttention` CUDA op against the container's
  PyTorch + CUDA.
- `scripts/download_pretrain.py` — fetches the Cityscapes-panoptic
  Swin-L checkpoint (`maskformer2_swin_large_IN21k_384_bs16_90k`) from
  fbaipublicfiles to `/workspace/FinalTry/pretrain/`.
- `src/datasets/idd_panoptic.py` — registers `idd_panoptic_train` and
  `idd_panoptic_val` with Detectron2's catalog. Custom RGB-path
  resolution because IDD's leftImg8bit is in scene subfolders, not the
  flat layout standard COCO-panoptic registration assumes.
- `configs/m2f_swinl_idd_panoptic.yaml` — training config inheriting
  Mask2Former's Cityscapes-panoptic Swin-L config; overrides
  `NUM_CLASSES=26`, IDD dataset names, 1280×720 input, lower LR
  (5e-5) for fine-tune, AMP enabled, 50k iters, eval every 5k.
- `scripts/train.py`, `scripts/predict.py`, `scripts/train.sh`,
  `scripts/predict.sh` — entry points. Predict writes per-image PNGs +
  COCO-panoptic JSON in the format our `evaluate.sh` already consumes.
- `scripts/_smoke_dataset.py` — CPU-only check that the registration
  loads, files exist, and metadata is sane. Run before paying GPU time.

### Workflow on the DGX

```
# 0. on laptop: git push / scp the FinalTry repo + dataset to
#    dgx-s-bmu-cse-230480@bmu-headnode:/home/.../clearsar/

# 1. on headnode:
kubectl apply -f FinalTry/k8s/pod_dev.yaml
kubectl exec -it idd-panoptic-dev -- bash

# 2. inside pod (one-time):
bash /workspace/FinalTry/scripts/setup_dgx.sh
python /workspace/FinalTry/scripts/download_pretrain.py
python /workspace/FinalTry/scripts/_smoke_dataset.py

# 3. shakedown (5 k iters, ~ 1 hr):
bash /workspace/FinalTry/scripts/train.sh SOLVER.MAX_ITER 5000

# 4. full fine-tune (detached as a Job):
kubectl apply -f /workspace/FinalTry/k8s/job_train.yaml
kubectl logs -f job/idd-panoptic-train

# 5. predict on test + submit
bash /workspace/FinalTry/scripts/predict.sh test
# upload outputs/test_pred_panoptic.json + test_pred_panoptic/ to leaderboard
```

### Risks (carry forward)

- **Detectron2 + PyTorch 2.5.0a (NGC alpha)** is a non-stable combo;
  if pip's `git+...` install of Detectron2 fails to compile, fall back
  to cloning d2 master and building with `pip install -e .`.
- **CUDA-op compile** (`MultiScaleDeformableAttention`) is the most
  fragile step. The NGC image has the right CUDA toolkit; if it bites
  we capture the error and adjust `setup.py`'s flags.
- **bs=2 at 1280×720 with Swin-L on 35 GB** — should fit with AMP, but
  is in the upper half of the slice's memory. If we OOM, options are
  (a) gradient checkpointing on Swin blocks, (b) drop crop size to
  1024, (c) request a bigger MIG slice.

---

## 2026-04-29 — Cloud GPU options + estimated costs

User-provided pricing menu (Paperspace-tier rates). Wall-time figures are
rough estimates for fine-tuning from a Cityscapes-pretrained checkpoint at
1280×720 — actual times vary ±30% with batch size, augmentation, and number
of iterations chosen. Costs are training-only (add a few extra hours for
interactive setup / debugging time).

### GPU tiers

| GPU | VRAM | Arch | Tensor cores | $/hr | $/hr×1.3 (with overhead) |
|---|---|---|---|---|---|
| RTX 5000 | 16 GB | Turing | 1st-gen | $0.82 | ~$1.07 |
| P6000 | 24 GB | Pascal | **none** | $1.10 | ~$1.43 |
| A5000 | 24 GB | Ampere | 3rd-gen | $1.38 | ~$1.79 |
| A6000 | 48 GB | Ampere | 3rd-gen | $1.89 | ~$2.46 |
| V100 16G | 16 GB | Volta | 1st-gen | $2.30 | ~$2.99 |
| V100 32G | 32 GB | Volta | 1st-gen | $2.30 | ~$2.99 |
| A100 40G | 40 GB | Ampere | 3rd-gen | $3.09 | ~$4.02 |
| A100 80G | 80 GB | Ampere | 3rd-gen | $3.18 | ~$4.13 |

Avoid:
- **V100 16G / V100 32G** at $2.30/hr — slower than the A5000 ($1.38)
  per dollar AND less VRAM (16 GB version). The 32 GB version is
  priced the same as the 16 GB. **Bad value either way.**
- **P6000** — Pascal has no Tensor cores. Real-world FP16/TF32 throughput
  is ~½ A5000, while costing only ~25 % less. Pay $0.28/hr more for the
  A5000 and finish 2× faster.

### Costed scenarios

Assumptions: Cityscapes-pretrained init; ~50k iterations for a baseline,
~100k for a leaderboard push. Per-iter throughput differs by GPU.

| Plan | GPU | Model | Train iters | Wall-time est. | Cost est. |
|---|---|---|---|---|---|
| **Cheap baseline** | RTX 5000 | Panoptic-DeepLab R50 (bs=2) | 50 k | 10–14 hr | **$8–12** |
| **Recommended baseline** | A5000 | Panoptic-DeepLab R50 (bs=4) | 50 k | 6–8 hr | **$8–11** |
| **Strong baseline** | A5000 | Mask2Former R50 (bs=2) | 50 k | 10–14 hr | **$14–19** |
| **Leaderboard push** | A6000 | Mask2Former Swin-B (bs=2) | 100 k | 18–24 hr | **$34–45** |
| **Fastest leaderboard** | A100 80G | Mask2Former Swin-L (bs=4) | 100 k | 10–14 hr | **$32–45** |

Add ~30 % overhead to every figure for interactive setup, env install,
data upload, debugging stops/starts. Most clouds bill per second and
charge zero while shut down — *always shut down between sessions*.

Storage + ingress: the merged IDD dataset is ~30 GB; first upload to the
cloud volume takes 30–90 min depending on link, billed at storage rate
(~$0.10/GB/month most providers).

### Recommendation

**A5000 ($1.38/hr)** is the right default for this project.

Reasons:
1. **24 GB VRAM** comfortably fits Panoptic-DeepLab R50 *and* Mask2Former
   R50 / Swin-T at bs ≥ 2 at 1280×720.
2. **Ampere + 3rd-gen tensor cores** → mixed-precision (FP16/TF32) gives
   real speedups on M2F's transformer decoder. The V100 has 1st-gen TC
   and is meaningfully slower per token.
3. **Best $/throughput in the menu** for our workload — A6000 is the
   same arch but 37 % more expensive without buying meaningful new
   capability for R50/Swin-T.

Use cheaper RTX 5000 *only* if budget is the dominant constraint and we
commit upfront to Panoptic-DeepLab R50 (M2F Swin-T fits 16 GB only with
careful tuning). Save A6000/A100 for a final leaderboard push if PQ on
the val split looks promising.

### Suggested staged plan

| Stage | GPU | Goal | Est. cost |
|---|---|---|---|
| 1. Pipeline shakedown | A5000 | Wire training loop, run 5 k iters, confirm loss decreases, evaluate on val. | $3–5 |
| 2. Baseline | A5000 | Full Panoptic-DeepLab R50 fine-tune from Cityscapes, evaluate on val. | $8–11 |
| 3. (Optional) push | A5000 or A6000 | Mask2Former R50 → Swin-B if PQ < target. | $14–45 |
| **Total**  |  |  | **~$25–60** |

If we run into memory issues at stage 2, fall back to RTX 5000 +
Panoptic-DeepLab only (≈$10 total) and skip stage 3.

---

## 2026-04-29 — Pre-training GT verification (deep check)

Before booking GPU time, ran `scripts/_verify_panoptic_gt.py` to cross-check
the generated GT against the canonical `anue_labels.py` spec and to round-trip
the panoptic PNG↔JSON encoding on a random sample.

**Categories:** 26/26 exact match against the level3Id spec.
- Things (9): `{4, 5, 6, 7, 8, 9, 10, 11, 12}` — person, rider, motorcycle,
  bicycle, autorickshaw, car, truck, bus, vehicle-fallback
- Stuff (17): `{0..3, 13..25}` — road, parking, sidewalk, rail-track, curb,
  wall, fence, guard-rail, billboard, traffic-sign/light, pole, obs-str-bar,
  building, bridge/tunnel, vegetation, sky/fallback-bg

**Annotation integrity (both splits):**
- 0 duplicate `segment_id` values within the same image (across 16,063 images).
- 0 `category_id` values outside the expected set.
- 0 `segment_id == 0` entries (would be invalid in COCO panoptic).
- 12 empty annotations (9 train, 3 val; 0.07 % of total) — scenes where all
  pixels are in `ignoreInEval` classes. Negligible.
- Train: 325,848 thing instance segs + 147,366 stuff segs.
- Val:    49,791 thing instance segs +  21,247 stuff segs.

**PNG ↔ JSON round-trip** (8 train + 8 val sampled, seed=42):
- 16 / 16 PNGs decoded and segment-id sets matched JSON exactly (no phantoms,
  no missing).
- Sampled thing segments: recomputed area + bbox from the decoded mask matched
  JSON values *byte-for-byte*. The NEAREST-resize patch re-write fully
  overwrote the BICUBIC residue.

**Side observations (not bugs):**
- Zero crowd-flagged thing segments (`iscrowd=1`) across 375k+ thing segs:
  IDD annotators always assign explicit instance IDs to thing pixels, so the
  `is_crowd = 1` branch in `cityscape_panoptic_gt.py` (triggered when pixel
  value < 1000) never fires. Consistent with high-quality annotation; not
  a code bug.
- Pixel encoding follows COCO panoptic (`segment_id = R + 256·G + 256²·B`),
  with `id_generator` producing colors close to the category palette plus
  small per-instance offsets.

**Verdict:** GT is correct. Safe to train against.

---

## 2026-04-29 — Decision: **Mask2Former + Swin-L**

User opted to skip the staged Panoptic-DeepLab → M2F path and go
directly to Mask2Former Swin-L for a leaderboard-grade run.

### Implications

- **Backbone size:** Swin-L is ~197 M params, plus ~30 M for the M2F
  pixel + transformer decoder. ≈230 M total trainable.
- **VRAM at 1280×720:**
  - bs=2, AMP/FP16: ~36–44 GB (fits A6000 / A100-40G tight / A100-80G)
  - bs=4, AMP/FP16: ~64–72 GB (only A100-80G)
- **GPU pick:** **A6000 ($1.89/hr)** is the cost-effective fit. Move to
  A100-80G ($3.18/hr) only if A6000 OOMs at bs=2 or if we want bs=4 to
  finish faster.
- **Compile-time gotcha:** Mask2Former uses a custom CUDA op (multi-scale
  deformable attention) that must be compiled against the cloud machine's
  CUDA + PyTorch versions. Cloud images usually have the CUDA toolkit
  pre-installed, but we'll budget time for the build.
- **Pretrained init (mandatory):** Training Swin-L from ImageNet/COCO
  scratch on a single GPU is impractical (days, $$$). We fine-tune from
  the **Cityscapes-panoptic Swin-L M2F checkpoint** — same domain, head
  is the only thing that needs adaptation (Cityscapes 19 → IDD 26 cats).
- **Cost estimate (training only):**
  - Sanity-check run (5 k iters, A6000): ~3 hr → **~$6**
  - Full fine-tune (~80 k iters, A6000, bs=2): ~22–28 hr → **~$42–53**
  - Same on A100-80G bs=4: ~12 hr → **~$38**
  - Add ~30 % for setup/debug overhead → **realistic $55–80 total**.

### Why this is defensible despite the cost

1. Swin-L M2F is currently the strongest publicly-supported panoptic
   model with a Cityscapes checkpoint we can warm-start from. The next
   tier up (OneFormer, kMaX) is more code complexity for marginal PQ.
2. The leaderboard pre-existing top scores (AutoNUE 2019) used much
   weaker baselines; M2F Swin-L should comfortably beat them.
3. We get one model to evaluate end-to-end. No baseline switch.

### Risk + mitigations

| Risk | Mitigation |
|---|---|
| OOM on A6000 at bs=2 | Use AMP / `torch.cuda.amp`, gradient checkpointing on Swin blocks; if still OOM, jump to A100-80G. |
| Custom-op compile fails on cloud image | Pin exact PyTorch + CUDA versions used by detectron2's prebuilt wheels; document in REPRO.md if it bites. |
| Training diverges from Cityscapes init | Lower LR by 10× from default M2F config; monitor val PQ every 5 k iters. |
| Run out of budget mid-train | Save checkpoint every 5 k iters; can resume on a fresh instance. |

### Path forward (next entries will cover each)

**Local prep (no GPU needed, free):**
1. Add `external/Mask2Former/` (clone Meta's repo) to our `external/`.
2. Write IDD-panoptic dataset registration code (`src/datasets/idd_panoptic.py`)
   that registers `idd_panoptic_train` / `idd_panoptic_val` with Detectron2's
   `DatasetCatalog`.
3. Author training config (`configs/m2f_swinl_idd_1280x720.yaml`)
   inheriting from Mask2Former's Cityscapes-panoptic Swin-L config:
   adjust num_classes 19 → 26, dataset names, weight paths.
4. Write helper scripts:
   - `scripts/download_pretrain.py` — fetch Cityscapes M2F Swin-L weights.
   - `scripts/train.py` — Detectron2 trainer wrapper.
   - `scripts/predict.py` — inference + write COCO-panoptic JSON
     compatible with our `scripts/evaluate.sh`.
5. Smoke-test locally: dataset registration loads, config parses, sample
   image preprocesses correctly (CPU only).

**Cloud (paid, ~$55–80 total):**
6. Spin up Paperspace **A6000** instance. Upload merged IDD + GT.
7. Install PyTorch (matching CUDA), Detectron2 (matching PyTorch),
   compile Mask2Former custom CUDA op.
8. Sanity-check: 5k-iter shakedown run, eval on val to confirm PQ
   trajectory.
9. Full fine-tune from Cityscapes Swin-L pretrain.
10. Predict on test split, package COCO-panoptic JSON, submit to
    [leaderboard](http://idd.insaan.iiit.ac.in/evaluation/submission/submit/).

---

## Open questions (carry forward)

- **GPU access?** Lab machine, Colab, Kaggle, or Paperspace? Decides batch
  size and viable model class.
- **Model choice.** Candidates to evaluate:
  - Panoptic-DeepLab — lighter, easier to get running on modest GPU.
  - Mask2Former — current SOTA on panoptic; needs Detectron2 + decent VRAM.
  - UPSNet — older, simpler reference baseline; useful for sanity-checking
    pipeline correctness before SOTA.
- **Submission cadence.** AutoNUE leaderboard accepts test-split predictions
  only; need to budget some attempts for hyperparameter tuning before a final
  submit.

---

## Submission log

_(populated when we start submitting to the leaderboard)_

| Date | Model | Train recipe | Val PQ | Test PQ (LB) | Notes |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
