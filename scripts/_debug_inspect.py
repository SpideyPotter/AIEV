"""Debug: inspect one source instance PNG and time the per-image work."""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "external", "public-code", "helpers"))
sys.path.insert(0, os.path.join(REPO, "external", "public-code", "preperation"))

DATA = os.path.join(REPO, "data", "idd_full", "gtFine", "train")
files = sorted(glob.glob(os.path.join(DATA, "*", "*_gtFine_instancelevel3Ids.png")))
print(f"found {len(files)} instance PNGs in {DATA}")

f = files[0]
print(f"\n--- file: {f}")

t0 = time.perf_counter()
img = Image.open(f)
print(f"open: {time.perf_counter()-t0:.3f}s  mode={img.mode}  size={img.size}")

t0 = time.perf_counter()
img2 = img.resize((1280, 720))
print(f"resize default: {time.perf_counter()-t0:.3f}s  mode={img2.mode}  size={img2.size}")

t0 = time.perf_counter()
img3 = img.resize((1280, 720), Image.NEAREST)
print(f"resize NEAREST: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
arr = np.array(img2)
print(f"np.array: {time.perf_counter()-t0:.3f}s  dtype={arr.dtype}  shape={arr.shape}")

t0 = time.perf_counter()
arr_n = np.array(img3)
print(f"np.array NEAREST: {time.perf_counter()-t0:.3f}s  dtype={arr_n.dtype}")

t0 = time.perf_counter()
u = np.unique(arr)
print(f"unique (default-resized): {time.perf_counter()-t0:.3f}s  count={len(u)}  range={u.min()}..{u.max()}")

t0 = time.perf_counter()
u_n = np.unique(arr_n)
print(f"unique (NEAREST-resized): {time.perf_counter()-t0:.3f}s  count={len(u_n)}  range={u_n.min()}..{u_n.max()}")

# Time the per-segment loop (mask + bbox) — this is the suspected bottleneck
print("\n--- per-segment work timing ---")
t0 = time.perf_counter()
for el in u:
    mask = arr == el
    area = int(np.sum(mask))
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
print(f"loop over {len(u)} segs (default-resized): {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
for el in u_n:
    mask = arr_n == el
    area = int(np.sum(mask))
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
print(f"loop over {len(u_n)} segs (NEAREST-resized): {time.perf_counter()-t0:.3f}s")
