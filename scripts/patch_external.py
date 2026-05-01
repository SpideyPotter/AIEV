#!/usr/bin/env python3
"""Apply local compatibility patches to external/public-code.

The upstream public-code (AutoNUE/public-code) was written against Python
3.7-era Pillow (< 7.0) and assumes Linux fork-based multiprocessing. On
modern Pillow + Windows we have to patch:

1. ``preperation/json2labelImg.py`` and ``preperation/json2instanceImg.py``
   — replace ``from PIL import PILLOW_VERSION`` (removed in Pillow 7) with
   a fallback to ``__version__``.
2. ``preperation/createLabels.py`` — pass ``args`` to multiprocessing
   workers via a Pool ``initializer``. On Windows, ``Pool`` uses *spawn*,
   which re-imports the module so the module-level ``args`` is None in
   workers and ``process_folder`` crashes with ``AttributeError``.

The script is idempotent: re-running on already-patched files is a no-op.
Run it whenever ``external/public-code`` is re-cloned.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUB = REPO / "external" / "public-code"

PILLOW_OLD = """try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)"""

PILLOW_NEW = """try:
    from PIL import PILLOW_VERSION
except ImportError:
    try:
        from PIL import __version__ as PILLOW_VERSION
    except ImportError:
        print("Please install the module 'Pillow' for image processing, e.g.")
        print("pip install pillow")
        sys.exit(-1)"""

INIT_OLD = """args = None


def process_folder(fn):
    global args"""

INIT_NEW = """args = None


def _init_worker(_args):
    # Windows multiprocessing uses spawn; workers re-import this module so the
    # module-level `args` would otherwise be None. Set it explicitly per worker.
    global args
    args = _args


def process_folder(fn):
    global args"""

POOL_OLD = "    pool = Pool(args.num_workers)"
POOL_NEW = "    pool = Pool(args.num_workers, initializer=_init_worker, initargs=(args,))"

PANOPTIC_INIT_OLD = """def process_image(working_idx):
    global file_list, categories_dic, output_folder
    f = file_list[working_idx]"""

PANOPTIC_INIT_NEW = """def _init_panoptic_worker(_file_list, _categories_dict, _output_folder):
    # Windows multiprocessing uses spawn; workers re-import this module so the
    # module-level globals set inside panoptic_converter would otherwise be
    # missing in workers. Set them explicitly per worker.
    global file_list, categories_dict, output_folder
    file_list = _file_list
    categories_dict = _categories_dict
    output_folder = _output_folder


def process_image(working_idx):
    global file_list, categories_dic, output_folder
    f = file_list[working_idx]"""

PANOPTIC_POOL_OLD = """    images = []
    annotations = []
    pool = Pool(num_workers)
    files = [x for x in range(len(file_list))]"""

PANOPTIC_POOL_NEW = """    images = []
    annotations = []
    pool = Pool(num_workers, initializer=_init_panoptic_worker,
                initargs=(file_list, categories_dict, out_folder))
    files = [x for x in range(len(file_list))]"""

# On Windows, glob.glob returns paths containing backslashes, which breaks
# `f.split('/')` based filename extraction. Use os.path.basename / dirname
# so the same code works on both POSIX and Windows.
PATH_OLD = """    file_name = f.split('/')[-1]
    image_id = file_name.rsplit('_', 2)[0]
    image_filename = '{}_{}_gtFine_panopticlevel3Ids.png'.format(
        f.split('/')[-2], image_id)"""

PATH_NEW = """    file_name = os.path.basename(f)
    image_id = file_name.rsplit('_', 2)[0]
    image_filename = '{}_{}_gtFine_panopticlevel3Ids.png'.format(
        os.path.basename(os.path.dirname(f)), image_id)"""

# Modern Pillow defaults Image.resize() to BICUBIC for all modes, including
# I;16 (instance label PNGs). Bicubic interpolation invents pixel values
# between real instance IDs, producing hundreds of phantom segments per
# image. This slowed the per-segment loop ~60x AND corrupted the GT.
RESIZE_OLD = """    img = Image.open(f)
    img = img.resize((1280, 720))"""

RESIZE_NEW = """    img = Image.open(f)
    # Must use NEAREST: BICUBIC (modern Pillow's default for resize) interpolates
    # instance IDs and produces hundreds of phantom segments per image, slowing
    # the per-segment loop ~60x and corrupting the panoptic GT.
    img = img.resize((1280, 720), Image.NEAREST)"""

PATCHES = [
    (PUB / "preperation" / "json2labelImg.py",        PILLOW_OLD,         PILLOW_NEW),
    (PUB / "preperation" / "json2instanceImg.py",     PILLOW_OLD,         PILLOW_NEW),
    (PUB / "preperation" / "createLabels.py",         INIT_OLD,           INIT_NEW),
    (PUB / "preperation" / "createLabels.py",         POOL_OLD,           POOL_NEW),
    (PUB / "preperation" / "cityscape_panoptic_gt.py", PANOPTIC_INIT_OLD, PANOPTIC_INIT_NEW),
    (PUB / "preperation" / "cityscape_panoptic_gt.py", PANOPTIC_POOL_OLD, PANOPTIC_POOL_NEW),
    (PUB / "preperation" / "cityscape_panoptic_gt.py", PATH_OLD,          PATH_NEW),
    (PUB / "preperation" / "cityscape_panoptic_gt.py", RESIZE_OLD,        RESIZE_NEW),
]


def apply(file: Path, old: str, new: str) -> str:
    if not file.exists():
        return f"SKIP (missing): {file}"
    text = file.read_text(encoding="utf-8")
    if new in text:
        return f"already patched: {file.relative_to(REPO)}"
    if old not in text:
        return f"NO MATCH (manual review needed): {file.relative_to(REPO)}"
    file.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"patched: {file.relative_to(REPO)}"


def main() -> int:
    failed = False
    for file, old, new in PATCHES:
        result = apply(file, old, new)
        print(result)
        if result.startswith("NO MATCH"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
