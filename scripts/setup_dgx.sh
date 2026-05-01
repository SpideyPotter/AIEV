#!/usr/bin/env bash
# One-time environment setup inside the NGC pytorch:24.10-py3 container.
#
# Creates a venv on the persistent volume at /workspace/.venv so that
# python packages survive pod restarts (the image's site-packages don't).
#
# Run inside the dev pod:
#   kubectl exec -it idd-panoptic-dev -- bash
#   bash /workspace/FinalTry/scripts/setup_dgx.sh

set -euo pipefail

REPO=${REPO:-/workspace/FinalTry}
VENV=${VENV:-/workspace/.venv}

echo "=== [1/6] Creating venv at $VENV ==="
if [ ! -d "$VENV" ]; then
    # --system-site-packages so we keep the NGC container's torch / cuda
    python3 -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "=== [2/6] Core deps (uses container's torch from system site-packages) ==="
python -m pip install \
    'numpy<2' \
    'opencv-python-headless' \
    'imageio' \
    'numpngw' \
    'tqdm' \
    'fvcore' \
    'cloudpickle' \
    'omegaconf' \
    'iopath' \
    'pycocotools' \
    'shapely' \
    'scipy' \
    'timm'

echo "=== [3/6] panopticapi (AutoNUE fork) ==="
python -m pip install 'panopticapi @ git+https://github.com/AutoNUE/panopticapi.git'

echo "=== [4/6] Detectron2 (build from source against container's torch) ==="
if ! python -c "import detectron2" 2>/dev/null; then
    python -m pip install --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git'
fi

echo "=== [5/6] Mask2Former clone ==="
M2F="$REPO/external/Mask2Former"
if [ ! -d "$M2F" ]; then
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F"
fi
python -m pip install -r "$M2F/requirements.txt" || true

echo "=== [6/6] Compile MultiScaleDeformableAttention CUDA op ==="
cd "$M2F/mask2former/modeling/pixel_decoder/ops"
python setup.py build install

cd "$REPO"
echo
echo "Done. Activate the venv in future sessions with:"
echo "    source $VENV/bin/activate"
echo
echo "Verify:"
echo "    python -c 'import torch, detectron2; print(torch.__version__, torch.cuda.is_available())'"
echo "    python scripts/_smoke_dataset.py"
