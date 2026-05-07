#!/usr/bin/env bash
# One-time environment setup inside the NGC pytorch:24.10-py3 container.
#
# Installs python packages into a persistent prefix on /workspace so they
# survive pod restarts (the image's /usr site-packages do not). We do NOT
# use a venv: the NGC image lacks python3-venv (ensurepip), and apt-installs
# wouldn't survive restarts anyway. Instead we install with --prefix into
# /workspace/.local and prepend that to PYTHONPATH/PATH.
#
# Run inside the dev pod:
#   kubectl exec -it idd-panoptic-dev -- bash
#   cd /workspace && bash scripts/setup_dgx.sh

set -euo pipefail

REPO=${REPO:-/workspace}
PREFIX=${PREFIX:-/workspace/.local}

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE="$PREFIX/lib/python${PY_VER}/site-packages"
mkdir -p "$SITE"

export PYTHONPATH="$SITE:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"

PIP_INSTALL=(python3 -m pip install --prefix="$PREFIX" --no-warn-script-location)

echo "=== [0/6] Python: $(python3 --version), prefix: $PREFIX ==="
python3 -m pip install --prefix="$PREFIX" --upgrade pip wheel setuptools

echo "=== [1/6] Core deps (uses container's torch from system site-packages) ==="
"${PIP_INSTALL[@]}" \
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

echo "=== [2/6] panopticapi (AutoNUE fork) ==="
"${PIP_INSTALL[@]}" 'panopticapi @ git+https://github.com/AutoNUE/panopticapi.git'

echo "=== [3/6] Detectron2 (build from source against container's torch) ==="
if ! python3 -c "import detectron2" 2>/dev/null; then
    "${PIP_INSTALL[@]}" --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git'
fi

echo "=== [4/6] Mask2Former clone ==="
M2F="$REPO/external/Mask2Former"
if [ ! -d "$M2F" ]; then
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F"
fi
"${PIP_INSTALL[@]}" -r "$M2F/requirements.txt" || true

echo "=== [5/6] Compile MultiScaleDeformableAttention CUDA op ==="
cd "$M2F/mask2former/modeling/pixel_decoder/ops"
# build into our prefix so the compiled .so is persistent
python3 setup.py build install --prefix="$PREFIX"

cd "$REPO"

echo "=== [6/6] Writing /workspace/.local/activate.sh ==="
cat > "$PREFIX/activate.sh" <<EOF
# Source this in future pod sessions:  source /workspace/.local/activate.sh
export PYTHONPATH="$SITE:\${PYTHONPATH:-}"
export PATH="$PREFIX/bin:\$PATH"
EOF

echo
echo "Done. In future sessions, activate with:"
echo "    source $PREFIX/activate.sh"
echo
echo "Verify:"
echo "    python3 -c 'import torch, detectron2; print(torch.__version__, torch.cuda.is_available())'"
echo "    python3 scripts/_smoke_dataset.py"
