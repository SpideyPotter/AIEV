#!/usr/bin/env bash
# Launch Mask2Former Swin-L IDD panoptic training inside the dev pod / Job.
#
# Usage (interactive, inside dev pod):
#   bash scripts/train.sh                 # full fine-tune
#   bash scripts/train.sh SOLVER.MAX_ITER 5000   # 5k-iter shakedown
#
# Extra args after the config are passed to detectron2 as --opts overrides.

set -euo pipefail

REPO=${REPO:-/workspace}
PREFIX=${PREFIX:-/workspace/.local}
CONFIG=${CONFIG:-$REPO/configs/m2f_swinl_idd_panoptic.yaml}

cd "$REPO"

# shellcheck source=/dev/null
source "$PREFIX/activate.sh"

export PYTHONPATH="$REPO:$REPO/external/Mask2Former:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python scripts/train.py \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    "$@"
