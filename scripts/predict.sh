#!/usr/bin/env bash
# Run prediction on a split inside the dev pod.
#
# Usage:
#   bash scripts/predict.sh val                    # predict val with model_final.pth
#   bash scripts/predict.sh test outputs/.../model_0049999.pth

set -euo pipefail

REPO=${REPO:-/workspace/FinalTry}
VENV=${VENV:-/workspace/.venv}
CONFIG=${CONFIG:-$REPO/configs/m2f_swinl_idd_panoptic.yaml}

SPLIT="${1:-val}"
WEIGHTS="${2:-$REPO/outputs/m2f_swinl_idd_panoptic/model_final.pth}"

cd "$REPO"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:$REPO/external/Mask2Former:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python scripts/predict.py \
    --config-file "$CONFIG" \
    --weights    "$WEIGHTS" \
    --split      "$SPLIT"
