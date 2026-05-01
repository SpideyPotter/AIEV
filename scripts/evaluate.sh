#!/usr/bin/env bash
# Run the official AutoNUE panoptic evaluation on a set of predictions.
#
# Usage:
#   bash scripts/evaluate.sh [split] [datadir] [pred_folder] [pred_json]
#
# Defaults:
#   split       = val
#   datadir     = <repo>/data/idd_full
#   pred_folder = <repo>/outputs/<split>_pred_panoptic
#   pred_json   = <repo>/outputs/<split>_pred_panoptic.json
#
# Run inside the autonue conda env:
#   conda activate autonue && bash scripts/evaluate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
    ROOT="$(cygpath -m "$ROOT")"
fi

SPLIT="${1:-val}"
DATADIR="${2:-$ROOT/data/idd_full}"
PRED_FOLDER="${3:-$ROOT/outputs/${SPLIT}_pred_panoptic}"
PRED_JSON="${4:-$ROOT/outputs/${SPLIT}_pred_panoptic.json}"

GT_FOLDER="$DATADIR/gtFine/${SPLIT}_panoptic"
GT_JSON="$DATADIR/gtFine/${SPLIT}_panoptic.json"

for path in "$GT_FOLDER" "$GT_JSON" "$PRED_FOLDER" "$PRED_JSON"; do
    if [ ! -e "$path" ]; then
        echo "Missing: $path" >&2
        exit 1
    fi
done

python -m panopticapi.evaluation \
    --gt_json_file   "$GT_JSON" \
    --pred_json_file "$PRED_JSON" \
    --gt_folder      "$GT_FOLDER" \
    --pred_folder    "$PRED_FOLDER"
