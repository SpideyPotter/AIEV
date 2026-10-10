#!/usr/bin/env bash
# Exp A4: inference-only sweep of Mask2Former's panoptic post-processing
# thresholds on the val split. Run inside a GPU pod.
#
# Usage:
#   bash scripts/sweep_thresholds.sh <config.yaml> <weights.pth> [out_dir]
#
# Writes <out_dir>/sweep_om<OBJ>_ov<OVL>.json per setting (default out_dir:
# the directory holding the weights) and prints a summary table.

set -euo pipefail

CONFIG=${1:?usage: sweep_thresholds.sh <config> <weights> [out_dir]}
WEIGHTS=${2:?usage: sweep_thresholds.sh <config> <weights> [out_dir]}
OUT=${3:-$(dirname "$WEIGHTS")}
OBJECT_THRESHOLDS=${OBJECT_THRESHOLDS:-"0.5 0.6 0.7 0.8"}
OVERLAP_THRESHOLDS=${OVERLAP_THRESHOLDS:-"0.8"}

cd "$(dirname "$0")/.."
source /workspace/.local/activate.sh
export PYTHONPATH="$PWD:$PWD/external/Mask2Former:${PYTHONPATH:-}"
mkdir -p "$OUT"

for om in $OBJECT_THRESHOLDS; do
    for ov in $OVERLAP_THRESHOLDS; do
        echo "=== OBJECT_MASK_THRESHOLD=$om OVERLAP_THRESHOLD=$ov ==="
        python3 scripts/eval_val_perclass_pq.py \
            --config-file "$CONFIG" --weights "$WEIGHTS" \
            --out-dir "/tmp/sweep_om${om}_ov${ov}/pngs" \
            --metrics-json "$OUT/sweep_om${om}_ov${ov}.json" \
            MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD "$om" \
            MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD "$ov" \
            2>&1 | grep -E "^(All|Things|Stuff) *:"
    done
done

python3 scripts/compare_experiments.py --glob "$OUT/sweep_*.json"
