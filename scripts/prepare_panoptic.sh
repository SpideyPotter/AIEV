#!/usr/bin/env bash
# Generate panoptic ground truth for the merged IDD dataset.
#
# Usage:
#   bash scripts/prepare_panoptic.sh [datadir] [num_workers]
#
# Defaults:
#   datadir     = <repo>/data/idd_full
#   num_workers = 4
#
# Run inside the autonue conda env:
#   conda activate autonue && bash scripts/prepare_panoptic.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# On git-bash / MSYS, convert MSYS path (/d/...) to mixed Windows form (D:/...)
# so native Windows python resolves PYTHONPATH and CLI args correctly.
# PSEP is the PYTHONPATH separator (`;` on Windows, `:` on Unix).
if command -v cygpath >/dev/null 2>&1; then
    ROOT="$(cygpath -m "$ROOT")"
    PSEP=";"
else
    PSEP=":"
fi

ANUE="${1:-$ROOT/data/idd_full}"
WORKERS="${2:-4}"

if [ ! -d "$ANUE/gtFine" ] || [ ! -d "$ANUE/leftImg8bit" ]; then
    echo "Error: $ANUE does not contain gtFine/ and leftImg8bit/." >&2
    echo "Run 'python scripts/merge_idd.py' first." >&2
    exit 1
fi

# helpers/ has the level3Id label definitions (anue_labels.py) used by
# json2labelImg.py at module-import time, so it must be on PYTHONPATH.
export PYTHONPATH="$ROOT/external/public-code/helpers${PYTHONPATH:+$PSEP$PYTHONPATH}"

cd "$ROOT/external/public-code"

python preperation/createLabels.py \
    --datadir    "$ANUE" \
    --id-type    level3Id \
    --panoptic   True \
    --num-workers "$WORKERS"

echo
echo "Panoptic GT written to:"
echo "  $ANUE/gtFine/train_panoptic/   + train_panoptic.json"
echo "  $ANUE/gtFine/val_panoptic/     + val_panoptic.json"
