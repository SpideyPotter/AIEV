#!/usr/bin/env bash
# Submit a from-scratch training Job for one experiment config.
#
# Usage (on the headnode):
#   bash scripts/submit_train_job.sh configs/m2f_r50_idd_scratch.yaml        # A0
#   bash scripts/submit_train_job.sh configs/m2f_r50_idd_scratch_a1_rfs.yaml  # A1
#   bash scripts/submit_train_job.sh <config> --dry-run   # print manifest only
#
# The job name is derived from the config file name, e.g.
# m2f_r50_idd_scratch_a1_rfs.yaml -> idd-r50-scratch-a1-rfs.

set -euo pipefail

CONFIG=${1:?usage: submit_train_job.sh <config.yaml> [--dry-run]}
DRY_RUN=${2:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
TEMPLATE="$REPO/k8s/job_train_scratch.yaml"

[ -f "$REPO/$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

STEM=$(basename "$CONFIG" .yaml)
JOB=$(echo "$STEM" | sed -e 's/^m2f_/idd-/' -e 's/_idd_/-/' -e 's/_/-/g')

[ "$CONFIG" = "configs/${STEM}.yaml" ] || { echo "config must live in configs/" >&2; exit 1; }

MANIFEST=$(sed \
    -e "s#idd-r50-scratch#${JOB}#g" \
    -e "s#STEM=m2f_r50_idd_scratch\$#STEM=${STEM}#" \
    "$TEMPLATE")

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "$MANIFEST"
    exit 0
fi

if kubectl get pod idd-panoptic-dev >/dev/null 2>&1; then
    echo "idd-panoptic-dev holds the only MIG slice; delete it first:" >&2
    echo "  kubectl delete pod idd-panoptic-dev" >&2
    exit 1
fi

echo "$MANIFEST" | kubectl apply -f -
echo "Submitted job/${JOB}. Follow with: kubectl logs -f job/${JOB}"
