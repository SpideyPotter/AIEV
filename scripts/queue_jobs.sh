#!/usr/bin/env bash
# Run training Jobs one after another on the single MIG slice.
#
# Usage (on the headnode; kubectl only, no GPU):
#   nohup setsid bash scripts/queue_jobs.sh configs/A.yaml configs/B.yaml ... \
#       >> outputs/logs/queue.log 2>&1 &
#
# Before each submission it waits until no idd-* Job is active and the MIG
# slice is free, then calls submit_train_job.sh. A config whose Job already
# exists (finished or not) is skipped, so the queue can be restarted safely.
# Stop it with: pkill -f queue_jobs.sh

set -uo pipefail
cd "$(dirname "$0")/.."
POLL=${POLL:-300}

busy() {
    kubectl get jobs -o jsonpath='{range .items[*]}{.metadata.name} {.status.active}{"\n"}{end}' \
        | awk '$1 ~ /^idd-/ && $2 > 0 {found=1} END {exit !found}' && return 0
    kubectl get pods --field-selector=status.phase!=Succeeded,status.phase!=Failed -o jsonpath='{range .items[*]}{.spec.containers[*].resources.limits}{"\n"}{end}' \
        | grep -q 'nvidia.com/mig'
}

for cfg in "$@"; do
    job=$(bash scripts/submit_train_job.sh "$cfg" --dry-run | awk '/^  name:/ {print $2; exit}')
    if kubectl get job "$job" >/dev/null 2>&1; then
        echo "$(date '+%F %T') skip $cfg: job/$job already exists"
        continue
    fi
    while busy; do sleep "$POLL"; done
    echo "$(date '+%F %T') submitting $cfg"
    bash scripts/submit_train_job.sh "$cfg" || echo "$(date '+%F %T') submit failed: $cfg"
    sleep 60
done
echo "$(date '+%F %T') queue empty"
