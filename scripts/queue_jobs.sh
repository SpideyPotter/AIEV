#!/usr/bin/env bash
# Run training Jobs one after another on the single MIG slice.
#
# Usage (on the headnode; kubectl only, no GPU):
#   nohup setsid bash scripts/queue_jobs.sh configs/A.yaml configs/B.yaml ... \
#       >> outputs/logs/queue.log 2>&1 &
#
# Before each submission it waits until every idd-* Job has finished, the MIG
# slice is free and the shared disk has MIN_FREE_GB free. It then waits for
# the submitted Job to finish and stops the whole queue if it failed (a
# failed Job between retries has no active pod, so "no active pod" is not
# "finished"). Completed Jobs are skipped; delete a failed Job to rerun it
# (it resumes from its last checkpoint).
# Stop it with: pkill -f queue_jobs.sh

set -uo pipefail
cd "$(dirname "$0")/.."
POLL=${POLL:-300}
MIN_FREE_GB=${MIN_FREE_GB:-10}

log() { echo "$(date '+%F %T') $*"; }

job_state() {  # Complete | Failed | Running | Missing
    local c
    c=$(kubectl get job "$1" -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type} {end}' 2>/dev/null) \
        || { echo Missing; return; }
    case "$c" in
        *Complete*) echo Complete ;;
        *Failed*) echo Failed ;;
        *) echo Running ;;
    esac
}

busy() {
    local j
    for j in $(kubectl get jobs -o name | sed 's#job.batch/##' | grep '^idd-'); do
        [ "$(job_state "$j")" = Running ] && return 0
    done
    kubectl get pods --field-selector=status.phase!=Succeeded,status.phase!=Failed \
        -o jsonpath='{range .items[*]}{.spec.containers[*].resources.limits}{"\n"}{end}' \
        | grep -q 'nvidia.com/mig'
}

free_gb() { df --output=avail -BG . | tail -1 | tr -dc '0-9'; }

for cfg in "$@"; do
    job=$(bash scripts/submit_train_job.sh "$cfg" --dry-run | awk '/^  name:/ {print $2; exit}')
    case "$(job_state "$job")" in
        Complete) log "skip $cfg: job/$job already complete"; continue ;;
        Failed) log "stop: job/$job failed earlier; kubectl delete job $job to rerun"; exit 1 ;;
        Running) log "job/$job already running; waiting for it" ;;
        Missing)
            while busy; do sleep "$POLL"; done
            while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
                log "waiting for disk: $(free_gb)G free < ${MIN_FREE_GB}G"
                sleep "$POLL"
            done
            log "submitting $cfg ($(free_gb)G free)"
            bash scripts/submit_train_job.sh "$cfg" || { log "submit failed: $cfg"; exit 1; }
            ;;
    esac
    while [ "$(job_state "$job")" = Running ]; do sleep "$POLL"; done
    state=$(job_state "$job")
    log "job/$job finished: $state"
    [ "$state" = Complete ] || { log "stop: kubectl logs job/$job"; exit 1; }
done
log "queue empty"
