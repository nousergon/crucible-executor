#!/bin/bash
# One-shot validation: executor upstream artifact-freshness gate (config#1725 Phase A).
# Runs `executor/main.py --dry-run` on ae-trading after weekday MorningEnrich +
# PredictorInference have populated the three gated deliverables.
#
# Invoked by upstream-gate-dryrun-validation.timer (2026-07-07 14:15 UTC) or manually:
#   bash infrastructure/ops/upstream-gate-dryrun-validation.sh

set -eo pipefail

REPO="/home/ec2-user/alpha-engine"
LOG="/var/log/upstream-gate-validation.log"
exec >>"$LOG" 2>&1

echo "=== upstream gate dry-run validation $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$REPO"
git pull --ff-only origin main

export FLOW_DOCTOR_ENABLED=1
export ALPHA_ENGINE_DEPLOYED=1
export PYTHONPATH="$REPO"
export AWS_REGION="${AWS_REGION:-us-east-1}"

set -a
# shellcheck disable=SC1091
source /home/ec2-user/.alpha-engine.env
set +a

source "$REPO/.venv/bin/activate"
"$REPO/infrastructure/wait-for-ibgateway.sh"

python "$REPO/executor/main.py" --dry-run
rc=$?

s3_key="_ssm_logs/upstream-gate-validation/$(date -u +%Y-%m-%d)/$(hostname)-$(date -u +%H%M%SZ).log"
aws s3 cp "$LOG" "s3://alpha-engine-research/${s3_key}" --only-show-errors || true
echo "=== exit $rc (log s3://${s3_key}) ==="
exit "$rc"
