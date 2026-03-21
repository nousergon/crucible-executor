#!/bin/bash
# Remove legacy executor cron jobs from any instance.
#
# All scheduling is now handled by EventBridge Scheduler:
#   - Trading instance start/stop: EventBridge → EC2 API
#   - Backtester spot launch: EventBridge → SSM RunCommand on micro
#   - Executor/daemon/EOD: systemd services on trading instance (boot-triggered)
#
# Run this on the micro instance to clean up old cron entries.
#
# Usage:
#   bash infrastructure/add-cron.sh

set -euo pipefail

echo "Removing legacy alpha-engine cron entries..."

EXISTING=$(crontab -l 2>/dev/null || true)
if [ -z "$EXISTING" ]; then
    echo "No crontab found — nothing to clean up"
    exit 0
fi

FILTERED=$(echo "$EXISTING" \
    | grep -v "alpha-engine/.*executor/main.py" \
    | grep -v "alpha-engine/.*executor/eod_reconcile.py" \
    | grep -v "alpha-engine/.*executor.daemon" \
    | grep -v "ec2 start-instances" \
    | grep -v "ec2 stop-instances" \
    | grep -v "spot_backtest.sh" \
    | grep -v "^CRON_TZ=America/Los_Angeles$" \
    || true)

echo "$FILTERED" | crontab -

echo "Legacy cron entries removed."
echo ""
echo "Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"
echo ""
echo "All scheduling is now via EventBridge Scheduler."
echo "Verify: aws scheduler list-schedules --region us-east-1"
