#!/bin/bash
# health_checker.sh — Check S3 health files and alert via Telegram on stale/failed modules.
#
# Designed to run via cron on the micro (dashboard) instance every 30 min during
# market hours. Checks each module's last_success timestamp against a per-module
# staleness threshold and sends a Telegram alert if stale or failed.
#
# Cron entry (UTC, covers 6 AM - 2 PM PT on weekdays):
#   */30 13-21 * * 1-5 /home/ec2-user/alpha-engine/infrastructure/health_checker.sh
#
# Requires: aws CLI, python3, curl
# Env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (from ~/.alpha-engine.env)

set -uo pipefail

# Load Telegram credentials
ENV_FILE="$HOME/.alpha-engine.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

BUCKET="alpha-engine-research"

# Module name → max staleness in hours
# executor: runs daily ~6:25 AM PT (13:25 UTC), stale after 4h
# predictor_inference: runs daily ~6:15 AM PT, stale after 4h
# eod_reconcile: runs daily ~1:20 PM PT (20:20 UTC), stale after 20h
# research: runs weekly Monday, stale after 8 days (192h)
# predictor_training: runs weekly Monday, stale after 8 days (192h)
# price_fetcher: runs daily ~4:35 PM ET, stale after 20h
declare -A MAX_HOURS=(
    [executor]=4
    [predictor_inference]=4
    [eod_reconcile]=20
    [price_fetcher]=20
    [research]=192
    [predictor_training]=192
)

send_alert() {
    local msg="$1"
    # Migrated 2026-05-20 (ROADMAP L146) — was an inline `curl` to the
    # Telegram bot API; now delegates to the canonical
    # ``alpha_engine_lib.alerts`` primitive (v0.21.0, lib #52) which
    # fans out to BOTH the SNS ``alpha-engine-alerts`` topic (→ email)
    # AND ``@nous_ergon_alerts_bot``. The Bash-side contract is
    # unchanged: ``send_alert "$msg"`` from anywhere in the script.
    #
    # Resolve a Python that has alpha_engine_lib installed — prefer the
    # repo-local venv (matches the dispatcher-cleanup pattern in
    # alpha-engine-backtester #231), fall back to whichever system
    # python3 is on PATH. ``|| echo`` is the same graceful-degrade
    # surface the pre-migration ``no Telegram`` branch provided.
    local _alert_python
    if [ -x "$(dirname "$0")/../.venv/bin/python" ]; then
        _alert_python="$(dirname "$0")/../.venv/bin/python"
    else
        _alert_python="$(command -v python3 || command -v python || echo python)"
    fi
    "$_alert_python" -m alpha_engine_lib.alerts publish \
        --message "$msg" \
        --severity error \
        --source alpha-engine/infrastructure/health_checker.sh \
        > /dev/null 2>&1 || echo "ALERT (alerts.publish failed): $msg"
}

alerts=""

for mod in "${!MAX_HOURS[@]}"; do
    max_h="${MAX_HOURS[$mod]}"
    health=$(aws s3 cp "s3://$BUCKET/health/${mod}.json" - 2>/dev/null)
    if [ -z "$health" ]; then
        continue  # no health file yet — module may not have run since health was added
    fi

    eval "$(echo "$health" | python3 -c "
import sys, json
from datetime import datetime, timezone
d = json.load(sys.stdin)
status = d.get('status', 'unknown')
ts = d.get('last_success', '')
if ts:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
else:
    age = 9999
print(f'status=\"{status}\"')
print(f'age_h={age:.1f}')
" 2>/dev/null)" || continue

    if [ "$status" = "failed" ]; then
        alerts="${alerts}\n- ${mod}: FAILED"
    elif (( $(echo "$age_h > $max_h" | bc -l 2>/dev/null || echo 0) )); then
        age_int="${age_h%.*}"
        alerts="${alerts}\n- ${mod}: stale (${age_int}h old, max ${max_h}h)"
    fi
done

if [ -n "$alerts" ]; then
    send_alert "$(echo -e "⚠️ Health Alert\n${alerts}")"
fi
