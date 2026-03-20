#!/bin/bash
# Register the executor + EOD cron jobs.
# Safe to run multiple times — replaces existing entries.
#
# Schedule (all times UTC, weekdays only):
#   13:30  executor/main.py       — morning trading loop
#   21:05  executor/eod_reconcile.py — EOD P&L + rationale email
#
# Both cron lines:
#   1. git pull --ff-only (auto-deploy latest code)
#   2. Source secrets from ~/.alpha-engine.env
#   3. Run the Python script
#
# Secrets file (~/.alpha-engine.env, chmod 600):
#   GMAIL_APP_PASSWORD=xxx
#   ANTHROPIC_API_KEY=yyy
#   TELEGRAM_BOT_TOKEN=xxx
#   TELEGRAM_CHAT_ID=xxx
#
# Usage:
#   bash infrastructure/add-cron.sh
#
# First-time setup (create the env file):
#   cat > ~/.alpha-engine.env << 'EOF'
#   GMAIL_APP_PASSWORD=your-app-password
#   ANTHROPIC_API_KEY=your-api-key
#   TELEGRAM_BOT_TOKEN=your-bot-token
#   TELEGRAM_CHAT_ID=your-chat-id
#   EOF
#   chmod 600 ~/.alpha-engine.env

set -euo pipefail

REPO_DIR="/home/ec2-user/alpha-engine"
ENV_FILE="/home/ec2-user/.alpha-engine.env"

# ── Validate env file exists ─────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Create it with GMAIL_APP_PASSWORD and ANTHROPIC_API_KEY, then chmod 600."
    exit 1
fi

# ── Build cron lines (source env file instead of inline secrets) ─────────────
SOURCE_ENV=". ${ENV_FILE} &&"

EXECUTOR_CRON="30 13 * * 1-5  cd ${REPO_DIR} && git pull --ff-only >> /var/log/executor.log 2>&1 && ${SOURCE_ENV} .venv/bin/python executor/main.py >> /var/log/executor.log 2>&1"
DAEMON_CRON="45 13 * * 1-5  cd ${REPO_DIR} && ${SOURCE_ENV} .venv/bin/python -m executor.daemon >> /var/log/daemon.log 2>&1"
EOD_CRON="5 21 * * 1-5  cd ${REPO_DIR} && git pull --ff-only >> /var/log/eod.log 2>&1 && ${SOURCE_ENV} .venv/bin/python executor/eod_reconcile.py >> /var/log/eod.log 2>&1"

# ── Replace existing entries ─────────────────────────────────────────────────
# Remove any existing alpha-engine executor/eod/daemon lines, then add new ones.
# Preserve backtester and other cron entries.
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "alpha-engine/.*executor/main.py" | grep -v "alpha-engine/.*executor/eod_reconcile.py" | grep -v "alpha-engine/.*executor.daemon" || true)

{
    echo "$FILTERED"
    echo "$EXECUTOR_CRON"
    echo "$DAEMON_CRON"
    echo "$EOD_CRON"
} | crontab -

echo "Executor cron jobs registered:"
echo "  Executor: weekdays 13:30 UTC (9:30 AM ET)"
echo "  Daemon:   weekdays 13:45 UTC (9:45 AM ET) — self-terminates at 4:00 PM ET"
echo "  EOD:      weekdays 21:05 UTC (4:05 PM ET)"
echo "  Secrets:  sourced from ${ENV_FILE}"
echo ""
echo "Current crontab:"
crontab -l
