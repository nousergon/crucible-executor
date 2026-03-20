#!/bin/bash
# Post-clone EC2 setup: venv, deps, log files, cron, config.
# Run after git clone — credentials must already be in ~/.netrc.
#
# Usage (from EC2, after cloning):
#   GMAIL_APP_PASSWORD=xxx ANTHROPIC_API_KEY=yyy bash ~/alpha-engine/infrastructure/setup-ec2.sh
#
# Prerequisites:
#   - ~/.netrc with GitHub PAT (for git pull in cron)
#   - IB Gateway running on port 4002 (ibgateway.service)
#   - config/risk.yaml must be created manually (gitignored)
#   - ~/.alpha-engine.env must include TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

set -euo pipefail

REPO_DIR="/home/ec2-user/alpha-engine"

echo "=== Alpha Engine Executor — EC2 setup ==="

cd "$REPO_DIR"

# ── 1. Virtualenv ─────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# ── 2. Log files ──────────────────────────────────────────────────────────────
for log in executor.log eod.log daemon.log; do
    sudo touch "/var/log/$log"
    sudo chown ec2-user:ec2-user "/var/log/$log"
done
echo "Log files ready: /var/log/executor.log, /var/log/eod.log, /var/log/daemon.log"

# ── 3. Config check ──────────────────────────────────────────────────────────
if [ ! -f config/risk.yaml ]; then
    echo ""
    echo "WARNING: config/risk.yaml not found."
    echo "  Copy config/risk.yaml.example and fill in values:"
    echo "    cp config/risk.yaml.example config/risk.yaml"
    echo ""
fi

# ── 4. Boot-pull service (auto-update all repos on instance start) ─────────
sudo bash "$REPO_DIR/infrastructure/install-boot-pull.sh"

# ── 5. Cron ───────────────────────────────────────────────────────────────────
bash "$REPO_DIR/infrastructure/add-cron.sh"

echo ""
echo "=== Setup complete ==="
echo "Test: cd $REPO_DIR && .venv/bin/python executor/main.py --dry-run"
echo "Logs: tail -f /var/log/executor.log"
