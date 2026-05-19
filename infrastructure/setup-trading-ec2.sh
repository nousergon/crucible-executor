#!/bin/bash
# Full setup for the trading EC2 instance (t3.small, market hours only).
#
# This instance runs IB Gateway (IBC + Xvfb), the morning batch (main.py),
# the intraday daemon, and EOD reconciliation. It is started/stopped
# daily by EventBridge Scheduler.
#
# Prerequisites:
#   - Amazon Linux 2023 AMI
#   - ~/.alpha-engine.env with secrets (GMAIL_APP_PASSWORD, ANTHROPIC_API_KEY,
#     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
#   - IB Gateway + IBC installed at ~/ibgateway and ~/ibc
#   - config/risk.yaml created manually (gitignored)
#
# Usage:
#   bash ~/alpha-engine/infrastructure/setup-trading-ec2.sh

set -euo pipefail

REPO_DIR="/home/ec2-user/alpha-engine"
ENV_FILE="/home/ec2-user/.alpha-engine.env"

echo "=== Alpha Engine Trading Instance Setup ==="

# ── 1. System packages ──────────────────────────────────────────────────────
echo "Installing system packages..."
sudo dnf install -y git python3.11 python3.11-pip xorg-x11-server-Xvfb 2>&1 | tail -3

# ── 2. Python venv ───────────────────────────────────────────────────────────
cd "$REPO_DIR"
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# ── 3. Log files ─────────────────────────────────────────────────────────────
for log in executor.log eod.log daemon.log; do
    sudo touch "/var/log/$log"
    sudo chown ec2-user:ec2-user "/var/log/$log"
done
echo "Log files ready"

# ── 4. Config check ─────────────────────────────────────────────────────────
if [ ! -f config/risk.yaml ]; then
    echo ""
    echo "WARNING: config/risk.yaml not found."
    echo "  cp config/risk.yaml.example config/risk.yaml"
    echo ""
fi

if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "WARNING: $ENV_FILE not found."
    echo ""
fi

if [ ! -d "/home/ec2-user/ibc" ]; then
    echo ""
    echo "WARNING: ~/ibc not found. Copy IBC installation from the micro instance."
    echo ""
fi

# ── 5. Boot-pull service ────────────────────────────────────────────────────
sudo bash "$REPO_DIR/infrastructure/install-boot-pull.sh"

# ── 6. Systemd services ────────────────────────────────────────────────────
# Only xvfb + ibgateway are autostarted on boot. The morning planner and
# daemon run EXCLUSIVELY from the weekday Step Function via SSM
# (RunMorningPlanner + RunDaemon steps) — the SF is the sole authority
# that starts the daemon, and it only reaches RunDaemon after
# RunMorningPlanner has written today's order book.
#
# NO alpha-engine-daemon.timer. It is deliberately absent from the repo.
# `boot-pull.sh` force-enables EVERY *.timer present in
# infrastructure/systemd/ on every boot (the "every shipped timer MUST be
# enabled" reconciliation). A daemon .timer in the repo therefore can NOT
# stay disabled — boot-pull re-enables it every boot and it fires on its
# wall clock regardless of whether the SF ran or succeeded. That is
# exactly the 2026-05-19 failure: the SF FAILED at MorningEnrich, yet the
# (boot-pull-re-enabled) daemon.timer started the daemon at 06:29 PT with
# no order book — and the 2026-05-05 stale-predictions incident before
# it. The .service unit is still shipped (SF `RunDaemon` does
# `systemctl restart` on it; services are NOT force-enabled by boot-pull,
# only timers) but there is no timer and no boot autostart, so the only
# way the daemon ever starts is the SF. If a daemon.timer ever reappears
# on an instance, boot-pull's orphan reconciliation (no repo source →
# `disable --now` + rm) retires it on the next boot.
SYSTEMD_DIR="$REPO_DIR/infrastructure/systemd"

for unit in xvfb.service ibgateway.service alpha-engine-morning.service \
            alpha-engine-daemon.service; do
    sudo cp "$SYSTEMD_DIR/$unit" /etc/systemd/system/
done

sudo systemctl daemon-reload

# Boot autostart — only the always-needed background services. SF
# orchestration is the sole authoritative path for the trading flow.
sudo systemctl enable xvfb.service
sudo systemctl enable ibgateway.service

echo ""
echo "=== Trading Instance Setup Complete ==="
echo ""
echo "Boot sequence (systemd):"
echo "  1. xvfb.service           — virtual display for IB Gateway"
echo "  2. ibgateway.service      — IB Gateway via IBC (needs 2FA on first login)"
echo ""
echo "Trading flow runs from the weekday Step Function (NOT boot-systemd):"
echo "  - RunMorningPlanner step → executor/main.py via SSM"
echo "  - RunDaemon step          → systemctl start alpha-engine-daemon.service"
echo ""
echo "Post-close data capture + EOD reconciliation also run via SF"
echo "(alpha-engine-eod-pipeline, triggered by EventBridge weekdays 13:05 PT)."
echo "SF chain: PostMarketData → EODReconcile → StopTradingInstance."
echo "Single authoritative path; if SF fails, no trades + SNS alerts fire."
echo ""
echo "First login: IB Gateway will send a 2FA push to your phone."
echo "Approve it within 2 minutes of instance start."
