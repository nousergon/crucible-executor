#!/bin/bash
# Full setup for the trading EC2 instance (t3.small, market hours only).
#
# This instance runs IB Gateway (IBC + Xvfb), the morning batch (main.py),
# the intraday daemon, and EOD reconciliation. It is started/stopped
# daily by EventBridge Scheduler.
#
# Prerequisites:
#   - Amazon Linux 2023 AMI
#   - ~/.netrc with GitHub PAT (for git pull)
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

# ── 2a. Configure git URL rewrite for private alpha-engine-lib ──────────────
# requirements.txt pins alpha-engine-lib from a private GitHub repo. pip
# needs an HTTPS auth path to clone it. The PAT lives in SSM at
# /alpha-engine/lib-token (SecureString); the EC2 instance role
# alpha-engine-executor-role grants ssm:GetParameter on /alpha-engine/*.
# Local shell var scope only; never exported, never logged. boot-pull.sh
# refreshes the same insteadOf rewrite on every boot so a ~/.gitconfig
# reset or fresh EBS recovers automatically.
echo "Fetching alpha-engine-lib PAT from SSM..."
AE_LIB_TOKEN=$(aws ssm get-parameter --name /alpha-engine/lib-token --with-decryption --query 'Parameter.Value' --output text --region us-east-1 2>/dev/null || echo "")
if [ -z "$AE_LIB_TOKEN" ]; then
    echo "ERROR: ssm:/alpha-engine/lib-token unreadable — required to pip install private alpha-engine-lib" >&2
    echo "       Create with: aws ssm put-parameter --name /alpha-engine/lib-token --type SecureString --value <PAT> --region us-east-1" >&2
    echo "       PAT must have Contents:read on cipher813/alpha-engine-lib." >&2
    exit 1
fi
git config --global url."https://x-access-token:${AE_LIB_TOKEN}@github.com/cipher813/alpha-engine-lib".insteadOf "https://github.com/cipher813/alpha-engine-lib"
unset AE_LIB_TOKEN

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
SYSTEMD_DIR="$REPO_DIR/infrastructure/systemd"

for unit in xvfb.service ibgateway.service alpha-engine-morning.service \
            alpha-engine-daemon.service alpha-engine-daemon.timer \
            alpha-engine-eod.service alpha-engine-eod.timer; do
    sudo cp "$SYSTEMD_DIR/$unit" /etc/systemd/system/
done

sudo systemctl daemon-reload

sudo systemctl enable xvfb.service
sudo systemctl enable ibgateway.service
sudo systemctl enable alpha-engine-morning.service
sudo systemctl enable alpha-engine-daemon.service
sudo systemctl enable alpha-engine-eod.timer

echo ""
echo "=== Trading Instance Setup Complete ==="
echo ""
echo "Boot sequence (systemd):"
echo "  1. xvfb.service           — virtual display for IB Gateway"
echo "  2. ibgateway.service      — IB Gateway via IBC (needs 2FA on first login)"
echo "  3. alpha-engine-morning   — order book planner (main.py)"
echo "  4. alpha-engine-daemon    — intraday order executor"
echo "  5. alpha-engine-eod       — EOD reconciliation (1:05 PM PT timer)"
echo ""
echo "First login: IB Gateway will send a 2FA push to your phone."
echo "Approve it within 2 minutes of instance start."
