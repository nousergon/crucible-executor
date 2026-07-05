#!/bin/bash
# Install the one-shot upstream-gate dry-run validation timer on ae-trading.
# Safe to re-run (idempotent). Does NOT enable recurring timers.
#
# Usage (on box or via SSM):
#   bash infrastructure/ops/install-upstream-gate-validation-timer.sh

set -euo pipefail

REPO="${REPO:-/home/ec2-user/alpha-engine}"
SYSTEMD_SRC="$REPO/infrastructure/systemd"

sudo cp "$SYSTEMD_SRC/upstream-gate-dryrun-validation.service" /etc/systemd/system/
sudo cp "$SYSTEMD_SRC/upstream-gate-dryrun-validation.timer" /etc/systemd/system/
sudo chmod +x "$REPO/infrastructure/ops/upstream-gate-dryrun-validation.sh"
sudo systemctl daemon-reload
sudo systemctl enable --now upstream-gate-dryrun-validation.timer
systemctl list-timers upstream-gate-dryrun-validation.timer --no-pager
