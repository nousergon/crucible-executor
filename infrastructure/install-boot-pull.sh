#!/bin/bash
# Install the boot-pull systemd service.
# Must run as root (sudo).
#
# Usage:
#   sudo bash /home/ec2-user/alpha-engine/infrastructure/install-boot-pull.sh

set -euo pipefail

SERVICE_FILE="/etc/systemd/system/boot-pull.service"
SCRIPT="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"
LOG="/var/log/boot-pull.log"

# Ensure log file exists with correct ownership
touch "$LOG"
chown ec2-user:ec2-user "$LOG"

# Ensure script is executable
chmod +x "$SCRIPT"

# Write systemd unit
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Pull latest Alpha Engine code on boot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ec2-user
ExecStart=/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh
TimeoutStartSec=120
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable boot-pull.service

echo "boot-pull.service installed and enabled."
echo "  Pulls all repos on every boot before cron jobs fire."
echo "  Logs: tail -f $LOG"
echo "  Test: sudo systemctl start boot-pull && cat $LOG"
