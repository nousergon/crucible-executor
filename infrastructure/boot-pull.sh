#!/bin/bash
# boot-pull.sh — Pull latest code for all Alpha Engine repos on EC2 boot.
#
# Runs as a systemd oneshot service (boot-pull.service) before any cron jobs.
# Safe to run manually too.
#
# Install:
#   sudo bash infrastructure/install-boot-pull.sh

set -uo pipefail

LOG="/var/log/boot-pull.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

log "=== boot-pull started ==="

REPOS=(
    /home/ec2-user/alpha-engine
    /home/ec2-user/alpha-engine-backtester
    /home/ec2-user/alpha-engine-dashboard
)

for repo in "${REPOS[@]}"; do
    if [ ! -d "$repo/.git" ]; then
        log "SKIP $repo (not cloned)"
        continue
    fi

    log "Pulling $repo ..."
    cd "$repo"
    if git pull --ff-only >> "$LOG" 2>&1; then
        log "OK   $repo — $(git log --oneline -1)"
    else
        log "WARN $repo — pull failed (merge conflict or network issue)"
    fi

    # Update pip deps if venv + requirements.txt exist
    if [ -f ".venv/bin/pip" ] && [ -f "requirements.txt" ]; then
        if .venv/bin/pip install --quiet -r requirements.txt >> "$LOG" 2>&1; then
            log "OK   $repo — deps updated"
        else
            log "WARN $repo — pip install failed"
        fi
    fi
done

# Sync systemd service files from repo (if alpha-engine has them)
SYSTEMD_SRC="/home/ec2-user/alpha-engine/infrastructure/systemd"
if [ -d "$SYSTEMD_SRC" ]; then
    CHANGED=false
    for unit in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        name=$(basename "$unit")
        if [ -f "/etc/systemd/system/$name" ]; then
            if ! diff -q "$unit" "/etc/systemd/system/$name" >/dev/null 2>&1; then
                sudo cp "$unit" /etc/systemd/system/
                log "OK   systemd: updated $name"
                CHANGED=true
            fi
        fi
    done
    if $CHANGED; then
        sudo systemctl daemon-reload
        log "OK   systemd: daemon-reload"
    fi
fi

log "=== boot-pull complete ==="
