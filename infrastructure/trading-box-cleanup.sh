#!/usr/bin/env bash
# trading-box-cleanup.sh — Reclaim disk on ae-trading (executor EC2).
#
# Safe to run while the box is stopped (next boot) or idle (no daemon).
# Does NOT touch IB Gateway, trades.db, or active venvs for executor/data.
#
# Usage (on box as ec2-user, or via SSM):
#   bash infrastructure/trading-box-cleanup.sh
#   bash infrastructure/trading-box-cleanup.sh --remove-orphan-repos
#
# --remove-orphan-repos  Also deletes legacy boot-pull clones that no
#                          longer belong on trading (dashboard, backtester,
#                          ad-hoc predictor checkouts). Saves ~1–2 GB.

set -uo pipefail

REMOVE_ORPHAN_REPOS=false
if [[ "${1:-}" == "--remove-orphan-repos" ]]; then
    REMOVE_ORPHAN_REPOS=true
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

log "=== trading-box-cleanup started ==="
df -h / /tmp /home 2>/dev/null || df -h /

# Ad-hoc / manual-run debris (2026-07-05 risk_model unblock session).
for path in \
    /home/ec2-user/predictor \
    /home/ec2-user/rm-cache \
    /home/ec2-user/probe-cache; do
    if [[ -d "$path" ]]; then
        log "Removing $path"
        rm -rf "$path"
    fi
done

# Ephemeral parquet / pip caches under /tmp (common failure mode on 8GB root).
find /tmp -maxdepth 1 -type d -name 'tmp*' -user ec2-user -mtime +0 -exec rm -rf {} + 2>/dev/null || true
find /tmp -maxdepth 1 -type d -name 'pip-*' -user ec2-user -mtime +0 -exec rm -rf {} + 2>/dev/null || true

if $REMOVE_ORPHAN_REPOS; then
    for path in \
        /home/ec2-user/alpha-engine-dashboard \
        /home/ec2-user/alpha-engine-backtester; do
        if [[ -d "$path/.git" ]]; then
            log "Removing orphan repo clone $path"
            rm -rf "$path"
        fi
    done
fi

log "=== trading-box-cleanup complete ==="
df -h / /tmp /home 2>/dev/null || df -h /
