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

# ── Configure git auth for private alpha-engine-lib ──────────────────────────
# requirements.txt in alpha-engine / alpha-engine-backtester pins
# alpha-engine-lib from a private GitHub repo. pip install needs an HTTPS
# auth path to clone it.
#
# The PAT lives in SSM as /alpha-engine/lib-token (SecureString). The EC2
# instance role alpha-engine-executor-role grants ssm:GetParameter on
# /alpha-engine/*. Fetching on every boot is idempotent and survives a
# fresh EBS or ~/.gitconfig reset. Local shell var scope only; never
# exported, never logged.
AE_LIB_TOKEN=$(aws ssm get-parameter --name /alpha-engine/lib-token --with-decryption --query 'Parameter.Value' --output text --region us-east-1 2>/dev/null || echo "")
if [ -n "$AE_LIB_TOKEN" ]; then
    git config --global url."https://x-access-token:${AE_LIB_TOKEN}@github.com/cipher813/alpha-engine-lib".insteadOf "https://github.com/cipher813/alpha-engine-lib"
    log "OK   git insteadOf rewrite configured for alpha-engine-lib (ssm:/alpha-engine/lib-token)"
else
    log "FAIL ssm:/alpha-engine/lib-token unreadable — pip install of alpha-engine-lib will fail"
fi
unset AE_LIB_TOKEN

REPOS=(
    /home/ec2-user/alpha-engine-config
    /home/ec2-user/alpha-engine
    /home/ec2-user/alpha-engine-backtester
    /home/ec2-user/alpha-engine-dashboard
    /home/ec2-user/alpha-engine-data
)

for repo in "${REPOS[@]}"; do
    if [ ! -d "$repo/.git" ]; then
        log "SKIP $repo (not cloned)"
        continue
    fi

    log "Pulling $repo ..."
    cd "$repo"
    PREV_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")
    if git fetch origin >> "$LOG" 2>&1 && git reset --hard origin/main >> "$LOG" 2>&1; then
        NEW_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")
        log "OK   $repo — $(git log --oneline -1)"

        # Deploy gate: syntax check only (no IB Gateway connection needed)
        if [ "$repo" = "/home/ec2-user/alpha-engine" ] && [ "$PREV_SHA" != "$NEW_SHA" ]; then
            if [ -f ".venv/bin/python" ] && [ -f "executor/main.py" ]; then
                log "GATE $repo — running syntax validation..."
                if .venv/bin/python -c "import ast; ast.parse(open('executor/main.py').read()); ast.parse(open('executor/daemon.py').read()); ast.parse(open('executor/eod_reconcile.py').read())" >> "$LOG" 2>&1; then
                    log "OK   $repo — syntax check passed"
                else
                    log "FAIL $repo — syntax check failed, rolling back to $PREV_SHA"
                    git reset --hard "$PREV_SHA" >> "$LOG" 2>&1
                    log "ROLLBACK $repo — reverted to $(git log --oneline -1)"
                fi
            fi
        fi
    else
        log "WARN $repo — fetch/reset failed (network issue?)"
    fi

    # Update pip deps if venv + requirements.txt exist.
    # flow-doctor is now pulled in transitively via
    # alpha-engine-lib[flow_doctor]; the previous bundled editable
    # install (pip install -e /home/ec2-user/flow-doctor) has been
    # removed so boot doesn't silently overwrite the lib-provided copy
    # with a local dev branch.
    if [ -f ".venv/bin/pip" ] && [ -f "requirements.txt" ]; then
        if .venv/bin/pip install --quiet -r requirements.txt >> "$LOG" 2>&1; then
            log "OK   $repo — deps updated"
        else
            log "WARN $repo — pip install failed"
        fi
    fi
done

# ── Restore trades.db from S3 if missing or empty ──────────────────────────
# The trading instance is stopped/started daily by EventBridge. If a new
# instance is launched (or EBS is recreated), trades.db will be missing and
# init_db() creates an empty one — losing all trade history and EOD data.
# This restores trades_latest.db from S3 as a safety net.
RISK_YAML="/home/ec2-user/alpha-engine/config/risk.yaml"
if [ -f "$RISK_YAML" ]; then
    # Parse db_path and trades_bucket from risk.yaml
    DB_PATH=$(grep -E '^\s*db_path:' "$RISK_YAML" | head -1 | sed 's/.*db_path:\s*["]*\([^"]*\)["]*\s*/\1/' | tr -d "'\"")
    TRADES_BUCKET=$(grep -E '^\s*trades_bucket:' "$RISK_YAML" | head -1 | sed 's/.*trades_bucket:\s*["]*\([^"]*\)["]*\s*/\1/' | tr -d "'\"")

    if [ -n "$DB_PATH" ] && [ -n "$TRADES_BUCKET" ]; then
        # Restore if db doesn't exist or is ≤ 20KB (empty schema only)
        DB_SIZE=0
        [ -f "$DB_PATH" ] && DB_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)

        if [ "$DB_SIZE" -le 20480 ]; then
            S3_KEY="trades/trades_latest.db"
            log "trades.db missing or empty (${DB_SIZE}B) — restoring from s3://${TRADES_BUCKET}/${S3_KEY}"
            if aws s3 cp "s3://${TRADES_BUCKET}/${S3_KEY}" "$DB_PATH" >> "$LOG" 2>&1; then
                NEW_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
                log "OK   trades.db restored (${NEW_SIZE}B)"
            else
                log "WARN trades.db restore failed — executor will start with empty db"
            fi
        else
            log "OK   trades.db exists (${DB_SIZE}B) — no restore needed"
        fi
    else
        log "WARN could not parse db_path/trades_bucket from risk.yaml"
    fi
else
    log "SKIP trades.db restore (no risk.yaml)"
fi

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

# Config files are now in the alpha-engine-config private repo (pulled above).
# Each module's config loader searches ~/alpha-engine-config/ first.

log "=== boot-pull complete ==="
