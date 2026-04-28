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
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" != "main" ]; then
        log "NOTE $repo — on branch '$CURRENT_BRANCH', will reset to origin/main (policy: boot always tracks main)"
    fi
    if git fetch origin >> "$LOG" 2>&1 \
       && git checkout -f main >> "$LOG" 2>&1 \
       && git reset --hard origin/main >> "$LOG" 2>&1; then
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

# Sync systemd service files from repo (if alpha-engine has them).
#
# Installs NEW units as well as updating existing ones. Prior versions of
# this block only touched units already in /etc/systemd/system/, which
# silently skipped newly-added repo units (e.g. alpha-engine-daily-data.timer
# added in PR #62 sat on disk for days before this was noticed). After a
# new .timer is copied and daemon-reload'd, it's enabled + started so the
# next boot picks it up and the current boot activates it immediately.
SYSTEMD_SRC="/home/ec2-user/alpha-engine/infrastructure/systemd"
if [ -d "$SYSTEMD_SRC" ]; then
    CHANGED=false
    for unit in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        name=$(basename "$unit")
        target="/etc/systemd/system/$name"
        if [ ! -f "$target" ]; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: installed $name (new)"
            CHANGED=true
        elif ! diff -q "$unit" "$target" >/dev/null 2>&1; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: updated $name"
            CHANGED=true
        fi
    done
    # Orphan reconciliation: any /etc/systemd/system/alpha-engine-*.{service,timer}
    # without a corresponding source in $SYSTEMD_SRC was removed from the
    # repo and must be retired here. Disable + remove. Safety: only matches
    # `alpha-engine-*` prefix, never touches unrelated system units.
    #
    # 2026-04-28: closes the asymmetry where adding a new timer was
    # self-healing (install/update/enable handled) but retiring one was
    # not — the systemd file lingered on disk and continued firing even
    # after deletion from the repo. After this pass, removing a unit
    # file from `infrastructure/systemd/` is the canonical retirement
    # path: next boot disables + deletes it.
    for installed in /etc/systemd/system/alpha-engine-*.service /etc/systemd/system/alpha-engine-*.timer; do
        [ -f "$installed" ] || continue  # glob may match nothing
        name=$(basename "$installed")
        if [ ! -f "$SYSTEMD_SRC/$name" ]; then
            sudo systemctl disable --now "$name" >> "$LOG" 2>&1 || true
            sudo rm -f "$installed"
            log "OK   systemd: orphan removed $name (no longer in repo)"
            CHANGED=true
        fi
    done

    if $CHANGED; then
        sudo systemctl daemon-reload
        log "OK   systemd: daemon-reload"
    fi

    # Reconcile timer enable state. Every timer shipped in the repo
    # MUST be enabled. `systemctl enable` is idempotent on already-
    # enabled timers, and recreates the `timers.target.wants/` symlink
    # on any timer that was disabled or whose symlink was lost. Fixes
    # three failure modes the prior new-install-only enable path could
    # not recover from:
    #
    # 1. Manual `systemctl disable` (intentional debugging, accidental).
    # 2. EBS volume state where unit files exist on disk but
    #    `timers.target.wants/` is empty.
    # 3. New timers added to `infrastructure/systemd/` after the
    #    one-shot `setup-trading-ec2.sh` run already completed — they
    #    get `cp`'d by boot-pull but the previous "enable only on first
    #    install" branch missed them because the target file already
    #    existed.
    #
    # 2026-04-21 SNDK EOD incident: `alpha-engine-eod.timer` was
    # disabled for an unknown reason. boot-pull's previous logic only
    # enabled timers inside the `[ ! -f "$target" ]` branch, so the
    # disabled state persisted for two boots. EOD emails silently
    # stopped firing until manual SSM intervention re-enabled the
    # timer. This reconciliation pass makes every boot self-healing
    # for timer-enable drift.
    for unit in "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        name=$(basename "$unit")
        if sudo systemctl enable --now "$name" >> "$LOG" 2>&1; then
            log "OK   systemd: enable reconciled $name"
        else
            log "WARN systemd: enable reconcile failed: $name"
        fi
    done
fi

# Config files are now in the alpha-engine-config private repo (pulled above).
# Each module's config loader searches ~/alpha-engine-config/ first.

log "=== boot-pull complete ==="
