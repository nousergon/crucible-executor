#!/usr/bin/env bash
# trader-provision.sh — make the v2 trader runnable on this box, idempotently.
#
# alpha-engine-config-I11545, Brian's ruling 1(a) of 2026-09-24: the v2 trader
# (nousergon/crucible-trader) runs on the executor box, beside IB Gateway
# paper. It is a SEPARATE system from the v1 executor in this repository; it
# shares only the box and the gateway.
#
# Run as the ExecStartPre of every alpha-engine-trader-*.service, not from
# boot-pull.sh, for two reasons:
#
#   * boot-pull.service has TimeoutStartSec=120 and gates the v1 session. A
#     first-time uv/python download or a trader dependency sync has no place
#     inside that budget, and a trader failure must never count as a v1
#     boot-pull failure (PULL_FAILURES pages "the executor may be running
#     stale code").
#   * boot-pull only ENABLES *.timer units. A provisioning service with no
#     timer would be installed and never run.
#
# So each trader unit provisions itself right before it runs, and a failure
# here fails THAT unit, visibly in its own journal, and nothing else.
#
# What it does, each step a no-op when already done:
#   1. uv at a pinned version, in its own venv under /opt/uv.
#   2. The crucible-trader checkout: cloned if absent (the box's
#      git-credential-nousergon-app mints the read token), then reset to
#      origin/main under the same git-sync lock boot-pull takes.
#   3. A CPython 3.12 for the trader's venvs (uv-managed if the OS has none).
#   4. The trader's locked environment with the `ib` extra, so the unit's own
#      run reads the release pin from a warm environment.
#
# It never reads or sets CRUCIBLE_TRADER_ORDER_ROUTING. Whether the trader
# routes orders is its environment file's decision (alpha-engine-config
# crucible-trader/trader.env, which keeps it `off`), not a deploy's.
set -euo pipefail

TRADER_DIR="${CRUCIBLE_TRADER_DIR:-/home/ec2-user/crucible-trader}"
TRADER_REMOTE="https://github.com/nousergon/crucible-trader.git"
# The uv the trader's CI runs is the uv the box runs; bump both together.
UV_VERSION="0.8.17"
UV_HOME="/opt/uv"
UV_BIN="/usr/local/bin/uv"
PROVISION_LOCK="${CRUCIBLE_TRADER_PROVISION_LOCK:-/home/ec2-user/.crucible-trader-provision.lock}"
GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"
GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"

say() { echo "trader-provision: $*"; }

# Two trader units never provision at once (a session and a reconcile can
# overlap on a late boot).
exec 9>"$PROVISION_LOCK"
flock -w 900 9

# 1. uv, pinned.
if [[ "$("$UV_BIN" --version 2>/dev/null || true)" != "uv ${UV_VERSION}"* ]]; then
    say "installing uv ${UV_VERSION} into ${UV_HOME}"
    [[ -x "$UV_HOME/bin/pip" ]] || sudo python3 -m venv "$UV_HOME"
    sudo "$UV_HOME/bin/pip" install --quiet --disable-pip-version-check "uv==${UV_VERSION}"
    sudo ln -sf "$UV_HOME/bin/uv" "$UV_BIN"
fi
say "$("$UV_BIN" --version)"

# 2. The checkout, on origin/main.
if [[ ! -d "$TRADER_DIR/.git" ]]; then
    say "cloning crucible-trader into ${TRADER_DIR}"
    git clone --quiet "$TRADER_REMOTE" "$TRADER_DIR"
fi
flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" bash -c '
    set -e
    git -C "$1" fetch --quiet origin main
    git -C "$1" checkout --quiet -f main
    git -C "$1" reset --quiet --hard origin/main
' _ "$TRADER_DIR"
head_sha="$(git -C "$TRADER_DIR" rev-parse HEAD)"
if [[ "$head_sha" != "$(git -C "$TRADER_DIR" rev-parse origin/main)" ]]; then
    say "crucible-trader HEAD ${head_sha} is not origin/main" >&2
    exit 1
fi
say "crucible-trader at $(git -C "$TRADER_DIR" log --oneline -1)"

# 3. CPython 3.12.
if ! "$UV_BIN" python find 3.12 >/dev/null 2>&1; then
    say "installing a uv-managed CPython 3.12"
    "$UV_BIN" python install 3.12
fi

# 4. The locked environment, with the IB extra.
"$UV_BIN" sync --quiet --frozen --extra ib --project "$TRADER_DIR"
say "environment synced (--frozen --extra ib)"
