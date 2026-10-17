#!/usr/bin/env bash
# ibc-master-client-id.sh — give IB Gateway the v2 reconcile's Master API client ID.
#
# alpha-engine-config-I12376, Brian's option (b) of 2026-10-17. The IB paper
# account on this box is SHARED: the v1 executor trades it (client ids 1, 2,
# 3) and the v2 trader reconciles it (crucible-trader `trader.reconcile`). IB
# returns executions to an API session for its own client id only -- except to
# the session whose client id is the gateway's Master API client ID, which
# receives every client's executions. Without that, the v2 reconcile saw none
# of v1's fills and graded each one as a share/cash discrepancy.
#
# So the reconcile connects on CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID, and this
# script makes the gateway agree: it sets IBC's `OverrideTwsMasterClientID`
# (IBC >= 3.x config.ini; IBC writes it into the gateway's API settings
# shortly after startup) to the SAME value, read from the trader's env file.
# One number, one place: alpha-engine-config crucible-trader/trader.env, which
# boot-pull.sh pulls before ibgateway.service starts.
#
# Run as the ExecStartPre of ibgateway.service, with systemd's `-` prefix: a
# failure here is logged in the gateway's journal and NEVER stops the gateway,
# because v1 trades through it. What a failure costs is the v2 reconcile's
# view of v1's fills, and that reconcile then fails loud on them -- it cannot
# pass on a missing fill.
#
# Edits exactly one key of config.ini in place (mode and owner kept) and never
# prints the file: it holds the IB login.
set -euo pipefail

TRADER_ENV="${CRUCIBLE_TRADER_ENV_FILE:-/home/ec2-user/alpha-engine-config/crucible-trader/trader.env}"
IBC_INI="${IBC_INI:-/home/ec2-user/ibc/config.ini}"
KEY="OverrideTwsMasterClientID"
VAR="CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID"
#: The v1 executor's client ids (morning batch, daemon, emergency shutdown).
V1_CLIENT_IDS=(1 2 3)

say() { echo "ibc-master-client-id: $*"; }
fail() { say "ERROR: $*" >&2; exit 1; }

[[ -r "$TRADER_ENV" ]] || fail "trader env file $TRADER_ENV is not readable; gateway master client id left unchanged"
[[ -f "$IBC_INI" && -w "$IBC_INI" ]] || fail "IBC config $IBC_INI is missing or not writable; gateway master client id left unchanged"

# systemd EnvironmentFile syntax (KEY=value, no quotes): read, never source.
env_value() {
    local name="$1"
    grep -E "^[[:space:]]*${name}=" "$TRADER_ENV" | tail -n 1 | cut -d= -f2- | tr -d '[:space:]' || true
}

master="$(env_value "$VAR")"
own="$(env_value CRUCIBLE_TRADER_IB_CLIENT_ID)"
[[ -n "$master" ]] || fail "$VAR is not set in $TRADER_ENV; gateway master client id left unchanged"
[[ "$master" =~ ^[0-9]+$ ]] || fail "$VAR=$master is not a non-negative integer"
for v1 in "${V1_CLIENT_IDS[@]}"; do
    [[ "$master" != "$v1" ]] || fail "$VAR=$master is a v1 executor client id; refusing"
done
[[ "$master" != "$own" ]] || fail "$VAR=$master equals the trader's order-capable CRUCIBLE_TRADER_IB_CLIENT_ID; refusing"

if grep -qE "^${KEY}=" "$IBC_INI"; then
    current="$(grep -E "^${KEY}=" "$IBC_INI" | tail -n 1 | cut -d= -f2- | tr -d '[:space:]')"
    if [[ "$current" == "$master" ]]; then
        say "$KEY already $master in $IBC_INI"
        exit 0
    fi
    sed -i -E "s/^${KEY}=.*/${KEY}=${master}/" "$IBC_INI"
else
    printf '\n# Set by crucible-executor infrastructure/ibc-master-client-id.sh (alpha-engine-config-I12376).\n%s=%s\n' \
        "$KEY" "$master" >> "$IBC_INI"
fi
say "$KEY set to $master in $IBC_INI (was '${current:-unset}')"
