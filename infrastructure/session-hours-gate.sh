#!/usr/bin/env bash
# The v2 trader session's hours gate (alpha-engine-config-I13544). Run by
# systemd as an ExecCondition= on two units:
#
#   session-hours-gate.sh            alpha-engine-trader-session.service
#   session-hours-gate.sh session-ran alpha-engine-v2-trade-alert.service
#
# WHY. alpha-engine-trader-session.service started at 20:36 ET on 2026-11-09
# (alpha-engine-config-I13447) and at 20:38 ET on 2026-11-10: the post-close
# reconcile's boot-pull replaced the changed session timer on a box up since
# the morning, and the timer fired at once -- on 11-10 even though boot-pull
# now stops a changed timer first. The trader held the run itself ("outside
# market hours"), but it still wrote a held run manifest for the next book day,
# and the crucible sweep paged that day as not served. A start outside the
# session's hours has to be refused before anything runs, whatever fired it.
#
# DEFAULT MODE: exit 0 when now is Monday-Friday, 09:25 <= time < 16:00
# America/New_York; otherwise exit 1. systemd reads exit 1-254 from an
# ExecCondition= as "condition failed": it runs no ExecStartPre= and no
# ExecStart=, records Result=exec-condition, and the unit ends INACTIVE, not
# failed -- so no OnFailure= page. NYSE holidays are NOT checked here: the
# trader skips them itself, and this gate stays a pure clock.
#
# SESSION-RAN MODE: the session's OnSuccess= still fires on such a skip
# (systemd 252 unit_notify starts OnSuccess= on any activating->inactive
# transition), which would send a "session ended" Telegram for a session that
# never ran. The trade alert's own ExecCondition= runs this mode: exit 1 when
# the session's last Result is exec-condition, exit 0 otherwise -- including
# when the Result cannot be read, because an alert too many beats a session
# ending nobody hears about.
#
# ON AN INTERNAL ERROR the default mode exits 255. systemd treats 255 from an
# ExecCondition= as a FAILURE, not a skip: nothing runs (closed), and the unit
# ends failed, so alpha-engine-trader-failed@ pages and the session watchdog
# sees a failed session (loud). Deliberately not exit 1: a gate that cannot
# read the clock must not quietly skip a trading day. The one quiet path left
# is this file or bash missing (env exits 126/127, read as a skip); then the
# session watchdog finds no session run today from 09:45 ET, retries twice and
# says so on Telegram (executor.v2_session_watchdog).
#
# Test hooks: SESSION_HOURS_GATE_NOW (epoch seconds) replaces the clock;
# SESSION_HOURS_GATE_SYSTEMCTL replaces systemctl. Never set on the box.
set -u

SESSION_UNIT="alpha-engine-trader-session.service"
OPEN_HHMM=925   # 09:25 ET: five minutes before the 09:30 timer, inclusive
CLOSE_HHMM=1600 # 16:00 ET: the close, exclusive (the 16:05 hard stop is a stop)

say() { echo "session-hours-gate: $*"; }

session_ran() {
    local systemctl result
    systemctl="${SESSION_HOURS_GATE_SYSTEMCTL:-/usr/bin/systemctl}"
    if ! result=$("$systemctl" show -P Result "$SESSION_UNIT" 2>/dev/null); then
        say "could not read $SESSION_UNIT Result; sending the alert anyway"
        return 0
    fi
    if [ "$result" = "exec-condition" ]; then
        say "skip: $SESSION_UNIT never ran (its hours gate refused the start); no alert"
        return 1
    fi
    return 0
}

hours() {
    local now stamp dow hhmm zone
    now="${SESSION_HOURS_GATE_NOW:-}"
    if [ -z "$now" ]; then
        now=$(date +%s) || { say "ERROR: cannot read the clock"; return 255; }
    fi
    case "$now" in
        '' | *[!0-9]*) say "ERROR: not an epoch time: '$now'"; return 255 ;;
    esac
    if ! stamp=$(TZ=America/New_York date -d "@$now" '+%u %H%M %Z %Y-%m-%d %H:%M' 2>/dev/null); then
        say "ERROR: date could not format $now in America/New_York"
        return 255
    fi
    read -r dow hhmm zone _ <<< "$stamp"
    # A missing zoneinfo makes date fall back to UTC without an error; a time
    # read in the wrong zone is a wrong answer, so it is an error here.
    case "$zone" in
        EST | EDT) ;;
        *) say "ERROR: America/New_York resolved to zone '$zone' (zoneinfo missing?)"; return 255 ;;
    esac
    case "$dow$hhmm" in
        [1-7][0-9][0-9][0-9][0-9]) ;;
        *) say "ERROR: unexpected date output '$stamp'"; return 255 ;;
    esac
    local when="${stamp#* * * }"
    if [ "$dow" -ge 6 ]; then
        say "refused: $when $zone is a weekend; the trader session runs Monday-Friday 09:25-16:00 ET"
        return 1
    fi
    if (( 10#$hhmm < OPEN_HHMM || 10#$hhmm >= CLOSE_HHMM )); then
        say "refused: $when $zone is outside the trader session's 09:25-16:00 ET hours; nothing started"
        return 1
    fi
    say "ok: $when $zone is inside 09:25-16:00 ET"
    return 0
}

case "${1:-}" in
    '') hours ;;
    session-ran) session_ran ;;
    *) say "ERROR: unknown mode '$1'"; exit 255 ;;
esac
exit $?
