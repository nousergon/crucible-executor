"""Regression: boot-pull.sh must enable every shipped timer on every boot.

2026-04-21 SNDK EOD incident: `alpha-engine-eod.timer` was disabled on
the trading instance for an unknown reason. boot-pull.sh's old logic
only ran `systemctl enable` inside the "new install" branch (`[ ! -f
"$target" ]`), so a unit file that existed on disk but was disabled
could never be re-enabled. EOD emails silently stopped firing for two
boots until manual SSM intervention.

The fix is a reconciliation pass that `systemctl enable`s every
`*.timer` shipped in `infrastructure/systemd/` on every boot.
`systemctl enable` is idempotent on already-enabled timers, and
restores the `timers.target.wants/` symlink on disabled-but-present
timers. Covers manual disables, EBS volume state drift, and
post-setup additions of new timer units.

These tests lock the reconciliation so a future well-intentioned
refactor doesn't re-introduce the enable-only-on-new-install bug.
"""

from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"


def _source() -> str:
    return _BOOT_PULL.read_text()


def test_boot_pull_exists():
    assert _BOOT_PULL.exists(), f"boot-pull.sh missing at {_BOOT_PULL}"


def test_reconcile_loop_enables_every_timer():
    """Must iterate every *.timer and call `systemctl enable` on each."""
    src = _source()
    # The reconciliation loop signature: a for-loop over *.timer files
    # that calls systemctl enable on each.
    assert 'for unit in "$SYSTEMD_SRC"/*.timer' in src, (
        "boot-pull.sh must have a dedicated loop over *.timer for the "
        "reconciliation pass — not just the combined service+timer "
        "install loop. Every boot must re-assert enable state on every "
        "shipped timer."
    )
    assert "systemctl enable" in src, (
        "reconcile loop must call `systemctl enable` (idempotent no-op "
        "when already enabled; re-creates timers.target.wants symlink "
        "when disabled)."
    )


def test_no_new_timers_only_enable_path():
    """The old NEW_TIMERS-only enable path must be gone.

    The 2026-04-21 bug was that `systemctl enable` only ran for timers
    whose target file did not exist on disk (i.e. first install).
    A disabled-but-present timer had no code path to re-enable.
    """
    src = _source()
    # NEW_TIMERS tracker must be gone — the reconcile loop replaced it.
    assert "NEW_TIMERS" not in src, (
        "NEW_TIMERS tracker found in boot-pull.sh — this is the "
        "'enable only on first install' bug that caused the 2026-04-21 "
        "EOD outage. Use a reconciliation loop over every timer "
        "shipped in infrastructure/systemd/ instead."
    )


# ── Orphan-removal pass (2026-04-28: EOD pipeline → SF cutover) ──────────────
# Adding a new timer was self-healing (install + enable handled by the
# reconciler) but retiring one was not — the unit file lingered on disk
# and continued firing even after deletion from the repo. Removing the
# alpha-engine-daily-data.* and alpha-engine-eod.* units to cut over to
# the EOD Step Function exposed this gap.


def test_orphan_removal_loop_exists():
    """boot-pull.sh must scan /etc/systemd/system for units (scoped by
    caller-supplied prefix) that no longer have a source in the repo, and
    disable + remove them. Without this pass, retiring a unit requires
    manual SSM.

    config#2352: this reconciliation was factored into the
    sync_systemd_units_from() function (parametrized by source dir + orphan
    prefixes) so the SAME logic also covers nousergon-data's systemd units
    (metron-intraday, systemd-unit-drift-check) without duplicating the
    whole block. The glob is now `/etc/systemd/system/${prefix}*` built
    from a caller-supplied variable rather than a literal `alpha-engine-*`
    string — assert the parametrized shape + that the alpha-engine-* call
    site still exists (see test_orphan_removal_safety_prefix below)."""
    src = _source()
    assert "/etc/systemd/system/${prefix}*.service" in src, (
        "boot-pull.sh must iterate /etc/systemd/system/${prefix}*.service "
        "(parametrized orphan glob) to find orphaned units."
    )
    assert "systemctl disable --now" in src, (
        "orphan-removal pass must call `systemctl disable --now` to "
        "stop active timers before removing the unit file."
    )
    assert 'rm -f "$installed"' in src or "rm -f $installed" in src, (
        "orphan-removal pass must `rm` the unit file from "
        "/etc/systemd/system after disabling — leaving it on disk "
        "lets `systemctl start` re-fire it manually."
    )


def test_orphan_removal_safety_prefix():
    """The orphan loop must only match a caller-supplied prefix — never an
    unscoped glob. Removing arbitrary units from /etc/systemd/system would
    brick the host. The alpha-engine repo's own systemd sync must still be
    called with the "alpha-engine-" prefix (its historical scope)."""
    src = _source()
    # The orphan loop must build its glob from a prefix variable, not a
    # bare wildcard — `${prefix}*.service`/`${prefix}*.timer`, never a
    # literal `/etc/systemd/system/*.service` that would sweep every unit
    # on the box.
    assert "${prefix}*.service" in src and "${prefix}*.timer" in src, (
        "orphan-removal globs must be built from a scoped $prefix variable, "
        "never an unscoped wildcard, to avoid disabling unrelated system units."
    )
    # alpha-engine's own call site must still pass "alpha-engine-" — the
    # historical safety scope for THIS repo's units.
    assert 'sync_systemd_units_from "/home/ec2-user/alpha-engine/infrastructure/systemd" "alpha-engine-"' in src, (
        "alpha-engine's systemd sync call site must still scope its orphan "
        "reconciliation to the alpha-engine- prefix."
    )


def test_retired_units_not_shipped():
    """The four units retired in the EOD-SF cutover must NOT exist in
    infrastructure/systemd/ (the orphan-removal loop on next boot will
    then disable + remove them on ae-trading)."""
    systemd_src = Path(__file__).parent.parent / "infrastructure" / "systemd"
    retired = [
        "alpha-engine-daily-data.service",
        "alpha-engine-daily-data.timer",
        "alpha-engine-eod.service",
        "alpha-engine-eod.timer",
    ]
    for name in retired:
        assert not (systemd_src / name).exists(), (
            f"{name} must not be in infrastructure/systemd/ — "
            f"the EOD Step Function (ne-postclose-trading-pipeline) is the "
            f"canonical path. If you re-add it, the SF triggers and the "
            f"systemd timer will both fire (duplicate emails / racing "
            f"writes against ArcticDB)."
        )


# ── nousergon-data systemd sync (config#2352) ────────────────────────────────
# metron-intraday.{service,timer} (and this issue's own
# systemd-unit-drift-check.{service,timer}) live in nousergon-data's
# infrastructure/systemd/, not this repo's. Prior to config#2352, boot-pull
# only ever looked at $HOME/alpha-engine/infrastructure/systemd — a merged
# nousergon-data unit edit silently never took effect (system_state/data.md:
# "boot-pull does NOT reinstall units"). These tests pin that a future
# refactor can't silently drop the second sync pass.


def test_nousergon_data_systemd_sync_call_site_exists():
    """boot-pull.sh must run sync_systemd_units_from against
    nousergon-data's infrastructure/systemd/ dir, scoped to the two unit
    families it ships (metron-intraday, systemd-unit-drift-check) — not an
    unscoped/wildcard prefix that could sweep unrelated units."""
    src = _source()
    assert (
        'sync_systemd_units_from "/home/ec2-user/alpha-engine-data/infrastructure/systemd" '
        '"metron-intraday" "systemd-unit-drift-check"' in src
    ), (
        "boot-pull.sh must sync nousergon-data's infrastructure/systemd/ "
        "(metron-intraday + systemd-unit-drift-check orphan prefixes) — "
        "without this call, a merged nousergon-data unit-file edit never "
        "reaches the trading box (config#2352)."
    )
