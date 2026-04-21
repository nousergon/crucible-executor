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
