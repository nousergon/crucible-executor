"""ae-trading boot-pull scope invariants.

The executor box must only git-sync repos the weekday/EOD Step Functions
actually invoke via SSM. Dashboard + backtester run on ae-dashboard and
Saturday spots — pulling them on trading wastes disk and pip time.
"""
from __future__ import annotations

from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"


def test_boot_pull_excludes_dashboard_and_backtester():
    src = _BOOT_PULL.read_text()
    assert "/home/ec2-user/alpha-engine-config" in src
    assert "/home/ec2-user/alpha-engine" in src
    assert "/home/ec2-user/alpha-engine-data" in src
    assert "/home/ec2-user/alpha-engine-dashboard" not in src
    assert "/home/ec2-user/alpha-engine-backtester" not in src


def test_trading_box_cleanup_script_exists():
    path = Path(__file__).parent.parent / "infrastructure" / "trading-box-cleanup.sh"
    assert path.is_file()
    text = path.read_text()
    assert "alpha-engine-dashboard" in text
    assert "predictor" in text


def test_boot_pull_reclaims_foreign_owned_files_before_git_reset():
    """Ownership reclaim must precede the git fetch/reset block.

    2026-07-06 incident (config#1811): a feature branch's sudo timer-install
    step left infrastructure/ops/ root-owned inside the ec2-user checkout;
    `git reset --hard origin/main` failed with "unable to unlink ...
    Permission denied" on every boot, the box silently ran 4-commits-stale
    code on a stray branch, and the day's pipeline burned ~40 min before the
    executor's deploy-drift preflight refused. The reclaim block makes that
    failure mode structurally impossible; this test pins its presence AND
    its ordering (reclaim before reset — after would be useless).
    """
    src = _BOOT_PULL.read_text()
    assert "-not -user ec2-user" in src, "foreign-ownership detection missing"
    assert 'chown -R ec2-user:ec2-user "$repo"' in src, "ownership reclaim missing"
    reclaim_pos = src.index("-not -user ec2-user")
    reset_pos = src.index("git reset --hard origin/main")
    assert reclaim_pos < reset_pos, "ownership reclaim must run BEFORE git reset"


def test_boot_pull_git_sync_runs_under_shared_flock():
    """config#1944: the per-repo git fetch/checkout/reset must run under a
    shared advisory flock so boot-pull.service can't race the weekday
    CodeFreshnessGate / ChronicGapSelfHeal (nousergon-data
    step_function_daily.json) on .git/index.lock.

    2026-07-08 ne-preopen-trading FailExecution: boot-pull's `git reset --hard`
    held alpha-engine-data/.git/index.lock while the gate's checkout/reset ran
    -> "Another git process seems to be running" (exit 128) -> no orders placed.
    The flock is window-free (kernel mutex) and auto-releases on process death.
    This pins that a future edit can't silently drop back to bare, race-prone
    git calls.
    """
    import re

    src = _BOOT_PULL.read_text()
    # The lock must live in ec2-user's HOME, not /var/lock: /var/lock ->
    # /run/lock is root:root 0755, so an ec2-user boot-pull cannot create a
    # lock file there. The nousergon-data gate flocks this SAME path.
    assert "/home/ec2-user/.ae-git-sync.lock" in src, (
        "git-sync flock must use the shared /home/ec2-user/.ae-git-sync.lock "
        "path (the nousergon-data CodeFreshnessGate uses the identical inode)."
    )
    # A bounded flock must wrap the index-mutating reset (window-free), and the
    # bound must be > boot-pull's own 120s TimeoutStartSec so a genuinely stuck
    # writer fails loud rather than the flock timing out prematurely.
    assert re.search(r"flock -w \S+ \S+ bash -c '[^']*git reset --hard origin/main", src), (
        "the git fetch/checkout/reset group must run under `flock -w <wait> "
        "<lock> bash -c '...'` so the whole index mutation is serialized."
    )


def test_boot_pull_git_sync_lock_wait_exceeds_boot_pull_timeout():
    """The flock wait budget must exceed boot-pull.service's TimeoutStartSec
    (120s) so the gate can outwait a full boot-pull git-sync rather than the
    flock timing out and failing a healthy run."""
    src = _BOOT_PULL.read_text()
    assert 'GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"' in src, (
        "flock wait must default to 150s (> boot-pull.service TimeoutStartSec=120)."
    )
