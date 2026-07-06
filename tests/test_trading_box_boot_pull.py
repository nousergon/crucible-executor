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
