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
