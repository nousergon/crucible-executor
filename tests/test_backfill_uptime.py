"""Unit tests for executor/backfill_uptime.py interval-based computation."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytz

from executor import backfill_uptime as bf


_ET = pytz.timezone("US/Eastern")
_UTC = pytz.utc


def _et(y, m, d, H, M) -> datetime:
    return _ET.localize(datetime(y, m, d, H, M)).astimezone(_UTC)


def test_full_session_covered_by_single_interval():
    """Daemon runs from 9:00 ET to 16:30 ET — full market window is covered."""
    day = date(2026, 4, 13)
    intervals = [(_et(2026, 4, 13, 9, 0), _et(2026, 4, 13, 16, 30))]

    m = bf.compute_backfill_metrics(intervals, day)
    assert m["active_minutes"] == 390
    assert m["connected_minutes"] == 390
    assert m["market_minutes"] == 390
    assert m["uptime_pct"] == 1.0
    assert m["crashes"] == 0
    assert m["source"] == "journalctl_intervals"


def test_interval_fully_outside_market_hours_ignored():
    """Pre-market and post-close intervals contribute zero minutes."""
    day = date(2026, 4, 13)
    intervals = [
        (_et(2026, 4, 13, 7, 0), _et(2026, 4, 13, 9, 0)),    # pre-market
        (_et(2026, 4, 13, 17, 0), _et(2026, 4, 13, 18, 0)),  # post-close
    ]
    m = bf.compute_backfill_metrics(intervals, day)
    assert m["active_minutes"] == 0
    assert m["uptime_pct"] == 0.0


def test_interval_clipped_at_market_open():
    """Interval spanning pre-market to mid-session counts only from 9:30 ET."""
    day = date(2026, 4, 13)
    intervals = [(_et(2026, 4, 13, 8, 0), _et(2026, 4, 13, 10, 30))]  # 8:00 → 10:30 ET
    m = bf.compute_backfill_metrics(intervals, day)
    # 9:30 to 10:30 = 60 minutes
    assert m["active_minutes"] == 60


def test_interval_clipped_at_market_close():
    """Interval spanning mid-session to post-close counts only up to 16:00 ET."""
    day = date(2026, 4, 13)
    intervals = [(_et(2026, 4, 13, 15, 0), _et(2026, 4, 13, 17, 0))]  # 15:00 → 17:00 ET
    m = bf.compute_backfill_metrics(intervals, day)
    # 15:00 to 16:00 = 60 minutes
    assert m["active_minutes"] == 60


def test_crash_and_recovery_counts_as_restart():
    """Two intervals within market hours = one crash/restart between them."""
    day = date(2026, 4, 13)
    intervals = [
        (_et(2026, 4, 13, 9, 30), _et(2026, 4, 13, 12, 0)),   # 150 min
        (_et(2026, 4, 13, 12, 30), _et(2026, 4, 13, 16, 0)),  # 210 min
    ]
    m = bf.compute_backfill_metrics(intervals, day)
    assert m["active_minutes"] == 150 + 210
    assert m["crashes"] == 1


def test_empty_intervals_yields_zero():
    m = bf.compute_backfill_metrics([], date(2026, 4, 13))
    assert m["active_minutes"] == 0
    assert m["connected_minutes"] == 0
    assert m["uptime_pct"] == 0.0
    assert m["crashes"] == 0


def test_quiet_daemon_still_counts_as_active():
    """Regression: old backfill undercounted quiet daemons. Interval-based
    treats the whole start→stop span as active regardless of log cadence."""
    day = date(2026, 4, 13)
    # Daemon ran all market hours but logged only twice (startup + shutdown)
    intervals = [(_et(2026, 4, 13, 9, 29), _et(2026, 4, 13, 16, 1))]
    m = bf.compute_backfill_metrics(intervals, day)
    assert m["active_minutes"] == 390
    assert m["uptime_pct"] == 1.0
