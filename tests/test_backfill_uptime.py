"""Unit tests for executor/backfill_uptime.py."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytz

from executor import backfill_uptime as bf


_ET = pytz.timezone("US/Eastern")
_UTC = pytz.utc


def test_compute_backfill_metrics_buckets_to_market_minutes():
    day = date(2026, 4, 13)
    market_open_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    # 120 journal entries spaced 1 minute apart inside market hours
    timestamps = [
        (market_open_et + timedelta(minutes=i)).astimezone(_UTC) for i in range(120)
    ]
    m = bf.compute_backfill_metrics(timestamps, day)
    assert m["active_minutes"] == 120
    assert m["connected_minutes"] == 120  # best-effort equals active for backfill
    assert m["source"] == "journalctl_backfill"
    assert m["uptime_pct"] == round(120 / 390, 4)


def test_compute_backfill_metrics_ignores_outside_window():
    day = date(2026, 4, 13)
    # Pre-market 08:00 ET and post-close 17:00 ET — both should be excluded
    before = _ET.localize(datetime(2026, 4, 13, 8, 0)).astimezone(_UTC)
    after = _ET.localize(datetime(2026, 4, 13, 17, 0)).astimezone(_UTC)
    inside = _ET.localize(datetime(2026, 4, 13, 10, 0)).astimezone(_UTC)
    m = bf.compute_backfill_metrics([before, after, inside], day)
    assert m["active_minutes"] == 1


def test_compute_backfill_metrics_empty_yields_zero():
    m = bf.compute_backfill_metrics([], date(2026, 4, 13))
    assert m["active_minutes"] == 0
    assert m["uptime_pct"] == 0.0
