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


# ── Interval-extraction state machine tests ───────────────────────────────
#
# journal_intervals() makes a subprocess call, so these tests feed events
# through a tiny helper that replicates the state-machine walk. Keeping the
# state machine's logic in a module-level helper would be cleaner — but for
# now we exercise it by monkeypatching subprocess output.


def _fake_journalctl_output(events: list[tuple[datetime, str]]) -> str:
    """Render (ts, 'start'|'stop') tuples as JSON lines journalctl would emit."""
    import json
    lines = []
    for ts, kind in events:
        msg = (
            "Started alpha-engine-daemon.service - …"
            if kind == "start"
            else "Stopped alpha-engine-daemon.service - …"
        )
        lines.append(json.dumps({
            "MESSAGE": msg,
            "__REALTIME_TIMESTAMP": str(int(ts.timestamp() * 1_000_000)),
        }))
    return "\n".join(lines)


def test_journal_intervals_dedupes_consecutive_stops(monkeypatch):
    """systemd emits Deactivated + Stopped + Main-exited per real stop.
    State machine must collapse them into one transition."""
    day = date(2026, 4, 13)
    # Start, then 3 back-to-back stop messages (as systemd really emits),
    # then a fresh start, then one stop.
    events = [
        (_et(2026, 4, 13, 10, 0), "start"),
        (_et(2026, 4, 13, 12, 0), "stop"),
        (_et(2026, 4, 13, 12, 0, ), "stop"),  # duplicate
        (_et(2026, 4, 13, 12, 0), "stop"),  # duplicate
        (_et(2026, 4, 13, 13, 0), "start"),
        (_et(2026, 4, 13, 15, 0), "stop"),
    ]
    monkeypatch.setattr(
        bf.subprocess,
        "check_output",
        lambda *a, **kw: _fake_journalctl_output(events),
    )
    intervals = bf.journal_intervals(day)
    # Two real intervals, no triple-counted (day_start, 12:00) ones.
    assert len(intervals) == 2
    m = bf.compute_backfill_metrics(intervals, day)
    # 10:00 → 12:00 = 120 min; 13:00 → 15:00 = 120 min
    assert m["active_minutes"] == 240
    assert m["uptime_pct"] == round(240 / 390, 4)
    # No runaway percentages
    assert m["uptime_pct"] <= 1.0


def test_journal_intervals_daemon_already_running_at_day_start(monkeypatch):
    """If the first event is a stop, the daemon was running before day_start."""
    day = date(2026, 4, 13)
    events = [
        (_et(2026, 4, 13, 10, 0), "stop"),
        (_et(2026, 4, 13, 11, 0), "start"),
        (_et(2026, 4, 13, 15, 0), "stop"),
    ]
    monkeypatch.setattr(
        bf.subprocess,
        "check_output",
        lambda *a, **kw: _fake_journalctl_output(events),
    )
    intervals = bf.journal_intervals(day)
    assert len(intervals) == 2
    m = bf.compute_backfill_metrics(intervals, day)
    # Day-start-to-10:00 clipped to (9:30, 10:00) = 30 min
    # 11:00 to 15:00 = 240 min
    assert m["active_minutes"] == 30 + 240


def test_journal_intervals_daemon_still_running_at_day_end(monkeypatch):
    """If the daemon starts and never stops, close the interval at day end."""
    day = date(2026, 4, 13)
    events = [(_et(2026, 4, 13, 10, 0), "start")]
    monkeypatch.setattr(
        bf.subprocess,
        "check_output",
        lambda *a, **kw: _fake_journalctl_output(events),
    )
    intervals = bf.journal_intervals(day)
    assert len(intervals) == 1
    m = bf.compute_backfill_metrics(intervals, day)
    # 10:00 → 16:00 market close = 360 min
    assert m["active_minutes"] == 360
