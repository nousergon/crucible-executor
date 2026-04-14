"""Unit tests for executor/uptime_tracker.py."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import pytz

from executor import uptime_tracker as ut


_ET = pytz.timezone("US/Eastern")
_UTC = pytz.utc


def _make_tick_line(ts_utc: datetime, connected: bool) -> str:
    """Render a log line in the same format the executor writes."""
    return (
        f"{ts_utc.strftime('%Y-%m-%d %H:%M:%S')},000 "
        f"INFO [executor] DAEMON_TICK ib_connected={'true' if connected else 'false'}\n"
    )


def test_parse_tick_lines_roundtrip():
    day = date(2026, 4, 13)
    # 9:30 AM ET = 13:30 UTC (during EDT). Use April → EDT offset of 4h.
    t_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    t_utc = t_et.astimezone(_UTC)

    line = _make_tick_line(t_utc, connected=True)
    ticks = ut.parse_tick_lines([line, "some unrelated log\n"], day)

    assert len(ticks) == 1
    ts_et, connected = ticks[0]
    assert ts_et.hour == 9 and ts_et.minute == 30
    assert connected is True


def _make_json_tick_line(ts_utc: datetime, connected: bool) -> str:
    """Render a DAEMON_TICK line in the JSON format emitted by alpha_engine_lib.logging."""
    return json.dumps({
        "ts": ts_utc.isoformat(),
        "level": "INFO",
        "module": "daemon",
        "func": "run_daemon",
        "msg": f"DAEMON_TICK ib_connected={'true' if connected else 'false'}",
    }) + "\n"


def test_parse_tick_lines_json_format():
    """Parser handles the JSON format written by the new alpha_engine_lib.logging."""
    day = date(2026, 4, 13)
    t_et = _ET.localize(datetime(2026, 4, 13, 10, 15))
    t_utc = t_et.astimezone(_UTC)

    line = _make_json_tick_line(t_utc, connected=True)
    ticks = ut.parse_tick_lines([line, "{\"msg\": \"other\"}\n"], day)

    assert len(ticks) == 1
    ts_et, connected = ticks[0]
    assert ts_et.hour == 10 and ts_et.minute == 15
    assert connected is True


def test_parse_tick_lines_mixed_formats():
    """Text and JSON lines coexist during rollout — both get parsed."""
    day = date(2026, 4, 13)
    t1 = _ET.localize(datetime(2026, 4, 13, 9, 30)).astimezone(_UTC)
    t2 = _ET.localize(datetime(2026, 4, 13, 9, 31)).astimezone(_UTC)

    ticks = ut.parse_tick_lines(
        [_make_tick_line(t1, True), _make_json_tick_line(t2, False)],
        day,
    )
    assert len(ticks) == 2
    assert ticks[0][1] is True
    assert ticks[1][1] is False


def test_parse_tick_lines_filters_wrong_day():
    day = date(2026, 4, 13)
    # Tick from the next day's morning
    t_et = _ET.localize(datetime(2026, 4, 14, 9, 31))
    t_utc = t_et.astimezone(_UTC)
    line = _make_tick_line(t_utc, True)
    assert ut.parse_tick_lines([line], day) == []


def test_compute_metrics_full_connected_session():
    """One tick per minute for the full 6.5 hour window, all connected."""
    day = date(2026, 4, 13)
    market_open_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    ticks = [
        (market_open_et + timedelta(minutes=i), True) for i in range(390)
    ]

    m = ut.compute_metrics(ticks, day)
    assert m["active_minutes"] == 390
    assert m["connected_minutes"] == 390
    assert m["market_minutes"] == 390
    assert m["uptime_pct"] == 1.0
    assert m["service_restarts"] == 0


def test_compute_metrics_half_session_missing():
    """Daemon runs first half, dies at 12:45 ET — ~195 minutes up then gap."""
    day = date(2026, 4, 13)
    market_open_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    ticks = [(market_open_et + timedelta(minutes=i), True) for i in range(195)]

    m = ut.compute_metrics(ticks, day)
    assert m["connected_minutes"] == 195
    assert m["uptime_pct"] == round(195 / 390, 4)
    # No crashes recorded because no trailing tick exists to compare against;
    # mid-day silence is captured by the missing minutes, not the crash counter.
    assert m["service_restarts"] == 0


def test_compute_metrics_mid_session_crash():
    """Ticks stop at minute 100, resume at minute 200 — counts as one crash."""
    day = date(2026, 4, 13)
    market_open_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    before = [(market_open_et + timedelta(minutes=i), True) for i in range(100)]
    after = [(market_open_et + timedelta(minutes=i), True) for i in range(200, 390)]
    ticks = before + after

    m = ut.compute_metrics(ticks, day)
    assert m["connected_minutes"] == len(before) + len(after)
    assert m["service_restarts"] == 1


def test_compute_metrics_ib_disconnected_drops_connected_not_active():
    """Daemon up but IB disconnected — active_minutes counts, connected_minutes does not."""
    day = date(2026, 4, 13)
    market_open_et = _ET.localize(datetime(2026, 4, 13, 9, 30))
    ticks = [
        (market_open_et + timedelta(minutes=i), i >= 100) for i in range(390)
    ]

    m = ut.compute_metrics(ticks, day)
    assert m["active_minutes"] == 390
    assert m["connected_minutes"] == 290
    assert m["uptime_pct"] == round(290 / 390, 4)


def test_compute_metrics_empty_yields_zero():
    day = date(2026, 4, 13)
    m = ut.compute_metrics([], day)
    assert m["active_minutes"] == 0
    assert m["connected_minutes"] == 0
    assert m["uptime_pct"] == 0.0
    assert m["service_restarts"] == 0


def test_collect_uptime_skips_non_trading_day():
    # 2026-04-11 is a Saturday
    m = ut.collect_uptime(day=date(2026, 4, 11), log_path="/nonexistent")
    assert m.get("skipped") == "non_trading_day"


def test_collect_uptime_missing_log_returns_zero_record(tmp_path):
    m = ut.collect_uptime(day=date(2026, 4, 13), log_path=str(tmp_path / "nope.log"))
    assert m["date"] == "2026-04-13"
    assert m["connected_minutes"] == 0
    assert m["uptime_pct"] == 0.0
