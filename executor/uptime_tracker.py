"""
Uptime tracker — computes daemon uptime during NYSE regular-session hours.

Data source: DAEMON_TICK lines emitted by executor/daemon.py each poll cycle.
A minute is "up" when a tick fired within tolerance AND ib_connected=true.

Writes uptime/{date}.json to S3:
    {
      "date": "2026-04-13",
      "active_minutes": 312,        # daemon was running
      "connected_minutes": 295,     # daemon running AND IB Gateway connected
      "market_minutes": 390,        # 9:30-16:00 ET = 6.5h
      "uptime_pct": 0.756,          # connected_minutes / market_minutes
      "service_restarts": 2,        # gaps > 2.5x tick cadence during market hours
      "tick_cadence_sec": 60,
      "source": "tick_log"
    }

Assumes log timestamps are UTC (EC2 default). Intended to run on ae-trading
at 1:20 PM PT inside eod_reconcile.py, after the daemon has stopped.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, time
from typing import Iterable

import pytz

from executor.market_hours import is_trading_day

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_MARKET_MINUTES = 6 * 60 + 30  # 390

_TICK_RE = re.compile(r"DAEMON_TICK\s+ib_connected=(true|false)")
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]?\d*")

# The daemon's systemd unit redirects stdout/stderr to /var/log/daemon.log;
# DAEMON_TICK lines live there, not in executor.log (which holds main.py).
_DEFAULT_LOG = "/var/log/daemon.log"
_DEFAULT_TICK_SEC = 60
_GAP_TOLERANCE_MULT = 2.5


def _parse_ts(line: str) -> datetime | None:
    """Parse the leading UTC timestamp from a legacy-text log line."""
    m = _TS_RE.match(line)
    if not m:
        return None
    try:
        ts_naive = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return pytz.utc.localize(ts_naive)


def _parse_tick_line(line: str) -> tuple[datetime, bool] | None:
    """Extract (ts_utc, ib_connected) from a DAEMON_TICK line in either the
    legacy text format or the current JSON format emitted by
    alpha_engine_lib.logging. Returns None for anything else.
    """
    if "DAEMON_TICK" not in line:
        return None

    stripped = line.lstrip()
    if stripped.startswith("{") and '"msg"' in stripped:
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        msg = rec.get("msg", "")
        tick = _TICK_RE.search(msg)
        if not tick:
            return None
        ts_raw = rec.get("ts")
        if not isinstance(ts_raw, str):
            return None
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            return None
        ts_utc = pytz.utc.localize(ts) if ts.tzinfo is None else ts.astimezone(pytz.utc)
        return ts_utc, tick.group(1) == "true"

    tick = _TICK_RE.search(line)
    if not tick:
        return None
    ts_utc = _parse_ts(line)
    if ts_utc is None:
        return None
    return ts_utc, tick.group(1) == "true"


def parse_tick_lines(lines: Iterable[str], day: date) -> list[tuple[datetime, bool]]:
    """Return (ts_et, ib_connected) tuples for DAEMON_TICK lines falling on `day` (ET)."""
    ticks: list[tuple[datetime, bool]] = []
    for line in lines:
        parsed = _parse_tick_line(line)
        if parsed is None:
            continue
        ts_utc, connected = parsed
        ts_et = ts_utc.astimezone(_ET)
        if ts_et.date() != day:
            continue
        ticks.append((ts_et, connected))
    ticks.sort(key=lambda x: x[0])
    return ticks


def compute_metrics(
    ticks: list[tuple[datetime, bool]],
    day: date,
    tick_cadence: int = _DEFAULT_TICK_SEC,
    source: str = "tick_log",
) -> dict:
    """Intersect tick events with the market-hours window and compute metrics."""
    market_open_et = _ET.localize(datetime.combine(day, _MARKET_OPEN))
    market_close_et = _ET.localize(datetime.combine(day, _MARKET_CLOSE))

    window = [(ts, c) for ts, c in ticks if market_open_et <= ts < market_close_et]

    minutes_active: set[int] = set()
    minutes_connected: set[int] = set()
    for ts, connected in window:
        minute = int((ts - market_open_et).total_seconds() // 60)
        minutes_active.add(minute)
        if connected:
            minutes_connected.add(minute)

    gap_tol_sec = tick_cadence * _GAP_TOLERANCE_MULT
    service_restarts = 0
    if window:
        prev_ts = window[0][0]
        for ts, _ in window[1:]:
            if (ts - prev_ts).total_seconds() > gap_tol_sec:
                service_restarts += 1
            prev_ts = ts

    connected_minutes = len(minutes_connected)
    return {
        "date": day.isoformat(),
        "active_minutes": len(minutes_active),
        "connected_minutes": connected_minutes,
        "market_minutes": _MARKET_MINUTES,
        "uptime_pct": round(connected_minutes / _MARKET_MINUTES, 4),
        # service_restarts: count of daemon start↔stop transitions during
        # market hours. Mixes planned maintenance restarts with crash-and-
        # recover cycles; not a clean "crashes" signal. Retained in JSON
        # for analysis but not surfaced on the homepage.
        "service_restarts": service_restarts,
        "tick_cadence_sec": tick_cadence,
        "source": source,
    }


def collect_uptime(
    day: date | None = None,
    log_path: str = _DEFAULT_LOG,
    tick_cadence: int = _DEFAULT_TICK_SEC,
) -> dict:
    """Parse the executor log for `day` and return uptime metrics. No S3 I/O."""
    if day is None:
        day = datetime.now(_ET).date()

    if not is_trading_day(day):
        return {"date": day.isoformat(), "skipped": "non_trading_day"}

    if not os.path.exists(log_path):
        logger.warning("uptime_tracker: log not found at %s", log_path)
        return compute_metrics([], day, tick_cadence)

    with open(log_path, "r", errors="replace") as f:
        ticks = parse_tick_lines(f, day)

    logger.info("uptime_tracker: parsed %d ticks for %s", len(ticks), day)
    return compute_metrics(ticks, day, tick_cadence)


def write_to_s3(metrics: dict, bucket: str, s3_client=None) -> str:
    """Write uptime metrics to s3://{bucket}/uptime/{date}.json. Returns the key."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    key = f"uptime/{metrics['date']}.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )
    logger.info(
        "uptime_tracker: wrote s3://%s/%s (uptime=%.1f%%)",
        bucket, key, metrics.get("uptime_pct", 0) * 100,
    )
    return key


def run(bucket: str, day: date | None = None, log_path: str = _DEFAULT_LOG) -> dict:
    """End-to-end: collect uptime and write to S3. Returns metrics dict."""
    metrics = collect_uptime(day=day, log_path=log_path)
    if metrics.get("skipped"):
        logger.info("uptime_tracker: skipped %s (%s)", metrics["date"], metrics["skipped"])
        return metrics
    write_to_s3(metrics, bucket)
    return metrics
