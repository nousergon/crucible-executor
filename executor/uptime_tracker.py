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
      "crashes": 2,                 # gaps > 2.5x tick cadence during market hours
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

_DEFAULT_LOG = "/var/log/executor.log"
_DEFAULT_TICK_SEC = 60
_GAP_TOLERANCE_MULT = 2.5


def _parse_ts(line: str) -> datetime | None:
    """Parse the leading UTC timestamp from a log line."""
    m = _TS_RE.match(line)
    if not m:
        return None
    try:
        ts_naive = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return pytz.utc.localize(ts_naive)


def parse_tick_lines(lines: Iterable[str], day: date) -> list[tuple[datetime, bool]]:
    """Return (ts_et, ib_connected) tuples for DAEMON_TICK lines falling on `day` (ET)."""
    ticks: list[tuple[datetime, bool]] = []
    for line in lines:
        if "DAEMON_TICK" not in line:
            continue
        m = _TICK_RE.search(line)
        if not m:
            continue
        ts_utc = _parse_ts(line)
        if ts_utc is None:
            continue
        ts_et = ts_utc.astimezone(_ET)
        if ts_et.date() != day:
            continue
        ticks.append((ts_et, m.group(1) == "true"))
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
    crashes = 0
    if window:
        prev_ts = window[0][0]
        for ts, _ in window[1:]:
            if (ts - prev_ts).total_seconds() > gap_tol_sec:
                crashes += 1
            prev_ts = ts

    connected_minutes = len(minutes_connected)
    return {
        "date": day.isoformat(),
        "active_minutes": len(minutes_active),
        "connected_minutes": connected_minutes,
        "market_minutes": _MARKET_MINUTES,
        "uptime_pct": round(connected_minutes / _MARKET_MINUTES, 4),
        "crashes": crashes,
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
