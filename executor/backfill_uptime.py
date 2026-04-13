"""
Backfill uptime/{date}.json for historical trading days.

DAEMON_TICK log lines are new as of today, so historical days have no
minute-level ib_connected signal. This script parses systemd start/stop
events from the journal to reconstruct the intervals during which the
daemon service was Active, then intersects those intervals with NYSE
regular-session hours.

connected_minutes is set equal to active_minutes (best-effort — cannot
distinguish broker disconnects retroactively; Active periods with a dead
IB socket count as connected).

Runs on ae-trading (requires journalctl access).

Usage:
  python executor/backfill_uptime.py --start 2026-03-15 --end 2026-04-12
  python executor/backfill_uptime.py --start 2026-03-15 --end 2026-04-12 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import date, datetime, time, timedelta

import pytz

from executor import config_loader
from executor.market_hours import is_trading_day
from executor.uptime_tracker import _MARKET_MINUTES, write_to_s3

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_UTC = pytz.utc
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

_JOURNAL_UNIT = "alpha-engine-daemon.service"

# systemd emits these messages on unit state transitions. journalctl -u filters
# to this unit so we don't need to re-check the unit name in the message.
_START_RE = re.compile(r"Started .*alpha-engine-daemon")
_STOP_RE = re.compile(
    r"Stopped .*alpha-engine-daemon"
    r"|Deactivated successfully"
    r"|Main process exited"
    r"|Failed with result"
)


def journal_intervals(day: date) -> list[tuple[datetime, datetime]]:
    """Return Active intervals (UTC) for the daemon service on `day` (ET).

    Walks journalctl output in JSON format, pairs Start/Stop events into
    [start, end] intervals. Handles two edge cases:
      - Daemon already running at day start (stop appears before any start) →
        interval opens at day start.
      - Daemon still running at day end (start with no matching stop) →
        interval closes at day end.
    """
    start_et = _ET.localize(datetime.combine(day, time(0, 0)))
    end_et = start_et + timedelta(days=1)
    since_utc = start_et.astimezone(_UTC).strftime("%Y-%m-%d %H:%M:%S")
    until_utc = end_et.astimezone(_UTC).strftime("%Y-%m-%d %H:%M:%S")

    cmd = [
        "journalctl", "-u", _JOURNAL_UNIT,
        "--since", since_utc, "--until", until_utc,
        "--output", "json", "--no-pager", "--utc",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except FileNotFoundError:
        logger.error("journalctl not on PATH — backfill only works on ae-trading")
        return []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("journalctl failed for %s: %s", day, e)
        return []

    events: list[tuple[datetime, str]] = []
    for line in out.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("MESSAGE", "")
        ts_raw = entry.get("__REALTIME_TIMESTAMP")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromtimestamp(int(ts_raw) / 1_000_000, tz=_UTC)
        except (ValueError, TypeError):
            continue
        if _START_RE.search(msg):
            events.append((ts, "start"))
        elif _STOP_RE.search(msg):
            events.append((ts, "stop"))

    events.sort(key=lambda x: x[0])

    day_start_utc = start_et.astimezone(_UTC)
    day_end_utc = end_et.astimezone(_UTC)
    intervals: list[tuple[datetime, datetime]] = []
    current_start: datetime | None = None

    for ts, kind in events:
        if kind == "start":
            if current_start is None:
                current_start = ts
            # else: double-start (no stop between) — keep the earlier start
        else:  # stop
            if current_start is None:
                # Daemon was already running at day start
                intervals.append((day_start_utc, ts))
            else:
                # Only record if stop is after start (guards against clock skew)
                if ts > current_start:
                    intervals.append((current_start, ts))
                current_start = None

    # Unclosed interval: daemon still running at day end
    if current_start is not None:
        intervals.append((current_start, day_end_utc))

    return intervals


def compute_backfill_metrics(
    intervals: list[tuple[datetime, datetime]],
    day: date,
) -> dict:
    """Sum the overlap between each interval and the market-hours window."""
    market_open_et = _ET.localize(datetime.combine(day, _MARKET_OPEN))
    market_close_et = _ET.localize(datetime.combine(day, _MARKET_CLOSE))
    market_open_utc = market_open_et.astimezone(_UTC)
    market_close_utc = market_close_et.astimezone(_UTC)

    total_seconds = 0.0
    for interval_start, interval_end in intervals:
        clipped_start = max(interval_start, market_open_utc)
        clipped_end = min(interval_end, market_close_utc)
        if clipped_end > clipped_start:
            total_seconds += (clipped_end - clipped_start).total_seconds()

    active_minutes = int(total_seconds // 60)
    # Each interval after the first implies a restart/crash between them.
    crashes = max(0, len(intervals) - 1)

    return {
        "date": day.isoformat(),
        "active_minutes": active_minutes,
        "connected_minutes": active_minutes,
        "market_minutes": _MARKET_MINUTES,
        "uptime_pct": round(active_minutes / _MARKET_MINUTES, 4) if _MARKET_MINUTES else 0.0,
        "crashes": crashes,
        "tick_cadence_sec": 0,
        "source": "journalctl_intervals",
    }


def backfill_range(start: date, end: date, bucket: str, force: bool = False) -> list[dict]:
    """Backfill every trading day in [start, end] inclusive. Returns list of written metrics."""
    import boto3

    s3 = boto3.client("s3")
    written: list[dict] = []
    cur = start

    while cur <= end:
        if not is_trading_day(cur):
            logger.debug("Skipping %s: non-trading day", cur)
            cur += timedelta(days=1)
            continue

        key = f"uptime/{cur.isoformat()}.json"
        if not force:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                logger.info("Skipping %s: %s already present", cur, key)
                cur += timedelta(days=1)
                continue
            except s3.exceptions.ClientError:
                pass

        intervals = journal_intervals(cur)
        metrics = compute_backfill_metrics(intervals, cur)
        write_to_s3(metrics, bucket, s3_client=s3)
        written.append(metrics)
        cur += timedelta(days=1)

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill historical uptime data from systemd journal")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--force", action="store_true", help="Overwrite existing uptime/*.json")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = config_loader.load_config()
    bucket = config.get("trades_bucket", "alpha-engine-research")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    written = backfill_range(start, end, bucket, force=args.force)
    logger.info("Backfill complete: wrote %d day(s) to s3://%s/uptime/", len(written), bucket)


if __name__ == "__main__":
    main()
