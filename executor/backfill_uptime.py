"""
Backfill uptime/{date}.json for historical trading days.

DAEMON_TICK log lines are new as of today, so historical days have no
minute-level ib_connected signal. This script uses systemd journal entries
for alpha-engine-daemon.service as a proxy — any journal line during market
hours counts the minute as active. connected_minutes is set equal to
active_minutes (best-effort; cannot distinguish broker disconnects
retroactively).

Runs on ae-trading (requires journalctl access).

Usage:
  python executor/backfill_uptime.py --start 2026-03-15 --end 2026-04-12
  python executor/backfill_uptime.py --start 2026-03-15 --end 2026-04-12 --force
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import date, datetime, time, timedelta

import pytz

from executor import config_loader
from executor.market_hours import is_trading_day
from executor.uptime_tracker import _MARKET_MINUTES, write_to_s3

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

_JOURNAL_UNIT = "alpha-engine-daemon.service"


def journal_timestamps(day: date) -> list[datetime]:
    """Return UTC timestamps of all journal entries for the daemon service on `day` (ET)."""
    start_et = _ET.localize(datetime.combine(day, time(0, 0)))
    end_et = start_et + timedelta(days=1)
    since_utc = start_et.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    until_utc = end_et.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")

    cmd = [
        "journalctl", "-u", _JOURNAL_UNIT,
        "--since", since_utc, "--until", until_utc,
        "--output", "short-iso-precise", "--no-pager", "--utc",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except FileNotFoundError:
        logger.error("journalctl not on PATH — backfill only works on ae-trading")
        return []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("journalctl failed for %s: %s", day, e)
        return []

    timestamps: list[datetime] = []
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if not parts:
            continue
        token = parts[0]
        try:
            ts = datetime.fromisoformat(token.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        timestamps.append(ts)
    return timestamps


def compute_backfill_metrics(timestamps: list[datetime], day: date) -> dict:
    """Approximate minute-level uptime from journal timestamps. No ib_connected signal."""
    market_open_et = _ET.localize(datetime.combine(day, _MARKET_OPEN))
    market_close_et = _ET.localize(datetime.combine(day, _MARKET_CLOSE))

    minutes_active: set[int] = set()
    for ts in timestamps:
        ts_et = ts.astimezone(_ET)
        if not (market_open_et <= ts_et < market_close_et):
            continue
        minutes_active.add(int((ts_et - market_open_et).total_seconds() // 60))

    active = len(minutes_active)
    return {
        "date": day.isoformat(),
        "active_minutes": active,
        "connected_minutes": active,
        "market_minutes": _MARKET_MINUTES,
        "uptime_pct": round(active / _MARKET_MINUTES, 4) if _MARKET_MINUTES else 0.0,
        "crashes": 0,
        "tick_cadence_sec": 0,
        "source": "journalctl_backfill",
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

        timestamps = journal_timestamps(cur)
        metrics = compute_backfill_metrics(timestamps, cur)
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
