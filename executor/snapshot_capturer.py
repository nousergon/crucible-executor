"""
EOD snapshot capturer — reads live IB state once at end-of-day and
persists an immutable snapshot to S3 keyed by run_date.

This is Phase 2 of the EOD-SF cutover. Decouples capture from
reconciliation so `eod_reconcile.py` can read date-locked state from
S3 instead of reading current live IB state at write-time. The
architectural invariant: a row keyed by `run_date=X` must source its
inputs from observations made at time X. Live IB at write-time only
satisfied this by accident (because the timer happened to fire once
a day right after close); a snapshot makes it explicit.

Idempotent. Re-running on the same `run_date` overwrites the existing
snapshot. Hard-fails on IB connection failure or S3 write failure —
no silent fallback (the reconcile path depends on this snapshot).

SF orchestration: this script runs as the `CaptureSnapshot` step in
`ne-postclose-trading-pipeline`, between `PostMarketData` and
`EODReconcile`. Both depend on IB Gateway being up; the SF's
`StopTradingInstance` step (which kills IB) only fires after
EODReconcile completes.

S3 path: s3://alpha-engine-research/trades/snapshots/{run_date}.json

Schema (additive-only per CLAUDE.md S3 contract):
    {
      "run_date": "YYYY-MM-DD",
      "captured_at": ISO8601,
      "schema_version": 1,
      "account": {net_liquidation, total_cash, settled_cash,
                  accrued_interest, gross_position_value,
                  buying_power, unrealized_pnl, realized_pnl},
      "positions": {ticker: {shares, market_value, avg_cost,
                             unrealized_pnl, sector}},
      "accrued_dividends": {ticker: float},
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.dates import now_dual
from nousergon_lib.logging import setup_logging

from executor.config_loader import load_config
from executor.ibkr import IBKRClient

_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)

_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging(
    "snapshot",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _snapshot_key(run_date: str) -> str:
    return f"trades/snapshots/{run_date}.json"


def run(run_date: str | None = None) -> None:
    """Capture live IB state and write to S3 keyed by run_date.

    Default `run_date` resolves via `now_dual().trading_day` (NYSE-aware,
    Pacific-time "last completed trading day"). Explicit `run_date`
    arguments are accepted but are expected to match today — capture
    only makes sense for the current trading day since IB's account
    state is now-as-of, not historical.
    """
    today_trading_day = now_dual().trading_day
    if run_date is None:
        run_date = today_trading_day
        logger.info(
            "Snapshot capture | run_date=%s (resolved from now_dual().trading_day)",
            run_date,
        )
    else:
        if run_date != today_trading_day:
            raise RuntimeError(
                f"Snapshot capturer refusing run_date={run_date!r} "
                f"!= today's trading_day {today_trading_day!r}. "
                f"Snapshots can only be captured live (`get_account_snapshot()` "
                f"returns now-as-of state); a historical run_date would "
                f"persist today's state under yesterday's key."
            )
        logger.info(
            "Snapshot capture | run_date=%s (explicit; matches today's trading_day)",
            run_date,
        )

    config = load_config()
    bucket = config["trades_bucket"]

    # ── Connect to IB Gateway ─────────────────────────────────────────────
    ibkr = IBKRClient(
        host=config["ibkr_host"],
        port=config["ibkr_port"],
        client_id=config["ibkr_client_id"],
    )

    try:
        account = ibkr.get_account_snapshot()
        positions = ibkr.get_positions()
        accrued_dividends = ibkr.get_accrued_dividends_by_symbol()
    finally:
        ibkr.disconnect()

    payload = {
        "run_date": run_date,
        "captured_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "account": account,
        "positions": positions,
        "accrued_dividends": accrued_dividends,
    }

    # ── Write to S3 ─────────────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=config.get("aws_region", "us-east-1"))
    key = _snapshot_key(run_date)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(
        "Snapshot written | s3://%s/%s NAV=%s positions=%d dividends=%d",
        bucket,
        key,
        account.get("net_liquidation"),
        len(positions),
        len(accrued_dividends),
    )


def load_snapshot(bucket: str, run_date: str, region: str = "us-east-1") -> dict | None:
    """Load the snapshot for `run_date`. Returns None if not found.

    Used by `eod_reconcile.py` to substitute for the three live IB calls
    (`get_account_snapshot`, `get_positions`, `get_accrued_dividends_by_symbol`).
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        obj = s3.get_object(Bucket=bucket, Key=_snapshot_key(run_date))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        # 404 from raw HTTPClientError can also mean "not found" depending
        # on bucket config — try parsing first, surface anything else loud.
        if "NoSuchKey" in str(exc) or "404" in str(exc):
            return None
        raise
    return json.loads(obj["Body"].read())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Capture live IB state to S3 keyed by run_date. Defaults to "
            "today's trading_day via now_dual; --date must equal today "
            "(snapshots can only be captured live)."
        )
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD; must equal today's trading_day or the run aborts.",
    )
    args = parser.parse_args()
    run(run_date=args.date)
