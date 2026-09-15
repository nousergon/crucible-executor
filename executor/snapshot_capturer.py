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


def _reconcile_fills(ib, db_path: str, run_date: str) -> None:
    """Write today's session executions back onto any still-open trade row.

    Loud on failure, but never fatal: the snapshot this stage exists to
    capture is the ONE non-re-runnable artifact of the day (see ``run``),
    and a reconciliation error must not cost it. The EOD run re-checks
    ``unresolved_trades`` and names anything still open in
    ``data_warnings``, so a failure here is visible downstream.
    """
    from executor.fill_reconciliation import (
        LOCAL_DAEMON_LOG_PATH,
        reconcile_from_daemon_log_file,
        reconcile_from_ib,
    )
    from executor.trade_logger import init_db

    try:
        conn = init_db(db_path)
        try:
            result = reconcile_from_ib(conn, run_date, ib)
            # This session is on a different clientId than the daemon, so it
            # usually holds none of the daemon's executions (measured on the
            # 2026-09-14 replay). The daemon log on this box holds all of them.
            patched = list(result["patched"])
            if result["unresolved"]:
                try:
                    result = reconcile_from_daemon_log_file(conn, run_date, LOCAL_DAEMON_LOG_PATH)
                    patched += result["patched"]
                except FileNotFoundError:
                    logger.error(
                        "fill reconciliation backstop: no daemon log at %s — %d row(s) left "
                        "for the EOD data_warnings", LOCAL_DAEMON_LOG_PATH, len(result["unresolved"]),
                    )
        finally:
            conn.close()
    except Exception as err:  # noqa: BLE001 — (a) reconciliation failed; (c) ERROR log here + EOD data_warnings from unresolved_trades
        logger.error("fill reconciliation backstop failed for %s: %s", run_date, err)
        return
    logger.info(
        "fill reconciliation backstop | run_date=%s patched=%d unresolved=%d",
        run_date, len(patched), len(result["unresolved"]),
    )
    for row in result["unresolved"]:
        logger.warning(
            "fill reconciliation backstop: %s %s %s (order %s) still %s — no execution "
            "on the session or in the daemon log",
            row["action"], row["shares"], row["ticker"], row.get("ib_order_id"), row.get("status"),
        )


def _sweep_prior_sessions(config: dict, run_date: str) -> None:
    """Repair earlier sessions' trade rows that were never reconciled.

    The live reconciliation passes only ever see the session they run in, so
    rows written before that machinery existed are unreachable by them. This
    drains that back-catalogue from the archived daemon logs in S3, a bounded
    number of dates per run, each date swept at most once (a negative-cache
    marker records the outcome, so an order that genuinely never filled does
    not re-download a ~100 MB log every night).

    It deliberately does NOT re-derive those sessions' ``eod_pnl`` rows.
    ``eod_reconcile.run(..., run_audit=False)`` re-runs TODAY's whole
    derivation over an old session, restating observed columns the fill never
    touched. Measured on the 2026-09-14 replay: a $1.75 BRO fill correction on
    2026-05-21 re-derived that row through the NAV mark correction added in
    August (-$3,037 of NAV), the TWR self-heal then rewrote 2026-05-22's return
    but not its nav_change_usd, and TWR closure went from 0.3bp to 29.1bp —
    failing the run. The fill-dependent attribution of a corrected session is
    left stale and logged here until an attribution-only restatement exists
    (alpha-engine-config-I10824).
    """
    from executor.fill_reconciliation import sweep_unresolved_sessions
    from executor.trade_logger import init_db

    bucket = config.get("trades_bucket")
    if not bucket:
        logger.warning("fill reconciliation sweep skipped: no trades_bucket configured")
        return
    try:
        conn = init_db(config["db_path"])
    except Exception as err:  # noqa: BLE001 — (a) the sweep could not open the ledger; (b) today's snapshot and today's own reconciliation are unaffected; (c) ERROR log here, and the next run retries
        logger.error("fill reconciliation sweep could not open trades.db: %s", err)
        return
    try:
        result = sweep_unresolved_sessions(
            conn, bucket=bucket, before=run_date,
            region=config.get("aws_region", "us-east-1"),
        )
    except Exception as err:  # noqa: BLE001 — (a) the sweep failed; (b) today's snapshot is already captured and unaffected; (c) ERROR log here, and any row it would have fixed stays non-terminal, which the EOD run reports in data_warnings
        logger.error("fill reconciliation sweep failed: %s", err)
        conn.close()
        return
    finally:
        conn.close()

    for corrected_date in result["changed"]:
        logger.warning(
            "fill reconciliation sweep: %s trade rows corrected; its eod_pnl attribution is "
            "intentionally NOT re-derived and still reflects the old fills — a whole-row "
            "re-derive restates observed NAV (alpha-engine-config-I10824)",
            corrected_date,
        )


def run(run_date: str | None = None) -> None:
    """Capture live IB state and write to S3 keyed by run_date.

    Default `run_date` resolves via `now_dual().trading_day` (NYSE-aware,
    Pacific-time "last completed trading day"). Explicit `run_date`
    arguments are accepted but are expected to match today — capture
    only makes sense for the current trading day since IB's account
    state is now-as-of, not historical.

    IRREVERSIBILITY (alpha-engine-config-I5569 / I6705): the `--date`
    live-capture-only constraint enforced below is not a convenience
    guard — it is the reason this is the EOD pipeline's ONE
    non-re-runnable stage. Every other EOD step reads from an artifact
    or a historical API and can be replayed for a past date; this step
    reads live IB account/position state, which only exists NOW. Miss
    the window (crash before `CaptureSnapshot` runs, or before NYSE-local
    midnight for the day) and that day's snapshot is gone permanently —
    there is no historical source to backfill it from. The cost of a
    missed day was measured in alpha-engine-config-I5325. Mitigations
    in place: same-day bounded retry + irreversible-deadline paging
    inside the EOD SF's `CaptureSnapshot` state (nousergon-data-PR1260),
    and an independent pre-midnight positive existence check —
    `alpha-engine-eod-snapshot-existence-check`, scheduled separately so
    it still fires even if the EOD SF never reaches this step at all
    (nousergon-data-PR1265).
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
        # EOD backstop for fill reconciliation (alpha-engine-config-I10800):
        # ib_insync syncs the day's executions on connect, so this fresh
        # session holds every fill the daemon's own per-tick pass may have
        # missed (daemon died between the fill and its next tick). Runs
        # BEFORE EODReconcile reads trades.db, which is the whole point of
        # placing it here rather than in the reconcile itself — the
        # reconcile reads a snapshot and never opens an IB session.
        _reconcile_fills(ibkr.ib, config["db_path"], run_date)
        _sweep_prior_sessions(config, run_date)
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
