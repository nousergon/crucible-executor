"""auto_backfill_gap — verified-zero-fill carry-forward + reprice auto-backfill
for a skipped-session eod_pnl gap (config#1454).

When a weekday session is skipped (the daemon never ran → no
`CaptureSnapshot` → no `eod_pnl` row), ``reconcile_audit`` used to
unconditionally flag the gap for MANUAL backfill and refuse to
auto-synthesize a NAV — the ledger-replay tool (``backfill_eod_pnl``) can
drift from the broker's actual book and fabricate a wrong NAV (config#1276:
20 synthesized positions / +51% NAV vs the real 7).

This module implements a NARROWER, honest special case: a day on which
**zero fills executed** (confirmed against the trade ledger) is
deterministic to reconstruct — nothing traded, so the book is exactly the
prior day's book, re-marked at the gap date's authoritative closes. This was
manually verified for 2026-06-24 (config#1454): NAV $984,303.41 / daily
+0.28% / alpha +0.32%, gaps cleared.

The procedure, gated strictly on three conditions — ANY failure still flags
the gap for manual review; this auto-backfill NEVER runs for a day that
actually traded:

  (a) zero fills executed on the gap date (no trade ledger rows whose
      ACTUAL fill lands on the gap date — see ``_has_zero_fills``);
  (b) a prior trading day's snapshot/positions exist to carry forward;
  (c) authoritative ArcticDB closes exist for every ticker held in the
      prior snapshot, on the gap date.

Procedure once gated:
  1. Confirm zero fills (gate a).
  2. Carry the prior day's EOD snapshot positions + cash + accrued forward
     unchanged (nothing traded).
  3. Reprice the carried book at authoritative ArcticDB closes for the gap
     date (reusing ``executor.price_cache._open_universe_library`` /
     ``_open_macro_library`` — the same source ``eod_reconcile`` uses).
  4. Write a stand-in ``trades/snapshots/{date}.json`` with the computed
     net_liquidation + per-position market_value AND a ``_reconstructed``
     provenance block.
  5. Run ``eod_reconcile.run(date, send_email=False)`` → a schema-identical
     eod_pnl row; the next reconcile/audit then reports gaps=0 for this date.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class GapNotAutoBackfillable(Exception):
    """Raised when the strict 3-part gate is not satisfied. The caller must
    fall back to flagging the gap for MANUAL, position-verified backfill —
    never auto-synthesize a NAV outside this narrow, verified-zero-fill case."""


def _has_zero_fills(conn, gap_date: str) -> tuple[bool, list[dict]]:
    """Confirm no trade ledger rows actually FILLED on ``gap_date``.

    Checks the trade's real execution timestamp (``fill_time``'s calendar
    date) rather than the ledger's ``date``/``trading_day`` tag — a trade
    tagged trading_day=D-1 or D+1 but whose ``fill_time`` calendar date is
    actually ``gap_date`` must still count as a fill ON the gap date (see
    config#1454's second axis: trading_day-vs-calendar mismatches). Falls
    back to the ``date`` column for legacy rows with a NULL ``fill_time``.

    A row only counts as a "fill" when it wasn't a non-fill status
    (Rejected/error/cancelled/etc.) and it moved actual shares — mirrors
    ``executor.reconciliation_audit._shares_contributed``'s status gate.

    Returns ``(zero_fills, offending_rows)`` — offending_rows is empty when
    zero_fills is True, else the rows that disqualify the gap from
    auto-backfill (surfaced in the manual-review flag for the operator).
    """
    from executor.reconciliation_audit import _NON_FILL_STATUSES, _effective_trade_date

    rows = conn.execute(
        "SELECT trade_id, ticker, action, shares, filled_shares, status, "
        "date, fill_time FROM trades"
    ).fetchall()
    offending = []
    for trade_id, ticker, action, shares, filled_shares, status, date_, fill_time in rows:
        if status in _NON_FILL_STATUSES:
            continue
        qty = filled_shares if filled_shares is not None else shares
        if not qty:
            continue
        # Same effective-date resolution used by reconciliation_audit's
        # ledger-vs-IB comparison (config#1454): prefer the actual fill
        # timestamp's calendar date over the ledger's date/trading_day tag.
        fill_date = _effective_trade_date({"date": date_, "fill_time": fill_time})
        if fill_date == gap_date:
            offending.append({
                "trade_id": trade_id, "ticker": ticker, "action": action,
                "shares": qty, "status": status, "date": date_,
                "fill_time": fill_time,
            })
    return (len(offending) == 0), offending


def _prior_trading_day_snapshot(conn, gap_date: str) -> dict | None:
    """The most recent RECONCILED eod_pnl row's positions_snapshot + account
    state strictly before ``gap_date`` — the carry-forward anchor (gate b).

    Returns None when no prior row exists (cold start — nothing to carry
    forward, so this auto-backfill cannot apply)."""
    row = conn.execute(
        "SELECT date, positions_snapshot, total_cash, accrued_interest "
        "FROM eod_pnl WHERE positions_snapshot IS NOT NULL AND date < ? "
        "ORDER BY date DESC LIMIT 1",
        (gap_date,),
    ).fetchone()
    if not row or not row[1]:
        return None
    try:
        positions = json.loads(row[1])
    except (json.JSONDecodeError, TypeError):
        return None
    return {
        "date": row[0],
        "positions": positions,
        "total_cash": row[2],
        "accrued_interest": row[3],
    }


def _authoritative_closes(trades_bucket: str, tickers: list[str], gap_date: str) -> dict[str, float]:
    """Authoritative ArcticDB close per ticker on ``gap_date`` (gate c).

    Reuses the same universe/macro-library dispatch as ``eod_reconcile`` and
    ``backfill_eod_pnl`` (``_MACRO_SYMBOLS`` routes VIX/TNX/sector-ETFs/etc.
    to the macro library; everything else — including SPY — to universe).
    Returns only the tickers that resolved; the caller checks completeness
    against the full held-ticker set to enforce gate (c) strictly."""
    from executor.price_cache import (
        _MACRO_SYMBOLS,
        _open_macro_library,
        _open_universe_library,
    )

    universe_lib = _open_universe_library(trades_bucket)
    macro_lib = None
    target_ts = pd.Timestamp(gap_date).normalize()
    closes: dict[str, float] = {}
    for ticker in tickers:
        if ticker in _MACRO_SYMBOLS:
            if macro_lib is None:
                macro_lib = _open_macro_library(trades_bucket)
            lib = macro_lib
        else:
            lib = universe_lib
        try:
            df = lib.read(ticker).data
        except Exception as e:
            # Logged (not swallowed silently) so an ArcticDB-wide outage is
            # distinguishable in the logs from a genuine single-ticker gap —
            # gate (c)'s completeness check in check_gate() still refuses to
            # proceed either way (missing == missing), but the operator
            # investigating a MANUAL-flag needs to tell the two apart.
            logger.warning(
                "[auto_backfill_gap] ArcticDB read failed for %s on %s: %s",
                ticker, gap_date, e,
            )
            continue
        if df.empty or "Close" not in df.columns:
            logger.warning(
                "[auto_backfill_gap] %s frame empty or missing Close column "
                "(gap_date=%s)", ticker, gap_date,
            )
            continue
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        match = df[idx == target_ts]
        if match.empty:
            continue
        closes[ticker] = float(match["Close"].iloc[-1])
    return closes


def check_gate(conn, trades_bucket: str, gap_date: str) -> dict:
    """Evaluate the strict 3-part gate for ``gap_date``. Never raises.

    Returns a dict:
      {"eligible": bool, "reason": str | None,
       "prior_date": str | None, "prior_snapshot": dict | None,
       "closes": dict[str, float], "offending_fills": list[dict]}

    ``eligible`` is True only when ALL THREE conditions hold:
      (a) zero fills executed on gap_date,
      (b) a prior trading day's positions_snapshot exists,
      (c) authoritative ArcticDB closes exist for EVERY ticker held in that
          prior snapshot, on gap_date.
    Any single failure means the gap must still be flagged for manual
    review — this function never auto-synthesizes anything itself.
    """
    zero_fills, offending = _has_zero_fills(conn, gap_date)
    if not zero_fills:
        return {
            "eligible": False,
            "reason": (
                f"{len(offending)} trade(s) actually filled on {gap_date} — "
                "not a zero-fill day. Refusing to auto-backfill a day that "
                "traded; flag for manual, position-verified backfill."
            ),
            "prior_date": None, "prior_snapshot": None,
            "closes": {}, "offending_fills": offending,
        }

    prior = _prior_trading_day_snapshot(conn, gap_date)
    if prior is None:
        return {
            "eligible": False,
            "reason": (
                f"No prior reconciled eod_pnl positions_snapshot before "
                f"{gap_date} to carry forward — cold start. Flag for manual "
                "review."
            ),
            "prior_date": None, "prior_snapshot": None,
            "closes": {}, "offending_fills": [],
        }

    held_tickers = sorted(prior["positions"].keys())
    closes = _authoritative_closes(trades_bucket, held_tickers, gap_date) if held_tickers else {}
    missing = [t for t in held_tickers if t not in closes]
    if missing:
        return {
            "eligible": False,
            "reason": (
                f"Missing authoritative ArcticDB close(s) for {len(missing)} "
                f"held ticker(s) on {gap_date}: {missing}. Flag for manual "
                "review — cannot reprice the carried book without a "
                "complete, authoritative close set."
            ),
            "prior_date": prior["date"], "prior_snapshot": prior,
            "closes": closes, "offending_fills": [],
        }

    return {
        "eligible": True, "reason": None,
        "prior_date": prior["date"], "prior_snapshot": prior,
        "closes": closes, "offending_fills": [],
    }


def _snapshot_key(run_date: str) -> str:
    return f"trades/snapshots/{run_date}.json"


def build_reconstructed_snapshot(
    gap_date: str,
    prior_snapshot: dict,
    closes: dict[str, float],
    schema_version: int | str = 1,
) -> dict:
    """Build the stand-in snapshot for ``gap_date``: prior day's book
    (positions/cash/accrued), UNCHANGED (nothing traded), repriced at
    ``closes``. Carries the ``_reconstructed`` provenance block required by
    config#1454 — distinct from ``backfill_eod_pnl``'s ``synthesized: true``
    marker, since this is a narrower, gate-verified carry-forward, not a
    ledger replay."""
    prior_positions = prior_snapshot["positions"]
    positions: dict[str, dict] = {}
    positions_mv = 0.0
    for ticker, prior_pos in prior_positions.items():
        shares = prior_pos.get("shares") or 0
        close = closes[ticker]
        mv = shares * close
        positions_mv += mv
        # Rebuild explicitly from a carried IDENTITY subset (shares/avg_cost/
        # sector) rather than spreading the whole prior_pos dict — the prior
        # day's snapshot also carries DERIVED, day-specific fields
        # (market_value, unrealized_pnl, daily_return_pct/usd,
        # alpha_contribution_*, closing_price) computed against THAT day's
        # close; blindly carrying them forward would leave stale values in
        # the gap day's snapshot instead of the freshly recomputed ones set
        # below (eod_reconcile overwrites market_value/closing_price itself,
        # but the OTHER derived fields have no such downstream overwrite).
        positions[ticker] = {
            "shares": shares,
            "avg_cost": prior_pos.get("avg_cost") if prior_pos.get("avg_cost") is not None else close,
            "sector": prior_pos.get("sector", "Unknown"),
            "market_value": mv,
            "closing_price": close,
        }
    cash = float(prior_snapshot.get("total_cash") or 0.0)
    accrued = prior_snapshot.get("accrued_interest")
    nav = cash + positions_mv

    return {
        "run_date": gap_date,
        "captured_at": datetime.now(UTC).isoformat(),
        "schema_version": schema_version,
        "_reconstructed": {
            "method": "verified_zero_fill_carry_forward_reprice",
            "config_ref": "config#1454",
            "prior_trading_day": prior_snapshot["date"],
            "gate": {
                "zero_fills_confirmed": True,
                "prior_snapshot_exists": True,
                "authoritative_closes_complete": True,
            },
            "reconstructed_at": datetime.now(UTC).isoformat(),
        },
        "account": {
            "net_liquidation": nav,
            "total_cash": cash,
            "accrued_interest": accrued,
        },
        "positions": positions,
        "accrued_dividends": {},
    }


def attempt_auto_backfill(
    conn,
    *,
    gap_date: str,
    trades_bucket: str,
    region: str = "us-east-1",
    s3_client=None,
) -> dict:
    """Attempt the verified-zero-fill auto-backfill for ``gap_date``.

    Returns a result dict: ``{"backfilled": bool, "reason": str | None,
    "gate": dict}``. When ``backfilled`` is False the caller MUST fall back
    to flagging the gap for manual review — this function never partially
    applies a backfill.

    On success: writes the ``_reconstructed`` snapshot to
    ``trades/snapshots/{gap_date}.json`` and runs the canonical
    ``eod_reconcile.run(gap_date, send_email=False, run_audit=False)`` so the
    resulting eod_pnl row is schema-identical to a normal EOD run. Any
    reconcile failure propagates (the gate having passed does not guarantee
    the downstream reconcile succeeds — e.g. a transient S3/preflight
    failure) and the caller should treat it the same as a failed gate: flag
    for manual review rather than leaving a half-written state.
    """
    gate = check_gate(conn, trades_bucket, gap_date)
    if not gate["eligible"]:
        logger.warning(
            "[auto_backfill_gap] %s NOT eligible for auto-backfill: %s",
            gap_date, gate["reason"],
        )
        return {"backfilled": False, "reason": gate["reason"], "gate": gate}

    s3 = s3_client if s3_client is not None else boto3.client("s3", region_name=region)

    # Guard against clobbering a snapshot that already exists for gap_date —
    # a REAL (non-reconstructed) snapshot with no eod_pnl row would mean
    # eod_reconcile itself failed after CaptureSnapshot; that is a different
    # failure mode than a skipped session and must not be silently
    # overwritten with a carry-forward guess. A PRIOR reconstructed
    # snapshot (e.g. an earlier auto-backfill attempt that then failed
    # downstream in eod_run) is safe to recompute and overwrite. Reads via
    # the SAME s3 client used for the write below (not snapshot_capturer.
    # load_snapshot's own boto3.client) so callers/tests can inject one stub.
    try:
        obj = s3.get_object(Bucket=trades_bucket, Key=_snapshot_key(gap_date))
        existing_snap = json.loads(obj["Body"].read())
    except Exception as e:
        # NoSuchKey (or its string-form equivalent from a raw HTTPClientError)
        # means "no snapshot yet" — the expected, common case. Anything else
        # (permissions, transient S3 errors) must NOT be silently treated as
        # "safe to backfill"; surface it so the caller falls back to the
        # manual-review flag instead of masking an infra problem.
        if "NoSuchKey" in str(e) or "404" in str(e):
            existing_snap = None
        else:
            raise
    if existing_snap is not None and "_reconstructed" not in existing_snap:
        reason = (
            f"A real (non-reconstructed) snapshot already exists for {gap_date} "
            "with no corresponding eod_pnl row — this is a different failure "
            "mode than a skipped session (eod_reconcile likely failed after "
            "CaptureSnapshot). Refusing to overwrite; flag for manual review."
        )
        logger.warning("[auto_backfill_gap] %s", reason)
        return {"backfilled": False, "reason": reason, "gate": gate}

    from executor.snapshot_capturer import SCHEMA_VERSION
    snapshot = build_reconstructed_snapshot(
        gap_date, gate["prior_snapshot"], gate["closes"],
        schema_version=SCHEMA_VERSION,
    )

    s3.put_object(
        Bucket=trades_bucket,
        Key=_snapshot_key(gap_date),
        Body=json.dumps(snapshot, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(
        "[auto_backfill_gap] wrote _reconstructed snapshot s3://%s/%s "
        "(prior_trading_day=%s, nav=$%.2f)",
        trades_bucket, _snapshot_key(gap_date), gate["prior_date"],
        snapshot["account"]["net_liquidation"],
    )

    from executor.eod_reconcile import run as eod_run
    # A backfill is a post-hoc recovery of a PAST day: never resend that
    # day's EOD email, and never recurse into the reconcile_audit self-heal
    # that likely called us.
    eod_run(gap_date, send_email=False, run_audit=False)

    logger.info(
        "[auto_backfill_gap] %s auto-backfilled via verified-zero-fill "
        "carry-forward + reprice — eod_pnl row written.", gap_date,
    )
    return {"backfilled": True, "reason": None, "gate": gate}
