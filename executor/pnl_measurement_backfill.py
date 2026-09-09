"""Backfill the performance-measurement columns over the HISTORICAL eod_pnl series.

alpha-engine-config-I8188, second pass. The first pass (PR490/PR491/PR509)
closed all four defects on the FORWARD path and left the history untouched.
Measured against ``s3://alpha-engine-research/trades/eod_pnl.csv`` on
2026-08-31, 120 sessions, 2026-03-09 → 2026-08-28:

    commission_usd              populated on   6 / 120 sessions
    slippage_usd                populated on   6 / 120
    traded_notional_usd         populated on   6 / 120
    daily_return_gross_pct      populated on   6 / 120
    dividend_usd                populated on  79 / 120, and 0.00 on all 79
    spy_dividend_per_share      populated on   6 / 120

Six sessions is every session since the 2026-08-21 deploy. So the cost line,
the dividend line and the total-return benchmark exist prospectively and are
absent from the series any threshold would actually be set against — including
`alpha-engine-config-I9005`, the Crucible viability predicate this issue blocks.
A viability bar set on a 120-session record whose cost line covers 5% of it is
set against a number that is optimistic by an unmeasured amount, and BOTH
omissions run the same direction:

* no cost line  → implementation cost reads as $0 on 114 sessions;
* price-return SPY → the benchmark is understated by its distribution yield.

WHAT IS AND IS NOT RECONSTRUCTIBLE — measured against the live ledger
(``trades_latest.db``, 513 rows / 504 filled, 2026-03-13 → 2026-08-31), not
assumed:

    price_at_order  present on 504 / 504 filled rows  → slippage IS backfillable
    commission_usd  present on  35 / 504 filled rows, ALL of them 2026-08

So implementation shortfall is recoverable over the whole history and
commission is not. The commission leg is therefore written **NULL with
``commission_available = 0``**, never 0.0, and ``daily_return_gross_pct`` is
left NULL on those sessions rather than published with a $0.00 commission leg —
a gross return computed that way is a net return wearing a gross label. Whether
IB can be made to reissue historical commissionReports is a separate question;
until it is answered the honest state of that column is ABSENT.

RUN LOCATION. This is a manual one-off write to a production data repo. It runs
IN-REGION on EC2 (the trading box off-market-hours, or a data-spot instance),
never from a laptop — a full-history pass is dominated by S3 round-trip latency
and, for the dividend leg, by Polygon's 5-calls-per-minute limiter.

    python -m executor.pnl_measurement_backfill --dry-run      # default
    python -m executor.pnl_measurement_backfill --apply --costs --benchmark
    python -m executor.pnl_measurement_backfill --apply --dividends   # Polygon
    python -m executor.pnl_measurement_backfill --restate-marks       # plan only
    python -m executor.pnl_measurement_backfill --restate-marks --apply
    # IN-REGION only — the range discriminator reads ArcticDB:
    python -m executor.pnl_measurement_backfill --restate-marks \
        --range-source arcticdb

THE MARK-RESTATEMENT LEG (``--restate-marks``, alpha-engine-config-I9629) is
the one leg that rewrites a published NAV. It moves ``portfolio_nav`` off a
broker mark the reconstruction proves wrong, preserves the original in
``nav_ib_raw_usd``, and recomputes ``daily_return_pct``/``daily_alpha_pct``
across BOTH sessions each corrected NAV sits between. With ``--range-source
arcticdb`` its instrument IS the live gate's — the day's traded ``[Low, High]``,
read through the same loader ``eod_reconcile`` uses. Without it the instrument
is an upper bound and every row it writes says so — see section 5.

THE COST LEG IS NOW WIRED INTO THE EOD PATH (alpha-engine-config-I9614).
``heal_cost_columns`` runs on every reconciliation beside
``pnl_backfill.backfill_residual_sleeves``, so a session whose cost columns are
NULL heals with no operator step (principles §2.3). It was held back until the
2026-09-05 end of the clause-1 change-quiet window (`alpha-engine-config-I9041`),
and the evidence for wiring it rather than scheduling another manual run is the
measurement itself: the cost columns covered 6 of 120 sessions on 2026-08-31 and
12 of 126 on 2026-09-08 — the CLI legs were never run, and only the forward path
moved.

``fetch_live_anchor_window`` is the same move for the benchmark leg's vendor
anchor: two Polygon calls over ``run_date`` and its prior session rather than the
whole history, so the one genuinely third-party side of the reconciliation runs
DAILY instead of only when a human types the CLI.

THE DIVIDEND AND MARK-RESTATEMENT LEGS REMAIN MANUAL, and deliberately.
``--dividends`` issues one Polygon query per held ticker over the full history
under a 5-calls-per-minute limiter, and ``--restate-marks`` rewrites a published
NAV — a restatement, which is reserved (see `alpha-engine-config-I9613` for the
benchmark-restatement ruling of the same shape). Wiring either into a postclose
would put a rate-limited multi-minute fetch, or an unreviewed rewrite of the
record, on the critical path of the trading day's close.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from collections.abc import Mapping
from typing import Any

from executor.dividends import SPY_TICKER, accrue_position_dividends
from executor.pnl_integrity import (
    check_benchmark_vendor_anchor,
    check_custodian_marks,
    check_residual_bounds,
    check_session_axis_coverage,
    gross_net_returns,
    mark_materiality_usd,
    plan_nav_mark_correction,
    session_costs,
    verify_benchmark_chain_closes,
    verify_twr_closes,
)

logger = logging.getLogger(__name__)

# The whole persisted history. Unlike the residual-sleeve backfill — which only
# has to keep a 63-session gate window full — the audience for these columns is
# the full track record a viability threshold is set against, so a lookback
# shorter than the history would reintroduce the gap it exists to close.
FULL_HISTORY = 10_000


class NothingToBackfill(Exception):
    """The requested leg has no reconstructible input on this row."""


def _rows(conn: sqlite3.Connection, *, limit: int = FULL_HISTORY) -> list[dict[str, Any]]:
    """The eod_pnl series as dicts, OLDEST-first."""
    conn.row_factory = sqlite3.Row
    try:
        out = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM eod_pnl ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
    finally:
        conn.row_factory = None
    return out[::-1]


def _positions(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("positions_snapshot")
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _f(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cost lines
# ─────────────────────────────────────────────────────────────────────────────

def plan_cost_backfill(
    rows: list[dict[str, Any]],
    trades_by_date: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Plan the ``slippage_usd`` / ``traded_notional_usd`` / commission writes. PURE.

    One entry per session that currently carries no ``slippage_usd``. Each
    reuses :func:`executor.pnl_integrity.session_costs` — the SAME function the
    live path calls — so a backfilled cost line is computed by the code that
    computes a live one. Any divergence between the two would be a defect here,
    not a licence to approximate.

    ``commission_usd`` is carried through exactly as ``session_costs`` reports
    it, which is ``None`` when fills executed and none carried a commission.
    ``daily_return_gross_pct`` follows :func:`gross_net_returns` and is
    therefore None on the same rows.
    """
    plans: list[dict[str, Any]] = []
    prior_nav: float | None = None
    for row in rows:
        nav = _f(row.get("portfolio_nav"))
        date = str(row.get("date"))
        if row.get("slippage_usd") is None:
            costs = session_costs(trades_by_date.get(date) or [])
            split = gross_net_returns(
                nav_change_usd=_f(row.get("nav_change_usd")),
                prior_nav=prior_nav,
                commission_usd=costs["commission_usd"],
                slippage_usd=costs["slippage_usd"],
            )
            plans.append({
                "date": date,
                "commission_usd": costs["commission_usd"],
                "commission_available": 1 if costs["commission_available"] else 0,
                "slippage_usd": costs["slippage_usd"],
                "traded_notional_usd": costs["traded_notional_usd"],
                "daily_return_net_pct": split["daily_return_net_pct"],
                "daily_return_gross_pct": split["daily_return_gross_pct"],
                "n_fills": costs["n_fills"],
                "slippage_bps": costs["slippage_bps"],
            })
        if nav is not None:
            prior_nav = nav
    return plans


def apply_cost_backfill(
    conn: sqlite3.Connection, plans: list[dict[str, Any]],
) -> int:
    """Write the planned cost lines. Idempotent — planning skips populated rows."""
    for p in plans:
        conn.execute(
            "UPDATE eod_pnl SET commission_usd=?, commission_available=?, "
            "slippage_usd=?, traded_notional_usd=?, daily_return_net_pct=?, "
            "daily_return_gross_pct=? WHERE date=?",
            (
                p["commission_usd"], p["commission_available"], p["slippage_usd"],
                p["traded_notional_usd"], p["daily_return_net_pct"],
                p["daily_return_gross_pct"], p["date"],
            ),
        )
    conn.commit()
    return len(plans)


def heal_cost_columns(conn: sqlite3.Connection) -> dict[str, Any]:
    """Fill the missing cost columns on historical ``eod_pnl`` rows, idempotently.

    alpha-engine-config-I9614, deliverable 1. :func:`plan_cost_backfill` and
    :func:`apply_cost_backfill` were reachable ONLY from this module's CLI, and
    the CLI was never run: measured against
    ``s3://alpha-engine-research/trades/eod_pnl.csv`` on 2026-09-08,
    ``commission_usd`` / ``slippage_usd`` / ``traded_notional_usd`` are
    populated on **12 of 126 sessions** — every session since the 2026-08-21
    forward-path deploy and not one before it. The gap did not shrink between
    2026-08-31 (6/120) and 2026-09-08; it only grew forward, which is the
    signature of a backfill that needs a human and therefore never happens.

    A backfill that needs an operator step is a page, not a fix (principles
    §2.3 — detect, diagnose, act, VERIFY, close). This runs on every EOD
    reconciliation beside :func:`executor.pnl_backfill.backfill_residual_sleeves`,
    which is the repo's own SOTA pattern for exactly this, in the same
    function.

    Purely local: the inputs are the persisted ``eod_pnl`` rows and the
    ``trades`` ledger in the same sqlite file. No vendor call, no rate limit,
    no network. Planning skips any row that already carries ``slippage_usd``,
    so a converged history costs one query.

    Returns ``{"filled": int, "commission_absent": int, "slippage_usd_total":
    float}``. ``commission_usd`` is written NULL with ``commission_available=0``
    where the ledger carries no commission — never 0.0, which is the defect
    this whole issue is about (an absent measurement wearing a measured zero's
    clothes).
    """
    from executor.trade_logger import get_todays_trades

    rows = _rows(conn)
    if not rows:
        return {"filled": 0, "commission_absent": 0, "slippage_usd_total": 0.0}
    pending = {str(r["date"]) for r in rows if r.get("slippage_usd") is None}
    if not pending:
        return {"filled": 0, "commission_absent": 0, "slippage_usd_total": 0.0}
    trades_by_date = {d: get_todays_trades(conn, d) for d in sorted(pending)}
    plans = plan_cost_backfill(rows, trades_by_date)
    written = apply_cost_backfill(conn, plans)
    return {
        "filled": written,
        "commission_absent": sum(1 for p in plans if p["commission_usd"] is None),
        "slippage_usd_total": sum(p["slippage_usd"] for p in plans),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Total-return benchmark
# ─────────────────────────────────────────────────────────────────────────────

def plan_benchmark_restatement(
    rows: list[dict[str, Any]],
    *,
    vendor_closes: dict[str, float],
    vendor_dividends: dict[str, float],
    prior_session_of: Any,
) -> dict[str, Any]:
    """Plan a VENDOR-ANCHORED restatement of ``spy_return_pct``/``daily_alpha_pct``.

    Neither persisted column can repair the other — ``spy_close`` changes basis
    at 2026-03-20 and ``spy_return_pct`` is what it is being checked against
    (see ``pnl_integrity`` §6). The only well-founded restatement is the
    vendor's own series:

        spy_return_pct[t] = (close[t] + div[t]) / close[prior_session(t)] − 1

    with ``prior_session`` from the TRADING CALENDAR, not from adjacency in the
    table. That is what makes the three coverage gaps (2026-03-12, 2026-07-27
    missing; 2026-04-03 a market holiday with a row) span correctly instead of
    silently comparing a two-session move against a one-session one.

    A row the vendor cannot cover — no close for the session, or none for the
    prior session — is REFUSED and named, never filled from the persisted
    column it is supposed to be verifying.

    THIS IS A RESTATEMENT OF A PUBLISHED TRACK RECORD, not a self-heal, and the
    caller gates it behind an explicit flag. Every ``daily_alpha_pct`` ever
    reported moves. That is a ruling for Brian, not a repair an agent takes.
    """
    corrections: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for row in rows:
        date = str(row.get("date"))
        stored = _f(row.get("spy_return_pct"))
        port = _f(row.get("daily_return_pct"))
        close = _f(vendor_closes.get(date))
        try:
            prior = str(prior_session_of(date))
        except Exception as exc:  # noqa: BLE001
            refused.append({"date": date, "reason": f"calendar lookup failed: {exc}"})
            continue
        prior_close = _f(vendor_closes.get(prior))
        if close is None or not prior_close:
            refused.append({
                "date": date,
                "reason": (
                    f"vendor has no close for {date if close is None else prior} "
                    "— the benchmark leg cannot be rebuilt for this session, and "
                    "keeping the persisted value is the honest state"
                ),
            })
            continue
        to_pct = ((close + float(vendor_dividends.get(date, 0.0) or 0.0))
                  / prior_close - 1.0) * 100.0
        if stored is not None and abs(stored - to_pct) * 100.0 <= 0.5:
            continue  # already correct to half a basis point
        if port is None:
            refused.append({
                "date": date,
                "from_pct": stored,
                "to_pct": to_pct,
                "reason": (
                    "row carries no daily_return_pct, so daily_alpha_pct cannot "
                    "be restated alongside spy_return_pct — correcting one leg "
                    "alone leaves the row internally inconsistent"
                ),
            })
            continue
        corrections.append({
            "date": date,
            "prior_session": prior,
            "from_pct": stored,
            "to_pct": to_pct,
            "delta_pct": (stored - to_pct) if stored is not None else None,
            "from_alpha_pct": _f(row.get("daily_alpha_pct")),
            "to_alpha_pct": port - to_pct,
            "spy_dividend_per_share": float(vendor_dividends.get(date, 0.0) or 0.0),
        })
    return {"corrections": corrections, "refused": refused}


def plan_dividend_per_share_writes(
    rows: list[dict[str, Any]], spy_ex_dividends: dict[str, float],
) -> list[dict[str, Any]]:
    """Plan the additive ``spy_dividend_per_share`` writes. PURE.

    Written on every row that lacks it, INCLUDING the zeros: "nothing went ex
    in this interval" is a real measurement, and it is what lets a later reader
    tell an unmeasured session from a measured one. This write restates
    nothing — it names a quantity the column never carried.
    """
    return [
        {"date": str(r.get("date")),
         "spy_dividend_per_share": float(spy_ex_dividends.get(str(r.get("date")), 0.0))}
        for r in rows
        if r.get("spy_dividend_per_share") is None
    ]


def apply_dividend_per_share_writes(
    conn: sqlite3.Connection, writes: list[dict[str, Any]],
) -> int:
    for w in writes:
        conn.execute(
            "UPDATE eod_pnl SET spy_dividend_per_share=? WHERE date=?",
            (w["spy_dividend_per_share"], w["date"]),
        )
    conn.commit()
    return len(writes)


def apply_benchmark_restatement(
    conn: sqlite3.Connection, plan: dict[str, Any],
) -> int:
    """Write the vendor-anchored benchmark restatement. GATED — see the planner."""
    for c in plan.get("corrections", []):
        conn.execute(
            "UPDATE eod_pnl SET spy_return_pct=?, daily_alpha_pct=? WHERE date=?",
            (c["to_pct"], c["to_alpha_pct"], c["date"]),
        )
    conn.commit()
    return len(plan.get("corrections", []))


def map_ex_dividends_to_sessions(
    rows: list[dict[str, Any]], events: list[dict[str, Any]],
) -> dict[str, float]:
    """Map raw Polygon dividend events onto the persisted session grid. PURE.

    ``events`` is what ``PolygonClient.get_dividends`` returns. Each event is
    credited to the FIRST persisted session on or after its ex-date, so the
    distribution lands in the interval that spans it. An ex-date after the last
    persisted session is dropped and counted by the caller — crediting it to
    the last row would put a future distribution inside a closed interval.

    Mirrors ``executor.dividends._in_interval``'s half-open ``(prior, through]``
    convention: the return for session *t* spans ``(t-1, t]``, so a dividend
    going ex ON session *t* belongs to session *t*.
    """
    dates = sorted(str(r.get("date")) for r in rows if r.get("date"))
    out: dict[str, float] = {}
    for ev in events or []:
        ex = ev.get("ex_dividend_date")
        amount = ev.get("cash_amount")
        if not ex or amount in (None, ""):
            continue
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        target = next((d for d in dates if d >= str(ex)), None)
        if target is None:
            continue
        out[target] = out.get(target, 0.0) + amount
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Position dividends
# ─────────────────────────────────────────────────────────────────────────────

def plan_dividend_backfill(
    rows: list[dict[str, Any]],
    ex_dividends_by_session: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Plan ``dividend_usd`` per session from the persisted position snapshots. PURE.

    ``ex_dividends_by_session[date][ticker]`` is the per-share cash going ex in
    the interval ending on ``date``. Entitlement is taken from the PRIOR
    session's share count, which is why this runs on consecutive persisted rows
    and reuses :func:`executor.dividends.accrue_position_dividends` rather than
    reimplementing the settle-before-ex rule.

    A session whose snapshot is absent or unparseable is SKIPPED with a reason
    and left NULL — it is not written as 0.00. ``dividend_usd`` reading 0.00 on
    all 79 populated historical sessions is the exact defect being repaired;
    replacing an unknown with a zero here would recreate it one layer down.
    """
    plans: list[dict[str, Any]] = []
    prior_positions: dict[str, Any] | None = None
    for row in rows:
        date = str(row.get("date"))
        positions = _positions(row)
        per_share = ex_dividends_by_session.get(date) or {}
        if positions is None:
            plans.append({"date": date, "skipped": "no parseable positions_snapshot"})
            prior_positions = None
            continue
        if prior_positions is None:
            plans.append({"date": date, "skipped": "no prior snapshot for entitlement"})
            prior_positions = positions
            continue
        # accrue_position_dividends mutates; operate on a copy so planning is pure
        # with respect to the caller's rows.
        working = {t: dict(p) for t, p in positions.items()}
        accruals = accrue_position_dividends(working, prior_positions, per_share)
        plans.append({
            "date": date,
            "dividend_usd": sum(a["amount_usd"] for a in accruals),
            "n_accruals": len(accruals),
            "tickers": [a["ticker"] for a in accruals],
        })
        prior_positions = positions
    return plans


def apply_dividend_backfill(
    conn: sqlite3.Connection, plans: list[dict[str, Any]],
) -> int:
    """Write the planned ``dividend_usd`` figures. Skipped rows stay NULL."""
    n = 0
    for p in plans:
        if "dividend_usd" not in p:
            continue
        conn.execute(
            "UPDATE eod_pnl SET dividend_usd=?, dividend_accrual_available=1 "
            "WHERE date=?",
            (p["dividend_usd"], p["date"]),
        )
        n += 1
    conn.commit()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retroactive audit — run every gate over the history, loudly
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_mark_divergences(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the custodian-mark check for a session from what was PERSISTED.

    ``ib_mark_outside_range`` / ``ib_mark_range_error_usd`` — the fields
    :func:`executor.pnl_integrity.check_custodian_marks` consumes — were only
    added to ``positions_snapshot`` in 2026-08. Measured over the persisted
    history: 5 flagged position-days, ALL of them 2026-08-17 or later, against
    989 position-days carrying ``market_value`` and ``ib_market_value``. So the
    live gate reads a field the history does not have, and running it over the
    history reports zero breaches — a silence that would be indistinguishable
    from a clean record.

    THE INSTRUMENT IS DIFFERENT AND WEAKER, and is labelled as such rather than
    merged into the same count. The live gate measures the distance past that
    day's ArcticDB traded RANGE — a mark outside the range is provably wrong
    regardless of volatility. The traded range is not persisted, so the only
    reference available retroactively is the settled close (``market_value``).
    A settled close lies INSIDE the traded range, so this divergence is an
    upper bound on the range breach: it can flag a mark the live gate would
    pass (a genuine intraday move away from the close), and it can never miss
    one the live gate would raise. That direction is the safe one for an audit
    of a record nobody has ever checked, and it is why this leg is reported
    under its own key with its own instrument named.

    Materiality is the live gate's own ``mark_materiality_usd`` — max($500,
    15bp of NAV) — so the audit and the forward path agree on what "material"
    means even where they disagree on what is being measured.
    """
    from executor.pnl_integrity import mark_materiality_usd

    nav = _f(row.get("portfolio_nav"))
    positions = _positions(row) or {}
    if nav is None or not positions:
        return []
    materiality = mark_materiality_usd(nav)
    out: list[dict[str, Any]] = []
    for ticker, pos in positions.items():
        settled_mv = _f((pos or {}).get("market_value"))
        ib_mv = _f((pos or {}).get("ib_market_value"))
        if (pos or {}).get("ib_mark_corrected"):
            # alpha-engine-config-I10048. A schema-2.11 corrected row carries
            # the REPAIRED mark, so the divergence here is $0 by construction
            # and the row's NAV already excludes the error — there is no
            # residual contamination for this audit to name. The evidence that
            # the mark WAS wrong is not lost: it stays on the same row as
            # `ib_mark_outside_range`, `ib_mark_range_error_usd`,
            # `ib_market_value_raw` and the row's `nav_mark_correction_json`.
            continue
        if settled_mv is None or ib_mv is None:
            continue
        divergence = ib_mv - settled_mv
        if abs(divergence) <= materiality:
            continue
        out.append({
            "kind": "custodian_mark_divergence",
            "instrument": "broker MV vs settled-close MV (upper bound on the "
                          "traded-range breach the live gate measures)",
            "run_date": str(row.get("date")),
            "ticker": ticker,
            "settled_mv_usd": settled_mv,
            "ib_market_value_usd": ib_mv,
            "divergence_usd": divergence,
            "divergence_pct": (divergence / settled_mv * 100.0) if settled_mv else None,
            "materiality_usd": materiality,
            "nav": nav,
        })
    return out


def plan_non_trading_day_flags(
    rows: list[dict[str, Any]], coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn ``non_trading_day_row`` breaches into a persisted-flag plan. PURE.

    alpha-engine-config-I9615 deliverable 1: the live 2026-04-03 row (Good
    Friday) is not a byte-identical duplicate of 2026-04-02 — its
    ``portfolio_nav`` differs by +$216.77 and its ``created_at`` sits at the
    same time-of-day as an ordinary postclose run, so the postclose pipeline
    genuinely executed on a day the market never opened, marking against a
    carried-forward stale ``spy_close``. Deleting that row would destroy a
    real (if spurious) measurement; "Retain all archives" applies here too.
    The remediation is therefore a FLAG, not a delete — merged into the
    existing ``integrity_breach_json`` column (already written by the live
    path, see ``eod_reconcile.py``) rather than a new schema column.

    Returns one entry per flagged row: ``{date, breach}`` where ``breach`` is
    the ``non_trading_day_row`` dict from
    :func:`executor.pnl_integrity.check_session_axis_coverage`, ready to merge
    into that row's existing ``integrity_breach_json`` list.
    """
    return [
        {"date": b["date"], "breach": b}
        for b in coverage.get("breaches", [])
        if b.get("kind") == "non_trading_day_row"
    ]


def apply_non_trading_day_flags(
    conn: sqlite3.Connection, plans: list[dict[str, Any]],
) -> int:
    """Write the planned flags, MERGING into any existing ``integrity_breach_json``.

    Never overwrites a breach list the live path already wrote for that
    session — appends this breach if not already present (idempotent: running
    the audit twice does not duplicate the entry).
    """
    n = 0
    for p in plans:
        existing_raw = conn.execute(
            "SELECT integrity_breach_json FROM eod_pnl WHERE date=?", (p["date"],),
        ).fetchone()
        existing: list[Any] = []
        if existing_raw and existing_raw[0]:
            try:
                parsed = json.loads(existing_raw[0])
                if isinstance(parsed, list):
                    existing = parsed
            except (TypeError, ValueError):
                existing = []
        if any(e.get("kind") == "non_trading_day_row" for e in existing
               if isinstance(e, dict)):
            continue  # already flagged — idempotent
        existing.append(p["breach"])
        conn.execute(
            "UPDATE eod_pnl SET integrity_breach_json=? WHERE date=?",
            (json.dumps(existing, default=str), p["date"]),
        )
        n += 1
    conn.commit()
    return n


def audit_history(
    rows: list[dict[str, Any]],
    *,
    spy_ex_dividends: dict[str, float] | None = None,
    vendor_closes: dict[str, float] | None = None,
    prior_session_of: Any | None = None,
    is_trading_day: Any | None = None,
    next_trading_day: Any | None = None,
) -> dict[str, Any]:
    """Run the integrity gates over the WHOLE persisted series and report breaches.

    The forward path evaluates each gate on the session it is reconciling. None
    of them has ever been evaluated against the history, so a breach that
    happened before the gate existed has never been seen. This runs all of them
    retroactively — TWR closure, benchmark-chain closure, the session-axis
    coverage gate, the per-session and cumulative residual bounds, and the
    custodian-mark materiality gate — and returns every breach with its
    session.

    ``is_trading_day``/``next_trading_day`` feed
    :func:`executor.pnl_integrity.check_session_axis_coverage` — see that
    docstring. Both absent SKIPS the coverage gate (reported as
    ``session_axis_coverage_status: "NOT EVALUATED"``) rather than reporting a
    pass; a gate that did not run has not agreed with anything, same
    convention as the vendor anchor below.

    Returns a dict with ``breach_count``; the CLI exits NON-ZERO when it is
    nonzero, so a breach in the history is a failure rather than a paragraph in
    a report nobody reads.
    """
    twr = verify_twr_closes(rows)
    bench = verify_benchmark_chain_closes(
        rows,
        ex_dividends=spy_ex_dividends or {},
        prior_session_of=prior_session_of,
    )
    if is_trading_day is not None and next_trading_day is not None:
        coverage = check_session_axis_coverage(
            rows, is_trading_day=is_trading_day, next_trading_day=next_trading_day,
        )
        coverage_status = "evaluated"
    else:
        coverage = {"status": "n/a", "closes": None, "breaches": []}
        coverage_status = "NOT EVALUATED — no trading calendar supplied"
    # The vendor anchor is the only leg here that is not built from columns this
    # system wrote. Its ABSENCE is reported as a named degradation rather than
    # as a pass — a gate that could not run has not agreed with anything.
    if vendor_closes:
        anchor_breaches = check_benchmark_vendor_anchor(
            rows,
            vendor_closes=vendor_closes,
            vendor_dividends=spy_ex_dividends or {},
        )
        anchor_status = "evaluated"
    else:
        anchor_breaches = []
        anchor_status = "NOT EVALUATED — no vendor close series supplied"

    residual_breaches: list[dict[str, Any]] = []
    mark_breaches: list[dict[str, Any]] = []
    divergence_breaches: list[dict[str, Any]] = []
    true_window: list[float] = []
    for row in rows:
        nav = _f(row.get("portfolio_nav"))
        date = str(row.get("date"))
        true_residual = _f(row.get("unattributed_true_usd"))
        if nav is None or true_residual is None:
            continue
        true_window.append(true_residual)
        residual_breaches.extend(
            check_residual_bounds(
                unattributed_true_usd=true_residual,
                nav=nav,
                trailing_residuals_usd=true_window[:-1],
                run_date=date,
            )
        )
        positions = _positions(row) or {}
        flags = [
            {
                "ticker": t,
                "mark_error_usd": p.get("ib_mark_range_error_usd"),
                # The RAW broker mark when the row was repaired in-session
                # (schema 2.11 / alpha-engine-config-I10048): this flag list
                # describes what the custodian gate SAW, which is the number
                # IB sent, not the number NAV was restruck on.
                "ib_market_value": (
                    p.get("ib_market_value_raw")
                    if p.get("ib_mark_corrected")
                    else p.get("ib_market_value")
                ),
                "shares": p.get("shares"),
            }
            for t, p in positions.items()
            if p.get("ib_mark_outside_range")
            and p.get("ib_mark_range_error_usd") is not None
        ]
        mark_breaches.extend(check_custodian_marks(flags, nav=nav, run_date=date))
        divergence_breaches.extend(reconstruct_mark_divergences(row))

    breach_count = (
        (0 if twr.get("closes") is not False else 1)
        + (0 if bench.get("closes") is not False else 1)
        + len(residual_breaches)
        + len(mark_breaches)
        + len(divergence_breaches)
        + len(anchor_breaches)
        + len(coverage.get("breaches", []))
    )
    return {
        "n_sessions": len(rows),
        "twr_closure": twr,
        "benchmark_closure": bench,
        "benchmark_vendor_anchor_status": anchor_status,
        "benchmark_vendor_anchor_breaches": anchor_breaches,
        "session_axis_coverage_status": coverage_status,
        "session_axis_coverage": coverage,
        "residual_breaches": residual_breaches,
        "mark_breaches": mark_breaches,
        "mark_divergence_breaches": divergence_breaches,
        "breach_count": breach_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Restating the historical NAV series for reconstructed wrong marks
#    (alpha-engine-config-I9629)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY. ``plan_nav_mark_correction`` (alpha-engine-config-I9627, PR524) repairs a
# provably-wrong broker mark on the FORWARD path: NAV stops carrying the broker's
# number and is moved onto the settled closes the positions were already valued
# at. The live 2026-08-31 instance (DUOL, 708sh, $152.40 -> $148.36, -$2,860.32)
# is repaired and its row is restated. The history is not. ``audit_history``
# reconstructs five more over 989 position-days — AMD 2026-08-04, COIN
# 2026-07-30, LNTH 2026-06-26, MU 2026-08-26, SPY 2026-06-26 — and every one of
# them sits inside the published track record a viability threshold is set
# against.
#
# WHICH INSTRUMENT THE ROW CARRIES DEPENDS ON WHERE THE PASS RUNS, and the
# written row says which one it got. See ``reconstruct_mark_divergences``: the
# live gate measures the distance past that day's ArcticDB traded
# ``[Low, High]``; the traded range is not persisted on the eod_pnl row, so from
# the persisted data alone the only reference is the settled close. A settled
# close lies inside the traded range by construction, so a reconstruction that
# has only the close produces an UPPER BOUND on the range breach — it can
# over-flag and can never under-flag.
#
# THE DISCRIMINATOR, AND WHEN IT CAN BE EVALUATED.
# ``plan_nav_mark_correction`` refuses to move NAV towards a settled close that
# is itself outside the day's ``[Low, High]``, because then the reference data
# is what is wrong. Three options were available:
#
#   (a) fetch the range from ArcticDB — TAKEN, and only in-region
#       (``--range-source arcticdb``). It was originally rejected on the premise
#       that the planning pass runs on a laptop, where ``alpha-engine-data`` is
#       unreadable. That premise does not hold for the restatement itself: this
#       is a production data-repo write, so the run-location rule already
#       requires ``--apply`` to run IN-REGION on EC2 — exactly where the live
#       gate reads its own range. The read reuses ``price_cache``'s
#       ``_open_universe_library``/``_open_macro_library`` and the
#       ``_MACRO_SYMBOLS`` dispatch, so the historical pass and
#       ``eod_reconcile`` share one ArcticDB reader, never two;
#   (b) invent a plausible range (close +/- some volatility) — rejected
#       outright: a fabricated bound would let the discriminator return a verdict
#       it did not measure, which is the exact class of defect this module exists
#       to remove;
#   (c) pass the ONLY bound the persisted data justifies — the degenerate range
#       ``[settled_close, settled_close]`` — and record that the discriminator was
#       NOT evaluated.
#
# (c) REMAINS THE DEFAULT, so a laptop planning pass still works and its
# behaviour is unchanged: the degenerate range is trivially satisfied, the
# discriminator refuses nothing, and the row carries ``basis: "reconstructed"``
# with ``discriminator_evaluated: false``.
#
# (a) IS WHAT THE OPERATOR RUNS IN-REGION, and it buys two refusals the
# degenerate range structurally could not make:
#
#   * a broker mark that lies INSIDE the day's traded range is NOT provably
#     wrong and is refused — the settled-close cut-over (I9638) is where that
#     basis question belongs, not a NAV restatement;
#   * a settled close that is itself OUTSIDE the range means the reference data
#     is wrong, and NAV is not moved towards it. This is
#     ``plan_nav_mark_correction``'s own refusal, reached by the live path's own
#     code — the historical pass only surfaces it per name.
#
# A name whose range cannot be fetched is REFUSED for that session with the
# reason naming why; it NEVER silently falls back to the degenerate range, which
# would mix two instruments inside one restated NAV. A session where every
# material name is refused is not restated at all.
#
# Rows restated that way carry ``basis: "reconstructed_ranged"`` and
# ``discriminator_evaluated: true``, against ``basis: "reconstructed"`` for the
# degenerate default and ``basis: "live_gate"`` for a row the forward path
# corrected. A consumer tells all three apart by reading one field.
#
# RUN LOCATION. ``--range-source arcticdb`` requires an in-region reader
# (ArcticDB is unreadable from the laptop), which is the same constraint
# ``--apply`` already carries. Planning WITHOUT the flag stays laptop-runnable.
#
# NOT WIRED INTO THE EOD PATH. This is an operator CLI, for the same reason the
# other legs are: ``eod_reconcile.run`` is what the postclose SF executes, and
# the pipelines are change-quiet until the clause-1 reliability clock starts.
MARK_BASIS_RECONSTRUCTED = "reconstructed"
MARK_BASIS_RECONSTRUCTED_RANGED = "reconstructed_ranged"
MARK_BASIS_LIVE = "live_gate"

# Per-name and per-session verdicts. The operator reads these before --apply.
VERDICT_RESTATED = "restated"
VERDICT_REFUSED_INSIDE_RANGE = "refused-inside-range"
VERDICT_REFUSED_NO_RANGE = "refused-no-range"
VERDICT_REFUSED_CLOSE_OUTSIDE_RANGE = "refused-close-outside-range"
VERDICT_REFUSED_NOT_RECONSTRUCTIBLE = "refused-not-reconstructible"
VERDICT_REFUSED_BY_BOUND = "refused-by-bound"
VERDICT_REFUSED_MIXED = "refused-mixed"
VERDICT_ALREADY_RESTATED = "already-restated"

RANGE_SOURCE_NONE = "none"
RANGE_SOURCE_ARCTICDB = "arcticdb"

# Mirrors ``eod_reconcile``'s own default (``config["trades_bucket"]``).
DEFAULT_TRADES_BUCKET = "alpha-engine-research"

# A restated return that differs from the stored one by less than this is not
# reported as moving — it is float noise in a column persisted at REAL precision.
RETURN_RESTATEMENT_EPSILON_PCT = 1e-9

# How far a row's STORED daily_return_pct may sit from the return its own
# persisted NAV chain implies before the divergence is named. 0.5bp, the same
# tolerance ``plan_benchmark_restatement`` uses against the vendor series.
CHAIN_BASIS_TOLERANCE_PCT = 0.005


class RangeUnavailable(Exception):
    """The day's traded ``[Low, High]`` could not be fetched for one name.

    Carries the reason verbatim into the refusal row. Never caught into a
    fallback: a name whose range is unavailable is refused, not repriced
    against the degenerate range.
    """


class ArcticDBDayRangeSource:
    """The day's traded ``[Low, High]`` per (ticker, session), from ArcticDB.

    THE LIVE GATE'S OWN READER. ``eod_reconcile.run`` loads the same range with
    ``_open_universe_library`` / ``_open_macro_library`` and the
    ``_MACRO_SYMBOLS`` dispatch from :mod:`executor.price_cache`; this class
    calls those same helpers rather than opening a second ArcticDB path. The
    per-ticker frame is read once and cached, because the historical pass asks
    for many sessions of the same name.

    IN-REGION ONLY. ``alpha-engine-data`` is unreadable from a laptop, so this
    is constructed only for ``--range-source arcticdb``, which the CLI documents
    as an EC2-only option.

    ``universe_lib`` / ``macro_lib`` exist so a test can inject a library object
    of the same shape (``lib.read(ticker).data`` → a DataFrame indexed by date
    with ``Low``/``High`` columns) without an ArcticDB connection.
    """

    name = RANGE_SOURCE_ARCTICDB

    def __init__(
        self,
        trades_bucket: str = DEFAULT_TRADES_BUCKET,
        *,
        universe_lib: Any = None,
        macro_lib: Any = None,
    ) -> None:
        self._bucket = trades_bucket
        self._universe = universe_lib
        self._macro = macro_lib
        self._frames: dict[str, Any] = {}
        self._read_failures: dict[str, str] = {}

    def _library(self, ticker: str) -> Any:
        from executor.price_cache import (
            _MACRO_SYMBOLS,
            _open_macro_library,
            _open_universe_library,
        )

        if ticker in _MACRO_SYMBOLS:
            if self._macro is None:
                self._macro = _open_macro_library(self._bucket)
            return self._macro
        if self._universe is None:
            self._universe = _open_universe_library(self._bucket)
        return self._universe

    def _frame(self, ticker: str) -> Any:
        if ticker in self._frames:
            return self._frames[ticker]
        if ticker in self._read_failures:
            raise RangeUnavailable(self._read_failures[ticker])
        try:
            df = self._library(ticker).read(ticker).data
        except Exception as exc:  # noqa: BLE001 — re-raised as a NAMED refusal
            # (a) swallowed: nothing. The ArcticDB read failure for ONE ticker
            # is converted into a refusal that names the exception class and
            # message; (b) the primary deliverable — every other name's
            # verdict — survives; (c) recorded: the refusal row in the plan
            # JSON and the log summary, both read before --apply.
            reason = (
                f"ArcticDB read failed for {ticker} "
                f"({exc.__class__.__name__}: {exc}) — the day's traded range "
                "cannot be fetched, so this name is refused rather than "
                "repriced against the degenerate range"
            )
            self._read_failures[ticker] = reason
            raise RangeUnavailable(reason) from exc
        self._frames[ticker] = df
        return df

    def range_for(self, ticker: str, date: str) -> tuple[float, float]:
        """``(low, high)`` for ``ticker`` on ``date``, or raise RangeUnavailable."""
        import pandas as pd

        df = self._frame(ticker)
        if df is None or getattr(df, "empty", True):
            raise RangeUnavailable(
                f"ArcticDB holds an empty frame for {ticker} — no traded range "
                f"exists for {date}"
            )
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        match = df[idx == pd.Timestamp(date).normalize()]
        if match.empty:
            raise RangeUnavailable(
                f"ArcticDB carries no row for {ticker} on {date} — the day's "
                "traded range is unknown and the mark cannot be tested against it"
            )
        if "Low" not in match.columns or "High" not in match.columns:
            raise RangeUnavailable(
                f"ArcticDB row for {ticker} on {date} has no Low/High columns "
                "(the macro library is Close-only for some symbols, "
                "alpha-engine-config-I9637) — the mark is not range-checkable"
            )
        low = float(match["Low"].iloc[-1])
        high = float(match["High"].iloc[-1])
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise RangeUnavailable(
                f"ArcticDB row for {ticker} on {date} carries an unusable "
                f"traded range [{low}, {high}] — the mark is not range-checkable"
            )
        return low, high


def build_range_source(
    name: str, *, trades_bucket: str = DEFAULT_TRADES_BUCKET,
) -> Any | None:
    """``None`` for the degenerate default, a live reader for ``arcticdb``."""
    if name == RANGE_SOURCE_NONE:
        return None
    if name == RANGE_SOURCE_ARCTICDB:
        return ArcticDBDayRangeSource(trades_bucket)
    raise ValueError(f"unknown range source {name!r}")


def reconstruct_mark_correction_inputs(
    row: dict[str, Any], *, range_source: Any = None,
) -> dict[str, Any]:
    """Rebuild :func:`plan_nav_mark_correction`'s inputs from a persisted row. PURE.

    Returns ``{"flags", "settled_closes", "day_low", "day_high", "refused",
    "range_evaluated", "range_source"}``.

    ``range_source`` is ``None`` (the default) for the degenerate range
    ``[settled_close, settled_close]``, behaviour unchanged. Given a source with
    a ``range_for(ticker, date) -> (low, high)`` method — see
    :class:`ArcticDBDayRangeSource` — the day's real traded range is fetched per
    name and the discriminator becomes evaluable: a mark INSIDE the range is
    refused (not provably wrong), and a name whose range cannot be fetched is
    refused rather than falling back to the degenerate range.

    ``flags`` carries the same keys ``eod_reconcile._detect_ib_mark_outside_range``
    produces for the live path — ``ticker`` / ``shares`` / ``ib_mark`` /
    ``mark_error_usd`` — reconstructed from the ``market_value`` /
    ``ib_market_value`` / ``shares`` triple, at ``mark_materiality_usd``, the
    live gate's own floor.

    NOTHING IS SUBSTITUTED AND NOTHING IS ZEROED. A row with no parseable
    ``positions_snapshot``, and each individual name missing ``ib_market_value``
    (the pre-schema-2.1 state, which is most of the history) or missing a usable
    share count, is NAMED in ``refused`` and contributes no flag. A silent skip
    here would report a session as clean that was never examined.
    """
    nav = _f(row.get("portfolio_nav"))
    date = str(row.get("date"))
    out: dict[str, Any] = {
        "date": date,
        "flags": [],
        "settled_closes": {},
        "day_low": {},
        "day_high": {},
        "refused": [],
        "range_evaluated": range_source is not None,
        "range_source": getattr(range_source, "name", RANGE_SOURCE_NONE),
    }
    positions = _positions(row)
    if positions is None:
        out["refused"].append({
            "date": date, "ticker": None,
            "verdict": VERDICT_REFUSED_NOT_RECONSTRUCTIBLE,
            "reason": "positions_snapshot absent or unparseable — the session "
                      "cannot be examined and is NOT reported as clean",
        })
        return out
    if nav is None:
        out["refused"].append({
            "date": date, "ticker": None,
            "verdict": VERDICT_REFUSED_NOT_RECONSTRUCTIBLE,
            "reason": "row carries no portfolio_nav — there is nothing to restate "
                      "and no NAV to size the materiality floor against",
        })
        return out

    materiality = mark_materiality_usd(nav)
    for ticker, pos in sorted(positions.items()):
        pos = pos or {}
        settled_mv = _f(pos.get("market_value"))
        ib_mv = _f(pos.get("ib_market_value"))
        shares = _f(pos.get("shares"))
        if pos.get("ib_mark_corrected"):
            # alpha-engine-config-I10048. Since schema 2.11 a corrected name's
            # `ib_market_value` is the REPAIRED mark, so `ib_mv - settled_mv`
            # is $0 here by construction. Reading that as "in range" would
            # turn a session we KNOW carried a wrong mark into a clean one —
            # the exact detection blindness this reconstruction exists to
            # avoid. The row's NAV already excludes the error, so no
            # restatement is due; it is NAMED rather than silently skipped.
            out["refused"].append({
                "date": date, "ticker": ticker,
                "verdict": VERDICT_ALREADY_RESTATED,
                "reason": "the broker mark for this name was already corrected "
                          "in-session (alpha-engine-config-I9627) and the row's "
                          "portfolio_nav already excludes the error — no "
                          "restatement is due, and the $0 divergence here is "
                          "the repair, not a clean mark. Raw broker mark: "
                          f"{pos.get('ib_market_value_raw')}",
            })
            continue
        if ib_mv is None:
            out["refused"].append({
                "date": date, "ticker": ticker,
                "verdict": VERDICT_REFUSED_NOT_RECONSTRUCTIBLE,
                "reason": "no ib_market_value on this position (pre-schema-2.1 "
                          "snapshot) — the broker mark was never persisted, so "
                          "this name cannot be checked and is not assumed clean",
            })
            continue
        if settled_mv is None:
            out["refused"].append({
                "date": date, "ticker": ticker,
                "verdict": VERDICT_REFUSED_NOT_RECONSTRUCTIBLE,
                "reason": "no market_value on this position — there is no settled "
                          "reference to price the broker mark against",
            })
            continue
        if not shares:
            out["refused"].append({
                "date": date, "ticker": ticker,
                "verdict": VERDICT_REFUSED_NOT_RECONSTRUCTIBLE,
                "reason": "share count absent or zero — a per-share mark cannot be "
                          "reconstructed from the market-value pair without it",
            })
            continue
        divergence = ib_mv - settled_mv
        if abs(divergence) <= materiality:
            continue
        settled_close = settled_mv / shares
        ib_mark = ib_mv / shares
        if range_source is None:
            # See §5 option (c): the degenerate range is the only bound the
            # persisted data alone justifies, and the row records that the
            # discriminator was NOT evaluated.
            low, high = settled_close, settled_close
        else:
            try:
                low, high = range_source.range_for(ticker, date)
            except RangeUnavailable as exc:
                out["refused"].append({
                    "date": date, "ticker": ticker,
                    "verdict": VERDICT_REFUSED_NO_RANGE,
                    "ib_mark": ib_mark,
                    "settled_close": settled_close,
                    "mark_error_usd": divergence,
                    "reason": str(exc),
                })
                continue
            if low <= ib_mark <= high:
                # The refusal the degenerate range structurally could not make:
                # the broker priced this name somewhere it actually traded, so
                # the mark is not PROVABLY wrong and NAV is not moved off it.
                # The settled-close cut-over (alpha-engine-config-I9638) is
                # where a mark-basis preference belongs — not a NAV restatement.
                out["refused"].append({
                    "date": date, "ticker": ticker,
                    "verdict": VERDICT_REFUSED_INSIDE_RANGE,
                    "ib_mark": ib_mark,
                    "settled_close": settled_close,
                    "day_low": low, "day_high": high,
                    "mark_error_usd": divergence,
                    "reason": (
                        f"broker mark ${ib_mark:,.2f} lies INSIDE the day's "
                        f"traded range [${low:,.2f}, ${high:,.2f}] — the name "
                        f"diverges from the settled close by ${divergence:+,.0f} "
                        "but the mark is not provably wrong, so NAV is not "
                        "restated off it (the settled-close cut-over, "
                        "alpha-engine-config-I9638, is where that basis "
                        "question belongs)"
                    ),
                })
                continue
        out["flags"].append({
            "ticker": ticker,
            "shares": shares,
            "ib_mark": ib_mark,
            "mark_error_usd": divergence,
            "materiality_usd": materiality,
        })
        out["settled_closes"][ticker] = settled_close
        out["day_low"][ticker] = low
        out["day_high"][ticker] = high
    return out


def _mark_correction_payload(
    plan: dict[str, Any], *, inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``nav_mark_correction_json`` body for a RECONSTRUCTED restatement.

    The ``basis`` field is the one field a consumer reads to tell the three
    provenances apart — ``live_gate`` (forward path), ``reconstructed_ranged``
    (this pass, in-region, the live gate's own instrument) and ``reconstructed``
    (this pass, degenerate range, an upper bound). ``corrections`` already
    carries the ``day_low``/``day_high`` actually used per name, written by
    ``plan_nav_mark_correction`` itself.
    """
    inputs = inputs or {}
    ranged = bool(inputs.get("range_evaluated"))
    payload = dict(plan)
    payload["basis"] = (
        MARK_BASIS_RECONSTRUCTED_RANGED if ranged else MARK_BASIS_RECONSTRUCTED
    )
    payload["range_source"] = inputs.get("range_source", RANGE_SOURCE_NONE)
    payload["discriminator_evaluated"] = ranged
    if ranged:
        payload["instrument"] = (
            "broker mark vs the day's traded [Low, High] read from ArcticDB "
            "through the SAME loader executor.eod_reconcile uses — the live "
            "gate's own instrument, not an upper bound on it"
        )
        payload["discriminator_note"] = (
            "the day's traded range was fetched per name; a mark lying inside "
            "it was refused as not provably wrong, and plan_nav_mark_correction "
            "tested each settled close for lying inside the same range before "
            "NAV was moved towards it"
        )
        payload["day_ranges"] = {
            t: [inputs.get("day_low", {}).get(t), inputs.get("day_high", {}).get(t)]
            for t in plan.get("corrected_tickers", [])
        }
    else:
        payload["instrument"] = (
            "settled market value vs broker market value, both reconstructed from "
            "positions_snapshot — an UPPER BOUND on the traded-range breach the live "
            "gate measures, never an under-estimate of it"
        )
        payload["discriminator_note"] = (
            "the day's ArcticDB [Low, High] was not fetched (--range-source none), "
            "so the settled close could not be tested for lying inside it; "
            "day_low/day_high were passed as the degenerate range "
            "[settled_close, settled_close] and the reference-data discriminator "
            "refused nothing on this row"
        )
    payload["source"] = "executor.pnl_measurement_backfill --restate-marks"
    payload["tracker"] = "alpha-engine-config-I9629"
    return payload


def _session_verdict(names_refused: list[dict[str, Any]]) -> str:
    """One verdict for a session no name could be restated on."""
    kinds = {r.get("verdict") for r in names_refused if r.get("verdict")}
    if len(kinds) == 1:
        return kinds.pop()
    return VERDICT_REFUSED_MIXED


def plan_historical_mark_restatement(
    rows: list[dict[str, Any]], *, range_source: Any = None,
) -> dict[str, Any]:
    """Plan the NAV restatement AND the return-chain recomputation. PURE.

    ``rows`` is the eod_pnl series OLDEST-first, as :func:`_rows` returns it.

    The NAV leg reuses :func:`executor.pnl_integrity.plan_nav_mark_correction`
    unchanged — the same function the live path calls — so a restated historical
    row is corrected by the code that corrects a live one, including its
    ``mark_correction_bound_usd`` refusal. Only the day-range input differs, and
    that difference is recorded on every row it touches.

    THE CHAIN, NOT THE ROW. ``daily_return_pct[t] = (nav[t] - nav[t-1])/nav[t-1]``,
    so moving ``nav[t]`` moves session *t*'s return AND session *t+1*'s, and
    ``daily_alpha_pct = daily_return_pct - spy_return_pct`` moves with each. Every
    downstream row is recomputed off the RESTATED NAV series; ``spy_return_pct``
    is not touched (a separate, deliberately untaken decision).

    IDEMPOTENCE. A row already carrying ``nav_ib_raw_usd`` or
    ``nav_mark_correction_json`` has been restated — by an earlier run of this
    pass, or by the live path, which is what makes the 2026-08-31 DUOL row a
    no-op here. It is reported under ``already_restated`` and left untouched, so
    a second run writes nothing.

    THE VERDICT PER CANDIDATE SESSION is reported in ``session_verdicts`` —
    ``restated``, ``refused-inside-range``, ``refused-no-range``,
    ``refused-close-outside-range``, ``refused-by-bound`` or ``refused-mixed``,
    each carrying the numbers behind it. That is what the operator reads before
    ``--apply``. A session where every material name is refused produces a
    refusal verdict and is not restated.

    ``range_source`` is threaded to :func:`reconstruct_mark_correction_inputs`;
    ``None`` keeps the degenerate range and the unchanged default behaviour.
    """
    restatements: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    already: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    nav_new: dict[str, float] = {}

    for row in rows:
        date = str(row.get("date"))
        nav = _f(row.get("portfolio_nav"))
        if nav is not None:
            nav_new[date] = nav
        if row.get("nav_ib_raw_usd") is not None or row.get("nav_mark_correction_json"):
            already.append({
                "date": date,
                "basis": (MARK_BASIS_LIVE
                          if row.get("nav_mark_correction_json") and
                          row.get("nav_ib_raw_usd") is not None
                          else "unknown"),
                "reason": "row already carries a mark restatement — left untouched",
            })
            continue
        inputs = reconstruct_mark_correction_inputs(row, range_source=range_source)
        session_refused = list(inputs["refused"])
        refused.extend(inputs["refused"])
        if not inputs["flags"]:
            if session_refused:
                verdicts.append({
                    "date": date,
                    "verdict": _session_verdict(session_refused),
                    "range_source": inputs["range_source"],
                    "discriminator_evaluated": inputs["range_evaluated"],
                    "portfolio_nav": nav,
                    "names_restated": [],
                    "names_refused": session_refused,
                })
            continue
        plan = plan_nav_mark_correction(
            inputs["flags"],
            settled_closes=inputs["settled_closes"],
            day_low=inputs["day_low"],
            day_high=inputs["day_high"],
            nav=nav,
            run_date=date,
        )
        # Per-name refusals the LIVE PATH's own code made. The one that only
        # exists with a real range is the settled close lying outside it —
        # classified here from the same [low, high] that was passed in, never
        # by matching on the message text.
        for u in plan.get("unrepairable", []):
            tkr = u.get("ticker")
            close = inputs["settled_closes"].get(tkr)
            lo = inputs["day_low"].get(tkr)
            hi = inputs["day_high"].get(tkr)
            outside = (
                close is not None and lo is not None and hi is not None
                and not (lo <= close <= hi)
            )
            entry = {
                "date": date, "ticker": tkr,
                "verdict": (VERDICT_REFUSED_CLOSE_OUTSIDE_RANGE if outside
                            else VERDICT_REFUSED_NOT_RECONSTRUCTIBLE),
                "settled_close": close, "day_low": lo, "day_high": hi,
                "reason": u.get("why"),
            }
            session_refused.append(entry)
            refused.append(entry)
        if not plan["applied"]:
            entry = {
                "date": date, "ticker": None,
                "verdict": (VERDICT_REFUSED_BY_BOUND if plan.get("refused")
                            else _session_verdict(session_refused)),
                "reason": plan["message"] or "no correction could be proven",
                "refused_by_bound": bool(plan.get("refused")),
                "correction_usd": plan.get("correction_usd"),
                "bound_usd": plan.get("bound_usd"),
            }
            refused.append(entry)
            verdicts.append({
                "date": date,
                "verdict": entry["verdict"],
                "range_source": inputs["range_source"],
                "discriminator_evaluated": inputs["range_evaluated"],
                "portfolio_nav": nav,
                "correction_usd": plan.get("correction_usd"),
                "bound_usd": plan.get("bound_usd"),
                "names_restated": [],
                "names_refused": session_refused,
            })
            continue
        nav_new[date] = float(plan["nav_corrected"])
        restatements.append({
            "date": date,
            "nav_ib_raw_usd": nav,
            "portfolio_nav": float(plan["nav_corrected"]),
            "nav_mark_correction_usd": float(plan["correction_usd"]),
            "tickers": list(plan["corrected_tickers"]),
            "corrections": plan["corrections"],
            "nav_mark_correction_json": _mark_correction_payload(
                plan, inputs=inputs
            ),
        })
        verdicts.append({
            "date": date,
            "verdict": VERDICT_RESTATED,
            "range_source": inputs["range_source"],
            "discriminator_evaluated": inputs["range_evaluated"],
            "portfolio_nav": nav,
            "portfolio_nav_restated": float(plan["nav_corrected"]),
            "correction_usd": float(plan["correction_usd"]),
            "bound_usd": plan.get("bound_usd"),
            "names_restated": plan["corrections"],
            "names_refused": session_refused,
        })

    chain, chain_refused, chain_basis_mismatch = _plan_return_chain(rows, nav_new)
    return {
        "restatements": restatements,
        "refused": refused,
        "already_restated": already,
        "session_verdicts": verdicts,
        "verdict_counts": _verdict_counts(verdicts),
        "range_source": getattr(range_source, "name", RANGE_SOURCE_NONE),
        "discriminator_evaluated": range_source is not None,
        "chain": chain,
        "chain_refused": chain_refused,
        "chain_basis_mismatch": chain_basis_mismatch,
    }


def _verdict_counts(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    """Sessions per verdict. Absent verdicts are absent, never rendered zero."""
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    return counts


def _plan_return_chain(
    rows: list[dict[str, Any]], nav_new: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute ``daily_return_pct``/``daily_alpha_pct`` off the RESTATED NAVs. PURE.

    Walks the series oldest-first carrying BOTH chains — the persisted NAVs and
    the restated ones — so a row is reported only when the restatement actually
    moves its return. Both legs of a moved link are covered: the corrected
    session and the one after it.

    Two things are named rather than silently absorbed:

    * ``chain_refused`` — a row whose ``daily_return_pct`` moves but whose
      ``spy_return_pct`` is absent, so ``daily_alpha_pct`` cannot be restated
      alongside it. Same refusal ``plan_benchmark_restatement`` makes in the
      mirror direction: correcting one leg alone leaves the row internally
      inconsistent.
    * ``chain_basis_mismatch`` — a row whose STORED return already disagreed with
      its own persisted NAV chain by more than
      ``CHAIN_BASIS_TOLERANCE_PCT``. Recomputing it from the NAV chain therefore
      moves it by more than the mark correction alone, and that pre-existing
      discrepancy is reported so the delta is never read as this pass's doing.
    """
    chain: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    prior_old: float | None = None
    prior_new: float | None = None
    for row in rows:
        date = str(row.get("date"))
        old_nav = _f(row.get("portfolio_nav"))
        new_nav = nav_new.get(date, old_nav)
        if old_nav is None or new_nav is None:
            prior_old, prior_new = old_nav, new_nav
            continue
        if prior_old and prior_new:
            old_implied = (old_nav - prior_old) / prior_old * 100.0
            new_return = (new_nav - prior_new) / prior_new * 100.0
            if abs(new_return - old_implied) > RETURN_RESTATEMENT_EPSILON_PCT:
                stored = _f(row.get("daily_return_pct"))
                if stored is not None and abs(stored - old_implied) > CHAIN_BASIS_TOLERANCE_PCT:
                    mismatches.append({
                        "date": date,
                        "stored_daily_return_pct": stored,
                        "nav_chain_implied_pct": old_implied,
                        "reason": "stored daily_return_pct already disagreed with "
                                  "this row's own persisted NAV chain before any "
                                  "restatement — the recomputed value moves by more "
                                  "than the mark correction",
                    })
                spy = _f(row.get("spy_return_pct"))
                entry = {
                    "date": date,
                    "nav_from": old_nav,
                    "nav_to": new_nav,
                    "prior_nav_from": prior_old,
                    "prior_nav_to": prior_new,
                    "daily_return_pct_from": stored,
                    "daily_return_pct_to": new_return,
                    "daily_alpha_pct_from": _f(row.get("daily_alpha_pct")),
                    "daily_alpha_pct_to": None,
                    "spy_return_pct": spy,
                }
                if spy is None:
                    refused.append({
                        "date": date,
                        "daily_return_pct_to": new_return,
                        "reason": "row carries no spy_return_pct, so daily_alpha_pct "
                                  "cannot be restated alongside daily_return_pct — "
                                  "correcting one leg alone leaves the row "
                                  "internally inconsistent",
                    })
                else:
                    entry["daily_alpha_pct_to"] = new_return - spy
                    chain.append(entry)
        prior_old, prior_new = old_nav, new_nav
    return chain, refused, mismatches


def apply_historical_mark_restatement(
    conn: sqlite3.Connection, plan: dict[str, Any],
) -> dict[str, int]:
    """Write the planned NAV restatement and the recomputed return chain.

    Idempotent by planning: a row already carrying ``nav_ib_raw_usd`` or
    ``nav_mark_correction_json`` never reaches here, so a second run plans no
    restatement, computes no chain movement, and writes nothing.

    ``nav_ib_raw_usd`` holds the pre-restatement ``portfolio_nav``, so the whole
    restatement is reversible from the ledger alone.
    """
    n_nav = 0
    for r in plan.get("restatements", []):
        conn.execute(
            "UPDATE eod_pnl SET portfolio_nav=?, nav_ib_raw_usd=?, "
            "nav_mark_correction_usd=?, nav_mark_correction_json=? WHERE date=?",
            (
                r["portfolio_nav"], r["nav_ib_raw_usd"], r["nav_mark_correction_usd"],
                json.dumps(r["nav_mark_correction_json"], default=str), r["date"],
            ),
        )
        n_nav += 1
    n_chain = 0
    for c in plan.get("chain", []):
        conn.execute(
            "UPDATE eod_pnl SET daily_return_pct=?, daily_alpha_pct=? WHERE date=?",
            (c["daily_return_pct_to"], c["daily_alpha_pct_to"], c["date"]),
        )
        n_chain += 1
    conn.commit()
    return {"nav_rows_written": n_nav, "chain_rows_written": n_chain}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_spy_events(start: str) -> list[dict[str, Any]]:
    from polygon_client import PolygonClient

    return PolygonClient().get_dividends(SPY_TICKER, start=start)


def fetch_vendor_closes(start: str, end: str) -> dict[str, float]:
    """SPY daily closes from the vendor, keyed by session date.

    The independent anchor for
    :func:`executor.pnl_integrity.check_benchmark_vendor_anchor`. Uses
    ``PolygonClient.get_daily_bars``, whose own docstring states the prices are
    split-adjusted but **NOT dividend-adjusted** — which is the series required
    here, because the distributions are added explicitly from
    ``/v3/reference/dividends`` rather than folded into the price level.

    That distinction is the whole finding: ``PolygonClient`` also carries
    ``get_daily_bars_dividend_adjusted``, and the persisted ``spy_close``
    values through 2026-03-19 sit exactly 27.2bp below the unadjusted series —
    $1.797/$659.80, SPY's 2026-03-20 distribution. Mixing the two in one column
    is what made the benchmark leg unverifiable.
    """
    from polygon_client import PolygonClient

    bars = PolygonClient().get_daily_bars(SPY_TICKER, start, end, adjusted=True)
    if bars is None or bars.empty:
        raise RuntimeError(
            f"vendor returned no SPY bars for {start} → {end}; the benchmark "
            "anchor cannot be evaluated and is reported NOT EVALUATED"
        )
    return {
        idx.date().isoformat(): float(close)
        for idx, close in bars["Close"].items()
    }


def fetch_live_anchor_window(
    rows: list[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Vendor closes + SPY distributions for a SHORT live window. Raises on failure.

    alpha-engine-config-I9614, deliverable 3. The CLI legs above fetch the whole
    history on every invocation, which is correct for a one-off audit and wrong
    for a per-session gate: the live EOD path needs ``run_date`` and its prior
    trading session and nothing else, and ``PolygonClient`` is constructed at
    ``calls_per_min=5`` (free tier — see ``executor.dividends`` and
    alpha-engine-config-I10047, where a per-ticker loop on that limiter produced
    16 ``429`` lines and two names silently recorded as having no dividend).

    ``rows`` is the ordered pair (or short run) of persisted ``eod_pnl`` rows
    the anchor will be evaluated over; only ``date`` is read. Two vendor calls
    total, independent of window length.

    NEVER returns an empty map as if it were agreement — an unavailable vendor
    raises, and the caller records a NAMED degradation. That posture is
    ``executor.dividends.fetch_ex_dividends``'s, deliberately: the failure this
    whole issue is about is an absent measurement rendered as a passing one.
    """
    dates = sorted(str(r.get("date")) for r in rows if r.get("date"))
    if len(dates) < 2:
        raise RuntimeError(
            f"the vendor anchor needs at least two persisted sessions; got "
            f"{len(dates)} ({dates}) — NOT EVALUATED, not clean"
        )
    start, end = dates[0], dates[-1]
    closes = fetch_vendor_closes(start, end)
    events = _fetch_spy_events(start)
    return closes, map_ex_dividends_to_sessions(list(rows), events)


def _prior_session_of(date_str: str) -> str:
    import datetime as _dt

    from krepis.trading_calendar import previous_trading_day

    return previous_trading_day(_dt.date.fromisoformat(date_str)).isoformat()


def _is_trading_day(date_str: str) -> bool:
    """String-in/bool-out adapter feeding ``check_session_axis_coverage``."""
    import datetime as _dt

    from krepis.trading_calendar import is_trading_day

    return bool(is_trading_day(_dt.date.fromisoformat(date_str)))


def _next_trading_day(date_str: str) -> str:
    """String-in/string-out adapter feeding ``check_session_axis_coverage``."""
    import datetime as _dt

    from krepis.trading_calendar import next_trading_day

    return next_trading_day(_dt.date.fromisoformat(date_str)).isoformat()


def _fetch_holding_events(tickers: list[str], start: str) -> dict[str, list[dict]]:
    from polygon_client import PolygonClient

    client = PolygonClient()
    out: dict[str, list[dict]] = {}
    for t in sorted(set(tickers)):
        try:
            out[t] = client.get_dividends(t, start=start)
        except Exception:  # noqa: BLE001 — per-ticker isolation, mirrors fetch_ex_dividends
            # (a) swallowed: a third-party HTTP/credential failure on ONE ticker;
            # (b) the primary deliverable — every other ticker's accrual and the
            #     cost/benchmark legs — survives; (c) recorded: the ticker is
            #     absent from the returned map, the CLI prints the shortfall, and
            #     the row's dividend stays inside the bounded residual.
            logger.warning("[backfill] dividend fetch failed for %s", t, exc_info=True)
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="trades.db", help="path to the trades sqlite db")
    ap.add_argument("--apply", action="store_true", help="write (default is dry-run)")
    ap.add_argument("--costs", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="write spy_dividend_per_share (additive, restates nothing)")
    ap.add_argument(
        "--restate-benchmark", action="store_true",
        help="ALSO rewrite spy_return_pct/daily_alpha_pct from the vendor series. "
             "This restates a published track record and needs a ruling; without "
             "it the restatement is planned and printed but never written.",
    )
    ap.add_argument("--dividends", action="store_true")
    ap.add_argument(
        "--restate-marks", action="store_true",
        help="restate the historical NAV series for broker marks the persisted "
             "market-value pair proves wrong, and recompute the return chain "
             "across every session the correction moves. Plans and prints by "
             "default; writes only with --apply. Never selected by the no-flag "
             "default — this rewrites a published NAV. "
             "alpha-engine-config-I9629.",
    )
    ap.add_argument(
        "--range-source", choices=[RANGE_SOURCE_NONE, RANGE_SOURCE_ARCTICDB],
        default=RANGE_SOURCE_NONE,
        help="where --restate-marks gets each name's day traded [Low, High]. "
             "'none' (default) passes the degenerate range [settled_close, "
             "settled_close] and records discriminator_evaluated=false — the "
             "laptop-runnable planning default. 'arcticdb' reads the real range "
             "through the same loader eod_reconcile uses, so a mark INSIDE the "
             "range and a settled close OUTSIDE it are both refused by name. "
             "IN-REGION ONLY: alpha-engine-data is unreadable from a laptop.",
    )
    ap.add_argument(
        "--trades-bucket", default=DEFAULT_TRADES_BUCKET,
        help="bucket backing the ArcticDB libraries for --range-source arcticdb",
    )
    ap.add_argument("--audit", action="store_true", help="run the retroactive gates")
    ap.add_argument(
        "--flag-non-trading-rows", action="store_true",
        help="write the session-axis coverage gate's non_trading_day_row "
             "breaches into integrity_breach_json (merged, never overwritten). "
             "Requires --audit and --apply. Never deletes a row — "
             "alpha-engine-config-I9615.",
    )
    ap.add_argument("--json-out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --restate-marks is deliberately absent from this default: it is the only
    # leg that rewrites an already-published NAV, so it is never turned on by
    # running the module with no flags.
    if not any([args.costs, args.benchmark, args.dividends, args.audit,
                args.restate_marks]):
        args.costs = args.benchmark = args.audit = True

    conn = sqlite3.connect(args.db)
    rows = _rows(conn)
    if not rows:
        logger.error("eod_pnl is empty in %s — nothing to backfill", args.db)
        return 2
    start, end = str(rows[0]["date"]), str(rows[-1]["date"])
    report: dict[str, Any] = {
        "db": args.db, "n_sessions": len(rows), "window": [start, end],
        "applied": bool(args.apply),
        "benchmark_restated": bool(args.apply and args.restate_benchmark),
    }

    if args.costs:
        from executor.trade_logger import get_todays_trades

        trades_by_date = {str(r["date"]): get_todays_trades(conn, str(r["date"]))
                          for r in rows}
        plans = plan_cost_backfill(rows, trades_by_date)
        n_absent = sum(1 for p in plans if p["commission_usd"] is None)
        report["costs"] = {
            "sessions_planned": len(plans),
            "sessions_with_commission_ABSENT": n_absent,
            "slippage_usd_total": sum(p["slippage_usd"] for p in plans),
            "traded_notional_usd_total": sum(p["traded_notional_usd"] for p in plans),
        }
        if args.apply:
            report["costs"]["written"] = apply_cost_backfill(conn, plans)
        logger.info("costs: %s", json.dumps(report["costs"]))

    spy_map: dict[str, float] = {}
    vendor_closes: dict[str, float] = {}
    if args.benchmark or args.audit:
        try:
            spy_map = map_ex_dividends_to_sessions(rows, _fetch_spy_events(start))
            vendor_closes = fetch_vendor_closes(start, end)
        except Exception as exc:  # noqa: BLE001
            # NOT swallowed into a zero. The benchmark leg is REFUSED outright
            # when its vendor source is unavailable: writing a "total-return"
            # benchmark with an unverified zero distribution, or passing an
            # anchor check that never ran, is the defect this module exists to
            # remove wearing a green badge.
            logger.error("Vendor benchmark fetch failed (%s) — benchmark leg REFUSED", exc)
            report["benchmark"] = {"status": "refused", "reason": str(exc)}
            args.benchmark = False

    if args.benchmark:
        writes = plan_dividend_per_share_writes(rows, spy_map)
        restatement = plan_benchmark_restatement(
            rows, vendor_closes=vendor_closes, vendor_dividends=spy_map,
            prior_session_of=_prior_session_of,
        )
        report["benchmark"] = {
            "status": "planned",
            "spy_ex_dividends": spy_map,
            "spy_dividend_per_share_writes": len(writes),
            "restatement_corrections": restatement["corrections"],
            "restatement_refused": restatement["refused"],
        }
        if args.apply:
            report["benchmark"]["spy_dividend_per_share_written"] = (
                apply_dividend_per_share_writes(conn, writes)
            )
            if args.restate_benchmark:
                report["benchmark"]["restatement_written"] = (
                    apply_benchmark_restatement(conn, restatement)
                )
            else:
                logger.warning(
                    "%d benchmark restatement(s) planned and NOT written — "
                    "rerun with --restate-benchmark once the restatement is "
                    "ruled on. Every daily_alpha_pct on those sessions moves.",
                    len(restatement["corrections"]),
                )

    if args.restate_marks:
        rows = _rows(conn)  # re-read so the plan sees anything just written
        range_source = build_range_source(
            args.range_source, trades_bucket=args.trades_bucket
        )
        mark_plan = plan_historical_mark_restatement(
            rows, range_source=range_source
        )
        report["mark_restatement"] = {
            "status": "applied" if args.apply else "planned",
            "sessions_restated": len(mark_plan["restatements"]),
            "restatements": mark_plan["restatements"],
            "downstream_rows_moved": len(mark_plan["chain"]),
            "chain": mark_plan["chain"],
            "chain_refused": mark_plan["chain_refused"],
            "chain_basis_mismatch": mark_plan["chain_basis_mismatch"],
            "already_restated": mark_plan["already_restated"],
            "refused": mark_plan["refused"],
            "n_refused": len(mark_plan["refused"]),
            "range_source": mark_plan["range_source"],
            "discriminator_evaluated": mark_plan["discriminator_evaluated"],
            "session_verdicts": mark_plan["session_verdicts"],
            "verdict_counts": mark_plan["verdict_counts"],
            "basis": (MARK_BASIS_RECONSTRUCTED_RANGED
                      if mark_plan["discriminator_evaluated"]
                      else MARK_BASIS_RECONSTRUCTED),
        }
        if args.apply:
            report["mark_restatement"].update(
                apply_historical_mark_restatement(conn, mark_plan)
            )
        elif mark_plan["restatements"]:
            logger.warning(
                "%d session(s) planned to restate and %d downstream row(s) to "
                "recompute, NONE written — rerun with --apply. This is a "
                "production data-repo write and must run IN-REGION (EC2), never "
                "from a laptop.",
                len(mark_plan["restatements"]), len(mark_plan["chain"]),
            )
        logger.info(
            "mark restatement: %d session(s), %d downstream row(s), %d name(s) "
            "refused, %d row(s) already restated — range source %s, "
            "discriminator evaluated=%s, session verdicts %s",
            len(mark_plan["restatements"]), len(mark_plan["chain"]),
            len(mark_plan["refused"]), len(mark_plan["already_restated"]),
            mark_plan["range_source"], mark_plan["discriminator_evaluated"],
            json.dumps(mark_plan["verdict_counts"], sort_keys=True),
        )
        # One line per candidate session, with the numbers. This is what the
        # operator reads before --apply; the JSON carries the same verdicts.
        for v in mark_plan["session_verdicts"]:
            logger.info(
                "  %s %s: nav %s, correction %s, %d name(s) restated, "
                "%d name(s) refused%s",
                v["date"], v["verdict"], v.get("portfolio_nav"),
                v.get("correction_usd"), len(v.get("names_restated") or []),
                len(v.get("names_refused") or []),
                "".join(
                    f"\n      REFUSED {r.get('ticker')}: {r.get('reason')}"
                    for r in (v.get("names_refused") or [])
                ),
            )

    if args.dividends:
        tickers = [t for r in rows for t in (_positions(r) or {})]
        events = _fetch_holding_events(tickers, start)
        by_session: dict[str, dict[str, float]] = {}
        for tkr, evs in events.items():
            for date, amount in map_ex_dividends_to_sessions(rows, evs).items():
                by_session.setdefault(date, {})[tkr] = amount
        div_plans = plan_dividend_backfill(rows, by_session)
        report["dividends"] = {
            "tickers_fetched": len(events),
            "tickers_requested": len(set(tickers)),
            "sessions_planned": sum(1 for p in div_plans if "dividend_usd" in p),
            "sessions_skipped": [p for p in div_plans if "skipped" in p],
            "dividend_usd_total": sum(p.get("dividend_usd", 0.0) for p in div_plans),
        }
        if args.apply:
            report["dividends"]["written"] = apply_dividend_backfill(conn, div_plans)

    exit_code = 0
    if args.audit:
        rows = _rows(conn)  # re-read so the audit sees anything just written
        audit = audit_history(
            rows, spy_ex_dividends=spy_map, vendor_closes=vendor_closes,
            prior_session_of=_prior_session_of,
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        report["audit"] = audit
        if audit["breach_count"]:
            logger.error(
                "HISTORICAL INTEGRITY: %d breach(es) over %d sessions — "
                "TWR closes=%s, benchmark columns agree=%s, vendor anchor=%s "
                "(%d breach), session axis=%s (%d breach), residual=%d, "
                "mark flags=%d, mark divergences=%d",
                audit["breach_count"], audit["n_sessions"],
                audit["twr_closure"].get("closes"),
                audit["benchmark_closure"].get("closes"),
                audit["benchmark_vendor_anchor_status"],
                len(audit["benchmark_vendor_anchor_breaches"]),
                audit["session_axis_coverage_status"],
                len(audit["session_axis_coverage"].get("breaches", [])),
                len(audit["residual_breaches"]), len(audit["mark_breaches"]),
                len(audit["mark_divergence_breaches"]),
            )
            exit_code = 1

        if args.flag_non_trading_rows:
            plans = plan_non_trading_day_flags(rows, audit["session_axis_coverage"])
            report["non_trading_row_flags"] = {
                "planned": plans,
                "n_planned": len(plans),
            }
            if args.apply:
                report["non_trading_row_flags"]["written"] = (
                    apply_non_trading_day_flags(conn, plans)
                )
            elif plans:
                logger.warning(
                    "%d non-trading-day row(s) planned to flag and NOT written "
                    "— rerun with --apply. This is a production data-repo write "
                    "and must run IN-REGION (EC2), never from a laptop.",
                    len(plans),
                )

    payload = json.dumps(report, indent=2, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(payload)
    else:
        print(payload)
    conn.close()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
