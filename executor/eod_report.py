"""
EOD report artifact builder — Alpha Engine executor.

Produces the structured ``consolidated/{date}/eod_report.json`` artifact that
is the single source of truth for the end-of-day report rendered on the
private console (alpha-engine-dashboard ``views/19_EOD_Report.py``). This
*replaces* the prior ``consolidated/{date}/eod.html`` email-render archive:
the console page renders this payload, and the EOD email links to it instead
of inlining the whole report.

Alpha attribution methodology
-----------------------------
Daily dollar-alpha is decomposed on a **prior-NAV basis** into economically-
meaningful sleeves that sum EXACTLY to the headline alpha for arbitrary intraday
rotation. The headline dollar-alpha is, by construction::

    dollar_alpha = nav_change_usd - (spy_return/100) * prior_nav
                 = prior_nav * (daily_return - spy_return) / 100
                 = prior_nav * alpha / 100

Each sleeve = dollar P&L − the SPY opportunity cost on the *prior* capital that
earned it (``spy_frac := spy_return/100``)::

    position_i (held)   : daily_return_usd_i - spy_frac * prior_close_i * retained_i
    rotation (exited)   : Σ (exit_px - prior_close)*sold - spy_frac * prior_close*sold
    cash                : interest_usd       - spy_frac * idle_cash
    pricing & timing    : pricing_timing_usd                  (no SPY base)
    unattributed (true) : unattributed_usd - rotation_realized - pricing_timing

where ``retained_i := min(prior_shares_i, today_shares_i)`` (a trim's sold
portion is benchmarked in *rotation*; a same-day-entered name carries no prior
SPY base — the cash sleeve bore its opportunity cost), and ``idle_cash :=
prior_nav − Σ prior-holdings MV`` is genuine idle cash only. The SPY bases sum to
``prior_nav`` and the dollar parts sum to ``nav_change_usd`` (the EOD-reconcile
identity ``nav_change = position_pnl + interest + unattributed``), so the sleeves
sum to ``dollar_alpha`` exactly.

**Why the "pricing & timing" sleeve exists (the headline fix).** The book is
valued two ways: the headline NAV is IB ``NetLiquidation`` while per-position P&L
is settled close-to-close. Their day-over-day difference (IB intraday/unsettled
marks vs settled closes — e.g. the provisional-SPY case, config#1276) used to be
dumped wholesale into "Unattributed" (tens of bps even on no-trade days). It is
now isolated, honestly labeled, and ``Unattributed`` shrinks to the genuine
residual (untracked corporate actions / fees / FX). ``pricing_timing_usd`` is
computed by the producer as ``mark_basis(t) − mark_basis(t−1)`` where
``mark_basis = nav_ib − (cash + accrued + Σ settled_mv)`` (see
``executor/eod_reconcile.py``); when a prior input is unavailable the term is 0,
the gap stays in Unattributed, and a warning fires.

Dividends earned are folded into each position's ``daily_return_usd`` (hence into
``position_pnl_usd``); ``dividend_usd`` is informational only — so the reconcile
identity has no separate dividend term and dividends are neither double-counted
nor dropped.

This also fixes the old emailer's "α % of Total" column, which divided each
position's alpha by the *signed* grand-total alpha — so a position with genuinely
positive alpha rendered negative whenever the day's total alpha was negative, and
the table total (Σ $-alpha / NAV) never reconciled with the NAV-based headline.
See ``tests/test_eod_report.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import boto3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.1"

# Sell-side trade actions whose fills realize P&L on shares rotated out today.
_SELL_ACTIONS = {
    "SELL", "EXIT", "REDUCE", "COVER", "LIQUIDATION_SELL", "EMERGENCY_SELL",
}

# Buy-side trade actions whose fills establish entry price for shares added today.
_BUY_ACTIONS = frozenset({"ENTER", "BUY", "COVER"})

# Artifact written per trading day; the console EOD Report page reads it.
REPORT_KEY_TEMPLATE = "consolidated/{run_date}/eod_report.json"


def _prior_share_close(prior_pos: dict | None) -> tuple[float, float]:
    """Return ``(prior_shares, prior_close)`` for a prior-snapshot position.

    ``closing_price`` (the settled close persisted by the prior reconcile) is the
    canonical prior price; fall back to ``market_value / shares`` for legacy
    snapshots that predate it.
    """
    pp = prior_pos or {}
    try:
        shares = float(pp.get("shares", 0) or 0)
    except (TypeError, ValueError):
        shares = 0.0
    close = pp.get("closing_price")
    if close is None:
        try:
            mv = float(pp.get("market_value", 0) or 0)
        except (TypeError, ValueError):
            mv = 0.0
        close = (mv / shares) if shares else 0.0
    else:
        try:
            close = float(close)
        except (TypeError, ValueError):
            close = 0.0
    return shares, close


def _sell_exit_prices(trades_today: list[dict] | None) -> dict[str, float]:
    """Share-weighted average sell/exit fill price per ticker from today's trades."""
    agg: dict[str, list[float]] = {}
    for t in trades_today or []:
        action = str(t.get("action", "")).upper()
        if action not in _SELL_ACTIONS and "SELL" not in action:
            continue
        tkr = t.get("ticker")
        # Tolerate both the mapped ``price`` (``_trades_today``) and the raw
        # ``price_at_order`` column (``trade_logger.get_todays_trades``).
        raw_px = t.get("price")
        if raw_px is None:
            raw_px = t.get("price_at_order")
        try:
            sh = abs(float(t.get("shares") or 0))
            px = float(raw_px or 0)
        except (TypeError, ValueError):
            continue
        if not tkr or sh <= 0 or px <= 0:
            continue
        slot = agg.setdefault(tkr, [0.0, 0.0])
        slot[0] += sh * px
        slot[1] += sh
    return {k: v[0] / v[1] for k, v in agg.items() if v[1] > 0}


def _buy_entry_prices(trades_today: list[dict] | None) -> dict[str, float]:
    """Share-weighted average buy/enter fill price per ticker from today's trades."""
    agg: dict[str, list[float]] = {}
    for t in trades_today or []:
        action = str(t.get("action", "")).upper()
        if action not in _BUY_ACTIONS and "BUY" not in action:
            continue
        tkr = t.get("ticker")
        raw_px = t.get("price")
        if raw_px is None:
            raw_px = t.get("price_at_order")
        try:
            sh = abs(float(t.get("shares") or 0))
            px = float(raw_px or 0)
        except (TypeError, ValueError):
            continue
        if not tkr or sh <= 0 or px <= 0:
            continue
        slot = agg.setdefault(tkr, [0.0, 0.0])
        slot[0] += sh * px
        slot[1] += sh
    return {k: v[0] / v[1] for k, v in agg.items() if v[1] > 0}


def compute_rotation_realized(
    positions: dict,
    prior_positions: dict | None,
    trades_today: list[dict] | None,
) -> float:
    """Realized $ P&L on shares rotated OUT today (full exits + trims).

    ``Σ (exit_price − prior_close) × shares_sold`` over every prior-held ticker
    whose share count fell today. ``shares_sold`` is the prior-vs-today position
    delta (authoritative regardless of trade-action labels); ``exit_price`` is the
    share-weighted sell fill, falling back to the prior close (→ $0 realized,
    leaving the true realized in unattributed) when no sell fill is recorded.
    This $ is part of ``unattributed_usd`` today; the attribution lifts it into
    its own ``rotation`` sleeve so it no longer masquerades as cash drag.
    """
    exit_px = _sell_exit_prices(trades_today)
    realized = 0.0
    for tkr, pp in (prior_positions or {}).items():
        prior_shares, prior_close = _prior_share_close(pp)
        try:
            today_shares = float((positions.get(tkr) or {}).get("shares", 0) or 0)
        except (TypeError, ValueError):
            today_shares = 0.0
        sold = prior_shares - today_shares
        if sold <= 0:
            continue
        px = exit_px.get(tkr, prior_close)
        realized += (px - prior_close) * sold
    return realized


def compute_alpha_attribution(
    *,
    prior_nav: float | None,
    spy_return: float | None,
    positions: dict,
    prior_positions: dict | None,
    interest_usd: float,
    unattributed_usd: float,
    nav_change_usd: float | None,
    trades_today: list[dict] | None = None,
    pricing_timing_usd: float = 0.0,
    pricing_timing_available: bool = False,
) -> dict | None:
    """Additive daily-alpha decomposition into economically-meaningful sleeves
    that sum EXACTLY to the headline dollar-alpha for arbitrary rotation.

    Sleeves (see the module docstring for the algebra + tie-out proof):
    ``position`` (held, SPY-benchmarked on its *retained* prior MV), ``rotation``
    (shares sold out today), ``cash`` (interest − SPY on genuine idle cash),
    ``reconciliation`` ("Pricing & timing" — the IB-mark-vs-settled-close basis
    difference, no SPY base), and ``unattributed`` (the TRUE residual after
    rotation + pricing&timing are lifted out).

    Returns ``None`` when attribution is undefined (no prior NAV, or no SPY
    reference). Otherwise the ``components`` ``contrib_usd`` values sum to
    ``dollar_alpha`` (``residual_usd`` is the tie-out check).
    """
    if prior_nav is None or prior_nav <= 0 or spy_return is None:
        return None

    spy_frac = spy_return / 100.0
    nav_change = nav_change_usd or 0.0
    dollar_alpha = nav_change - spy_frac * prior_nav

    # ── Position sleeves (names held today) ───────────────────────────────────
    components: list[dict] = []
    sum_held_spy_base = 0.0
    for ticker, pos in sorted(positions.items()):
        daily_usd = float(pos.get("daily_return_usd", 0.0) or 0.0)
        prior_shares, prior_close = _prior_share_close(
            (prior_positions or {}).get(ticker)
        )
        try:
            today_shares = float(pos.get("shares", 0) or 0)
        except (TypeError, ValueError):
            today_shares = 0.0
        # Benchmark only the prior capital that was carried THROUGH the day; a
        # trim's sold portion is handled in rotation, a same-day entry has no
        # prior base (cash funded it).
        retained = min(prior_shares, today_shares)
        spy_base = prior_close * retained
        sum_held_spy_base += spy_base
        contrib = daily_usd - spy_frac * spy_base
        components.append({
            "label": ticker, "kind": "position",
            "contrib_usd": contrib, "contrib_bps": contrib / prior_nav * 1e4,
        })

    # ── Rotation sleeve (shares sold out today: full exits + trims) ───────────
    exit_px = _sell_exit_prices(trades_today)
    rotation_dollar = 0.0
    rotation_spy_base = 0.0
    rotated = False
    for tkr, pp in (prior_positions or {}).items():
        prior_shares, prior_close = _prior_share_close(pp)
        try:
            today_shares = float((positions.get(tkr) or {}).get("shares", 0) or 0)
        except (TypeError, ValueError):
            today_shares = 0.0
        sold = prior_shares - today_shares
        if sold <= 0:
            continue
        rotated = True
        px = exit_px.get(tkr, prior_close)
        rotation_dollar += (px - prior_close) * sold
        rotation_spy_base += prior_close * sold
    if rotated:
        rot_contrib = rotation_dollar - spy_frac * rotation_spy_base
        components.append({
            "label": "Rotation (exited)", "kind": "rotation",
            "contrib_usd": rot_contrib, "contrib_bps": rot_contrib / prior_nav * 1e4,
        })

    # ── Cash sleeve (genuine idle cash only) ──────────────────────────────────
    idle_cash = prior_nav - sum_held_spy_base - rotation_spy_base
    cash_contrib = float(interest_usd or 0.0) - spy_frac * idle_cash
    components.append({
        "label": "Cash", "kind": "cash",
        "contrib_usd": cash_contrib, "contrib_bps": cash_contrib / prior_nav * 1e4,
    })

    # ── Pricing & timing reconciliation (IB marks vs settled closes) ──────────
    pt = float(pricing_timing_usd or 0.0) if pricing_timing_available else 0.0
    if pricing_timing_available:
        components.append({
            "label": "Pricing & timing", "kind": "reconciliation",
            "contrib_usd": pt, "contrib_bps": pt / prior_nav * 1e4,
        })

    # ── True unattributed residual (rotation + pricing&timing lifted out) ─────
    unattr_true = float(unattributed_usd or 0.0) - rotation_dollar - pt
    components.append({
        "label": "Unattributed", "kind": "unattributed",
        "contrib_usd": unattr_true, "contrib_bps": unattr_true / prior_nav * 1e4,
    })

    summed = sum(c["contrib_usd"] for c in components)
    residual = dollar_alpha - summed

    return {
        "basis": "prior_nav",
        "prior_nav": prior_nav,
        "spy_return_pct": spy_return,
        "dollar_alpha": dollar_alpha,
        "alpha_pct": dollar_alpha / prior_nav * 100.0,
        "components": components,
        "rotation_realized_usd": rotation_dollar,
        "pricing_timing_usd": pt,
        "pricing_timing_available": bool(pricing_timing_available),
        "unattributed_true_usd": unattr_true,
        "idle_cash": idle_cash,
        "residual_usd": residual,
        "ties_to_headline": abs(residual) < 1.0,
    }


def _trades_today(conn: sqlite3.Connection, run_date: str) -> list[dict]:
    rows = conn.execute(
        "SELECT action, ticker, shares, price_at_order FROM trades "
        "WHERE date=? ORDER BY created_at",
        (run_date,),
    ).fetchall()
    return [
        {"action": a, "ticker": t, "shares": s, "price": p}
        for (a, t, s, p) in rows
    ]


def _trailing_history(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT date, portfolio_nav, daily_return_pct, spy_return_pct, "
        "daily_alpha_pct FROM eod_pnl ORDER BY date DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "date": d,
            "nav": nav,
            "daily_return_pct": ret,
            "spy_return_pct": spy,
            "daily_alpha_pct": alp,
        }
        for (d, nav, ret, spy, alp) in rows
    ]


def build_eod_report(
    *,
    run_date: str,
    nav: float,
    prior_nav: float | None,
    daily_return: float | None,
    spy_return: float | None,
    alpha: float | None,
    positions: dict,
    prior_positions: dict | None,
    conn: sqlite3.Connection,
    account_snapshot: dict | None = None,
    nav_reconciliation: dict | None = None,
    position_narratives: dict[str, str] | None = None,
    sector_attribution: dict | None = None,
    roundtrip_stats: dict | None = None,
    data_warnings: list[str] | None = None,
    generated_at: str | None = None,
    spy_close_provisional: bool = False,
) -> dict:
    """Assemble the structured EOD report payload (the ``eod_report.json`` artifact).

    ``spy_close_provisional`` marks the day's SPY close (and therefore the
    headline ``spy_return_pct``/``daily_alpha_pct``) as not-yet-settled. The
    same-day EOD run reads SPY from ArcticDB at ~4:20pm ET, which can still be
    a pre-settlement value (config#1276); the T+1 ``reconcile_audit`` pass
    re-derives it from the settled close and re-emits with the flag cleared.
    The console surfaces this so a human knows when a number is provisional.
    """
    acct = account_snapshot or {}
    recon = nav_reconciliation or {}
    narratives = position_narratives or {}

    trades_today = _trades_today(conn, run_date)
    attribution = compute_alpha_attribution(
        prior_nav=prior_nav,
        spy_return=spy_return,
        positions=positions,
        prior_positions=prior_positions,
        interest_usd=recon.get("interest_usd", 0.0) or 0.0,
        unattributed_usd=recon.get("unattributed_usd", 0.0) or 0.0,
        nav_change_usd=recon.get("nav_change_usd"),
        trades_today=trades_today,
        pricing_timing_usd=recon.get("pricing_timing_usd", 0.0) or 0.0,
        pricing_timing_available=bool(recon.get("pricing_timing_available", False)),
    )
    contrib_by_ticker = {
        c["label"]: c
        for c in (attribution["components"] if attribution else [])
        if c["kind"] == "position"
    }

    positions_out: list[dict] = []
    for ticker, pos in sorted(positions.items()):
        mv = float(pos.get("market_value", 0) or 0)
        contrib = contrib_by_ticker.get(ticker)
        positions_out.append({
            "ticker": ticker,
            "shares": pos.get("shares"),
            "market_value": mv,
            "pct_nav": (mv / nav * 100.0) if nav else None,
            "daily_return_pct": pos.get("daily_return_pct"),
            "daily_return_usd": pos.get("daily_return_usd"),
            "alpha_contrib_usd": contrib["contrib_usd"] if contrib else None,
            "alpha_contrib_bps": contrib["contrib_bps"] if contrib else None,
            "sector": pos.get("sector", "Unknown"),
            "rationale": narratives.get(ticker),
            # ── Per-ticker price-source traceability (schema 2.1) ──────────
            "prior_shares": pos.get("prior_shares"),
            "retained_shares": pos.get("retained_shares"),
            "added_shares": pos.get("added_shares"),
            "prior_price": pos.get("prior_price"),
            "entry_price": pos.get("entry_price"),
        })

    sector_out = [
        {
            "sector": sector,
            "weight_pct": data.get("weight", 0.0) * 100.0,
            "contribution_pct": data.get("contribution", 0.0),
            "positions": data.get("positions", 0),
        }
        for sector, data in sorted(
            (sector_attribution or {}).items(),
            key=lambda kv: abs(kv[1].get("contribution", 0.0)),
            reverse=True,
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "generated_at": generated_at,
        "summary": {
            "nav": nav,
            "prior_nav": prior_nav,
            "daily_return_pct": daily_return,
            "spy_return_pct": spy_return,
            "daily_alpha_pct": alpha,
            "spy_close_provisional": bool(spy_close_provisional),
            "dollar_alpha": attribution["dollar_alpha"] if attribution else None,
            "cash": acct.get("total_cash"),
            "positions_mv": acct.get("gross_position_value"),
            "unrealized_pnl": acct.get("unrealized_pnl"),
            "realized_pnl": acct.get("realized_pnl"),
            "accrued_interest": acct.get("accrued_interest"),
        },
        "nav_reconciliation": {
            "nav_change_usd": recon.get("nav_change_usd"),
            "position_pnl_usd": recon.get("position_pnl_usd"),
            "interest_usd": recon.get("interest_usd"),
            "dividend_usd": recon.get("dividend_usd"),
            "unattributed_usd": recon.get("unattributed_usd"),
            "pricing_timing_usd": recon.get("pricing_timing_usd"),
            "pricing_timing_available": recon.get("pricing_timing_available"),
            "rotation_realized_usd": (
                attribution.get("rotation_realized_usd") if attribution else None
            ),
            "unattributed_true_usd": (
                attribution.get("unattributed_true_usd") if attribution else None
            ),
        },
        "data_warnings": list(data_warnings or []),
        "alpha_attribution": attribution,
        "positions": positions_out,
        "sector_attribution": sector_out,
        "trades_today": trades_today,
        "roundtrip_stats": roundtrip_stats,
        "trailing_history": _trailing_history(conn),
    }


def write_eod_report(
    report: dict,
    *,
    trades_bucket: str,
    run_date: str,
) -> str | None:
    """Persist the report artifact to S3. Returns the key on success, else None.

    Non-fatal: a failed report write must not break EOD reconciliation. The
    failure is logged at WARNING (the console page surfaces the absence via
    its own freshness check), consistent with the artifact-archival posture
    of the old ``eod.html`` write.
    """
    if not trades_bucket:
        return None
    key = REPORT_KEY_TEMPLATE.format(run_date=run_date)
    try:
        boto3.client("s3").put_object(
            Bucket=trades_bucket,
            Key=key,
            Body=json.dumps(report, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("EOD report artifact written to s3://%s/%s", trades_bucket, key)
        return key
    except Exception as e:  # noqa: BLE001 — best-effort archival, page surfaces absence
        logger.warning("EOD report artifact write failed (non-fatal): %s", e)
        return None
