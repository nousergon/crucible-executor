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
Daily dollar-alpha is decomposed on a **prior-NAV basis** so the per-sleeve
contributions sum EXACTLY to the headline alpha (``port_return - spy_return``).
The headline dollar-alpha is, by construction::

    dollar_alpha = nav_change_usd - (spy_return/100) * prior_nav
                 = prior_nav * (daily_return - spy_return) / 100
                 = prior_nav * alpha / 100

Each sleeve's additive contribution::

    position_i        : daily_return_usd_i - (spy_return/100) * prior_mv_i
    cash & rotation   : interest_usd       - (spy_return/100) * prior_cash_residual
    unattributed      : unattributed_usd

where ``prior_cash_residual := prior_nav - Σ prior_mv_i`` (over *today's* held
tickers) is defined as the residual sleeve so the prior weights sum to
``prior_nav`` exactly. Because ``nav_change_usd = position_pnl_usd +
interest_usd + unattributed_usd`` (the EOD-reconcile identity), the sleeve
contributions sum to ``dollar_alpha`` regardless of intraday rotation — capital
that was in positions exited today lands in the cash-&-rotation sleeve, so the
identity still closes. See ``tests/test_eod_report.py::
test_attribution_ties_to_headline``.

This is the fix for the old emailer's "α % of Total" column, which divided each
position's alpha by the *signed* grand-total alpha — so a position with genuinely
positive alpha rendered negative whenever the day's total alpha was negative, and
the table total (Σ $-alpha / NAV) never reconciled with the NAV-based headline.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import boto3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Artifact written per trading day; the console EOD Report page reads it.
REPORT_KEY_TEMPLATE = "consolidated/{run_date}/eod_report.json"


def compute_alpha_attribution(
    *,
    prior_nav: float | None,
    spy_return: float | None,
    positions: dict,
    prior_positions: dict | None,
    interest_usd: float,
    unattributed_usd: float,
    nav_change_usd: float | None,
) -> dict | None:
    """Additive daily-alpha decomposition that ties to the headline alpha.

    Returns ``None`` when attribution is undefined (first trading day with no
    prior NAV, or no SPY reference). Otherwise returns a dict with a
    ``components`` list whose ``contrib_usd`` values sum to ``dollar_alpha``
    (within floating-point tolerance), plus a ``residual_usd`` tie-out check.
    """
    if prior_nav is None or prior_nav <= 0 or spy_return is None:
        return None

    spy_frac = spy_return / 100.0
    dollar_alpha = prior_nav * ((nav_change_usd or 0.0) / prior_nav - spy_frac)

    def _prior_mv(tkr: str) -> float:
        pp = (prior_positions or {}).get(tkr) or {}
        try:
            return float(pp.get("market_value", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    sum_prior_mv = sum(_prior_mv(t) for t in positions)
    # Residual sleeve: prior cash + the prior market value of any position
    # exited/rotated out today. Defined so Σ prior weights == prior_nav exactly.
    prior_cash_residual = prior_nav - sum_prior_mv

    components: list[dict] = []
    for ticker, pos in sorted(positions.items()):
        daily_usd = float(pos.get("daily_return_usd", 0.0) or 0.0)
        contrib = daily_usd - spy_frac * _prior_mv(ticker)
        components.append({
            "label": ticker,
            "kind": "position",
            "contrib_usd": contrib,
            "contrib_bps": contrib / prior_nav * 1e4,
        })

    cash_contrib = float(interest_usd or 0.0) - spy_frac * prior_cash_residual
    components.append({
        "label": "Cash & rotation",
        "kind": "cash",
        "contrib_usd": cash_contrib,
        "contrib_bps": cash_contrib / prior_nav * 1e4,
    })

    unattr = float(unattributed_usd or 0.0)
    components.append({
        "label": "Unattributed",
        "kind": "unattributed",
        "contrib_usd": unattr,
        "contrib_bps": unattr / prior_nav * 1e4,
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
) -> dict:
    """Assemble the structured EOD report payload (the ``eod_report.json`` artifact)."""
    acct = account_snapshot or {}
    recon = nav_reconciliation or {}
    narratives = position_narratives or {}

    attribution = compute_alpha_attribution(
        prior_nav=prior_nav,
        spy_return=spy_return,
        positions=positions,
        prior_positions=prior_positions,
        interest_usd=recon.get("interest_usd", 0.0) or 0.0,
        unattributed_usd=recon.get("unattributed_usd", 0.0) or 0.0,
        nav_change_usd=recon.get("nav_change_usd"),
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
        },
        "data_warnings": list(data_warnings or []),
        "alpha_attribution": attribution,
        "positions": positions_out,
        "sector_attribution": sector_out,
        "trades_today": _trades_today(conn, run_date),
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
