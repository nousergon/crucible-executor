"""Reference-rate showcase artifact — the public, illustrative-only cross-repo
contract Metron consumes to render the "Reference Rate" demo portfolio.

This is a PRODUCT CONTRACT (per the M0 slot discipline in ~/Development/CLAUDE.md):
a versioned, purpose-built artifact at a stable key. Metron MUST consume THIS, never
reach around it into the executor's producer-private `trades/snapshots/` path.

Disclosure scope (deliberately minimal — illustrative only, no claims):
  * current positions (ticker / shares / avg_cost / market_value / sector / asset_type)
  * portfolio NAV (net_liquidation only)
  * a NAV-vs-SPY history series (the illustrative performance curve)

What it MUST NOT carry (kept private — strategy edge / assumptions / internals):
  * scoring weights, model params, predictions, signal logic
  * buying power, settled cash, realized/unrealized P&L attribution, accrued interest

No performance claims and no stated objective live in this artifact or its consumer
copy — it is a "reference rate", illustrative and demo-only.

S3 path: s3://alpha-engine-research/metron/reference_rate.json

Schema (additive-only per CLAUDE.md S3 contract):
    {
      "schema_version": 1,
      "as_of": "YYYY-MM-DD",            # run_date (trading_day)
      "generated_at": ISO8601 UTC,
      "label": "Reference Rate",
      "disclaimer": str,                # illustrative-only, no-claims notice
      "base_currency": "USD",
      "account": {"net_liquidation": float},
      "positions": [
        {"ticker": str, "shares": float, "avg_cost": float,
         "market_value": float, "sector": str, "asset_type": "STK" | "ETF"}
      ],
      "nav_history": [
        {"date": "YYYY-MM-DD", "nav": float, "spy_close": float | null}
      ],
    }
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REFERENCE_RATE_KEY = "metron/reference_rate.json"
LABEL = "Reference Rate"
DISCLAIMER = (
    "Illustrative reference portfolio. Not investment advice; "
    "no representation is made as to performance."
)

# Broad-market index/ETF core positions the portfolio-optimizer holds as the
# no-conviction fill (SPY primarily; VOO/IVV/SPLG are substitutes the optimizer
# may swap to). Single source of truth shared with ``eod_reconcile._index_etf_sector``
# — a ticker added here is simultaneously tagged with the right sector label AND
# reported to Metron as ETF, not equity. New core-ETF substitutes get added here.
INDEX_ETF_TICKERS = frozenset({"SPY", "VOO", "IVV", "SPLG"})


def _asset_type_for(ticker: str) -> str:
    """IBKR-vocabulary asset category (STK/ETF) for a held ticker."""
    return "ETF" if ticker in INDEX_ETF_TICKERS else "STK"

# History depth published in the artifact (trading days). ~2y is enough for the
# illustrative NAV-vs-SPY curve without bloating the object; the executor's full
# eod_pnl history stays internal.
_NAV_HISTORY_MAX_ROWS = 504


def build_payload(
    positions: dict[str, dict[str, Any]],
    nav: float | None,
    nav_history: list[dict[str, Any]],
    run_date: str,
) -> dict[str, Any]:
    """Build the reference-rate artifact from the EOD reconcile's in-memory state.

    Pure (no I/O) so it is unit-testable. ``positions`` is the enriched EOD positions
    dict (ticker -> {shares, market_value, avg_cost, sector, ...}); only the disclosed
    subset of fields is copied through. ``nav_history`` is a list of
    {date, nav, spy_close} rows (oldest-first); it is truncated to the most recent
    ``_NAV_HISTORY_MAX_ROWS``. Internal attribution fields (daily_return_*,
    alpha_contribution_*, unrealized_pnl) are intentionally NOT copied.
    """
    out_positions: list[dict[str, Any]] = []
    for ticker, pos in sorted(positions.items()):
        shares = pos.get("shares")
        if not shares:  # closed / zero-share rows are not holdings
            continue
        out_positions.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": pos.get("avg_cost"),
                "market_value": pos.get("market_value"),
                "sector": pos.get("sector") or "Unknown",
                "asset_type": _asset_type_for(ticker),
            }
        )

    history = [
        {"date": row["date"], "nav": row.get("nav"), "spy_close": row.get("spy_close")}
        for row in nav_history
        if row.get("date") is not None and row.get("nav") is not None
    ]
    history = history[-_NAV_HISTORY_MAX_ROWS:]

    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": run_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "label": LABEL,
        "disclaimer": DISCLAIMER,
        "base_currency": "USD",
        "account": {"net_liquidation": nav},
        "positions": out_positions,
        "nav_history": history,
    }


def nav_history_from_eod_df(eod_df) -> list[dict[str, Any]]:
    """Extract the {date, nav, spy_close} history rows from the eod_pnl DataFrame.

    Tolerant of missing columns / NaNs — a row with no NAV is dropped (the artifact
    never fabricates a value). Oldest-first; ``build_payload`` truncates the tail.
    """
    rows: list[dict[str, Any]] = []
    if eod_df is None or len(eod_df) == 0:
        return rows
    for _, r in eod_df.iterrows():
        nav = r.get("portfolio_nav")
        date_val = r.get("date")
        if date_val is None or nav is None or (isinstance(nav, float) and nav != nav):
            continue
        spy = r.get("spy_close")
        spy = None if (spy is None or (isinstance(spy, float) and spy != spy)) else float(spy)
        rows.append({"date": str(date_val), "nav": float(nav), "spy_close": spy})
    return rows


def publish(s3, bucket: str, payload: dict[str, Any]) -> None:
    """Write the artifact to S3. Raises on failure — the caller wraps this best-effort
    (it is secondary observability hung off the already-committed EOD path)."""
    s3.put_object(
        Bucket=bucket,
        Key=REFERENCE_RATE_KEY,
        Body=json.dumps(payload, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(
        "Reference-rate artifact written | s3://%s/%s as_of=%s positions=%d nav_history=%d",
        bucket,
        REFERENCE_RATE_KEY,
        payload.get("as_of"),
        len(payload.get("positions", [])),
        len(payload.get("nav_history", [])),
    )
