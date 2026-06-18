"""Daemon-vs-IB reconciliation-integrity audit (config#859).

Produces ``s3://{trades_bucket}/trades/{date}/reconciliation_audit.json`` for
the evaluator report-card ``reconciliation_integrity`` component (Executor
tile, criticality=critical).

The audit compares the system's OWN trade ledger (the ``trades`` table — what
the daemon *recorded* it executed) against IB's actual broker state (the EOD
snapshot — what the broker really holds). Two independent checks:

  A. **Position parity** — the headline ``reconciliation_match_rate``.
     Reconstruct net shares per ticker from the ledger (Σ signed
     ``filled_shares``) and compare to IB's reported positions. Catches the
     failures worth catching: fills the system never recorded, positions that
     drifted, the daemon believing it holds X while IB holds Y.
  B. **Daily-delta integrity** — a supporting per-day signal. Today's IB
     position changes (today snapshot vs prior snapshot) vs today's recorded
     ledger fills. Sidesteps the historical-baseline problem in (A).

DELIBERATELY NOT a NAV tautology: computing a "daemon NAV" as
``Σ(IB position market_value) + IB cash`` and comparing it to IB
``net_liquidation`` reconciles IB against ITSELF — it is structurally always
~0 and grades GREEN forever (false confidence, worse than honest N/A). The
ledger is the only independent source, so the metric is built on it.

Known reconstruction caveats (surfaced in the artifact, not hidden):
  - Positions predating the trade ledger reconstruct short (baseline gap).
  - Corporate actions (splits/spinoffs) change IB shares with no ledger
    trade → an expected mismatch until the ledger is split-adjusted.
  - These are why (B)'s daily-delta — which needs no historical baseline —
    is reported alongside (A).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Trade-action sign conventions (see executor/daemon.py action vocabulary).
_BUY_ACTIONS = frozenset({"ENTER", "COVER", "BUY"})
_SELL_ACTIONS = frozenset(
    {"EXIT", "REDUCE", "SELL", "LIQUIDATION_SELL", "EMERGENCY_SELL"}
)
# Statuses that mean the order did NOT result in shares changing hands —
# excluded from ledger reconstruction. Anything else (Filled / ok / legacy
# NULL) contributes its filled_shares (or intended shares as a fallback).
_NON_FILL_STATUSES = frozenset(
    {"Rejected", "rejected", "error", "failed", "cancelled", "Cancelled", "pending"}
)

_RECON_TOLERANCE_SHARES = 0  # exact share parity; whole-share equities


def _shares_contributed(row: dict[str, Any]) -> int:
    """Signed shares a single trade row contributes to a ticker's net position.

    Uses the ACTUAL ``filled_shares`` when present (the real executed
    quantity, incl. partial/zero fills); falls back to the intended
    ``shares`` only for legacy rows that predate the filled_shares column,
    and only when the status doesn't mark it a non-fill. Unknown actions
    contribute 0 (logged) rather than silently guessing a sign.
    """
    status = row.get("status")
    if status in _NON_FILL_STATUSES:
        return 0
    filled = row.get("filled_shares")
    qty = filled if filled is not None else row.get("shares")
    if not qty:
        return 0
    action = (row.get("action") or "").upper()
    if action in _BUY_ACTIONS:
        return int(qty)
    if action in _SELL_ACTIONS:
        return -int(qty)
    logger.warning(
        "[reconciliation_audit] unknown trade action %r (ticker=%s) — "
        "contributing 0 shares; add it to _BUY_ACTIONS/_SELL_ACTIONS",
        row.get("action"), row.get("ticker"),
    )
    return 0


def reconstruct_ledger_positions(
    conn, *, as_of_date: Optional[str] = None, on_date: Optional[str] = None
) -> dict[str, int]:
    """Net shares per ticker reconstructed from the ``trades`` ledger.

    ``as_of_date`` (inclusive upper bound) reconstructs the cumulative
    position through that date — used for the (A) position-parity check.
    ``on_date`` restricts to trades dated exactly that day — used for the
    (B) daily-delta check. Pass at most one. Tickers netting to 0 are
    dropped.
    """
    if as_of_date and on_date:
        raise ValueError("pass at most one of as_of_date / on_date")
    sql = "SELECT ticker, action, shares, filled_shares, status FROM trades"
    params: tuple = ()
    if on_date is not None:
        sql += " WHERE date = ?"
        params = (on_date,)
    elif as_of_date is not None:
        sql += " WHERE date <= ?"
        params = (as_of_date,)
    net: dict[str, int] = {}
    for ticker, action, shares, filled_shares, status in conn.execute(sql, params):
        if not ticker:
            continue
        delta = _shares_contributed({
            "ticker": ticker, "action": action, "shares": shares,
            "filled_shares": filled_shares, "status": status,
        })
        if delta:
            net[ticker] = net.get(ticker, 0) + delta
    return {t: s for t, s in net.items() if s != 0}


def _ib_shares(positions: dict[str, Any]) -> dict[str, int]:
    """Extract {ticker: int shares} from an IB positions snapshot, dropping
    zero/closed positions."""
    out: dict[str, int] = {}
    for ticker, pos in (positions or {}).items():
        shares = pos.get("shares") if isinstance(pos, dict) else None
        if shares:
            out[ticker] = int(round(float(shares)))
    return out


def build_reconciliation_audit(
    conn,
    *,
    today_positions: dict[str, Any],
    prior_positions: Optional[dict[str, Any]],
    run_date: str,
    ib_nav: Optional[float] = None,
    generated_at: Optional[str] = None,
) -> dict[str, Any]:
    """Build the reconciliation-audit payload (pure — no I/O).

    ``reconciliation_match_rate`` (the report-card metric) is the fraction of
    positions where the ledger-reconstructed net shares equal IB's reported
    shares, over the UNION of tickers that either side holds. A ticker IB
    holds but the ledger can't explain (or vice-versa) is a mismatch — that
    is the integrity signal.
    """
    ledger = reconstruct_ledger_positions(conn, as_of_date=run_date)
    ib = _ib_shares(today_positions)

    # ── A. Position parity (headline) ──
    universe = sorted(set(ledger) | set(ib))
    mismatches: list[dict[str, Any]] = []
    n_matched = 0
    for t in universe:
        lg, ibq = ledger.get(t, 0), ib.get(t, 0)
        if abs(lg - ibq) <= _RECON_TOLERANCE_SHARES:
            n_matched += 1
        else:
            mismatches.append({
                "ticker": t, "ledger_shares": lg, "ib_shares": ibq,
                "delta": ibq - lg,
                "kind": (
                    "ib_only" if lg == 0 else
                    "ledger_only" if ibq == 0 else "share_mismatch"
                ),
            })
    # Empty universe (no positions either side) is a vacuous match → 1.0.
    match_rate = 1.0 if not universe else round(n_matched / len(universe), 4)

    # ── B. Daily-delta integrity (supporting) ──
    daily: dict[str, Any] = {"computed": False}
    if prior_positions is not None:
        prior_ib = _ib_shares(prior_positions)
        ledger_today = reconstruct_ledger_positions(conn, on_date=run_date)
        d_universe = sorted(set(prior_ib) | set(ib) | set(ledger_today))
        d_mismatches: list[dict[str, Any]] = []
        d_matched = 0
        for t in d_universe:
            ib_delta = ib.get(t, 0) - prior_ib.get(t, 0)
            led_delta = ledger_today.get(t, 0)
            if ib_delta == led_delta:
                d_matched += 1
            elif ib_delta != 0 or led_delta != 0:
                d_mismatches.append({
                    "ticker": t, "ib_delta": ib_delta, "ledger_delta": led_delta,
                })
            else:
                d_matched += 1
        daily = {
            "computed": True,
            "match_rate": (
                1.0 if not d_universe else round(d_matched / len(d_universe), 4)
            ),
            "n_tickers": len(d_universe),
            "n_matched": d_matched,
            "mismatches": d_mismatches,
        }

    status = "OK" if match_rate >= 1.0 else "DRIFT"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "date": run_date,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        # Headline metric consumed by the evaluator reconciliation_integrity grader.
        "reconciliation_match_rate": match_rate,
        "status": status,
        "n_positions": len(universe),
        "n_matched": n_matched,
        "n_mismatched": len(mismatches),
        "position_parity": {
            "ledger_positions": ledger,
            "ib_positions": ib,
            "mismatches": mismatches,
        },
        "daily_delta": daily,
        # Informational only — NAV parity is NOT the metric (an IB-derived
        # daemon NAV would be a tautology). Recorded for operator context.
        "ib_nav": ib_nav,
        "caveats": [
            "Ledger reconstruction nets signed filled_shares from the trades "
            "table; positions predating the ledger reconstruct short.",
            "Corporate actions (splits/spinoffs) change IB shares with no "
            "ledger trade and surface as expected mismatches until the ledger "
            "is split-adjusted — cross-check the daily_delta signal, which "
            "needs no historical baseline.",
        ],
    }
    return payload


def write_reconciliation_audit(
    payload: dict[str, Any], *, bucket: str, run_date: str,
    region: str = "us-east-1", s3_client: Any | None = None,
) -> str:
    """Write the audit payload to ``trades/{run_date}/reconciliation_audit.json``.

    Returns the S3 key written. Sibling of ``trades/execution_quality`` /
    ``trades/{date}/`` artifacts.
    """
    import boto3

    key = f"trades/{run_date}/reconciliation_audit.json"
    client = s3_client if s3_client is not None else boto3.client("s3", region_name=region)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    return key
