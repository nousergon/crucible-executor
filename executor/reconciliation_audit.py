"""Daemon-vs-IB reconciliation-integrity audit (config#859).

Produces ``s3://{trades_bucket}/trades/{date}/reconciliation_audit.json`` for
the evaluator report-card ``reconciliation_integrity`` component (Executor
tile, criticality=critical).

The audit compares the system's OWN trade ledger (the ``trades`` table — what
the daemon *recorded* it executed) against IB's actual broker state (the EOD
snapshot — what the broker really holds). The integrity checks:

  A. **Anchored position parity** — the headline ``reconciliation_match_rate``
     (config#1301). ``expected[t] = prior_broker_snapshot[t] + today's
     recorded ledger fills``, compared to IB's actual book today. A mismatch
     is TODAY's unexplained change — a fill the system never recorded, an
     untracked same-day corporate action, or fresh drift. Anchoring on the
     broker's own prior-day positions (ground truth) means the metric is NOT
     dominated by the pre-ledger baseline gap; it is the honest day-over-day
     integrity signal (mirrors the NAV anchoring of config#1281 / PR #296).
  B. **Cumulative-ledger parity** — a DIAGNOSTIC (``cumulative_ledger_parity``):
     net shares reconstructed from the entire ledger from inception vs IB.
     Structurally depressed by positions predating the ledger (baseline gap)
     and un-split-adjusted corporate actions — this is exactly why it was
     demoted from the headline (it graded a false ~0.20 DRIFT while the live
     book was ~aligned). Retained for operator visibility and as the
     cold-start fallback headline when no prior snapshot exists to anchor on.

DELIBERATELY NOT a NAV tautology: computing a "daemon NAV" as
``Σ(IB position market_value) + IB cash`` and comparing it to IB
``net_liquidation`` reconciles IB against ITSELF — it is structurally always
~0 and grades GREEN forever (false confidence, worse than honest N/A). The
ledger is the only independent source, so the metric is built on it.

Known reconstruction caveats (surfaced in the artifact, not hidden):
  - Positions predating the trade ledger reconstruct short (baseline gap) —
    these depress the cumulative diagnostic (B) but NOT the anchored headline
    (A), which starts from the broker's prior snapshot.
  - Corporate actions (splits/spinoffs) change IB shares with no ledger trade
    → an expected mismatch on the action day until the ledger is
    split-adjusted (follow-up); both (A) and (B) see it on that day.
  - This baseline asymmetry is exactly why (A) anchors on the prior broker
    snapshot + today's fills rather than replaying the ledger from inception.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

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


def _effective_trade_date(row: dict[str, Any]) -> Optional[str]:
    """The calendar date a trade row should be attributed to for a
    day-level ledger-vs-IB comparison: the ACTUAL fill timestamp's calendar
    date when available, else the ledger's ``date`` tag (legacy rows with no
    ``fill_time``).

    config#1454 (position-reconciliation trading_day-vs-calendar artifact):
    an order tagged ``date``/``trading_day`` = D that actually fills on D+1
    (e.g. a late/overnight execution reported the next morning) previously
    made the (B) daily-delta / anchored-parity check compare IB's actual
    book — which only reflects fills that settled by that day's close —
    against a ledger delta keyed on the ORDER's trading_day tag rather than
    the fill's real date. That produced a false mismatch on BOTH D (ledger
    counted a fill IB hadn't received yet) and D+1 (IB reflected a fill the
    ledger attributed to D). Keying on the fill's own timestamp instead of
    the trading_day tag fixes the root cause: the ledger delta for a given
    day now reflects exactly the fills IB could have received by that day.
    """
    fill_time = row.get("fill_time")
    if fill_time:
        try:
            return pd.Timestamp(fill_time).date().isoformat()
        except (ValueError, TypeError):
            pass
    return row.get("date")


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

    Both bounds are evaluated against each row's EFFECTIVE date — the real
    fill timestamp's calendar date (``_effective_trade_date``), not the
    ledger's ``date``/``trading_day`` tag. This closes the config#1454
    trading_day-vs-calendar artifact: a trade tagged trading_day=D that
    actually filled on D+1 must compare against IB's book on D+1 (when the
    fill was real), not D (when it wasn't in IB's book yet).

    Deliberately NOT narrowed by a SQL date filter first: the tag and the
    true fill date can in principle drift by more than a day (e.g. a fill
    reported several days late), and a tight SQL pre-filter would silently
    reproduce exactly the bug this function fixes by excluding those rows
    before the Python effective-date check ever sees them. The full-table
    scan is deliberate and cheap at this executor's single-portfolio ledger
    scale; correctness here matters more than a SQL-side bound.
    """
    if as_of_date and on_date:
        raise ValueError("pass at most one of as_of_date / on_date")

    sql = (
        "SELECT ticker, action, shares, filled_shares, status, date, "
        "fill_time FROM trades"
    )
    net: dict[str, int] = {}
    for ticker, action, shares, filled_shares, status, date_, fill_time in conn.execute(sql):
        if not ticker:
            continue
        eff_date = _effective_trade_date({"date": date_, "fill_time": fill_time})
        if on_date is not None and eff_date != on_date:
            continue
        if as_of_date is not None and (eff_date is None or eff_date > as_of_date):
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


def same_day_split_ratios(
    splits_by_ticker: dict[str, list[dict[str, Any]]], run_date: str
) -> dict[str, float]:
    """Map ``{ticker: cumulative split ratio}`` for splits whose ex-date
    (``execution_date``) is exactly ``run_date`` (config#1682).

    ``splits_by_ticker`` is the raw Polygon ``/v3/reference/splits`` payload
    keyed by ticker (see :meth:`PolygonClient.get_splits`). The ratio is
    ``split_to / split_from`` — the factor IB multiplies the held share count
    by on the ex-date (2-for-1 → 2.0; 1-for-10 reverse → 0.1). If a ticker has
    more than one split dated the same day (rare) the ratios compound. Malformed
    rows (missing / zero / non-numeric ``split_from``/``split_to``) are skipped
    with a warning rather than silently defaulting to 1.0 — a swallowed bad
    ratio would re-introduce exactly the false mismatch this closes.

    Pure: no I/O. The caller fetches the splits (see
    :func:`fetch_same_day_split_ratios`) and passes the result in, keeping
    :func:`build_reconciliation_audit` free of network calls.
    """
    ratios: dict[str, float] = {}
    for ticker, splits in (splits_by_ticker or {}).items():
        for s in splits or []:
            if s.get("execution_date") != run_date:
                continue
            try:
                to, frm = float(s["split_to"]), float(s["split_from"])
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "[reconciliation_audit] malformed split for %s on %s: %r "
                    "— skipping (not defaulting to 1.0)", ticker, run_date, s,
                )
                continue
            if to <= 0 or frm <= 0:
                logger.warning(
                    "[reconciliation_audit] non-positive split ratio for %s on "
                    "%s: split_to=%s split_from=%s — skipping", ticker, run_date,
                    s.get("split_to"), s.get("split_from"),
                )
                continue
            ratios[ticker] = ratios.get(ticker, 1.0) * (to / frm)
    return ratios


def _apply_split_ratios(
    shares: dict[str, int], ratios: dict[str, float]
) -> dict[str, int]:
    """Rebase a pre-corporate-action share map onto today's post-action basis.

    For each ticker with a same-day split ratio, the pre-action holding
    (the prior broker snapshot, or the from-inception ledger reconstruction)
    is multiplied by the ratio so it lines up with IB's post-split book. Tickers
    with no same-day action pass through unchanged. Returns a new dict (does not
    mutate the input).
    """
    if not ratios:
        return dict(shares)
    out = dict(shares)
    for ticker, ratio in ratios.items():
        if ticker in out:
            out[ticker] = int(round(out[ticker] * ratio))
    return out


def fetch_same_day_split_ratios(
    tickers, run_date: str, *, client: Any | None = None
) -> dict[str, float]:
    """I/O wiring helper: fetch same-day split ratios for ``tickers`` via Polygon.

    Separate from the pure :func:`build_reconciliation_audit` builder so the
    latter stays testable without network. Best-effort by contract — any failure
    (missing ``POLYGON_API_KEY``, rate limit, HTTP error) returns ``{}`` and logs,
    because a same-day split is rare and the reconciliation audit is secondary
    observability that must never abort the EOD path. Inject ``client`` (a
    ``PolygonClient``-shaped object exposing ``get_splits``) in tests.
    """
    tickers = [t for t in (tickers or []) if t]
    if not tickers:
        return {}
    try:
        if client is None:
            from polygon_client import PolygonClient  # lazy: optional dep / key

            client = PolygonClient()
        splits_by_ticker: dict[str, list[dict[str, Any]]] = {}
        for t in tickers:
            try:
                splits_by_ticker[t] = client.get_splits(t, start=run_date)
            except Exception:  # noqa: BLE001 — per-ticker isolation
                logger.warning(
                    "[reconciliation_audit] split fetch failed for %s; "
                    "treating as no same-day action", t, exc_info=True,
                )
        return same_day_split_ratios(splits_by_ticker, run_date)
    except Exception:  # noqa: BLE001 — client construction / key absent
        logger.warning(
            "[reconciliation_audit] same-day split fetch unavailable "
            "(no POLYGON_API_KEY or client error); proceeding un-adjusted",
            exc_info=True,
        )
        return {}


def _anchored_parity(
    prior_ib: dict[str, int], ib: dict[str, int], ledger_today: dict[str, int]
) -> tuple[float, list[str], int, list[dict[str, Any]]]:
    """Absolute-position parity ANCHORED on the prior broker snapshot + today's
    recorded fills (config#1301).

    ``expected[t] = prior_ib[t] + ledger_today[t]`` is what the system should
    hold today IF its ledger captured every fill, started from the broker's
    own prior-day EOD positions (ground truth). A mismatch vs the actual IB
    book is therefore TODAY's unexplained change — an unrecorded fill, an
    untracked same-day corporate action, or fresh drift — NOT an artifact of
    pre-ledger history. This trusts the broker's prior snapshot as the
    baseline rather than replaying the ledger from inception, mirroring the
    prior-snapshot anchoring shipped for NAV in config#1281 (PR #296).

    Returns ``(match_rate, universe, n_matched, mismatches)``.
    """
    universe = sorted(set(prior_ib) | set(ib) | set(ledger_today))
    mismatches: list[dict[str, Any]] = []
    n_matched = 0
    for t in universe:
        expected = prior_ib.get(t, 0) + ledger_today.get(t, 0)
        actual = ib.get(t, 0)
        if abs(expected - actual) <= _RECON_TOLERANCE_SHARES:
            n_matched += 1
        else:
            mismatches.append({
                "ticker": t,
                "prior_ib_shares": prior_ib.get(t, 0),
                "ledger_today_shares": ledger_today.get(t, 0),
                "expected_shares": expected,
                "ib_shares": actual,
                "delta": actual - expected,
                "kind": (
                    "ib_only" if expected == 0 else
                    "missing_in_ib" if actual == 0 else "share_mismatch"
                ),
            })
    rate = 1.0 if not universe else round(n_matched / len(universe), 4)
    return rate, universe, n_matched, mismatches


def _cumulative_parity(
    ledger: dict[str, int], ib: dict[str, int]
) -> tuple[float, list[str], int, list[dict[str, Any]]]:
    """Full-ledger-replay parity: net shares reconstructed from the ENTIRE
    trades ledger (from inception) vs IB's current book.

    Structurally depressed by positions predating the ledger (baseline gap)
    and un-split-adjusted corporate actions, so it is no longer the headline
    metric (config#1301) — retained as a diagnostic and as the cold-start
    fallback when no prior broker snapshot is available to anchor on.

    Returns ``(match_rate, universe, n_matched, mismatches)``.
    """
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
    rate = 1.0 if not universe else round(n_matched / len(universe), 4)
    return rate, universe, n_matched, mismatches


def build_reconciliation_audit(
    conn,
    *,
    today_positions: dict[str, Any],
    prior_positions: Optional[dict[str, Any]],
    run_date: str,
    ib_nav: Optional[float] = None,
    generated_at: Optional[str] = None,
    corporate_actions: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Build the reconciliation-audit payload (pure — no I/O).

    The headline ``reconciliation_match_rate`` (the report-card metric) is
    ANCHORED parity (config#1301): the fraction of tickers where the broker's
    prior-day positions plus today's recorded ledger fills equal IB's actual
    book today. This is the honest day-over-day integrity signal — it catches
    unrecorded fills and fresh drift while NOT penalising the pre-ledger
    baseline gap or un-split-adjusted corporate actions that structurally
    depressed the old from-inception replay (which graded a false ~0.20 DRIFT
    while the live book was ~aligned).

    Anchoring requires a prior broker snapshot. When ``prior_positions`` is
    ``None`` (genuinely unavailable — e.g. the very first EOD, or a missing/
    corrupt snapshot), the headline falls back to the from-inception
    ``cumulative`` replay (``anchored: false``) so the metric is never built
    on a phantom empty baseline. The cumulative replay is ALWAYS computed and
    surfaced under ``cumulative_ledger_parity`` as a diagnostic.

    ``corporate_actions`` (config#1682) is ``{ticker: same-day split ratio}``
    for splits/spinoffs whose ex-date is ``run_date`` — actions that change IB's
    share count with no corresponding ledger trade and would otherwise register
    as false mismatches on the action day. The ratio (``split_to/split_from``,
    e.g. 2.0 for a 2-for-1) rebases the *pre-action* share baselines onto today's
    post-action IB basis: the prior broker snapshot for the anchored headline,
    and the from-inception reconstruction for the cumulative diagnostic. Today's
    ledger fills are already recorded in post-split terms (IB reports post-split
    quantities), so only the pre-action carry is rebased. The applied ratios are
    echoed under ``corporate_actions_applied``. Fetched out-of-band and passed in
    (see :func:`fetch_same_day_split_ratios`) to keep this builder pure.
    """
    ratios = corporate_actions or {}
    ib = _ib_shares(today_positions)
    ledger_cum = _apply_split_ratios(
        reconstruct_ledger_positions(conn, as_of_date=run_date), ratios
    )
    cum_rate, cum_universe, cum_matched, cum_mismatches = _cumulative_parity(
        ledger_cum, ib
    )

    anchored = prior_positions is not None
    daily: dict[str, Any] = {"computed": False}
    if anchored:
        prior_ib = _apply_split_ratios(_ib_shares(prior_positions), ratios)
        ledger_today = reconstruct_ledger_positions(conn, on_date=run_date)
        match_rate, universe, n_matched, mismatches = _anchored_parity(
            prior_ib, ib, ledger_today
        )
        position_parity = {
            "basis": "anchored",
            "prior_ib_positions": prior_ib,
            "ledger_today": ledger_today,
            "ib_positions": ib,
            "mismatches": mismatches,
        }
        # daily_delta is the delta-view of the same anchored reconciliation
        # (expected==actual ⟺ ib_delta==ledger_delta) — kept for consumers
        # that read the per-day signal directly.
        daily = {
            "computed": True,
            "match_rate": match_rate,
            "n_tickers": len(universe),
            "n_matched": n_matched,
            "mismatches": [
                {
                    "ticker": m["ticker"],
                    "ib_delta": m["ib_shares"] - m["prior_ib_shares"],
                    "ledger_delta": m["ledger_today_shares"],
                }
                for m in mismatches
            ],
        }
    else:
        # Cold-start fallback: no prior snapshot to anchor on.
        match_rate, universe, n_matched, mismatches = (
            cum_rate, cum_universe, cum_matched, cum_mismatches
        )
        position_parity = {
            "basis": "cumulative_ledger",
            "ledger_positions": ledger_cum,
            "ib_positions": ib,
            "mismatches": mismatches,
        }

    status = "OK" if match_rate >= 1.0 else "DRIFT"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "date": run_date,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        # Headline metric consumed by the evaluator reconciliation_integrity grader.
        "reconciliation_match_rate": match_rate,
        "anchored": anchored,
        "status": status,
        "n_positions": len(universe),
        "n_matched": n_matched,
        "n_mismatched": len(mismatches),
        "position_parity": position_parity,
        "daily_delta": daily,
        # Diagnostic only (config#1301): from-inception ledger replay vs IB.
        # Structurally depressed by the pre-ledger baseline gap + un-split-
        # adjusted corporate actions — NOT the headline. Surfaced so operators
        # can still see the cumulative divergence and its drivers.
        "cumulative_ledger_parity": {
            "match_rate": cum_rate,
            "n_positions": len(cum_universe),
            "n_matched": cum_matched,
            "n_mismatched": len(cum_mismatches),
            "mismatches": cum_mismatches,
            "note": (
                "Full-ledger replay from inception vs IB; structurally "
                "depressed by positions predating the ledger (baseline gap) "
                "and un-split-adjusted corporate actions. Diagnostic only — "
                "the headline reconciliation_match_rate anchors on the prior "
                "broker snapshot + today's fills instead."
            ),
        },
        # Informational only — NAV parity is NOT the metric (an IB-derived
        # daemon NAV would be a tautology). Recorded for operator context.
        "ib_nav": ib_nav,
        # Same-day split ratios applied to the pre-action baselines (config#1682).
        # Empty {} when no ticker had an ex-date == run_date corporate action.
        "corporate_actions_applied": dict(ratios),
        "caveats": [
            "Headline parity anchors on the prior broker snapshot + today's "
            "recorded fills (config#1301): a mismatch is TODAY's unexplained "
            "change, not pre-ledger history.",
            (
                "Same-day corporate actions (splits/spinoffs) change IB shares "
                "with no ledger trade; their split ratios are applied to the "
                "pre-action baselines (prior snapshot + from-inception ledger) "
                "so they no longer register as false mismatches (config#1682). "
                "See corporate_actions_applied for what was rebased on this day."
            )
            if ratios else (
                "Same-day corporate actions (splits/spinoffs) change IB shares "
                "with no ledger trade; none had an ex-date on this run_date, so "
                "no split adjustment was applied (config#1682)."
            ),
            "When no prior broker snapshot is available the headline falls "
            "back to the from-inception cumulative replay (anchored=false); "
            "see cumulative_ledger_parity for the always-computed diagnostic.",
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
