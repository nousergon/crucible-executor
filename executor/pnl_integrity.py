"""Performance-measurement integrity gates for the EOD P&L path.

Every function here is PURE (no IO, no IB, no S3, no SQLite) so each gate is
unit-testable without mocking ``eod_reconcile.run``'s IO. ``eod_reconcile``
owns dispatch — logging, flow-doctor paging, and raising.

Why this module exists (alpha-engine-config-I8188). Four measured defects, all
against ``s3://alpha-engine-research/trades/eod_pnl.csv`` (115 sessions,
2026-03-09 → 2026-08-21):

1. ``nav_change_usd = position_pnl_usd + interest_usd + dividend_usd +
   unattributed_usd`` held on every session **because ``unattributed_usd`` is
   defined as the remainder**. The identity was tautological and could never
   fail, so −$20,293 cumulative accumulated with nothing firing.
2. Chain-linking ``daily_return_pct`` gave +3.8261% against a NAV ratio of
   +3.6526% — 17.4bp of drift that is identically zero by construction absent
   external flows.
3. There was no transaction-cost line anywhere in the schema, so gross and net
   performance were the same number and neither was labelled.
4. ``ib_mark_outside_range`` existed but only flagged: 24 of 229 position-days
   (10.5%) differed from the broker mark by more than 0.5% with nothing
   failing.

A fifth defect, measured against the same file (alpha-engine-config-I9025),
sits beside defect 2 rather than inside it: defect 2's TWR-closure gate
compares the chain-linked ``daily_return_pct`` series against the NAV RATIO
(``nav[-1]/nav[0] − 1``); it never touches the chain-linked
``nav_change_usd / prior_nav`` series, which is a different pair. Reconstructed
over the full 119-session history (2026-03-09 → 2026-08-27):

    ``nav_change_usd`` is NULL on 41 of 119 sessions (2026-03-09 →
    2026-05-05, before PR490 added the named P&L lines) while
    ``daily_return_pct`` is populated on all 119. On every one of the 78
    sessions carrying BOTH values, ``daily_return_pct`` and
    ``nav_change_usd / prior_nav × 100`` agree to floating-point precision
    (delta = 0.0 exactly, all 78) — including 2026-07-28, the one session
    whose ``input_closure_usd`` is materially nonzero ($4,061.25, 40.4bp of
    NAV; the evaluator's own 25bp ``INPUT_CLOSURE_NAV_BPS`` bound already
    excludes that session from the attribution window it publishes).

So the drift a naive whole-history chain shows is a **day-set mismatch**
(cause (b) of I9025's three candidates), not a stale stored value (cause (a)
— there is no disagreeing row to self-heal) and not an accumulating residual
(cause (c) — the per-session delta is exactly zero whenever both legs exist).
Per I9025 deliverable 3, the fix for cause (b) is to make the two series
cover the same sessions BY CONSTRUCTION — ``verify_nav_change_basis_closes``
below excludes a row missing ``nav_change_usd`` from BOTH chains rather than
letting the ``daily_return_pct`` chain silently run ahead of it — not to widen
the tolerance.

The residual-bound derivation below rests on one measurement that the issue
body did NOT have, and which changes what the bound is FOR. Reconstructing
``rotation_realized_usd`` (realized P&L on shares rotated out, which
``eod_report.compute_rotation_realized`` already computes but which was never
persisted) over the 74 sessions carrying attribution columns:

    Σ unattributed_usd (raw)              = −$20,293
    Σ rotation_realized_usd               = −$20,815
    Σ residual after lifting rotation out = **+$522**  (mean +$7/session)

    turnover days (57):  raw mean −$406  →  ex-rotation mean −$41
    no-turnover days (17): raw mean +$168 →  ex-rotation mean +$168

So the residual "tracking turnover" was realized rotation P&L — mechanically
turnover-linked — not unmodelled trading costs. The bound therefore applies to
``unattributed_true_usd`` (after the rotation and pricing&timing sleeves are
lifted out), which is the number that should actually be ~0, and the sleeves
are persisted alongside it so the CSV carries the decomposition rather than one
undifferentiated plug.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Residual bounds — per-session and cumulative
# ─────────────────────────────────────────────────────────────────────────────
#
# DERIVATION (measured, 74 sessions with attribution columns, NAV ≈ $1.03M).
#
# Per-session. |unattributed_usd − rotation_realized_usd| distribution:
#   p50 = $754 (7bp of NAV) · p90 = $2,789 (27bp) · p95 = $3,560 (35bp) ·
#   p99 = $8,573 (83bp). The p99 day is 2026-08-04, whose residual (−$9,713)
#   is dominated by a single stale IB mark on AMD (+$8,906, 8.26% off the
#   settled close) — i.e. the pricing&timing sleeve, which this gate lifts out
#   separately. A HARD raise must therefore sit ABOVE the ordinary
#   settlement/mark noise band and only fire on a residual no named sleeve can
#   absorb. max($5,000, 50bp of NAV) ≈ $5,150 at today's NAV clears p95 by 1.4x
#   and fires on 1 of 74 historical sessions (1.4%) — the one session that was
#   genuinely broken.
#
#   The pre-existing SOFT warning (max($100, 5bp of NAV), data_warnings →
#   EOD email only) is UNCHANGED and still fires first. Same two-band shape
#   as NAV_HARD_GATE_TOLERANCE_* in eod_reconcile: a hard gate set at the soft
#   band's rate pages exactly as often as the email already flags and trains
#   the operator to ignore it.
#
# Cumulative. The whole point of defect 1 is that per-session-invisible,
# same-sign residuals compounded to −$20,293 (−1.97% of NAV) with no detector.
# A per-session bound structurally cannot catch that. Measured true drift over
# 74 sessions after the sleeves are lifted: +$522 = 5bp of NAV. A trailing
# 63-session (one quarter) window bounded at max($10,000, 100bp of NAV) is
# ~20x the measured drift — it cannot fire on the observed behaviour of a
# correctly-attributed book, and it would have fired on the RAW plug within
# ~30 sessions.
RESIDUAL_HARD_PER_SESSION_USD_FLOOR = 5_000.0
RESIDUAL_HARD_PER_SESSION_NAV_BPS = 50.0
RESIDUAL_CUMULATIVE_WINDOW_SESSIONS = 63
RESIDUAL_HARD_CUMULATIVE_USD_FLOOR = 10_000.0
RESIDUAL_HARD_CUMULATIVE_NAV_BPS = 100.0


def residual_per_session_tolerance_usd(nav: float) -> float:
    """Dollar bound on one session's TRUE residual at this NAV level."""
    return max(
        RESIDUAL_HARD_PER_SESSION_USD_FLOOR,
        RESIDUAL_HARD_PER_SESSION_NAV_BPS / 10_000.0 * nav,
    )


def residual_cumulative_tolerance_usd(nav: float) -> float:
    """Dollar bound on the trailing-window cumulative residual at this NAV."""
    return max(
        RESIDUAL_HARD_CUMULATIVE_USD_FLOOR,
        RESIDUAL_HARD_CUMULATIVE_NAV_BPS / 10_000.0 * nav,
    )


def check_residual_bounds(
    *,
    unattributed_true_usd: float | None,
    nav: float | None,
    trailing_residuals_usd: Sequence[float] | None = None,
    run_date: str = "",
    basis_is_true_residual: bool = True,
) -> list[dict[str, Any]]:
    """Bound the P&L residual per-session AND cumulatively.

    ``unattributed_true_usd`` is the residual AFTER the rotation-realized and
    pricing&timing sleeves are lifted out — the number that should be ~0. Pass
    the raw plug only if the sleeves are unavailable (the caller records that
    as a degradation).

    ``trailing_residuals_usd`` is the persisted history of the same quantity,
    OLDEST-first and EXCLUDING today; today's value is appended here so the
    cumulative check always includes the session being written. The window and
    today's value MUST be the same quantity: the raw plug carries realized
    rotation P&L and sums to −$20,293 over the measurement window by
    construction, so one plug inside a true-residual sum breaches a bound
    derived for a quantity measured at +$522.

    ``basis_is_true_residual`` is False when the caller had to fall back to the
    raw plug for TODAY. The per-session bound still applies — loudly, and
    labelled — but the cumulative check is SKIPPED rather than run on a mixed
    window, and the caller records the skip. A gate that cannot be evaluated on
    one basis reports that it was not evaluated; it does not report a pass.

    Returns a list of breach dicts (empty when both bounds hold). Each breach
    carries ``kind`` (``"per_session"`` / ``"cumulative"``), the measured
    value, the tolerance, and a ``message`` the caller raises with. Returning
    rather than raising keeps the gate testable and lets the caller persist the
    EOD row BEFORE failing the run — a lost artifact is a worse outcome than a
    red pipeline, and the breach is itself persisted (``residual_breach``).
    """
    if unattributed_true_usd is None or not nav:
        return []

    breaches: list[dict[str, Any]] = []

    per_tol = residual_per_session_tolerance_usd(nav)
    if abs(unattributed_true_usd) > per_tol:
        breaches.append({
            "kind": "per_session",
            "run_date": run_date,
            "value_usd": unattributed_true_usd,
            "tolerance_usd": per_tol,
            "nav": nav,
            "pct_of_nav": unattributed_true_usd / nav * 100,
            "message": (
                f"P&L residual bound breached for {run_date}: unattributed "
                f"${unattributed_true_usd:+,.0f} "
                f"({unattributed_true_usd / nav * 100:+.3f}% of NAV) exceeds the "
                f"per-session bound ${per_tol:,.0f} "
                f"({RESIDUAL_HARD_PER_SESSION_NAV_BPS:.0f}bps of NAV). This is the "
                "residual AFTER rotation-realized and pricing&timing are lifted "
                "out — no named sleeve explains it."
            ),
        })

    if not basis_is_true_residual:
        return breaches

    window = list(trailing_residuals_usd or [])[-(RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1):]
    window.append(float(unattributed_true_usd))
    cumulative = sum(window)
    cum_tol = residual_cumulative_tolerance_usd(nav)
    if abs(cumulative) > cum_tol:
        breaches.append({
            "kind": "cumulative",
            "run_date": run_date,
            "value_usd": cumulative,
            "tolerance_usd": cum_tol,
            "nav": nav,
            "n_sessions": len(window),
            "pct_of_nav": cumulative / nav * 100,
            "message": (
                f"Cumulative P&L residual bound breached as of {run_date}: "
                f"${cumulative:+,.0f} ({cumulative / nav * 100:+.3f}% of NAV) over "
                f"the trailing {len(window)} session(s) exceeds "
                f"${cum_tol:,.0f} ({RESIDUAL_HARD_CUMULATIVE_NAV_BPS:.0f}bps of "
                "NAV). Same-sign residuals are accumulating — a systematic "
                "attribution gap, not day-to-day noise."
            ),
        })

    return breaches


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction costs — commission and implementation shortfall
# ─────────────────────────────────────────────────────────────────────────────

# The trade-action vocabulary, classified by what the fill does to the POSITION
# (COVER increases it — buying stock back to close a short — which is why it is
# buy-side here even though eod_report prices it on the exit leg).
#
# These must cover every action the executor actually emits. They did not:
# LIQUIDATION_SELL and EMERGENCY_SELL are real actions in eod_report's own
# vocabulary and were absent here, so a forced exit fell through the side
# classification and its implementation shortfall was dropped from the
# slippage line while its notional still counted — diluting slippage_bps on
# exactly the days most likely to have the worst of it. TRIM/ADD are the
# reverse case: named here, emitted nowhere. Unclassified fills are now
# COUNTED and surfaced rather than skipped in silence.
_BUY_ACTIONS = {"BUY", "ENTER", "ADD", "COVER"}
_SELL_ACTIONS = {
    "SELL", "EXIT", "REDUCE", "TRIM", "LIQUIDATION_SELL", "EMERGENCY_SELL",
}


def session_costs(trades_today: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Explicit transaction costs for one session, sourced from FILLS.

    Two named lines, both POSITIVE = a cost to the book:

    ``commission_usd``
        Σ of the per-execution commission IB reports on each fill
        (``Fill.commissionReport.commission``, captured onto ``trades`` at
        order time). This is a real cash debit and is already inside
        ``nav_change_usd``; naming it is what lets net be distinguished from
        gross.

    ``slippage_usd``
        Implementation shortfall against the ARRIVAL price
        (``price_at_order``, the mark at order submission):
        ``Σ side · shares · (fill_price − price_at_order)``, side +1 for buys
        and −1 for sells. Measured over the whole live window (468 fills):
        **+6.4bp of traded notional** — a real and entirely unrecorded cost. Unlike commission this NEVER debited NAV (the fill
        price IS the book cost); it is the difference between the return the
        book earned and the return the decision would have earned at arrival.
        Gross-of-cost return is therefore explicitly hypothetical and labelled
        as such.

    ``commission_available`` is False when fills executed and no filled row
    carried a commission figure — fail loud: a $0 commission line and an ABSENT
    commission line are different facts and must not render identically. The
    pre-existing code treated the absence as "paper-account commissions are
    trivial" and dropped it silently, which is exactly how the cost line came
    not to exist.

    ``commission_usd`` is therefore ``None`` — never ``0.0`` — on that path
    (alpha-engine-config-I8188 deliverable 2, second pass). Returning the flag
    beside a 0.0 was the same defect one layer down: the flag lived only in a
    ``data_warnings`` string while the PERSISTED column read a measured zero,
    and it did so on 1 of the first 6 live sessions. A session with NO fills
    keeps ``0.0``, which is a real measurement: nothing traded, so nothing was
    charged.
    """
    n_fills = 0
    n_with_commission = 0
    n_unclassified = 0
    n_no_arrival = 0
    commission = 0.0
    slippage = 0.0
    notional = 0.0
    for t in trades_today or []:
        shares = t.get("filled_shares")
        if shares in (None, ""):
            shares = t.get("shares")
        fill_price = t.get("fill_price")
        if shares in (None, "") or fill_price in (None, ""):
            continue
        try:
            shares = float(shares)
            fill_price = float(fill_price)
        except (TypeError, ValueError):
            continue
        if shares <= 0 or fill_price <= 0:
            continue
        n_fills += 1
        notional += shares * fill_price

        comm = t.get("commission_usd")
        if comm not in (None, ""):
            try:
                commission += abs(float(comm))
                n_with_commission += 1
            except (TypeError, ValueError):
                pass

        arrival = t.get("price_at_order")
        action = str(t.get("action") or "").upper()
        if action not in (_BUY_ACTIONS | _SELL_ACTIONS):
            n_unclassified += 1
            continue
        if arrival in (None, ""):
            n_no_arrival += 1
            continue
        try:
            arrival = float(arrival)
        except (TypeError, ValueError):
            continue
        if arrival <= 0:
            continue
        side = 1.0 if action in _BUY_ACTIONS else -1.0
        slippage += side * shares * (fill_price - arrival)

    commission_available = n_fills == 0 or n_with_commission > 0
    return {
        # None, not 0.0, when fills executed and IB attached no commission to
        # any of them. See the docstring: a rendered zero is indistinguishable
        # from a measured one, which is the defect class this module exists for.
        "commission_usd": commission if commission_available else None,
        "slippage_usd": slippage,
        "traded_notional_usd": notional,
        "n_fills": n_fills,
        "n_fills_with_commission": n_with_commission,
        # Fills excluded from the slippage leg, by reason. A dropped fill still
        # sits in traded_notional_usd, so a silent drop understates slippage_bps
        # rather than leaving it absent — the caller surfaces these.
        "n_fills_unclassified_action": n_unclassified,
        "n_fills_without_arrival_price": n_no_arrival,
        "commission_available": commission_available,
        "slippage_bps": (slippage / notional * 10_000.0) if notional else None,
    }


def gross_net_returns(
    *,
    nav_change_usd: float | None,
    prior_nav: float | None,
    commission_usd: float | None,
    slippage_usd: float,
) -> dict[str, Any]:
    """Split the day's return into net-of-cost and gross-of-cost.

    ``daily_return_net_pct`` is the return the book actually earned — the
    existing ``daily_return_pct``, restated under a name that says which side
    of costs it is on. Commissions have already debited NAV and fills already
    happened at the fill price, so the NAV-based number IS net. It does NOT
    depend on the commission being known, and is therefore published whenever
    there is a prior NAV.

    ``daily_return_gross_pct`` adds both cost lines back: the return the same
    decisions would have produced with zero commission and zero implementation
    shortfall. It is a counterfactual, not a second measurement of the book —
    the shortfall leg never touched NAV.

    ``commission_usd=None`` means the commission is ABSENT, not zero
    (``session_costs`` returns None when fills executed and IB attached no
    commissionReport). Gross is then ``None`` with ``gross_unavailable_reason``
    set, rather than a gross figure computed as though the missing leg were
    $0.00 — that number would be a net return wearing a gross label, and would
    understate the cost of implementation by exactly the amount nobody
    measured. ``total_cost_usd`` is None on the same path for the same reason.

    Both returns are None when there is no prior NAV (first session).
    """
    total_cost = (
        commission_usd + slippage_usd if commission_usd is not None else None
    )
    reason = (
        None if commission_usd is not None
        else "commission absent (no commissionReport on any fill) — a gross "
             "return computed with a $0.00 commission leg would be a net "
             "return wearing a gross label"
    )
    if nav_change_usd is None or not prior_nav:
        return {
            "daily_return_net_pct": None,
            "daily_return_gross_pct": None,
            "total_cost_usd": total_cost,
            "gross_available": False,
            "gross_unavailable_reason": reason or "no prior NAV (first session)",
        }
    net = nav_change_usd / prior_nav * 100.0
    if commission_usd is None:
        return {
            "daily_return_net_pct": net,
            "daily_return_gross_pct": None,
            "total_cost_usd": None,
            "gross_available": False,
            "gross_unavailable_reason": reason,
        }
    gross = (nav_change_usd + commission_usd + slippage_usd) / prior_nav * 100.0
    return {
        "daily_return_net_pct": net,
        "daily_return_gross_pct": gross,
        "total_cost_usd": total_cost,
        "gross_available": True,
        "gross_unavailable_reason": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWR closure — chain-linked daily returns must equal the NAV ratio
# ─────────────────────────────────────────────────────────────────────────────
#
# Absent external cash flows these are identically equal by construction, since
# every stored daily_return_pct is (nav − prior_nav) / prior_nav against the
# immediately preceding persisted row. Any drift is therefore a STORED value
# that no longer agrees with the NAV series it was derived from.
#
# MEASURED CAUSE of the live 17.4bp drift — it is not external flows, and it is
# not distributed: it is ONE row. On 2026-04-07 the stored daily_return_pct is
# +0.026827% while the persisted NAVs imply −0.140311%. Its implied prior NAV
# is $1,007,786.32; the persisted 2026-04-06 row says $1,009,473.08. The
# 2026-04-06 NAV was corrected AFTER the 2026-04-07 row was written (that row's
# created_at is 21:42Z against the usual 20:20Z — a re-run) and 04-07's stored
# return was never recomputed. Recomputing that single row moves the chain-link
# from +3.8261% to +3.6526%, matching the NAV ratio to 3e-5pp.
TWR_CLOSURE_TOLERANCE_BPS = 1.0
# A self-heal must not be able to launder an arbitrary rewrite of the track
# record. A correction larger than this on a single session is not a stale
# stored value, it is a different NAV series (an external flow, a restated
# snapshot, a mis-keyed row) — that raises instead of being silently rewritten.
TWR_SELF_HEAL_MAX_CORRECTION_PCT = 1.0


def nav_implied_returns(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """For each row, the daily return its own NAV series implies.

    ``rows`` is the persisted eod_pnl series OLDEST-first, each carrying
    ``date``, ``portfolio_nav`` and ``daily_return_pct``. The first row has no
    predecessor so its implied return is None.
    """
    out: list[dict[str, Any]] = []
    prior_nav: float | None = None
    for r in rows:
        nav = r.get("portfolio_nav")
        try:
            nav = float(nav) if nav not in (None, "") else None
        except (TypeError, ValueError):
            nav = None
        stored = r.get("daily_return_pct")
        try:
            stored = float(stored) if stored not in (None, "") else None
        except (TypeError, ValueError):
            stored = None
        implied = None
        if nav is not None and prior_nav:
            implied = (nav / prior_nav - 1.0) * 100.0
        out.append({
            "date": r.get("date"),
            "portfolio_nav": nav,
            "stored_pct": stored,
            "implied_pct": implied,
            "delta_pct": (
                stored - implied
                if (stored is not None and implied is not None)
                else None
            ),
        })
        if nav is not None:
            prior_nav = nav
    return out


def verify_twr_closes(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_bps: float = TWR_CLOSURE_TOLERANCE_BPS,
) -> dict[str, Any]:
    """Assert chain-linked TWR equals the NAV ratio to ``tolerance_bps``.

    Returns a dict with ``closes`` (bool), both figures, the drift in bps, and
    ``offenders`` — the rows whose stored return disagrees with the return
    their own NAV series implies, which is where the drift comes from.

    ``status`` is ``"n/a"`` when there are fewer than two usable rows.
    """
    detail = [d for d in nav_implied_returns(rows) if d["portfolio_nav"] is not None]
    if len(detail) < 2:
        return {"status": "n/a", "closes": None, "offenders": []}

    chain = 1.0
    for d in detail[1:]:
        if d["stored_pct"] is None:
            return {
                "status": "n/a",
                "closes": None,
                "offenders": [],
                "reason": f"row {d['date']} has no stored daily_return_pct",
            }
        chain *= 1.0 + d["stored_pct"] / 100.0
    chain_pct = (chain - 1.0) * 100.0

    first, last = detail[0]["portfolio_nav"], detail[-1]["portfolio_nav"]
    if not first:
        return {"status": "n/a", "closes": None, "offenders": []}
    nav_ratio_pct = (last / first - 1.0) * 100.0

    drift_bps = (chain_pct - nav_ratio_pct) * 100.0
    offenders = [
        d for d in detail[1:]
        if d["delta_pct"] is not None and abs(d["delta_pct"]) * 100.0 > tolerance_bps
    ]
    return {
        "status": "ok",
        "closes": abs(drift_bps) <= tolerance_bps,
        "chain_linked_pct": chain_pct,
        "nav_ratio_pct": nav_ratio_pct,
        "drift_bps": drift_bps,
        "tolerance_bps": tolerance_bps,
        "n_sessions": len(detail),
        "offenders": offenders,
        "message": (
            f"TWR does not close: chain-linked daily_return_pct = "
            f"{chain_pct:+.4f}% vs NAV ratio {nav_ratio_pct:+.4f}% over "
            f"{len(detail)} sessions — {drift_bps:+.1f}bp of drift against a "
            f"{tolerance_bps:.0f}bp tolerance. Absent external flows these are "
            f"identically equal by construction. Disagreeing row(s): "
            + ", ".join(
                f"{o['date']} stored {o['stored_pct']:+.6f}% vs NAV-implied "
                f"{o['implied_pct']:+.6f}%"
                for o in offenders[:5]
            )
            + ("" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)")
        ),
    }


def plan_twr_self_heal(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_bps: float = TWR_CLOSURE_TOLERANCE_BPS,
    max_correction_pct: float = TWR_SELF_HEAL_MAX_CORRECTION_PCT,
) -> dict[str, Any]:
    """Plan the corrections that would make TWR close, without applying them.

    A stored ``daily_return_pct`` that disagrees with the persisted NAV series
    is stale — the NAV series is ground truth (it is what the broker reported)
    and the stored percentage is a derived value that was never recomputed
    after an upstream NAV correction. Rewriting it is a repair, and it CLOSES
    THE LOOP without an operator step; the alternative — a detector that goes
    red until a human runs a backfill — is a page, not a fix.

    Two guards keep the repair from becoming a licence to restate history:

    * any single correction larger than ``max_correction_pct`` is REFUSED and
      surfaces in ``refused`` — at that size the disagreement is a different
      NAV series (an external flow, a restated snapshot), not a stale
      derived value, and it must be ruled on rather than rewritten;
    * ``spy_return_pct``/``daily_alpha_pct`` are NOT touched — only the leg
      whose ground truth is the persisted NAV.

    Returns ``{"corrections": [...], "refused": [...]}`` where each correction
    is ``{date, from_pct, to_pct, delta_pct}``.
    """
    corrections: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for d in nav_implied_returns(rows):
        if d["delta_pct"] is None:
            continue
        if abs(d["delta_pct"]) * 100.0 <= tolerance_bps:
            continue
        entry = {
            "date": d["date"],
            "from_pct": d["stored_pct"],
            "to_pct": d["implied_pct"],
            "delta_pct": d["delta_pct"],
        }
        if abs(d["delta_pct"]) > max_correction_pct:
            entry["reason"] = (
                f"correction {d['delta_pct']:+.4f}pp exceeds the "
                f"{max_correction_pct:.2f}pp self-heal ceiling — this is a NAV-series "
                "disagreement (external flow / restated snapshot), not a stale "
                "derived value"
            )
            refused.append(entry)
        else:
            corrections.append(entry)
    return {"corrections": corrections, "refused": refused}


# ─────────────────────────────────────────────────────────────────────────────
# 3b. TWR closure, second arm — the ``nav_change_usd`` basis (I9025)
# ─────────────────────────────────────────────────────────────────────────────
#
# A THIRD arm of the TWR-closure discipline (the first two are
# ``verify_twr_closes`` above: chain-linked ``daily_return_pct`` vs the NAV
# ratio). This compares the chain-linked ``daily_return_pct`` series against
# the chain-linked ``nav_change_usd / prior_nav`` series — a pair
# ``verify_twr_closes`` never touches, and the one ``return_chain_basis_gap``
# publishes at ``s3://alpha-engine-research/evaluator/latest/attribution.json``
# (alpha-engine-config-I9025).
#
# MEASURED CAUSE. Reconstructed over the full 119-session ``eod_pnl.csv``
# history: ``nav_change_usd`` is NULL on 41 sessions (2026-03-09 →
# 2026-05-05, before PR490 added the named P&L lines) while
# ``daily_return_pct`` is populated throughout. On every one of the 78
# sessions carrying BOTH values — 2026-07-28 (the sole session with a
# material ``input_closure_usd``, $4,061.25 / 40.4bp) included — the two
# figures agree to floating-point precision: delta = 0.0 exactly, all 78.
# There is no disagreeing row to self-heal (cause (a), ruled out) and no
# accumulating same-sign residual (cause (c), ruled out: the per-session
# delta is exactly zero, not small-and-consistent). The cause is (b): the two
# series cover DIFFERENT DAY SETS, purely because one column started later
# than the other.
#
# So the fix is not a tolerance and not a self-heal — it is coverage. A row
# missing ``nav_change_usd`` is excluded from BOTH chains here, never left in
# one and dropped from the other; ``coverage_gap_sessions`` names those rows
# for observability (a schema-coverage gap is itself worth seeing) without
# counting them as drift. ``plan_twr_self_heal`` is NOT extended for this arm
# — there is nothing to rewrite; a self-heal here would be inventing a NAV
# change for a session that never persisted one.
NAV_CHANGE_BASIS_TOLERANCE_BPS = TWR_CLOSURE_TOLERANCE_BPS


def nav_change_implied_returns(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """For each row, the return its own ``nav_change_usd`` implies.

    ``rows`` is the persisted eod_pnl series OLDEST-first, each carrying
    ``date``, ``portfolio_nav``, ``daily_return_pct`` and ``nav_change_usd``.
    A row missing ``nav_change_usd`` (or with no persisted prior NAV) gets
    ``implied_pct=None`` and, when ``daily_return_pct`` IS present, is flagged
    ``coverage_gap=True`` — the day-set mismatch that is I9025's measured
    cause, surfaced rather than silently absorbed into one chain only.
    """
    out: list[dict[str, Any]] = []
    prior_nav: float | None = None
    for r in rows:
        nav = r.get("portfolio_nav")
        try:
            nav = float(nav) if nav not in (None, "") else None
        except (TypeError, ValueError):
            nav = None
        stored = r.get("daily_return_pct")
        try:
            stored = float(stored) if stored not in (None, "") else None
        except (TypeError, ValueError):
            stored = None
        nav_change = r.get("nav_change_usd")
        try:
            nav_change = float(nav_change) if nav_change not in (None, "") else None
        except (TypeError, ValueError):
            nav_change = None
        implied = None
        if nav_change is not None and prior_nav:
            implied = nav_change / prior_nav * 100.0
        out.append({
            "date": r.get("date"),
            "portfolio_nav": nav,
            "stored_pct": stored,
            "nav_change_usd": nav_change,
            "implied_pct": implied,
            "coverage_gap": (
                stored is not None and nav_change is None and prior_nav is not None
            ),
            "delta_pct": (
                stored - implied
                if (stored is not None and implied is not None)
                else None
            ),
        })
        if nav is not None:
            prior_nav = nav
    return out


def verify_nav_change_basis_closes(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_bps: float = NAV_CHANGE_BASIS_TOLERANCE_BPS,
) -> dict[str, Any]:
    """Assert chain-linked TWR equals the chain-linked ``nav_change_usd`` basis.

    The third TWR-closure arm (alpha-engine-config-I9025) — see the module
    docstring and the block comment above for the measured cause. Both chains
    are built ONLY over rows carrying both ``daily_return_pct`` and
    ``nav_change_usd``: a row missing either is excluded from both by
    construction (never from just one), so the comparison can never register
    day-set coverage as drift. Rows excluded for missing ``nav_change_usd``
    while carrying ``daily_return_pct`` are named in
    ``coverage_gap_sessions`` — a data-quality signal, not a closure failure.

    ``status`` is ``"n/a"`` when there are fewer than two comparable rows.
    """
    detail = nav_change_implied_returns(rows)
    coverage_gap_sessions = [d["date"] for d in detail if d["coverage_gap"]]
    comparable = [
        d for d in detail
        if d["stored_pct"] is not None and d["implied_pct"] is not None
    ]
    if len(comparable) < 1:
        return {
            "status": "n/a",
            "closes": None,
            "offenders": [],
            "coverage_gap_sessions": coverage_gap_sessions,
        }

    chain_stored = 1.0
    chain_implied = 1.0
    for d in comparable:
        chain_stored *= 1.0 + d["stored_pct"] / 100.0
        chain_implied *= 1.0 + d["implied_pct"] / 100.0
    stored_chain_pct = (chain_stored - 1.0) * 100.0
    implied_chain_pct = (chain_implied - 1.0) * 100.0
    drift_bps = (stored_chain_pct - implied_chain_pct) * 100.0

    offenders = [
        d for d in comparable
        if d["delta_pct"] is not None and abs(d["delta_pct"]) * 100.0 > tolerance_bps
    ]

    return {
        "status": "ok",
        "closes": abs(drift_bps) <= tolerance_bps,
        "stored_return_chain_pct": stored_chain_pct,
        "nav_change_basis_chain_pct": implied_chain_pct,
        "drift_bps": drift_bps,
        "tolerance_bps": tolerance_bps,
        "n_sessions": len(comparable),
        "offenders": offenders,
        "coverage_gap_sessions": coverage_gap_sessions,
        "message": (
            f"TWR does not close against the nav_change_usd basis: "
            f"chain-linked daily_return_pct = {stored_chain_pct:+.4f}% vs "
            f"chain-linked nav_change_usd/prior_nav = {implied_chain_pct:+.4f}% "
            f"over {len(comparable)} sessions — {drift_bps:+.1f}bp of drift "
            f"against a {tolerance_bps:.0f}bp tolerance. Disagreeing row(s): "
            + ", ".join(
                f"{o['date']} stored {o['stored_pct']:+.6f}% vs nav_change-"
                f"implied {o['implied_pct']:+.6f}%"
                for o in offenders[:5]
            )
            + ("" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)")
            + (
                f". {len(coverage_gap_sessions)} session(s) excluded from both "
                "chains for missing nav_change_usd."
                if coverage_gap_sessions else ""
            )
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Custodian mark divergence — promoted from flag to failure
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS BEING COMPARED matters, and it decides the shape of the test.
# ``market_value`` is the CANONICAL valuation (settled ArcticDB close x shares).
# ``ib_market_value`` is IB Gateway's paper-account mark — a <=15-minute DELAYED
# quote captured at ~4:05pm ET. Measured over the 229 position-days carrying
# both, |mv - ib_mv| / ib_mv runs p50 = 0.085%, p90 = 0.52%, p95 = 0.79%. That
# body of the distribution is just one name's last-quarter-hour price move
# (sigma_15min ~ sigma_daily x sqrt(15/390) ~ 0.4% for the names actually held),
# which is why the pre-existing 0.5% band flags 24 of 229 (10.5%).
#
# So a flat percentage is the WRONG INSTRUMENT to promote to a failure: it fails
# high-volatility names for being volatile and passes stale marks on quiet ones,
# and any threshold picked to make today's data pass would be worthless.
#
# The repo already has the right test. ``eod_reconcile._detect_ib_mark_outside_range``
# (config#6349/#6818) flags an IB mark that fell outside that day's own ArcticDB
# [Low, High]. That is volatility-normalised with no arbitrary tolerance at all:
# a mark outside the range the security actually traded in is PROVABLY wrong
# however volatile the name is. This gate promotes THAT check rather than
# inventing a parallel percentage band beside it.
#
# The only remaining question is materiality, and the answer reuses the number
# the NAV three-way hard gate already declares (15bps of NAV) with the soft
# data_warnings floor beneath it ($500, ``NAV_BREACH_RESIDUAL_FLOOR_USD``)
# rather than minting a fourth threshold.
#
# Against the 8 out-of-range marks in the live window the separation is clean —
# the distribution is bimodal, and the gap between the two modes is 4x wide:
#   2026-08-04 AMD   mark $479.00 vs [$502.20, $530.13]  -$5,220  50.4bp  RAISE
#   2026-07-30 COIN  mark $154.45 vs [$159.31, $164.78]  -$2,999  29.7bp  RAISE
#   2026-06-26 LNTH  mark $105.12 vs [$108.51, $111.46]  -$2,532  25.5bp  RAISE
#   2026-07-29 SPY   ...                                   -$584   5.9bp  pass
#   2026-07-29 COIN  ...                                   -$420   4.2bp  pass
#   2026-08-21 DECK / 2026-07-16 TWLO / 2026-08-17 SPY   <=$102  <=1.0bp  pass
# The three that raise are genuinely wrong marks — AMD 2026-08-04 is also the
# single worst residual session in the entire window (-$9,713). The five that
# pass are edge-of-range rounding.
MARK_HARD_MATERIALITY_USD_FLOOR = 500.0
MARK_HARD_MATERIALITY_NAV_BPS = 15.0


def mark_materiality_usd(nav: float) -> float:
    """Dollar materiality floor for an out-of-range broker mark at this NAV."""
    return max(
        MARK_HARD_MATERIALITY_USD_FLOOR,
        MARK_HARD_MATERIALITY_NAV_BPS / 10_000.0 * nav,
    )


def check_custodian_marks(
    mark_range_flags: Sequence[Mapping[str, Any]] | None,
    *,
    nav: float | None,
    run_date: str = "",
    corrected_tickers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Promote an out-of-range broker mark from a flag to a FAILURE.

    ``mark_range_flags`` is what ``eod_reconcile._detect_ib_mark_outside_range``
    returns: one entry per held ticker whose ``ib_market_value / shares`` fell
    outside that day's ArcticDB ``[Low, High]``, carrying ``mark_error_usd``
    (signed, the distance past the breached bound x shares).

    Every flag stays a flag — the observational surface that measured the
    distribution above is not deleted by promoting the check. A breach is
    returned only when the mark error is also MATERIAL against NAV, so a
    provably-wrong mark on a trivially small position does not halt the
    pipeline.

    ``corrected_tickers`` (alpha-engine-config-I9627) names the marks whose
    error ``plan_nav_mark_correction`` has already REMOVED from NAV. A halt
    exists to stop the book carrying a number known to be wrong; once the wrong
    number is gone the halt has nothing left to protect, and re-raising on it
    would fail every future EOD for a condition the pipeline has just repaired.
    The flag, the raw broker mark and the correction are all still persisted, so
    nothing about the event becomes unobservable. A name NOT in this list — one
    whose settled close could not prove the error, or one whose correction the
    bound refused — breaches exactly as before.
    """
    if not mark_range_flags or not nav:
        return []
    repaired = set(corrected_tickers or ())
    materiality_usd = mark_materiality_usd(nav)
    breaches: list[dict[str, Any]] = []
    for flag in mark_range_flags:
        if flag.get("ticker") in repaired:
            continue
        error_usd = flag.get("mark_error_usd")
        try:
            error_usd = float(error_usd)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(error_usd) or abs(error_usd) <= materiality_usd:
            continue
        breaches.append({
            "kind": "custodian_mark",
            "run_date": run_date,
            "ticker": flag.get("ticker"),
            "ib_mark": flag.get("ib_mark"),
            "day_low": flag.get("day_low"),
            "day_high": flag.get("day_high"),
            "shares": flag.get("shares"),
            "error_usd": error_usd,
            "materiality_usd": materiality_usd,
            "pct_of_nav": error_usd / nav * 100.0,
            "message": (
                f"Custodian mark failure for {flag.get('ticker')} on {run_date}: "
                f"broker mark ${flag.get('ib_mark'):,.2f} fell outside the day's "
                f"traded range [${flag.get('day_low'):,.2f}, "
                f"${flag.get('day_high'):,.2f}] — ${error_usd:+,.0f} "
                f"({error_usd / nav * 100:+.3f}% of NAV) past the breached bound, "
                f"above the ${materiality_usd:,.0f} materiality floor "
                f"({MARK_HARD_MATERIALITY_NAV_BPS:.0f}bps of NAV). A mark outside "
                "the range the security actually traded in is wrong regardless of "
                "volatility, and every return, risk metric and attribution number "
                "is computed on top of these marks."
            ),
        })
    return breaches


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Repairing a provably-wrong broker mark (alpha-engine-config-I9627)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY. ``check_custodian_marks`` above detects an IB portfolio mark that fell
# outside the day's own traded range and halts the EOD pipeline on it. That is
# a correct verdict with no remediation attached: the mark is the broker's, the
# pipeline cannot make IB re-send it, and a rerun on the same snapshot re-raises
# the identical breach. The postclose SF has therefore been failing on a
# condition no operator action can clear — 2026-08-31 (DUOL, $1,975 past the
# breached bound) is the live instance, and ``pnl_measurement_backfill``
# reconstructs five more over the history.
#
# The repair the pipeline can make, and until now did not, is to stop carrying
# the wrong number. ``eod_reconcile`` ALREADY prices every position off the
# settled ArcticDB close — ``pos["market_value"]`` is overridden two lines after
# the IB mark is captured. Only the headline NAV kept the broker's mark, because
# NAV is read whole from IB ``NetLiquidation``. So the book values one name at
# two different prices on the same day, and the gate fires on the difference.
#
# Correcting NAV by exactly the proven error puts the headline on the same
# prices the positions already use. It is not a tolerance and not a suppression:
# every flag still fires, every flag is still persisted, and a name whose error
# cannot be PROVEN is not touched.
#
# THE DISCRIMINATOR. A settled close is a price the security actually traded at,
# so it lies inside that day's own ``[Low, High]`` by construction. When it does
# not, the reference data is what is wrong — not the broker mark — and moving
# NAV towards it would import the error rather than remove it. That name is left
# uncorrected and the gate raises on it, which is the correct outcome: an
# ArcticDB row disagreeing with itself is a data-repo defect, not a broker one.
#
# THE BOUND. Modelled on ``plan_twr_self_heal``: a correction larger than the
# bound is REFUSED and raises, because at that size the disagreement is not a
# stale mark but a different book (a share-count error, a corporate action we
# did not model, a wrong snapshot). Every instance measured to date sits far
# below it — DUOL 2026-08-31 $2,860 (28bp), AMD 2026-08-04 $5,220 (50bp) — so
# the bound refuses nothing that has ever legitimately occurred while still
# refusing a book-scale disagreement.
MARK_CORRECTION_MAX_NAV_BPS = 100.0
MARK_CORRECTION_MAX_USD_FLOOR = 10_000.0


#: Paging threshold for an APPLIED correction. The repair itself is automatic
#: and sound below ``MARK_CORRECTION_MAX_NAV_BPS``; what an operator needs to be
#: woken for is a correction large enough to move a reported figure. Below this
#: it is the routine after-hours drift of the broker's position mark, logged at
#: WARNING (tracked, persisted on the report, not paged). Measured 2026-09-28:
#: HOOD's IB mark $116.01 sat $0.05 under a $116.06 session low — IB marks
#: positions at the last print including extended hours, and the snapshot is
#: taken after 16:00 ET — a $394 (3.8bp) correction that paged at ERROR. The
#: instances that justified the correction machinery (DUOL 28bp, AMD 50bp) stay
#: above it and still page.
MARK_CORRECTION_PAGE_NAV_BPS = 10.0


def mark_correction_log_level(plan: Mapping[str, Any], *, announce: bool = True) -> int:
    """Logging level for an APPLIED correction's message.

    ``announce`` is False when this reconcile is not the first report of the
    same inputs (a second scheduler over the same snapshot, or a
    ``reconcile_audit`` re-reconcile of a past day): the correction was already
    reported, so it is recorded at INFO rather than paged again.
    """
    if not announce:
        return logging.INFO
    nav = plan.get("nav_raw")
    total = abs(float(plan.get("correction_usd") or 0.0))
    if not nav or not math.isfinite(float(nav)):
        return logging.ERROR
    if total / abs(float(nav)) * 10_000.0 >= MARK_CORRECTION_PAGE_NAV_BPS:
        return logging.ERROR
    return logging.WARNING


def mark_correction_bound_usd(nav: float) -> float:
    """Largest total NAV mark correction that may be applied automatically."""
    return max(
        MARK_CORRECTION_MAX_USD_FLOOR,
        MARK_CORRECTION_MAX_NAV_BPS / 10_000.0 * nav,
    )


def plan_nav_mark_correction(
    mark_range_flags: Sequence[Mapping[str, Any]] | None,
    *,
    settled_closes: Mapping[str, float] | None,
    day_low: Mapping[str, float] | None,
    day_high: Mapping[str, float] | None,
    nav: float | None,
    run_date: str = "",
) -> dict[str, Any]:
    """Plan the NAV repair for every PROVABLY-wrong broker mark.

    Pure: computes and explains the correction, applies nothing. The caller
    substitutes ``nav_corrected`` for the broker NAV when ``applied`` is True.

    ``corrected_tickers`` is what ``check_custodian_marks`` consumes: a name in
    it has had its error removed from NAV and no longer justifies halting the
    pipeline. Every other flagged name still breaches.
    """
    out: dict[str, Any] = {
        "applied": False,
        "refused": False,
        "run_date": run_date,
        "nav_raw": nav,
        "nav_corrected": nav,
        "correction_usd": 0.0,
        "bound_usd": None,
        "corrections": [],
        "corrected_tickers": [],
        "unrepairable": [],
        "message": "",
    }
    if not mark_range_flags or not nav or not math.isfinite(float(nav)):
        return out
    closes = settled_closes or {}
    lows = day_low or {}
    highs = day_high or {}
    bound = mark_correction_bound_usd(float(nav))
    out["bound_usd"] = bound

    planned: list[dict[str, Any]] = []
    total = 0.0
    for flag in mark_range_flags:
        ticker = flag.get("ticker")
        try:
            shares = float(flag.get("shares") or 0)
            ib_mark = float(flag.get("ib_mark"))
        except (TypeError, ValueError):
            out["unrepairable"].append(
                {"ticker": ticker, "why": "share count or broker mark unreadable"}
            )
            continue
        close = closes.get(ticker)
        lo, hi = lows.get(ticker), highs.get(ticker)
        if not shares or close is None or lo is None or hi is None:
            out["unrepairable"].append(
                {"ticker": ticker,
                 "why": "settled close or traded range unavailable for this name"}
            )
            continue
        close = float(close)
        if not math.isfinite(close) or not (float(lo) <= close <= float(hi)):
            # The reference data disagrees with itself — see THE DISCRIMINATOR.
            out["unrepairable"].append(
                {"ticker": ticker,
                 "why": (
                     f"settled close ${close:,.2f} is itself outside the day's "
                     f"traded range [${float(lo):,.2f}, ${float(hi):,.2f}] — the "
                     "reference data is wrong, not the broker mark, so NAV is "
                     "not moved towards it"
                 )}
            )
            continue
        delta = shares * (close - ib_mark)
        total += delta
        planned.append({
            "ticker": ticker,
            "shares": shares,
            "ib_mark": ib_mark,
            "settled_close": close,
            "day_low": float(lo),
            "day_high": float(hi),
            "correction_usd": delta,
        })

    if not planned:
        out["message"] = (
            f"NAV mark correction NOT APPLIED for {run_date}: none of the "
            f"{len(mark_range_flags)} out-of-range mark(s) could be proven "
            "against a settled close inside the day's own traded range. Every "
            "flagged name still breaches."
        )
        return out

    if abs(total) > bound:
        out["refused"] = True
        out["correction_usd"] = total
        out["corrections"] = planned
        out["message"] = (
            f"NAV mark correction REFUSED for {run_date}: the total correction "
            f"${total:+,.0f} ({total / float(nav) * 100:+.3f}% of NAV) exceeds the "
            f"${bound:,.0f} bound ({MARK_CORRECTION_MAX_NAV_BPS:.0f}bps of NAV). A "
            "disagreement this large is not a stale mark — it is a different book "
            "(a share count, an unmodelled corporate action, a wrong snapshot). "
            "NAV is left as the broker reported it and the custodian gate raises."
        )
        return out

    out["applied"] = True
    out["correction_usd"] = total
    out["corrections"] = planned
    out["corrected_tickers"] = [p["ticker"] for p in planned]
    out["nav_corrected"] = float(nav) + total
    names = ", ".join(
        f"{p['ticker']} ${p['ib_mark']:,.2f}->${p['settled_close']:,.2f} "
        f"(${p['correction_usd']:+,.0f})"
        for p in planned
    )
    out["message"] = (
        f"NAV mark correction APPLIED for {run_date}: ${total:+,.0f} "
        f"({total / float(nav) * 100:+.3f}% of NAV) — broker NAV ${float(nav):,.2f} "
        f"-> ${out['nav_corrected']:,.2f}. {len(planned)} broker mark(s) fell "
        f"outside the day's traded range and were repriced to the settled close "
        f"the positions were already valued at: {names}. The raw broker NAV, the "
        "per-name correction and the original flags are all persisted."
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Custodian-mark check COVERAGE (alpha-engine-config-I9637)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY. `check_custodian_marks` and `plan_nav_mark_correction` both act only on
# what `_detect_ib_mark_outside_range` was ABLE to evaluate. Neither has any way
# to know what it could not evaluate, and until I9637 nothing did: the detector
# skipped an uncheckable name silently, so a NAV carrying an unverified broker
# mark was indistinguishable, on every surface, from a NAV whose every mark had
# been checked and passed.
#
# THE MEASURED HOLE. The ArcticDB `macro` library is Close-only — measured
# 2026-08-31, `XLK`/`SPY`/`GLD`/`VIX` each return `cols=['Close']`. Every
# symbol in `price_cache._MACRO_SYMBOLS` (the eleven sector ETFs plus GLD, USO,
# VIX, VIX3M, TNX, IRX) therefore has no traded [Low, High] to check against.
# The 2026-08-31 book held twelve universe-routed names and no macro-routed
# ones, so live coverage was 12/12 — but that was a property of the day's
# holdings, not of the gate, and no artifact recorded it either way.
#
# WHAT THIS DOES AND DELIBERATELY DOES NOT DO. It publishes coverage and warns
# when an unchecked position is MATERIAL at the same 15bp-of-NAV floor the mark
# gate itself uses. It does NOT halt: the cause is a schema limitation in a
# different repo's data library, and halting the trading pipeline on a gap this
# repo cannot close would be a fail-closed posture pointed at the wrong system.
# Closing the gap properly means OHLC in the macro library — filed separately.
# `principles.md` §2.7: no data is never rendered as green. Making the absence
# countable is what turns it from invisible into a tracked number.


def check_mark_coverage(
    positions: Mapping[str, Mapping[str, Any]] | None,
    *,
    nav: float | None,
    run_date: str = "",
) -> dict[str, Any]:
    """Report which held marks the range check could and could not evaluate.

    Reads the ``ib_mark_range_checked`` / ``ib_mark_range_uncheckable_reason``
    stamps ``eod_reconcile._detect_ib_mark_outside_range`` leaves on every
    position. Pure; returns a record for the artifact plus any warnings.
    """
    out: dict[str, Any] = {
        "run_date": run_date,
        "held": 0,
        "checked": 0,
        "unchecked": 0,
        "coverage_pct": None,
        "unchecked_names": [],
        "unchecked_market_value_usd": 0.0,
        "unchecked_material": False,
        "materiality_usd": (mark_materiality_usd(nav) if nav else None),
        "warnings": [],
    }
    if not positions:
        return out
    for ticker, pos in positions.items():
        out["held"] += 1
        if pos.get("ib_mark_range_checked"):
            out["checked"] += 1
            continue
        out["unchecked"] += 1
        try:
            mv = abs(float(pos.get("market_value") or 0.0))
        except (TypeError, ValueError):
            mv = 0.0
        out["unchecked_market_value_usd"] += mv
        out["unchecked_names"].append({
            "ticker": ticker,
            "market_value_usd": mv,
            "reason": pos.get("ib_mark_range_uncheckable_reason"),
        })
    out["coverage_pct"] = (
        out["checked"] / out["held"] * 100.0 if out["held"] else None
    )
    if not out["unchecked"]:
        return out

    names = ", ".join(
        f"{u['ticker']} (${u['market_value_usd']:,.0f})"
        for u in out["unchecked_names"]
    )
    base = (
        f"Custodian-mark check coverage {out['checked']}/{out['held']} for "
        f"{run_date}: {out['unchecked']} held position(s) could NOT be "
        f"range-checked — {names}. Their broker marks reach NAV unverified."
    )
    if nav and out["unchecked_market_value_usd"] > mark_materiality_usd(nav):
        out["unchecked_material"] = True
        out["warnings"].append(
            base + f" This is MATERIAL: ${out['unchecked_market_value_usd']:,.0f} "
            f"of unverified market value exceeds the "
            f"${mark_materiality_usd(nav):,.0f} materiality floor "
            f"({MARK_HARD_MATERIALITY_NAV_BPS:.0f}bps of NAV) the mark gate "
            "itself uses. NAV is published on marks the gate did not see."
        )
    else:
        out["warnings"].append(base)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attribution closure — the NAV mark-basis LEVEL
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY (alpha-engine-config-I8188, class sweep). ``eod_report`` published a field
# named ``ties_to_headline``, computed as ``abs(dollar_alpha − Σ components) <
# $1``, and ``eod_reconcile`` logged "investigate before trusting per-position
# contributions" when it was False. Expanding the algebra, every constructed
# term cancels — the rotation sleeve against itself, the pricing&timing sleeve
# against itself, and the per-position SPY bases telescoping to ``prior_nav``
# through the DEFINITION ``idle_cash := prior_nav − Σ held base − Σ rotated
# base``. What is left is exactly
#
#     residual = nav_change_usd − position_pnl_usd − interest_usd
#                              − unattributed_usd
#
# which is ``eod_reconcile``'s own defining equation for ``unattributed_usd``.
# It is zero by construction. The check could never fail no matter what the
# attribution did, and it asserted validation power it did not have — the same
# class as I8188 defect 1 and I8307, one layer up.
#
# There is no non-tautological identity available here: any equation containing
# the plug is satisfied by the plug's definition. The escape is to check a
# quantity measured from a SECOND, independent source. That quantity is the
# NAV mark basis:
#
#     nav_basis_level = nav_ib − (cash + accrued + Σ shares·settled_close)
#
# ``nav_ib`` is IB's NetLiquidation; the right-hand side is rebuilt from the
# broker cash balance and ArcticDB settled closes. They are independent
# measurements of the same book, so their difference CAN be non-zero, and a
# non-zero value means a share count, a settled close, or the broker NAV is
# wrong — precisely the condition under which the per-position contributions
# should not be trusted.
#
# DETECTION BLINDNESS THIS CLOSES. ``_check_nav_three_way_hard_gate`` already
# gates ``pricing_timing_usd = nav_basis_level(t) − nav_basis_level(t−1)``.
# A basis error that is CONSTANT across two sessions — a persistently wrong
# settled close, a share count wrong on both days, an unmodelled cash line —
# cancels exactly in that difference and is invisible to it. The level gate
# below is what sees it.
#
# CALIBRATION (measured over the 92 of 119 live sessions whose snapshot carries
# closing_price + shares for every name, NAV ≈ $1.03M):
#   |nav_basis_level| p50 $477 (4.6bp) · p90 $1,974 (19.9bp) ·
#   p95 $2,723 (27.1bp) · p99/max $8,125 (78.4bp, 2026-08-04 — the AMD stale
#   mark already known from the residual work).
# HARD = max($5,000, 50bp of NAV) fires on 1 of 92 sessions (1.1%): the one
# session that was genuinely broken. SOFT = max($1,000, 20bp) sits at ~p90 and
# is email-only, matching the two-band shape used by the residual and
# three-way gates.
ATTRIBUTION_BASIS_HARD_USD_FLOOR = 5_000.0
ATTRIBUTION_BASIS_HARD_NAV_BPS = 50.0
ATTRIBUTION_BASIS_SOFT_USD_FLOOR = 1_000.0
ATTRIBUTION_BASIS_SOFT_NAV_BPS = 20.0


def attribution_basis_tolerance_usd(nav: float, *, hard: bool = True) -> float:
    """Dollar bound on the NAV mark-basis LEVEL at this NAV."""
    if hard:
        return max(
            ATTRIBUTION_BASIS_HARD_USD_FLOOR,
            ATTRIBUTION_BASIS_HARD_NAV_BPS / 10_000.0 * nav,
        )
    return max(
        ATTRIBUTION_BASIS_SOFT_USD_FLOOR,
        ATTRIBUTION_BASIS_SOFT_NAV_BPS / 10_000.0 * nav,
    )


def nav_basis_level_usd(
    *,
    nav: float | None,
    total_cash: float | None,
    accrued_interest: float | None,
    positions: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Rebuild NAV from cash + settled marks and return the gap to broker NAV.

    Returns ``{"available": bool, "nav_basis_usd": float | None,
    "settled_mv_usd": float | None, "n_positions": int,
    "n_positions_unpriced": int, "reason": str | None}``.

    ``available`` is False — and the level is None, never a silent 0 — when the
    broker NAV or cash is missing, or when ANY held name lacks a settled close
    or a share count. A partial sum would understate the settled MV and
    manufacture a basis gap out of missing data, which is the failure mode this
    gate exists to distinguish from a real one.
    """
    n_positions = 0
    n_unpriced = 0
    settled_mv = 0.0
    for _tkr, pos in (positions or {}).items():
        n_positions += 1
        close = pos.get("closing_price")
        shares = pos.get("shares")
        if close in (None, "") or shares in (None, ""):
            n_unpriced += 1
            continue
        try:
            settled_mv += float(close) * float(shares)
        except (TypeError, ValueError):
            n_unpriced += 1
    if nav is None or total_cash is None:
        reason = (
            "broker NAV or cash balance absent — the settled-NAV rebuild has no "
            "left-hand side to compare against"
        )
        return {
            "available": False, "nav_basis_usd": None, "settled_mv_usd": None,
            "n_positions": n_positions, "n_positions_unpriced": n_unpriced,
            "reason": reason,
        }
    if n_unpriced:
        reason = (
            f"{n_unpriced} of {n_positions} held name(s) carry no settled close "
            "or share count — the settled-MV leg would be understated, so the "
            "basis level is NOT EVALUATED rather than reported wrong"
        )
        return {
            "available": False, "nav_basis_usd": None, "settled_mv_usd": None,
            "n_positions": n_positions, "n_positions_unpriced": n_unpriced,
            "reason": reason,
        }
    basis = float(nav) - (float(total_cash) + float(accrued_interest or 0.0) + settled_mv)
    return {
        "available": True,
        "nav_basis_usd": basis,
        "settled_mv_usd": settled_mv,
        "n_positions": n_positions,
        "n_positions_unpriced": 0,
        "reason": None,
    }


def check_attribution_closure(
    *,
    nav_basis: Mapping[str, Any],
    nav: float | None,
    components: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = "",
) -> list[dict[str, Any]]:
    """Non-tautological closure checks for the daily alpha attribution.

    Two independent failure modes, both of which CAN fire:

    ``attribution_arithmetic``
        A component's ``contrib_usd`` is absent or non-finite. A NaN silently
        poisons every downstream sum; this names the component.

    ``attribution_basis_level``
        The NAV mark-basis LEVEL exceeds :func:`attribution_basis_tolerance_usd`
        — broker NAV and the settled rebuild disagree by more than the
        settlement/mark noise band, so the per-position sleeves rest on marks
        that do not add up to the book. See this section's header for why the
        existing day-over-day gate cannot see this.

    Returns a list of breach dicts (empty = clean), same shape as the residual
    and custodian-mark gates so the caller dispatches all three identically.
    """
    breaches: list[dict[str, Any]] = []

    for idx, comp in enumerate(components or []):
        value = comp.get("contrib_usd")
        finite = False
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if finite:
            continue
        label = comp.get("label", f"#{idx}")
        breaches.append({
            "kind": "attribution_arithmetic",
            "run_date": run_date,
            "label": label,
            "kind_of_component": comp.get("kind"),
            "contrib_usd": value,
            "message": (
                f"Alpha attribution component {label!r} ({comp.get('kind')}) on "
                f"{run_date} carries a non-finite contribution ({value!r}). Every "
                "sleeve total, the per-name table and the sector rollup are "
                "computed on top of it."
            ),
        })

    if not nav:
        return breaches
    if not nav_basis.get("available"):
        # Fail loud as NOT EVALUATED — an unmeasurable gate is not a passing
        # gate. Recorded as a breach-shaped record with severity 'unevaluated'
        # so the caller surfaces it rather than reading silence as health.
        breaches.append({
            "kind": "attribution_basis_level",
            "severity": "unevaluated",
            "run_date": run_date,
            "nav_basis_usd": None,
            "tolerance_usd": attribution_basis_tolerance_usd(nav),
            "nav": nav,
            "message": (
                f"NAV mark-basis level NOT EVALUATED on {run_date}: "
                f"{nav_basis.get('reason')}. The per-position attribution is "
                "unverified for this session — this is an absent measurement, "
                "not a clean one."
            ),
        })
        return breaches

    basis = float(nav_basis["nav_basis_usd"])
    tolerance = attribution_basis_tolerance_usd(nav)
    if abs(basis) <= tolerance:
        return breaches
    breaches.append({
        "kind": "attribution_basis_level",
        "severity": "breach",
        "run_date": run_date,
        "nav_basis_usd": basis,
        "settled_mv_usd": nav_basis.get("settled_mv_usd"),
        "n_positions": nav_basis.get("n_positions"),
        "tolerance_usd": tolerance,
        "tolerance_bps": ATTRIBUTION_BASIS_HARD_NAV_BPS,
        "nav": nav,
        "pct_of_nav": basis / nav * 100.0,
        "message": (
            f"NAV mark-basis level breach on {run_date}: broker NetLiquidation "
            f"exceeds the settled rebuild (cash + accrued + Σ shares·settled "
            f"close) by ${basis:+,.0f} ({basis / nav * 100:+.3f}% of NAV), past "
            f"the ${tolerance:,.0f} bound "
            f"({ATTRIBUTION_BASIS_HARD_NAV_BPS:.0f}bps of NAV). Two independent "
            "measurements of the same book disagree, so a share count, a settled "
            "close, or the broker NAV is wrong — the per-position alpha sleeves "
            "rest on those marks. A CONSTANT basis error cancels in the "
            "day-over-day three-way gate and is invisible to it."
        ),
    })
    return breaches



# ─────────────────────────────────────────────────────────────────────────────
# 6. The BENCHMARK leg — the half of alpha nothing verified
#    (alpha-engine-config-I8188 defects 1 and 3, second pass)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY. Every gate above bounds the PORTFOLIO leg. ``plan_twr_self_heal`` states
# the exclusion in its own docstring — "``spy_return_pct``/``daily_alpha_pct``
# are NOT touched". Correct as a self-heal rule (the persisted NAV is not
# ground truth for the benchmark) and it left the benchmark leg with NO check
# of any kind. ``daily_alpha_pct`` is ``daily_return_pct − spy_return_pct``, so
# half of every alpha figure the system publishes came from a series nothing
# verified against anything.
#
# MEASURED 2026-08-31 over the full 120-session ``eod_pnl.csv`` history
# (2026-03-09 → 2026-08-28), and the two internal columns DISAGREE:
#
#     chain-linked stored spy_return_pct      = +15.5242%
#     persisted spy_close, first → last       = +13.7425%
#     gap                                     = **178.2bp**
#
# — 10.2x defect 4's 17.4bp portfolio TWR drift, on the other leg.
#
# THE OBVIOUS FIX WAS WRONG, and this comment records that because the wrong
# version would have looked right. The gap localises almost entirely to ONE
# pair, 2026-03-11 → 2026-03-13: stored ``spy_return_pct = −0.5660%`` against a
# close-implied −2.0759%. That is the exact shape of defect 4 (a stale stored
# return after an upstream correction), so the first draft of this module made
# ``spy_close`` ground truth and rewrote the stored return to match. Checked
# against the vendor before shipping: SPY closed 666.06 on 2026-03-12 and
# 662.29 on 2026-03-13, i.e. −0.5660%. **The stored return was right.**
#
# Two real defects were hiding under it, and neither is a stale value:
#
#   (a) A SESSION-COVERAGE GAP. ``eod_pnl`` has no 2026-03-12 row at all, so
#       the close-implied leg silently spanned two sessions while the stored
#       leg spanned one. Three such pairs exist over the history — 2026-03-12
#       and 2026-07-27 are missing sessions, and 2026-04-03 (Good Friday)
#       carries a row for a day the market never opened. This is I9025's cause
#       (b) again, on a third pair of series.
#
#   (b) THE ``spy_close`` COLUMN IS NOT ONE SERIES. Every persisted close
#       through 2026-03-19 sits exactly **−0.272%** below the vendor's, and
#       every close from 2026-03-20 matches it to 0.000%. 0.272% is
#       $1.797 / $659.80 — SPY's 2026-03-20 distribution. The early rows are
#       dividend-BACK-ADJUSTED and the later ones are raw: the basis changes
#       mid-column. A constant offset cancels inside a ratio, which is why 118
#       of 119 pairs agree to a median of exactly 0.0bp and nothing ever
#       surfaced it.
#
# So neither internal column can be ground truth for the other, and a gate
# built from the two of them can only ever say that they disagree. The
# resolution is the one ``check_attribution_closure`` already uses for the NAV:
# a SECOND, INDEPENDENT measurement. ``check_benchmark_vendor_anchor`` compares
# both persisted columns against the vendor's own price series and its declared
# cash distributions, which is a quantity neither column can define into
# agreement.
#
# PROVENANCE of the anchor: Polygon ``/v2/aggs`` daily closes and
# ``/v3/reference/dividends`` (``ex_dividend_date`` + ``cash_amount``) — the
# same vendor and the same vendored ``PolygonClient`` the live
# ``executor.dividends.fetch_ex_dividends`` path already uses for the
# total-return leg. Two SPY distributions fall in the live window: 2026-03-20
# at $1.796999 and 2026-06-18 at $1.903516 per share.
BENCHMARK_CHAIN_TOLERANCE_BPS = 1.0
# Per-row divergence between a persisted close and the vendor's. 5bp is well
# inside a real quote difference and well outside the 0.0bp the 112 same-basis
# rows actually show; the 8 back-adjusted rows sit at 27.2bp.
BENCHMARK_CLOSE_DIVERGENCE_TOLERANCE_BPS = 5.0
# Whole-window drift of the chain-linked stored return against the vendor's
# total-return index. 25bp is ~1.5x the portfolio leg's own historical worst
# (17.4bp) and 7x below the 178bp this gate was built on.
BENCHMARK_ANCHOR_TOLERANCE_BPS = 25.0


def _f(value: Any) -> float | None:
    """Float or None — never a silent 0.0 for an absent value."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def benchmark_implied_returns(
    rows: Sequence[Mapping[str, Any]],
    *,
    ex_dividends: Mapping[str, float] | None = None,
    prior_session_of: Any | None = None,
) -> list[dict[str, Any]]:
    """Per row: the SPY total return its own persisted closes imply, and whether
    the pair those closes span is CONTIGUOUS.

    ``rows`` is the eod_pnl series OLDEST-first. ``ex_dividends`` maps a session
    date to the SPY cash distribution going ex in the interval ending on it, and
    is consulted only where the row carries no ``spy_dividend_per_share``.

    ``prior_session_of`` is a callable ``(date_str) -> date_str`` naming the
    trading session that should immediately precede a given one — pass
    ``krepis.trading_calendar.previous_trading_day``. Without it every adjacent
    pair is assumed contiguous, which is correct for a synthetic fixture and
    WRONG for the live series: three pairs there are not (2026-03-12 and
    2026-07-27 have no row; 2026-04-03 has a row for a market holiday). A
    non-contiguous pair spans two sessions on the close leg and one on the
    stored leg, and comparing them is a category error, not a finding.
    """
    ex_dividends = ex_dividends or {}
    out: list[dict[str, Any]] = []
    prior_close: float | None = None
    prior_date: str | None = None
    for r in rows:
        date = str(r.get("date"))
        close = _f(r.get("spy_close"))
        stored = _f(r.get("spy_return_pct"))
        per_share = _f(r.get("spy_dividend_per_share"))
        if per_share is None:
            per_share = float(ex_dividends.get(date, 0.0) or 0.0)
        contiguous = True
        if prior_date is not None and prior_session_of is not None:
            try:
                contiguous = str(prior_session_of(date)) == prior_date
            except Exception:  # noqa: BLE001
                # (a) swallowed: a calendar lookup on a malformed date string;
                # (b) survives: every other pair is still evaluated; (c) recorded:
                # the pair is marked NON-contiguous, i.e. EXCLUDED, which is the
                # conservative direction — an unverifiable pair is never counted
                # as agreement.
                logger.warning("[benchmark] calendar lookup failed for %s", date)
                contiguous = False
        implied = None
        if close is not None and prior_close and contiguous:
            implied = ((close + per_share) / prior_close - 1.0) * 100.0
        out.append({
            "date": date,
            "spy_close": close,
            "spy_dividend_per_share": per_share,
            "stored_pct": stored,
            "implied_pct": implied,
            "contiguous": contiguous,
            "delta_pct": (
                stored - implied
                if (stored is not None and implied is not None)
                else None
            ),
        })
        if close is not None:
            prior_close = close
        prior_date = date
    return out


def verify_benchmark_chain_closes(
    rows: Sequence[Mapping[str, Any]],
    *,
    ex_dividends: Mapping[str, float] | None = None,
    prior_session_of: Any | None = None,
    tolerance_bps: float = BENCHMARK_CHAIN_TOLERANCE_BPS,
) -> dict[str, Any]:
    """INTERNAL consistency: stored ``spy_return_pct`` vs the persisted closes.

    Chain-links both legs over the SAME pairs — a pair that is non-contiguous,
    or whose row is missing a close or a stored return, is excluded from BOTH
    and named in ``excluded_pairs``, never left in one and dropped from the
    other.

    This gate is DETECTION ONLY and has no self-heal. It says the two internal
    columns disagree; it cannot say which is wrong, and on the live series it
    was the close column, not the return column (see the section comment). Use
    :func:`check_benchmark_vendor_anchor` to resolve a disagreement.
    """
    detail = benchmark_implied_returns(
        rows, ex_dividends=ex_dividends, prior_session_of=prior_session_of,
    )
    usable = [
        d for d in detail[1:]
        if d["stored_pct"] is not None and d["implied_pct"] is not None
    ]
    excluded = [
        {"date": d["date"],
         "reason": ("non-contiguous session pair" if not d["contiguous"]
                    else "row missing spy_close or spy_return_pct")}
        for d in detail[1:]
        if d["stored_pct"] is None or d["implied_pct"] is None
    ]
    if not usable:
        return {"status": "n/a", "closes": None, "offenders": [],
                "excluded_pairs": excluded}

    stored_chain = 1.0
    implied_chain = 1.0
    for d in usable:
        stored_chain *= 1.0 + d["stored_pct"] / 100.0
        implied_chain *= 1.0 + d["implied_pct"] / 100.0
    stored_pct = (stored_chain - 1.0) * 100.0
    implied_pct = (implied_chain - 1.0) * 100.0
    drift_bps = (stored_pct - implied_pct) * 100.0
    offenders = sorted(
        (d for d in usable if abs(d["delta_pct"]) * 100.0 > tolerance_bps),
        key=lambda d: abs(d["delta_pct"]), reverse=True,
    )
    return {
        "status": "ok",
        "closes": abs(drift_bps) <= tolerance_bps,
        "chain_linked_stored_pct": stored_pct,
        "chain_linked_implied_pct": implied_pct,
        "drift_bps": drift_bps,
        "tolerance_bps": tolerance_bps,
        "n_pairs": len(usable),
        "excluded_pairs": excluded,
        "offenders": offenders,
        "message": (
            f"Benchmark columns disagree: chain-linked spy_return_pct = "
            f"{stored_pct:+.4f}% vs {implied_pct:+.4f}% implied by the persisted "
            f"spy_close series over the same {len(usable)} contiguous pairs — "
            f"{drift_bps:+.1f}bp against a {tolerance_bps:.0f}bp tolerance. "
            f"daily_alpha_pct is daily_return_pct MINUS this series. Neither "
            f"column is ground truth for the other; resolve with the vendor "
            f"anchor. Disagreeing pair(s): "
            + ", ".join(
                f"{o['date']} stored {o['stored_pct']:+.6f}% vs close-implied "
                f"{o['implied_pct']:+.6f}%"
                for o in offenders[:5]
            )
            + ("" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)")
        ),
    }


def check_benchmark_vendor_anchor(
    rows: Sequence[Mapping[str, Any]],
    *,
    vendor_closes: Mapping[str, float],
    vendor_dividends: Mapping[str, float] | None = None,
    close_tolerance_bps: float = BENCHMARK_CLOSE_DIVERGENCE_TOLERANCE_BPS,
    anchor_tolerance_bps: float = BENCHMARK_ANCHOR_TOLERANCE_BPS,
) -> list[dict[str, Any]]:
    """A SECOND, INDEPENDENT measurement of the benchmark leg. Returns breaches.

    ``vendor_closes`` maps every trading session in the window to that vendor's
    own close; ``vendor_dividends`` maps an ex-date to the cash distribution
    per share. Both come from outside this system, which is the entire point:
    ``spy_close`` and ``spy_return_pct`` are both written by the same producer
    on the same run, so no equation built from the two of them can fail for the
    reason that matters — they can be wrong together.

    Two breach kinds:

    ``benchmark_close_divergence``
        A persisted ``spy_close`` that differs from the vendor's by more than
        ``close_tolerance_bps``. On the live history this fires on the 8 rows
        through 2026-03-19, all at 27.2bp and all in the same direction: those
        closes are dividend-back-adjusted while every later row is raw.

    ``benchmark_anchor_drift``
        The chain-linked stored ``spy_return_pct`` against the vendor's
        TOTAL-return index over the same window. The vendor index is built from
        the vendor's own consecutive sessions, so a session the book failed to
        persist is included in the benchmark — which is correct, and is the
        point: a missing trading day does not pause the benchmark, and a
        comparison that silently drops it flatters or penalises the book by
        that day's market move.

    ``vendor_closes`` missing a session the book persisted is itself reported,
    as ``benchmark_vendor_coverage`` — an anchor with holes is not an anchor,
    and it is never treated as agreement.
    """
    vendor_dividends = vendor_dividends or {}
    breaches: list[dict[str, Any]] = []
    dated = [r for r in rows if r.get("date")]
    if not dated or not vendor_closes:
        return breaches

    missing = [str(r["date"]) for r in dated if str(r["date"]) not in vendor_closes]
    if missing:
        breaches.append({
            "kind": "benchmark_vendor_coverage",
            "missing_sessions": missing,
            "n_missing": len(missing),
            "message": (
                f"The benchmark anchor does not cover {len(missing)} persisted "
                f"session(s) ({', '.join(missing[:5])}"
                + ("" if len(missing) <= 5 else f", +{len(missing) - 5} more")
                + "). An anchor with holes cannot verify the series it anchors, "
                "and its silence on those sessions is not agreement."
            ),
        })

    for r in dated:
        date = str(r["date"])
        persisted = _f(r.get("spy_close"))
        vendor = _f(vendor_closes.get(date))
        if persisted is None or not vendor:
            continue
        divergence_bps = (persisted / vendor - 1.0) * 10_000.0
        if abs(divergence_bps) <= close_tolerance_bps:
            continue
        breaches.append({
            "kind": "benchmark_close_divergence",
            "run_date": date,
            "persisted_spy_close": persisted,
            "vendor_spy_close": vendor,
            "divergence_bps": divergence_bps,
            "tolerance_bps": close_tolerance_bps,
            "message": (
                f"Persisted spy_close on {date} is {persisted:.4f} against the "
                f"vendor's {vendor:.2f} — {divergence_bps:+.1f}bp, past the "
                f"{close_tolerance_bps:.0f}bp tolerance. The benchmark leg of "
                "every alpha figure on this session rests on this number."
            ),
        })

    first, last = str(dated[0]["date"]), str(dated[-1]["date"])
    sessions = sorted(d for d in vendor_closes if first <= d <= last)
    if len(sessions) >= 2:
        index = 1.0
        for a, b in zip(sessions, sessions[1:], strict=False):
            c0, c1 = _f(vendor_closes[a]), _f(vendor_closes[b])
            if not c0 or c1 is None:
                continue
            index *= (c1 + float(vendor_dividends.get(b, 0.0) or 0.0)) / c0
        vendor_tr_pct = (index - 1.0) * 100.0

        chain = 1.0
        n = 0
        for r in dated[1:]:
            stored = _f(r.get("spy_return_pct"))
            if stored is None:
                continue
            chain *= 1.0 + stored / 100.0
            n += 1
        stored_pct = (chain - 1.0) * 100.0
        drift_bps = (stored_pct - vendor_tr_pct) * 100.0
        if abs(drift_bps) > anchor_tolerance_bps:
            breaches.append({
                "kind": "benchmark_anchor_drift",
                "window": [first, last],
                "chain_linked_stored_pct": stored_pct,
                "vendor_total_return_pct": vendor_tr_pct,
                "vendor_sessions": len(sessions),
                "persisted_sessions": n + 1,
                "drift_bps": drift_bps,
                "tolerance_bps": anchor_tolerance_bps,
                "message": (
                    f"The published benchmark series drifts from the vendor's "
                    f"own total return over {first} → {last}: chain-linked "
                    f"spy_return_pct = {stored_pct:+.4f}% against "
                    f"{vendor_tr_pct:+.4f}% ({drift_bps:+.0f}bp, tolerance "
                    f"{anchor_tolerance_bps:.0f}bp). The vendor index spans "
                    f"{len(sessions)} sessions and the book persisted {n + 1}; "
                    "a session the book missed still moved the market, so the "
                    "difference is carried straight into cumulative alpha."
                ),
            })
    return breaches


# ─────────────────────────────────────────────────────────────────────────────
# 6. Session-axis coverage — the eod_pnl date axis IS the trading calendar
# ─────────────────────────────────────────────────────────────────────────────
#
# alpha-engine-config-I9615. ``benchmark_implied_returns`` above already
# computes a per-PAIR ``contiguous`` flag and excludes a non-contiguous pair
# from both benchmark chains — but that check is private to the benchmark
# gate. ``verify_twr_closes`` and ``verify_nav_change_basis_closes`` chain-link
# the SAME date axis and check contiguity NOT AT ALL: they would silently
# chain across the exact three gaps this issue measured (2026-03-12 and
# 2026-07-27 missing; 2026-04-03 — Good Friday — an extra row) had those gaps
# happened to move ``daily_return_pct``/``nav_change_usd`` enough to breach
# tolerance. Per the issue's deliverable 3, this lifts the contiguity test
# itself into a standalone, NAMED, breachable condition every chained gate can
# call, rather than each one rediscovering it independently (or not).
#
# This is DETECTION ONLY. It never drops, reorders or synthesizes a row — the
# caller decides what a breach means (see ``pnl_measurement_backfill``'s
# ``plan_non_trading_day_flags``, which turns a ``non_trading_day_row`` breach
# into a persisted, reviewable flag rather than a silent delete: "Retain all
# archives" applies to a spurious row as much as a correct one, and the
# production write is a manual, in-region call in any case).
def check_session_axis_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    is_trading_day: Any,
    next_trading_day: Any,
) -> dict[str, Any]:
    """Assert the persisted ``date`` axis is exactly the NYSE trading calendar.

    ``rows`` is the eod_pnl series in any order; only ``date`` is read. Pass
    ``krepis.trading_calendar.is_trading_day`` / ``.next_trading_day`` (or
    ``nousergon_lib.trading_calendar``'s re-export) as string-in/string-or-bool
    -out callables: ``is_trading_day(date_str) -> bool``,
    ``next_trading_day(date_str) -> date_str``. This module stays pure — it
    never imports a calendar itself, matching ``prior_session_of`` elsewhere in
    this file.

    Two breach kinds, named separately because they are different mistakes:

    ``non_trading_day_row``
        A persisted row for a date the calendar says was never a session — the
        live 2026-04-03 case (Good Friday). Every gate that chain-links
        adjacent rows as adjacent sessions treats it as a real trading day.

    ``missing_session``
        A trading session between the first and last persisted date with no
        row at all — the live 2026-03-12 and 2026-07-27 cases. Enumerated by
        walking ``next_trading_day`` across the full window, so a gap is named
        by its OWN date rather than inferred from the pair that straddles it.

    A calendar-lookup failure on one row is NOT silently absorbed into either
    "covered" or "not covered" — it is its own breach kind
    (``calendar_lookup_failed``) so a malformed date string is visible rather
    than indistinguishable from a clean session.

    ``status`` is ``"n/a"`` on an empty series (nothing to walk).
    """
    dated = sorted({str(r["date"]) for r in rows if r.get("date")})
    if not dated:
        return {"status": "n/a", "closes": None, "breaches": []}

    breaches: list[dict[str, Any]] = []

    for d in dated:
        try:
            trading = bool(is_trading_day(d))
        except Exception as exc:  # noqa: BLE001
            # (a) swallowed: a calendar lookup on a malformed persisted date;
            # (b) survives: every other date is still checked; (c) recorded:
            # its own breach kind, never folded into "not a trading day".
            logger.warning("[session_axis] is_trading_day failed for %s", d)
            breaches.append({
                "kind": "calendar_lookup_failed",
                "date": d,
                "reason": str(exc),
                "message": f"Calendar lookup failed for persisted date {d}: {exc}",
            })
            continue
        if not trading:
            breaches.append({
                "kind": "non_trading_day_row",
                "date": d,
                "message": (
                    f"eod_pnl carries a row for {d}, which the NYSE trading "
                    "calendar says was never a session. Every chain-linked "
                    "gate (TWR closure, nav_change basis, benchmark closure) "
                    "treats adjacent rows as adjacent sessions unless this row "
                    "is excluded — it contributes a spurious session to all "
                    "of them."
                ),
            })

    if len(dated) >= 2:
        cursor = dated[0]
        present = set(dated)
        seen = {cursor}
        while cursor < dated[-1]:
            try:
                nxt = str(next_trading_day(cursor))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[session_axis] next_trading_day failed for %s", cursor)
                breaches.append({
                    "kind": "calendar_lookup_failed",
                    "date": cursor,
                    "reason": str(exc),
                    "message": f"Calendar lookup failed walking forward from {cursor}: {exc}",
                })
                break
            if nxt in seen:
                # A calendar that does not advance would spin forever; treat as
                # a lookup failure rather than looping.
                breaches.append({
                    "kind": "calendar_lookup_failed",
                    "date": nxt,
                    "reason": "next_trading_day did not advance",
                    "message": (
                        f"next_trading_day({cursor}) returned a date already "
                        "seen while walking the window — refusing to loop."
                    ),
                })
                break
            cursor = nxt
            seen.add(cursor)
            if cursor > dated[-1]:
                break
            if cursor not in present:
                breaches.append({
                    "kind": "missing_session",
                    "date": cursor,
                    "message": (
                        f"NYSE trading session {cursor} has no eod_pnl row. "
                        "Every chain-linked gate treats its neighbors as "
                        "adjacent sessions unless this gap is named — it is "
                        "what let a two-session close-price move be compared "
                        "against a one-session stored return."
                    ),
                })

    return {
        "status": "ok",
        "closes": not breaches,
        "first_date": dated[0],
        "last_date": dated[-1],
        "n_rows": len(dated),
        "breaches": breaches,
        "message": (
            "eod_pnl date axis matches the NYSE trading calendar exactly over "
            f"{dated[0]} → {dated[-1]} ({len(dated)} sessions)."
            if not breaches else
            f"eod_pnl date axis diverges from the NYSE trading calendar over "
            f"{dated[0]} → {dated[-1]}: {len(breaches)} breach(es) — "
            + ", ".join(f"{b['kind']}:{b['date']}" for b in breaches[:8])
            + ("" if len(breaches) <= 8 else f" (+{len(breaches) - 8} more)")
        ),
    }
