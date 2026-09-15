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

**Per-ticker allocation of the pricing & timing sleeve (config#2046).** The
aggregate telescopes exactly into a per-name sum, since
``mark_basis(t) = Σ_{held today} (ib_market_value_i − market_value_i)`` and
``mark_basis(t−1) = Σ_{held yesterday} (ib_market_value_i − market_value_i)``
(settled) — both persisted per position since schema 2.1 (crucible-executor
PR343). So ``pricing_timing_usd = Σ_i [basis_today_i − basis_prior_i]`` over
every ticker held on either day (a same-day entry/exit nets against a zero
baseline on the day it wasn't held). Each name's slice is folded into its own
sleeve — held/entered names into their ``position`` component, fully-exited
names into ``rotation`` — so "Pricing & Timing" as a *generic, unattributed*
bucket disappears on any day schema-2.1 data is complete. When either day's
``ib_market_value``/``market_value`` is missing for a name (pre-PR343 legacy
snapshot), that name's slice is deliberately left OUT of the per-ticker split
— never guessed — and falls through to the ``reconciliation`` component,
which now represents only the genuinely-unattributable leftover (per
feedback_no_silent_fails: attribute what you can name, label what you can't).

**Unattributed is intentionally NOT decomposed per ticker.** Unlike pricing &
timing, the true residual (fees, FX, untracked corporate actions, snapshot
timing mismatches) has no per-position basis in the data the executor
currently captures — force-allocating it to a ticker would be a fabricated
number, not a derived one. It stays a portfolio-level sleeve until a producer
change gives it one (see config#2046 discussion).

Dividends earned are folded into each position's ``daily_return_usd`` (hence into
``position_pnl_usd``); ``dividend_usd`` is informational only — so the reconcile
identity has no separate dividend term and dividends are neither double-counted
nor dropped.

That last sentence was true of the MECHANISM and false of the RESULT until
alpha-engine-config-I8188: the only dividend source was IB's per-symbol accrual,
which a paper account never populates, so ``dividend_usd`` was $0.00 on all 115
live sessions while the book held seven payers. The dividends were not dropped
from the RETURN (NAV is IB NetLiquidation, which the cash reaches) — they were
dropped from the ATTRIBUTION and silently credited to ``unattributed_usd``.
``executor/dividends.py`` now accrues them explicitly on the ex-date from the
Polygon corporate-actions feed, and ``spy_return_pct`` is TOTAL return on the
benchmark leg so both sides of the comparison use the same return definition.

This also fixes the old emailer's "α % of Total" column, which divided each
position's alpha by the *signed* grand-total alpha — so a position with genuinely
positive alpha rendered negative whenever the day's total alpha was negative, and
the table total (Σ $-alpha / NAV) never reconciled with the NAV-based headline.
See ``tests/test_eod_report.py``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Mapping
from typing import Any

import boto3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.12"
# 2.12 (alpha-engine-config-I9638): ``nav_reconciliation`` gains the NAV-basis
# shadow block — ``nav_basis`` (which figure ``summary.nav`` is struck on,
# ``ib_netliq`` or ``settled_close``), ``nav_ib_usd`` (post-mark-correction IB
# NetLiquidation), ``nav_settled_usd`` (cash + accrued + Σ shares·settled
# close), ``nav_basis_diff_usd``/``nav_basis_diff_bps``, plus
# ``mark_basis_delta_usd`` (the day-over-day IB-vs-settled divergence, which
# equals ``pricing_timing_usd`` only while the basis is ``ib_netliq``) and the
# absent-measurement fields ``nav_basis_unavailable_reason`` /
# ``nav_settled_fallback_tickers``. Both figures are published under either
# basis: this is the shadow stage of the ruled cut-over, and a reader grading
# it must never have to infer which definition a row's NAV carries.
# 2.11 (alpha-engine-config-I10048): each position row gains
# ``ib_market_value_raw``, ``ib_mark_corrected`` and ``ib_mark_correction_usd``.
# ``ib_market_value`` CHANGES MEANING on a corrected name only: it is now the
# mark the headline NAV was struck on (``shares × settled_close``) rather than
# the broker's raw number, which moves to ``ib_market_value_raw``. This is the
# only non-additive step since 2.4, and it is the point: 2.8 corrected NAV and
# left the position row raw, so the two disagreed and the NEXT day's mark-basis
# attribution differenced a corrected NAV against a raw prior mark, re-surfacing
# the correction as ``nav_identity_residual_usd`` (+$1,605, HOOD 2026-09-03) and
# mislabelling the breach ``reconcile_defect``. Every range flag
# (``ib_mark_outside_range``, ``ib_mark_range_error_usd``,
# ``ib_mark_range_checked``) is stamped BEFORE the correction and is unchanged,
# so a repaired row is never readable as one that was never wrong.
# 2.7 (alpha-engine-config-I8188, second pass): additive only.
# ``transaction_costs`` gains ``gross_available`` and
# ``gross_unavailable_reason``. When IB attaches no commissionReport to any
# fill, ``commission_usd`` is now None rather than 0.0 and
# ``daily_return_gross_pct`` is SUPPRESSED rather than published with a $0.00
# commission leg — that figure was a net return wearing a gross label. The
# ``commission_available`` flag already existed and stays; what changes is that
# the FIGURES now agree with it instead of contradicting it. Measured: 1 of the
# first 6 live sessions carrying the column persisted commission_usd = 0.0 with
# commission_available = 0.
# 2.6 (alpha-engine-config-I9085): additive only. Each position row gains
# ``mark_basis_usd`` (``ib_market_value − market_value``) and
# ``ib_mark_off_close_pct``. The pre-existing ``ib_mark_outside_range`` flag
# is a strict SUBSET of these — a mark inside the day's traded range can
# still be off the settled close by a full percent and still moves
# ``pricing_timing_usd`` dollar for dollar. No 2.5 field changed meaning.
# 2.5 (alpha-engine-config-I8188, class sweep): additive only. The
# ``alpha_attribution`` block gains ``identity_is_tautological``,
# ``identity_residual_note``, ``components_finite`` and ``component_sum_usd``.
# ``residual_usd`` and ``ties_to_headline`` are RETAINED unchanged — they are
# zero/True by construction and are now labelled as such rather than removed,
# so a consumer reading them still parses while a consumer TRUSTING them can
# see, in the artifact itself, that they validate nothing.
# 2.4 (alpha-engine-config-I8188): adds the ``transaction_costs`` block
# (named commission/slippage lines plus the gross-of-cost vs net-of-cost
# split) and the ``integrity`` block (residual-bound / custodian-mark /
# TWR-closure gate outcomes, dividend-accrual availability, and the explicit
# declaration that ``spy_return_pct`` is now TOTAL return, not price return).
# Both are additive; no 2.3 field changed meaning.
#
# 2.9 (alpha-engine-config-I9637) adds ``integrity.mark_check_coverage`` — the
# custodian-mark gate's own coverage. The ArcticDB macro library is Close-only,
# so a macro-routed holding has no traded range to check against and its mark
# reached NAV unverified AND unrecorded. Additive.
#
# 2.10 (alpha-engine-config-I9637) carries ``ib_mark_range_checked`` and
# ``ib_mark_range_uncheckable_reason`` onto each POSITION row. 2.9 published
# the book-level count but not the per-name stamp, so a reader inspecting one
# position still could not tell "checked and in range" from "never evaluated" —
# the ambiguity relocated rather than closed. Additive.
#
# 2.8 (alpha-engine-config-I9627) adds ``integrity.nav_mark_correction``: the
# broker NAV as received, the per-name repair applied to any provably-wrong
# mark, and whether the repair was applied or refused. ``summary.nav`` is the
# CORRECTED figure on a correction day, so without this block the published
# headline could not be traced back to what IB sent. Additive.

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


def _trade_attribution_price(t: Mapping[str, Any] | dict) -> tuple[float | None, bool]:
    """Price a trade row contributes to the attribution, and whether it is a FILL.

    Order of authority: ``fill_price`` (the broker execution, share-weighted
    VWAP) → ``price`` (the mapped ``_trades_today`` shape) → ``price_at_order``
    (the arrival price — an ESTIMATE the daemon wrote before the fill arrived,
    or the reference price a Working row was logged with). The second element
    is False whenever the fallback was taken, so a caller can report that this
    row's implementation shortfall is inside the residual rather than inside
    the position/rotation sleeve it belongs to.

    Reading ``price_at_order`` first is how a $3.33 gap between a REDUCE's
    estimate and its real fill reached the unattributed residual and failed the
    postclose pipeline on 2026-09-14 (alpha-engine-config-I10800); it also put
    every session's slippage into the residual by construction, since
    ``session_costs`` measures shortfall against the same arrival price.
    """
    for key, is_fill in (("fill_price", True), ("price", False), ("price_at_order", False)):
        raw = t.get(key)
        if raw in (None, ""):
            continue
        try:
            px = float(raw)
        except (TypeError, ValueError):
            continue
        if px > 0:
            return px, is_fill
    return None, False


def _weighted_prices(trades_today: list[dict] | None, actions, side_word: str) -> dict[str, float]:
    agg: dict[str, list[float]] = {}
    for t in trades_today or []:
        action = str(t.get("action", "")).upper()
        if action not in actions and side_word not in action:
            continue
        tkr = t.get("ticker")
        px, _ = _trade_attribution_price(t)
        try:
            sh = abs(float(t.get("shares") or 0))
        except (TypeError, ValueError):
            continue
        if not tkr or sh <= 0 or px is None:
            continue
        slot = agg.setdefault(tkr, [0.0, 0.0])
        slot[0] += sh * px
        slot[1] += sh
    return {k: v[0] / v[1] for k, v in agg.items() if v[1] > 0}


def _sell_exit_prices(trades_today: list[dict] | None) -> dict[str, float]:
    """Share-weighted average sell/exit fill price per ticker from today's trades."""
    return _weighted_prices(trades_today, _SELL_ACTIONS, "SELL")


def _buy_entry_prices(trades_today: list[dict] | None) -> dict[str, float]:
    """Share-weighted average buy/enter fill price per ticker from today's trades."""
    return _weighted_prices(trades_today, _BUY_ACTIONS, "BUY")


def arrival_priced_trades(trades_today: list[dict] | None) -> list[dict]:
    """Rows the attribution had to price at the ARRIVAL price, not a fill.

    Each entry names the row (``ticker``, ``action``, ``shares``, ``status``,
    ``price``) so the EOD run can put it in ``data_warnings``: the row's
    shortfall against its real fill is sitting in the unattributed residual,
    and if the residual bounds gate fires, this list is the first suspect.
    """
    out: list[dict] = []
    for t in trades_today or []:
        px, is_fill = _trade_attribution_price(t)
        if px is None or is_fill:
            continue
        out.append({
            "ticker": t.get("ticker"),
            "action": t.get("action"),
            "shares": t.get("shares"),
            "status": t.get("status"),
            "price": px,
        })
    return out


def compute_pricing_timing_by_ticker(
    positions: dict,
    prior_positions: dict | None,
) -> tuple[dict[str, float], int]:
    """Full-book per-ticker decomposition of the pricing & timing term.

    Returns ``(by_ticker, uncovered)`` where ``by_ticker[tkr]`` is that
    name's day-over-day mark basis
    ``(ib_market_value - market_value)_t - (ib_market_value - market_value)_{t-1}``
    and ``uncovered`` counts names skipped because one of the two days'
    snapshots lacked ``ib_market_value``/``market_value``. Summed, the map is
    the full-book mark-basis quantity ``pricing_timing_usd`` is built from —
    see the module docstring's telescoping proof.

    A name is included only when BOTH days carry the fields; otherwise its
    dollars are left out entirely and fall through to the caller's residual,
    never guessed. ``uncovered`` makes that omission visible to callers that
    need to know the sum is a lower bound (alpha-engine-config-I8733).
    """
    by_ticker: dict[str, float] = {}
    uncovered = 0
    for tkr in set(positions or {}) | set((prior_positions or {}).keys()):
        pos_t = (positions or {}).get(tkr)
        pos_p = (prior_positions or {}).get(tkr)
        if pos_t is None:
            basis_today = 0.0
        else:
            ib_t, mv_t = pos_t.get("ib_market_value"), pos_t.get("market_value")
            basis_today = (
                (ib_t - mv_t) if (ib_t is not None and mv_t is not None) else None
            )
        if pos_p is None:
            basis_prior = 0.0
        else:
            ib_p, mv_p = pos_p.get("ib_market_value"), pos_p.get("market_value")
            basis_prior = (
                (ib_p - mv_p) if (ib_p is not None and mv_p is not None) else None
            )
        if basis_today is None or basis_prior is None:
            uncovered += 1
            continue
        delta = basis_today - basis_prior
        if delta:
            by_ticker[tkr] = delta
    return by_ticker, uncovered


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


def _is_finite(value) -> bool:
    """True only for a real, finite number — None/str/NaN/inf are all False."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    ``position`` (held, SPY-benchmarked on its *retained* prior MV, plus its
    own slice of the pricing&timing basis gap when attributable), ``rotation``
    (shares sold out today, plus the exited names' slice of that gap),
    ``cash`` (interest − SPY on genuine idle cash), ``reconciliation``
    ("Pricing & timing" — only the leftover portion no ticker could be
    attributed, no SPY base), and ``unattributed`` (the TRUE residual after
    rotation + pricing&timing are lifted out; deliberately NOT per-ticker —
    see module docstring).

    Returns ``None`` when attribution is undefined (no prior NAV, or no SPY
    reference). Otherwise the ``components`` ``contrib_usd`` values sum to
    ``dollar_alpha`` (``residual_usd`` is the tie-out check).
    """
    if prior_nav is None or prior_nav <= 0 or spy_return is None:
        return None

    spy_frac = spy_return / 100.0
    nav_change = nav_change_usd or 0.0
    dollar_alpha = nav_change - spy_frac * prior_nav

    # ── Per-ticker pricing & timing decomposition (config#2046) ──────────────
    # See the module docstring for the telescoping proof. A name's slice is
    # computed only when BOTH days' ``ib_market_value``/``market_value`` are
    # present for it; otherwise it is left out of the map entirely and its
    # dollars fall through to the ``reconciliation`` residual below — never
    # guessed.
    pt = float(pricing_timing_usd or 0.0) if pricing_timing_available else 0.0
    pt_by_ticker = (
        compute_pricing_timing_by_ticker(positions, prior_positions)[0]
        if pricing_timing_available
        else {}
    )
    pt_attributed_total = sum(pt_by_ticker.values())

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
        position_alpha = daily_usd - spy_frac * spy_base
        pt_contrib = pt_by_ticker.get(ticker, 0.0)
        contrib = position_alpha + pt_contrib
        components.append({
            "label": ticker, "kind": "position",
            "contrib_usd": contrib, "contrib_bps": contrib / prior_nav * 1e4,
            "position_alpha_usd": position_alpha,
            "pricing_timing_usd": pt_contrib,
        })

    # ── Rotation sleeve (shares sold out today: full exits + trims) ───────────
    exit_px = _sell_exit_prices(trades_today)
    rotation_spy_base = 0.0
    rotation_dollar = 0.0
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
        ticker_realized = (px - prior_close) * sold
        rotation_dollar += ticker_realized
        rotation_spy_base += prior_close * sold
        rotation_alpha = ticker_realized - spy_frac * prior_close * sold
        # Pricing & timing: only include if fully exited (trim's pt is in its position row).
        pt_contrib = pt_by_ticker.get(tkr, 0.0) if tkr not in positions else 0.0
        rot_contrib = rotation_alpha + pt_contrib
        components.append({
            "label": tkr, "kind": "rotation",
            "contrib_usd": rot_contrib, "contrib_bps": rot_contrib / prior_nav * 1e4,
            "rotation_alpha_usd": rotation_alpha,
            "pricing_timing_usd": pt_contrib,
        })

    # ── Cash sleeve (genuine idle cash only) ──────────────────────────────────
    idle_cash = prior_nav - sum_held_spy_base - rotation_spy_base
    cash_contrib = float(interest_usd or 0.0) - spy_frac * idle_cash
    components.append({
        "label": "Cash", "kind": "cash",
        "contrib_usd": cash_contrib, "contrib_bps": cash_contrib / prior_nav * 1e4,
    })

    # ── Pricing & timing reconciliation residual ──────────────────────────────
    # Only the slice no ticker could be attributed (see decomposition above) —
    # ~$0 whenever both days' snapshots carry schema-2.1 ib_market_value.
    pt_unattributable = pt - pt_attributed_total
    if pricing_timing_available:
        components.append({
            "label": "Pricing & timing", "kind": "reconciliation",
            "contrib_usd": pt_unattributable,
            "contrib_bps": pt_unattributable / prior_nav * 1e4,
        })

    # ── True unattributed residual (rotation + pricing&timing lifted out) ─────
    # Deliberately portfolio-level, not per-ticker — see module docstring.
    unattr_true = float(unattributed_usd or 0.0) - rotation_dollar - pt
    components.append({
        "label": "Unattributed", "kind": "unattributed",
        "contrib_usd": unattr_true, "contrib_bps": unattr_true / prior_nav * 1e4,
    })

    summed = sum(c["contrib_usd"] for c in components)
    # ``residual`` is IDENTICALLY ZERO — it is not a check (I8188 class sweep).
    # Expanding the algebra, the rotation sleeve cancels against itself, the
    # pricing&timing sleeve cancels against itself, and the per-position SPY
    # bases telescope to ``prior_nav`` through the DEFINITION of ``idle_cash``
    # just above. What survives is
    #     nav_change_usd − position_pnl_usd − interest_usd − unattributed_usd
    # which is ``eod_reconcile``'s defining equation for ``unattributed_usd``.
    # Both fields are RETAINED (S3 schema rule: add, never remove) but are now
    # labelled for what they are; the check with actual power over these
    # sleeves is ``pnl_integrity.check_attribution_closure``, dispatched by
    # ``eod_reconcile``, which compares the broker NAV against an independently
    # rebuilt settled NAV.
    residual = dollar_alpha - summed
    components_finite = all(
        _is_finite(c.get("contrib_usd")) for c in components
    )

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
        "pricing_timing_by_ticker": pt_by_ticker,
        "pricing_timing_unattributable_usd": pt_unattributable,
        "unattributed_true_usd": unattr_true,
        "idle_cash": idle_cash,
        "residual_usd": residual,
        "ties_to_headline": abs(residual) < 1.0,
        # ── Honesty fields (alpha-engine-config-I8188 class sweep) ──────────
        # ``ties_to_headline`` is TRUE BY CONSTRUCTION and validates nothing;
        # a consumer must not read it as evidence the attribution is sound.
        "identity_is_tautological": True,
        "identity_residual_note": (
            "residual_usd is zero by construction — unattributed_usd is the "
            "plug of the very identity it closes. Read attribution_closure "
            "in the integrity breaches instead."
        ),
        # This one CAN be False: a NaN/inf/absent contribution poisons every
        # downstream sum and is not visible in residual_usd (NaN propagates
        # into `summed`, and abs(NaN) < 1.0 is False only by accident of
        # ordering, never by design).
        "components_finite": components_finite,
        "component_sum_usd": summed,
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
    integrity_breaches: list | None = None,
    nav_mark_correction: dict | None = None,
    mark_check_coverage: dict | None = None,
    twr_closure: dict | None = None,
    dividend_accrual_available: bool | None = None,
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
            # Schema 2.3 (config#6349/#6818): IB's raw mark before the
            # settled-close override, so a NAV hard-gate breach is
            # self-diagnosing from the artifact instead of a by-hand trace.
            "ib_market_value": pos.get("ib_market_value"),
            # Schema 2.9 (alpha-engine-config-I10048): when the I9627 mark
            # correction applies, `ib_market_value` above is the mark the
            # headline NAV was actually struck on (`shares × settled_close`),
            # not the broker's. The broker's number is preserved here so the
            # artifact still shows what IB sent, and `ib_mark_corrected` says
            # the row was repaired rather than never wrong. Before this the
            # two disagreed: NAV moved and the position row did not, so the
            # next day's mark-basis attribution differenced a corrected NAV
            # against a raw prior mark and re-surfaced the correction as
            # `nav_identity_residual_usd` (+$1,605, HOOD 2026-09-03).
            "ib_market_value_raw": pos.get("ib_market_value_raw"),
            "ib_mark_corrected": pos.get("ib_mark_corrected", False),
            "ib_mark_correction_usd": pos.get("ib_mark_correction_usd"),
            # `ib_mark_outside_range` alone cannot say whether the check RAN
            # (alpha-engine-config-I9637). Its `, False` default makes an
            # unevaluated mark read exactly like one that was checked and
            # passed — the same ambiguity `mark_check_coverage` closes for the
            # book as a whole, still open at the row level until this field
            # travels with it. Measured on the 2026-08-31 artifact: every
            # position carried `ib_mark_outside_range: false` and no reader
            # could tell that from `ib_mark_range_checked: null`.
            "ib_mark_outside_range": pos.get("ib_mark_outside_range", False),
            "ib_mark_range_checked": pos.get("ib_mark_range_checked"),
            "ib_mark_range_uncheckable_reason": pos.get(
                "ib_mark_range_uncheckable_reason"
            ),
            "ib_mark_range_error_usd": pos.get("ib_mark_range_error_usd"),
            # Schema 2.4 (alpha-engine-config-I9085): the out-of-range flag
            # above is a strict SUBSET of "the IB mark is not the settled
            # close". A mark inside the day's traded range but off the close
            # moves `pricing_timing_usd` dollar for dollar and is invisible to
            # that flag — MU 2026-08-27, mark $923.07 vs close $935.39
            # (-1.32%, in range), carried 75% of a -$4,393 hard-gate breach
            # with zero tickers flagged.
            "mark_basis_usd": pos.get("mark_basis_usd"),
            "ib_mark_off_close_pct": pos.get("ib_mark_off_close_pct"),
            "pct_nav": (mv / nav * 100.0) if nav else None,
            "daily_return_pct": pos.get("daily_return_pct"),
            "daily_return_usd": pos.get("daily_return_usd"),
            "alpha_contrib_usd": contrib["contrib_usd"] if contrib else None,
            "alpha_contrib_bps": contrib["contrib_bps"] if contrib else None,
            # Schema 2.2 (config#2046): breakdown of alpha_contrib_usd into the
            # pure settled-close economic alpha vs this name's own slice of the
            # pricing&timing (IB-mark-vs-settled) basis gap.
            "position_alpha_usd": contrib.get("position_alpha_usd") if contrib else None,
            "pricing_timing_contrib_usd": contrib.get("pricing_timing_usd") if contrib else None,
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
            "dividend_timing_usd": recon.get("dividend_timing_usd"),
            "dividend_receivable_usd": recon.get("dividend_receivable_usd"),
            "unattributed_usd": recon.get("unattributed_usd"),
            "pricing_timing_usd": recon.get("pricing_timing_usd"),
            "pricing_timing_available": recon.get("pricing_timing_available"),
            # ── NAV basis shadow (schema 2.12, alpha-engine-config-I9638) ──
            # Both candidate NAV figures and their gap, published on every
            # run under either basis. `nav_basis` names which one
            # `summary.nav` — and therefore daily_return_pct,
            # daily_alpha_pct and every position weight — is struck on.
            "nav_basis": recon.get("nav_basis"),
            "nav_ib_usd": recon.get("nav_ib_usd"),
            "nav_settled_usd": recon.get("nav_settled_usd"),
            "nav_basis_diff_usd": recon.get("nav_basis_diff_usd"),
            "nav_basis_diff_bps": recon.get("nav_basis_diff_bps"),
            # Non-null only when the settled rebuild could NOT be made, or was
            # made with a broker mark standing in for a missing settled close.
            # An absent measurement is reported as absent, never as agreement.
            "nav_basis_unavailable_reason": recon.get("nav_basis_unavailable_reason"),
            "nav_settled_fallback_tickers": recon.get("nav_settled_fallback_tickers"),
            "mark_basis_delta_usd": recon.get("mark_basis_delta_usd"),
            "rotation_realized_usd": (
                attribution.get("rotation_realized_usd") if attribution else None
            ),
            "unattributed_true_usd": (
                attribution.get("unattributed_true_usd") if attribution else None
            ),
            # Schema 2.2 (config#2046): the leftover slice of pricing&timing
            # no ticker could be attributed — ~0 when schema-2.1 data is
            # complete for every held/exited name.
            "pricing_timing_unattributable_usd": (
                attribution.get("pricing_timing_unattributable_usd") if attribution else None
            ),
        },
        # ── Explicit transaction costs (alpha-engine-config-I8188) ──────────
        # Named cost lines and the gross/net split they make possible. Before
        # this block there was no transaction-cost figure anywhere in the P&L
        # artifacts, so gross and net performance were the same number and
        # neither was labelled. ``commission_available`` is carried so an
        # ABSENT commission figure never renders as a measured $0.00.
        "transaction_costs": {
            "commission_usd": recon.get("commission_usd"),
            "commission_available": recon.get("commission_available"),
            "slippage_usd": recon.get("slippage_usd"),
            "slippage_bps": recon.get("slippage_bps"),
            "traded_notional_usd": recon.get("traded_notional_usd"),
            "total_cost_usd": recon.get("total_cost_usd"),
            "daily_return_net_pct": recon.get("daily_return_net_pct"),
            "daily_return_gross_pct": recon.get("daily_return_gross_pct"),
            "gross_available": recon.get("gross_available"),
            "gross_unavailable_reason": recon.get("gross_unavailable_reason"),
        },
        # Outcome of the integrity gates, so the surface a human reads carries
        # the same verdict the pipeline exit code does.
        "integrity": {
            "breaches": list(integrity_breaches or []),
            # Schema 2.8 (alpha-engine-config-I9627): the broker-mark repair.
            # `nav` above is the CORRECTED figure whenever `applied` is true,
            # so the artifact must carry the raw broker NAV beside it or the
            # published number cannot be traced to what IB sent.
            "nav_mark_correction": nav_mark_correction,
            # Schema 2.9 (alpha-engine-config-I9637): how many held marks the
            # range check could actually evaluate. Without it, a NAV carrying
            # an unverified broker mark reads identically to one whose every
            # mark was checked and passed.
            "mark_check_coverage": mark_check_coverage,
            "twr_closure": twr_closure,
            "dividend_accrual_available": dividend_accrual_available,
            "spy_return_basis": "total_return",
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
