"""
EOD reconciliation — runs at 4:05pm ET after market close.

Captures portfolio NAV, computes daily return vs. SPY, writes to eod_pnl table.

Cron:  5 21 * * 1-5  python /home/ec2-user/alpha-engine/executor/eod_reconcile.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time as _time
from datetime import date, timedelta

import boto3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.dates import now_dual
from nousergon_lib.logging import guard_entrypoint, setup_logging
from nousergon_lib.trading_calendar import previous_trading_day

from executor import reference_rate
from executor.dividends import (
    SPY_TICKER,
    accrue_position_dividends,
    fetch_ex_dividends,
    spy_total_return_pct,
)
from executor.eod_emailer import send_eod_email
from executor.eod_report import build_eod_report, write_eod_report
from executor.market_hours import is_trading_day
from executor.pnl_backfill import backfill_residual_sleeves
from executor.pnl_integrity import (
    RESIDUAL_CUMULATIVE_WINDOW_SESSIONS,
    check_attribution_closure,
    check_benchmark_vendor_anchor,
    check_custodian_marks,
    check_mark_coverage,
    check_residual_bounds,
    gross_net_returns,
    nav_basis_level_usd,
    plan_nav_mark_correction,
    plan_twr_self_heal,
    session_costs,
    verify_nav_change_basis_closes,
    verify_twr_closes,
)
from executor.pnl_measurement_backfill import fetch_live_anchor_window, heal_cost_columns
from executor.trade_logger import (
    backup_to_s3,
    dividend_receivable_usd,
    get_entry_trade,
    get_todays_trades,
    init_db,
    log_eod,
    record_dividend_accrual,
    settle_due_dividend_accruals,
)

# See executor/main.py for the rationale on IB Error 10197 / 10349 suppression.
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)

_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging("eod", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

# ── Close provenance (alpha-engine-config-I10360) ────────────────────────────
#
# ArcticDB's Close for a given trading day is written TWICE by different
# vendors, by design (alpha-engine-data collectors/daily_closes.py):
#
#   ~4:05 PM ET, run_date      — ``yfinance_only`` EOD pass. yfinance's
#                                same-day 1d bar, ``auto_adjust=False``.
#                                Polygon is skipped because its free tier
#                                returns 403 for same-day grouped-daily.
#   ~5:30 AM PT, run_date + 1  — ``polygon_only`` morning pass. Polygon
#                                grouped-daily, corporate-action ADJUSTED,
#                                with true VWAP. ``_SOURCE_PRIORITY`` ranks
#                                polygon 3 vs yfinance 1, so it deliberately
#                                OVERWRITES the row.
#
# This reconcile runs inside the postclose Step Function, i.e. between those
# two writes — so the close it freezes is ALWAYS the provisional yfinance
# print, and the T+1 ``reconcile_audit`` pass ALWAYS reads the polygon one.
# The value did not "revise"; the SOURCE was replaced on a schedule.
#
# Recording which vendor produced the close we froze is what lets the audit
# tell that expected substitution apart from a genuine data-integrity event.
# Without it the audit compares two vendors while asserting one source, and
# classifies the difference as ``corruption`` on ~60% of trading days.
#
# The ``universe`` library carries the per-row ``source`` column; the ``macro``
# library stores Close ONLY, so macro-routed symbols degrade to "unknown" and
# the audit keeps its pre-existing magnitude band for them (loud degradation,
# not a silent pass) — tracked as alpha-engine-config-I10362.
CLOSE_SOURCE_UNKNOWN = "unknown"


def _row_close_source(match: pd.DataFrame) -> str:
    """Vendor provenance of the matched ArcticDB row, or ``CLOSE_SOURCE_UNKNOWN``.

    Never raises and never guesses: an absent column, a null cell or an empty
    string all resolve to "unknown", which the audit treats as "cannot tell"
    and therefore holds to the STRICTER pre-provenance classification.
    """
    if "source" not in match.columns:
        return CLOSE_SOURCE_UNKNOWN
    raw = match["source"].iloc[-1]
    if raw is None:
        return CLOSE_SOURCE_UNKNOWN
    if isinstance(raw, float) and raw != raw:  # NaN, without a pd.isna try/except
        return CLOSE_SOURCE_UNKNOWN
    text = str(raw).strip().lower()
    return CLOSE_SOURCE_UNKNOWN if text in ("", "nan", "none", "<na>") else text


from executor.config_loader import (  # noqa: E402 -- must follow setup_logging above
    NAV_BASIS_IB_NETLIQ,
    NAV_BASIS_SETTLED_CLOSE,
    load_config,
    resolve_nav_basis,
)

# ── NAV three-way reconcile — hard-gate tolerance (config#2457) ────────────
# `pricing_timing_usd` (mark_basis_today − mark_basis_prior, computed below)
# is the three-way divergence signal between the broker-reported NAV (IB
# NetLiquidation, the independent witness) and the system/settled NAV (cash +
# accrued + Σ settled position market values). It has fed `data_warnings`
# (EOD email only) since config#1276 at a SOFT threshold of max($500, 5bps of
# NAV) — appropriate for "a human notices this in the email eventually."
#
# A HARD gate that pages flow-doctor (→ Telegram/GitHub/changelog) needs a
# WIDER band than the soft warning, or it pages exactly as often as the email
# already flags and trains the operator to ignore it — this is the same
# lesson reconcile_audit.py's PAGE_THRESHOLD_BPS encodes for the SPY
# self-heal pass (config#2145: an in-band 1.46bp correction paged identically
# to a real incident). 3x the soft-warning floor and rate is a deliberate,
# named choice: routine settlement-lag / mark-timing noise stays inside the
# existing soft band (email-visible, not paged); only a divergence large
# enough to plausibly indicate a broken NAV input (bad IB session, stale
# settled marks, a real reconciliation break) pages.
NAV_HARD_GATE_TOLERANCE_USD_FLOOR = 2500.0
NAV_HARD_GATE_TOLERANCE_NAV_BPS = 15.0  # 0.15% of NAV



def _log_paged(msg: str, *args) -> None:
    """Log a condition that ``fd.report`` pages in the same breath.

    WARNING, not ERROR, on purpose (alpha-engine-config-I10049): the root
    ``FlowDoctorHandler`` (attached at ERROR by ``setup_logging``) turns every
    ERROR record into its own flow-doctor report, so an ERROR line beside an
    ``fd.report`` of the same condition pages the operator TWICE and auto-files
    two issues for one event — measured 2026-09-04, reports 0.2 ms apart
    (I10018 + I10019). The severity of record lives on the ``fd.report`` call,
    which also carries the classification-dependent severity and the context
    dict the log line cannot. The text stays in the run log at WARNING so a
    reader of the box log still sees it; only the duplicate page is removed.
    """
    logger.warning(msg, *args)

def _nav_hard_gate_tolerance_usd(nav: float) -> float:
    """Dollar tolerance for the NAV three-way hard gate at this NAV level."""
    return max(NAV_HARD_GATE_TOLERANCE_USD_FLOOR, NAV_HARD_GATE_TOLERANCE_NAV_BPS / 10000.0 * nav)


def _check_nav_three_way_hard_gate(
    *,
    pricing_timing_usd: float,
    pricing_timing_available: bool,
    nav: float | None,
    run_date: str,
    residual_usd: float | None = None,
    attribution_ok: bool = False,
) -> dict | None:
    """NAV three-way reconcile hard-gate decision (config#2457).

    ``pricing_timing_usd`` is the broker-reported-NAV (IB NetLiquidation) vs.
    settled/system-NAV (cash + accrued + Σ settled position market values)
    divergence, day-over-day differenced so a constant cash/accrued offset
    cancels — see the ``nav_reconciliation`` block in ``run()`` for the full
    derivation.

    **Re-based onto the unexplained residual (alpha-engine-config-I9087).**
    Before this the gate fired whenever the RAW term crossed
    ``_nav_hard_gate_tolerance_usd`` — 8 of 48 sessions (16.7%,
    2026-06-22..2026-08-28), almost all of them routine delayed-feed
    (``reqMarketDataType(3)``) mark staleness rather than a real reconcile
    break, the "trained to ignore it" path the module's own docstring warns
    against (config#2145). ``residual_usd`` — the portion of the mark-basis
    divergence NOT attributed by ``_attribute_mark_basis_divergence`` to
    names whose IB mark demonstrably is not the settled close — measured
    three orders of magnitude smaller (−$30.7 / +$8.4) on the two sessions
    with full attribution coverage.

    Two independent triggers, EITHER of which fires the gate:

    1. ``residual_usd`` exceeds ``NAV_BREACH_RESIDUAL_FLOOR_USD`` while
       ``attribution_ok`` is True — the unexplained-residual hard gate this
       issue exists to install.
    2. The RAW term still crosses the (unwidened) ``tolerance_usd`` — the
       raw term is RETAINED, never removed, as the ``broker_data_quality``
       signal on the warning tier (the caller resolves final severity from
       ``_classify_nav_breach``; this function only decides whether
       anything is reportable at all).

    **Fail closed (required, not optional — alpha-engine-config-I9087).**
    When ``attribution_ok`` is False — ``_attribute_mark_basis_divergence``
    raised, returned partial coverage, or produced no attribution at all —
    the residual cannot be trusted, so trigger 1 above falls back to the RAW
    term against ``tolerance_usd`` (never a wider band; the caller forces
    the classification to ``reconcile_defect``/error in this case). A
    defect in the attribution codepath can therefore never silently
    suppress a real breach: attribution failure can only ever make the gate
    MORE likely to fire, never less.

    Returns ``None`` when the gate does not fire (no data, or within
    tolerance on both triggers). Returns a dict describing the breach when
    it does — the caller (``run()``) is responsible for logging + paging
    flow-doctor; kept as a pure decision function so it's unit-testable
    without mocking the rest of ``run()``'s IO (snapshot/DB/S3).

    Deliberately does NOT fire when ``pricing_timing_available`` is False
    (missing prior snapshot) — that path already gets its own honesty
    warning in ``data_warnings``, and paging on "we don't have enough data
    to check" would be a false-alarm generator, not a real divergence
    signal.
    """
    if not pricing_timing_available or not nav:
        return None
    tolerance_usd = _nav_hard_gate_tolerance_usd(nav)
    raw_breach = abs(pricing_timing_usd) > tolerance_usd
    if attribution_ok:
        residual_breach = (
            residual_usd is not None
            and abs(residual_usd) > NAV_BREACH_RESIDUAL_FLOOR_USD
        )
    else:
        # Fail closed: the residual is not trustworthy, so the only signal
        # left is the raw term against its own (unwidened) tolerance —
        # identical to the pre-I9087 gate for this one case.
        residual_breach = raw_breach
    if not raw_breach and not residual_breach:
        return None
    return {
        "run_date": run_date,
        "pricing_timing_usd": pricing_timing_usd,
        "pricing_timing_pct_of_nav": pricing_timing_usd / nav * 100,
        "tolerance_usd": tolerance_usd,
        "tolerance_bps": NAV_HARD_GATE_TOLERANCE_NAV_BPS,
        "nav": nav,
        "residual_usd": residual_usd,
        "residual_breach": residual_breach,
        "raw_breach": raw_breach,
        "attribution_ok": attribution_ok,
        "message": (
            f"NAV three-way reconcile breach for {run_date}: pricing & timing "
            f"divergence ${pricing_timing_usd:+,.0f} "
            f"({pricing_timing_usd / nav * 100:+.3f}% of NAV) exceeds hard-gate "
            f"tolerance ${tolerance_usd:,.0f} ({NAV_HARD_GATE_TOLERANCE_NAV_BPS:.1f}"
            "bps of NAV). Broker-reported NAV (IB NetLiquidation) vs settled/"
            "system NAV (cash + accrued + settled position MV) diverged beyond "
            "tolerance."
        ),
    }


# ── IB mark-outside-range detection (config#6349/#6818) ────────────────────
# IB Gateway's delayed-feed portfolio mark for a held ticker occasionally
# lands outside the day's own traded [Low, High] range (six historical
# instances catalogued in config#6349, e.g. AMD 2026-08-04: IB mark $479.00
# vs day low $502.20). NetLiquidation — and therefore pricing_timing_usd —
# inherits that bad mark wholesale. This never repriced positions (the
# settled close stays canonical for valuation, see the override two lines
# above `pos["closing_price"] = current_price`); it only names the culprit
# so a hard-gate page is diagnosable in one glance instead of a four-day
# by-hand trace.
NAV_BREACH_RESIDUAL_FLOOR_USD = 500.0  # matches the pre-existing soft data_warnings floor (config#1276)

# ── Mark-basis divergence attribution (alpha-engine-config-I9085) ──────────
# `_detect_ib_mark_outside_range` is a POINT test on today's book, but
# `pricing_timing_usd` is a day-over-day DIFFERENCE of the mark basis. Every
# bad mark therefore produces TWO hard-gate breaches — one on the day it
# lands, and an equal-and-opposite one on the day it reverts — and the
# point detector is structurally blind to the second. Measured pairs:
# AMD 2026-08-04 (−$9,560) / 2026-08-05 (+$8,906); MU 2026-08-26 (+$1,980,
# FLAGGED out of range) / 2026-08-27 (−$3,293, ZERO tickers flagged).
#
# The reversion day is the one an operator gets no help on: the alert named
# tickers only out of `ib_mark_range_flags`, so the 2026-08-27 page carried a
# portfolio total and not one name, which is the exact operator failure
# config#6349 deliverable 2 was filed to end.
#
# The independent, non-tautological test for "IB's mark is not the close" is
# the per-name distance between the IB mark and that day's SETTLED CLOSE —
# a property of the DATA, not of our arithmetic. `[Low, High]` is a strict
# subset of it: a mark can sit inside the day's traded range and still be
# stale by a full percent (MU 2026-08-27: mark $923.07 vs settled close
# $935.39, −1.32%, inside the range, −$1,331 of basis on 108 shares).
IB_MARK_OFF_CLOSE_PCT_FLOOR = 0.10  # 10bp — beyond this the mark is not the close
MARK_BASIS_TOP_CONTRIBUTORS = 5     # names carried into the alert text


def _settled_position_value_usd(
    positions: dict,
    closing_prices: dict,
) -> tuple[float, list[str]]:
    """``Σ shares × settled close`` over the held book, and the names that missed.

    Returns ``(settled_mv_usd, fallback_tickers)``. A held name with no
    ArcticDB settled close falls back to the broker's ``market_value`` — the
    same fallback the positions loop below takes — and its ticker is RETURNED
    rather than swallowed, because under ``nav_basis: settled_close`` that
    fallback means the headline NAV silently contains a broker mark. The
    caller decides what to do with the list; it is never dropped.

    Runs BEFORE the positions loop applies the settled-close override, so it
    reads ``closing_prices`` directly rather than the post-override
    ``market_value``. The two agree by construction for every name that has a
    close (alpha-engine-config-I9638).
    """
    settled_mv = 0.0
    fallback: list[str] = []
    for ticker, pos in (positions or {}).items():
        try:
            shares = float(pos.get("shares", 0) or 0)
        except (TypeError, ValueError):
            shares = 0.0
        if not shares:
            continue
        close = closing_prices.get(ticker)
        if close is None:
            fallback.append(ticker)
            settled_mv += float(pos.get("market_value", 0) or 0)
            continue
        settled_mv += float(close) * shares
    return settled_mv, fallback


def _mark_basis_usd(pos: dict | None) -> float | None:
    """Day's mark basis for one position: ``ib_market_value - market_value``.

    ``None`` when either side is absent (pre-schema-2.1 snapshot) — never
    guessed as zero, which would silently under-state the divergence.
    """
    if pos is None:
        return 0.0
    ib_mv, mv = pos.get("ib_market_value"), pos.get("market_value")
    if ib_mv is None or mv is None:
        return None
    return float(ib_mv) - float(mv)


def _off_close_pct(pos: dict | None) -> float | None:
    """|IB mark − settled close| as a percent of the settled close."""
    if not pos:
        return None
    shares = pos.get("shares") or 0
    ib_mv, mv = pos.get("ib_market_value"), pos.get("market_value")
    if not shares or ib_mv is None or mv is None or not mv:
        return None
    return abs((float(ib_mv) - float(mv)) / float(mv)) * 100.0


# ── Corrected-mark bookkeeping (alpha-engine-config-I10048) ────────────────
# `plan_nav_mark_correction` (I9627) removes a provably-wrong broker mark from
# the HEADLINE NAV, but PR524 left the per-position `ib_market_value` at the
# raw broker number. The book therefore published a NAV struck on one price
# and a position row carrying another, and the whole mark-basis apparatus
# (`_mark_basis_usd`, `_off_close_pct`, `_attribute_mark_basis_divergence`,
# `compute_pricing_timing_by_ticker`) reads the position rows.
#
# Measured cost, 2026-09-03: HOOD's 2026-09-02 correction was −$1,605.
# `pricing_timing_usd` differences the NAV-level basis and so used the
# CORRECTED prior NAV (−$2,517), while the per-name attribution differenced
# the RAW prior position mark (−$4,122 full book, −$4,004 explained). The
# difference is the correction to the dollar: `nav_identity_residual_usd` came
# out at +$1,605, `residual_usd` at +$1,487, and the breach was labelled
# `reconcile_defect` and paged at ERROR instead of `broker_data_quality` at
# WARNING — on the day AFTER every correction, i.e. exactly when the class the
# classifier exists to sort is most active.
#
# The repair is bookkeeping, not tolerance: the corrected position carries the
# mark the headline NAV was actually struck on, and the raw broker mark is
# preserved beside it so no evidence is lost. `_detect_ib_mark_outside_range`
# and `check_mark_coverage` both run BEFORE this and keep their stamps
# (`ib_mark_outside_range`, `ib_mark_range_error_usd`,
# `ib_mark_range_checked`); nothing re-runs the detector on the corrected
# value, so "repaired" can never be read back as "never wrong".
def _apply_mark_correction_to_positions(
    positions: dict,
    corrections: list[dict] | None,
) -> list[str]:
    """Write an APPLIED mark correction onto the positions it repaired.

    Mutates each corrected position: ``ib_market_value`` becomes
    ``shares × settled_close`` (the mark NAV was struck on),
    ``ib_market_value_raw`` preserves the broker's number,
    ``ib_mark_correction_usd`` carries the per-name repair and
    ``ib_mark_corrected`` is True. Returns the tickers written.

    Called BEFORE the settled-close override in the positions loop, so
    ``market_value`` converges on the same number and the name's
    ``mark_basis_usd`` lands at $0 by construction.
    """
    written: list[str] = []
    for c in corrections or []:
        tkr = c.get("ticker")
        pos = (positions or {}).get(tkr)
        if pos is None:
            # A correction naming a name that is not in the book is a contract
            # violation, not a tolerable skip: the flags the plan was built
            # from came from this very dict one call earlier.
            raise RuntimeError(
                f"NAV mark correction names {tkr!r}, which is absent from the "
                "positions book it was planned from. The correction cannot be "
                "written to the position and NAV would publish on a mark no "
                "position row carries."
            )
        shares = float(c.get("shares") or 0)
        settled_close = float(c["settled_close"])
        if "ib_market_value_raw" not in pos:
            pos["ib_market_value_raw"] = pos.get("ib_market_value")
        pos["ib_market_value"] = shares * settled_close
        pos["ib_mark_correction_usd"] = float(c.get("correction_usd") or 0.0)
        pos["ib_mark_corrected"] = True
        written.append(tkr)
    return written


def _restate_prior_positions_for_mark_correction(
    prior_positions: dict | None,
    nav_mark_correction_json: str | None,
) -> list[str]:
    """Backward compatibility for snapshots written BEFORE I10048.

    A prior-day row persisted by PR524 (2026-08-31 → 2026-09-03) carries a
    CORRECTED ``portfolio_nav`` and a RAW per-position ``ib_market_value``.
    The correction itself was persisted whole on that row as
    ``nav_mark_correction_json``, so the corrected prior mark is DERIVED —
    ``shares × settled_close`` from the plan's own ``corrections`` list — not
    guessed from the artifact or reverse-engineered from the NAV delta.

    Mutates ``prior_positions`` in place and returns the tickers restated.
    A row that already carries ``ib_mark_corrected`` (written forward by
    :func:`_apply_mark_correction_to_positions`) is left untouched.
    """
    if not prior_positions or not nav_mark_correction_json:
        return []
    try:
        plan = json.loads(nav_mark_correction_json)
    except (json.JSONDecodeError, TypeError):
        # Fail loud enough to be seen, but not fatal: an unparseable
        # provenance blob must not stop today's reconcile. The classifier
        # simply reverts to the pre-fix behaviour for this one prior day and
        # the log names it, rather than the run dying on a history artifact.
        logger.error(
            "Prior-day nav_mark_correction_json is unparseable — the prior "
            "positions are NOT restated and the mark-basis attribution will "
            "carry the previous day's correction as a residual."
        )
        return []
    if not isinstance(plan, dict) or not plan.get("applied"):
        return []
    restated: list[str] = []
    for c in plan.get("corrections") or []:
        tkr = c.get("ticker")
        pos = prior_positions.get(tkr)
        if pos is None or pos.get("ib_mark_corrected"):
            continue
        try:
            shares = float(c.get("shares") or 0)
            settled_close = float(c["settled_close"])
        except (TypeError, ValueError, KeyError):
            continue
        if "ib_market_value_raw" not in pos:
            pos["ib_market_value_raw"] = pos.get("ib_market_value")
        pos["ib_market_value"] = shares * settled_close
        pos["mark_basis_usd"] = _mark_basis_usd(pos)
        pos["ib_mark_off_close_pct"] = _off_close_pct(pos)
        pos["ib_mark_correction_usd"] = float(c.get("correction_usd") or 0.0)
        pos["ib_mark_corrected"] = True
        pos["ib_mark_corrected_source"] = "prior_row_nav_mark_correction_json"
        restated.append(tkr)
    return restated


def _attribute_mark_basis_divergence(
    *,
    positions: dict,
    prior_positions: dict | None,
    top_n: int = MARK_BASIS_TOP_CONTRIBUTORS,
) -> dict:
    """Per-name attribution of the day-over-day mark-basis term.

    Returns ``{"contributors": [...], "explained_usd": float | None,
    "covered_usd": float, "uncovered_names": int}``.

    ``contributors`` is every held-or-previously-held name with a non-zero
    basis delta, ranked by ``abs(delta)``, each carrying today's and the
    prior day's basis and off-close percentage plus a ``reversion`` flag
    (the two days' bases are materially opposite in sign — the signature of
    a bad mark unwinding, which is what makes the second breach unnameable
    by the point detector).

    ``explained_usd`` sums the deltas of names whose IB mark demonstrably
    is not the settled close on EITHER day — off-close beyond
    ``IB_MARK_OFF_CLOSE_PCT_FLOOR``. This is the term the classifier tests
    against, and unlike the full-book sum it is a FILTERED subset: it can
    genuinely fail to explain the breach, which is the whole point. It is
    ``None`` when no name on either day carries schema-2.1
    ``ib_market_value`` (nothing to test against; the caller falls back).
    """
    contributors: list[dict] = []
    explained_usd = 0.0
    covered_usd = 0.0
    uncovered = 0
    any_basis = False
    for tkr in sorted(set(positions or {}) | set((prior_positions or {}).keys())):
        pos_t = (positions or {}).get(tkr)
        pos_p = (prior_positions or {}).get(tkr)
        basis_t = _mark_basis_usd(pos_t)
        basis_p = _mark_basis_usd(pos_p)
        if basis_t is None or basis_p is None:
            uncovered += 1
            continue
        any_basis = True
        delta = basis_t - basis_p
        if not delta:
            continue
        covered_usd += delta
        off_t = _off_close_pct(pos_t)
        off_p = _off_close_pct(pos_p)
        off_close = max(
            (v for v in (off_t, off_p) if v is not None), default=None,
        )
        mark_is_not_close = (
            off_close is not None and off_close >= IB_MARK_OFF_CLOSE_PCT_FLOOR
        )
        if mark_is_not_close:
            explained_usd += delta
        contributors.append({
            "ticker": tkr,
            "contrib_usd": delta,
            "basis_today_usd": basis_t,
            "basis_prior_usd": basis_p,
            "off_close_pct_today": off_t,
            "off_close_pct_prior": off_p,
            "mark_is_not_close": mark_is_not_close,
            # A bad mark unwinding: materially non-zero on both days and
            # opposite in sign. Names the PRIOR day's mark as today's cause.
            "reversion": (
                basis_t * basis_p < 0
                and min(abs(basis_t), abs(basis_p)) >= NAV_BREACH_RESIDUAL_FLOOR_USD / 10
            ),
        })
    contributors.sort(key=lambda c: -abs(c["contrib_usd"]))
    return {
        "contributors": contributors[:top_n],
        "explained_usd": explained_usd if any_basis else None,
        "covered_usd": covered_usd,
        "uncovered_names": uncovered,
    }


def _format_mark_basis_contributors(contributors: list[dict]) -> str:
    """Always-present ticker detail for the hard-gate alert.

    Unlike `_format_mark_range_detail` this does not depend on any name
    crossing a traded-range boundary, so a reversion-day breach names its
    culprits too.
    """
    parts = []
    for c in contributors:
        off_t = c["off_close_pct_today"]
        off_p = c["off_close_pct_prior"]
        pct_p = "n/a" if off_p is None else f"{off_p:.2f}%"
        pct_t = "n/a" if off_t is None else f"{off_t:.2f}%"
        parts.append(
            f"{c['ticker']} ${c['contrib_usd']:+,.0f} "
            f"(mark-basis ${c['basis_prior_usd']:+,.0f}→${c['basis_today_usd']:+,.0f}, "
            f"off-close {pct_p}→{pct_t}"
            f"{', REVERSION of prior-day mark' if c['reversion'] else ''})"
        )
    return "; ".join(parts)


def _detect_ib_mark_outside_range(
    *,
    positions: dict,
    day_low: dict[str, float],
    day_high: dict[str, float],
) -> list[dict]:
    """Flag held tickers whose ``ib_market_value / shares`` fell outside
    that day's ArcticDB ``[Low, High]``.

    Mutates each flagged position with ``ib_mark_outside_range=True`` (so
    the flag reaches ``eod_report.json`` via the ``positions`` dict) and
    returns the flag list the hard-gate call site uses to name tickers in
    the alert text and classify the breach.

    **Records negative evidence** (alpha-engine-config-I9637). Every position
    is stamped with ``ib_mark_range_checked``, and an unchecked one carries
    ``ib_mark_range_uncheckable_reason``. Before this, a name the check could
    not evaluate looked exactly like a name that passed: the loop `continue`d
    and the position dict was left untouched, so ``ib_mark_outside_range``
    read False for "verified in range" and for "never looked at" alike.

    That is not hypothetical. The ArcticDB ``macro`` library is **Close-only**
    — measured 2026-08-31, ``XLK``/``SPY``/``GLD``/``VIX`` all return
    ``cols=['Close']`` — so ``day_low``/``day_high`` are never populated for a
    macro-routed held name (``price_cache._MACRO_SYMBOLS``: the sector ETFs,
    GLD, USO, VIX, VIX3M, TNX, IRX). Every one of those marks flows into NAV
    unverified, and the gate that exists to stop a wrong mark reaching NAV
    reported nothing at all about them.
    """
    flags: list[dict] = []
    for ticker, pos in positions.items():
        shares = pos.get("shares", 0) or 0
        ib_mv = pos.get("ib_market_value")
        lo = day_low.get(ticker)
        hi = day_high.get(ticker)
        reason = None
        if not shares:
            reason = "no share count on the position"
        elif ib_mv is None:
            reason = "no IB market value (pre-schema-2.1 snapshot)"
        elif lo is None or hi is None:
            reason = (
                "no traded [Low, High] for this ticker on this date — the "
                "ArcticDB macro library is Close-only, so a macro-routed "
                "holding cannot be range-checked"
            )
        if reason is not None:
            pos["ib_mark_range_checked"] = False
            pos["ib_mark_range_uncheckable_reason"] = reason
            continue
        pos["ib_mark_range_checked"] = True
        ib_mark = ib_mv / shares
        if lo <= ib_mark <= hi:
            # Explicit negative evidence: checked, and in range.
            pos["ib_mark_outside_range"] = False
            continue
        mark_error_usd = shares * (ib_mark - lo if ib_mark < lo else ib_mark - hi)
        pos["ib_mark_outside_range"] = True
        pos["ib_mark_range_error_usd"] = mark_error_usd
        flags.append({
            "ticker": ticker,
            "ib_mark": ib_mark,
            "day_low": lo,
            "day_high": hi,
            "shares": shares,
            "mark_error_usd": mark_error_usd,
        })
    return flags


def _classify_nav_breach(
    pricing_timing_usd: float,
    mark_range_flags: list[dict],
    *,
    full_book_mark_basis_usd: float | None = None,
    full_book_uncovered_names: int = 0,
    mark_divergence_explained_usd: float | None = None,
) -> dict:
    """Classify a hard-gate breach as a broker data-quality event rather
    than a NAV reconcile defect (config#6349 deliverable 4) — a different
    remediation (chase the broker feed) than a code defect (chase the
    reconcile math).

    **Two independent tests, both of which must pass to earn the softer
    `broker_data_quality` label** (alpha-engine-config-I9085):

    1. **Explanation.** ``mark_divergence_explained_usd`` is the mark-basis
       delta summed over ONLY those names whose IB mark demonstrably is not
       that day's settled close — off-close beyond
       ``IB_MARK_OFF_CLOSE_PCT_FLOOR`` on either day (see
       ``_attribute_mark_basis_divergence``). Its residual against the
       breach is the unexplained dollars.

    2. **NAV identity tie-out.** ``full_book_mark_basis_usd`` is
       ``Σ_positions Δ(ib_market_value − market_value)``. Differenced
       against ``pricing_timing_usd`` — which is
       ``Δ(nav_ib − cash − accrued − Σ market_value)`` — cash, accrued and
       the settled leg all cancel, leaving
       ``Δ(nav_ib − cash − accrued − Σ ib_market_value)``: whether IB's own
       reported NetLiquidation equals the sum of its own components. That
       is a real control (it catches an IB sleeve we do not model), and it
       is reported as ``nav_identity_residual_usd``.

    **Why the shape changed.** Shipped as of I8733, test 2 WAS the
    explanation test — ``explained`` was set True unconditionally on the
    full-book basis and the residual was the identity above. But that
    residual is ~0 by construction on any ordinary equities book. Measured
    over all 48 sessions with an artifact (2026-06-22 → 2026-08-28):
    ``|pricing_timing_unattributable_usd| <= $63.54`` on every single day
    and ``<= $10`` on 45 of 48, against a $500 floor. The
    ``reconcile_defect`` branch was therefore UNREACHABLE from the call
    site, and a genuine reconcile defect would have paged at severity
    ``warning`` under a "chase the broker feed" label. The unit tests could
    not catch it because they hand-fed ``full_book_mark_basis_usd`` values
    the call site cannot produce. Same failure class as the tautological
    attribution tie-out replaced by alpha-engine-config-I8188.

    Test 1 is non-tautological precisely because it is a FILTERED subset
    keyed on a property of the data rather than of our arithmetic: a breach
    moved by something other than a stale broker mark lands in its residual
    and classifies ``reconcile_defect``.

    ``mark_divergence_explained_usd=None`` — no schema-2.1
    ``ib_market_value`` on either day — falls back to the flagged
    out-of-range subset, the pre-#8733 behaviour, which never over-explains.

    The test is on ``abs(residual)`` and so is SIGN-SYMMETRIC: the term is
    a day-over-day difference and its recorded instances split roughly even
    high/low, so a one-sided guard would miss half the class (premise
    correction on config#6819, 2026-08-26).
    """
    total_mark_error_usd = sum(f["mark_error_usd"] for f in mark_range_flags)
    flagged_subset_residual_usd = pricing_timing_usd - total_mark_error_usd
    nav_identity_residual_usd = (
        None if full_book_mark_basis_usd is None
        else pricing_timing_usd - full_book_mark_basis_usd
    )
    if mark_divergence_explained_usd is None:
        basis = "flagged_subset"
        residual_usd = flagged_subset_residual_usd
        explained = bool(mark_range_flags)
    else:
        basis = "off_close_marks"
        residual_usd = pricing_timing_usd - mark_divergence_explained_usd
        explained = True
    fully_explained = explained and abs(residual_usd) <= NAV_BREACH_RESIDUAL_FLOOR_USD
    identity_holds = (
        nav_identity_residual_usd is None
        or abs(nav_identity_residual_usd) <= NAV_BREACH_RESIDUAL_FLOOR_USD
    )
    return {
        "classification": (
            "broker_data_quality"
            if (fully_explained and identity_holds)
            else "reconcile_defect"
        ),
        "attribution_basis": basis,
        "total_mark_error_usd": total_mark_error_usd,
        "flagged_subset_residual_usd": flagged_subset_residual_usd,
        "full_book_mark_basis_usd": full_book_mark_basis_usd,
        "full_book_uncovered_names": full_book_uncovered_names,
        "mark_divergence_explained_usd": mark_divergence_explained_usd,
        "nav_identity_residual_usd": nav_identity_residual_usd,
        "nav_identity_holds": identity_holds,
        "residual_usd": residual_usd,
    }


def _format_mark_range_detail(mark_range_flags: list[dict]) -> str:
    """Ticker-by-ticker detail string for the hard-gate alert text — the
    single most important thing config#6349's investigation had to
    reconstruct by hand across three sessions before this existed."""
    return "; ".join(
        f"{f['ticker']} mark ${f['ib_mark']:.2f} vs day range "
        f"[${f['day_low']:.2f}, ${f['day_high']:.2f}] (${f['mark_error_usd']:+,.0f})"
        for f in mark_range_flags
    )


def _compute_unattributed_residual_pct(
    unattributed_usd: float | None,
    nav: float | None,
) -> float | None:
    """Headline metric for the *P&L attribution* row of the Phase 2
    transparency-inventory: ``unattributed_usd / portfolio_nav × 100``.

    Returns None when either input is None or when nav is zero/falsy
    (a divide-by-zero protection — a NAV of 0 means we can't compute a
    meaningful residual % regardless of the dollar amount).

    The inventory gate is ≤1%. Sign is preserved — a negative value
    means position-pnl + interest exceeded the actual NAV change
    (typically an unaccounted fee). Consumers should compare on
    absolute value when alarming.
    """
    if unattributed_usd is None or not nav:
        return None
    return (unattributed_usd / nav) * 100.0


def _spy_close(run_date: str, config: dict | None = None) -> float:
    """Fetch SPY close for run_date from ArcticDB macro library.

    SPY lives in the `macro` library (per alpha-engine-data's
    daily_append writer), NOT the `universe` library. Reading from
    universe was a bug: universe has only the per-ticker watchlist
    symbols, no index ETFs.

    ArcticDB is the single source of truth — no parquet, polygon, or
    yfinance fallback. Hard-fails if SPY is missing, stale, or has no
    close for run_date, because EOD alpha is meaningless without a
    reliable SPY reference.
    """
    from executor.price_cache import _open_macro_library
    bucket = (config or {}).get("trades_bucket", "alpha-engine-research")
    macro = _open_macro_library(bucket)
    try:
        df = macro.read("SPY").data
    except Exception as e:
        raise RuntimeError(f"ArcticDB read failed for SPY: {e}") from e
    if df.empty or "Close" not in df.columns:
        raise RuntimeError("ArcticDB SPY frame empty or missing Close column")
    target = pd.Timestamp(run_date).normalize()
    idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
    matches = df[idx == target]
    if matches.empty:
        raise RuntimeError(
            f"ArcticDB has no SPY close for {run_date} (latest: "
            f"{pd.Timestamp(df.index[-1]).date()})"
        )
    close = float(matches["Close"].iloc[-1])
    logger.info("[data_source=arcticdb] SPY close for %s: $%.2f", run_date, close)
    return close


def _load_signals_from_s3(bucket: str, run_date: str, max_lookback: int = 14) -> tuple[dict, str | None]:
    """Load signals.json from S3, falling back to prior days.

    Research runs weekly (Saturday) and writes signals with the trading-day
    date, so the lookback must span weekends + a missed-cycle buffer. The
    14-day default matches the morning planner's
    ``read_signals_with_fallback`` window — EOD reconciliation should use the
    same last-good signals the planner traded on, not a tighter window. (The
    prior 5-day window paged a misleading flow-doctor ERROR on 2026-06-25
    when the only staleness was a missed Saturday cycle the planner had
    already tolerated.) Whether the required signals.json was actually
    PRODUCED is owned by the central artifact-freshness monitor
    (research_signals in ARTIFACT_REGISTRY.yaml), not this consumer.
    """
    s3 = boto3.client("s3")
    start = date.fromisoformat(run_date)
    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        dt = str(candidate)
        try:
            obj = s3.get_object(Bucket=bucket, Key=f"signals/{dt}/signals.json")
            if days_back > 0:
                logger.info("No signals for %s — using %s (%d day(s) old)", run_date, dt, days_back)
            return json.loads(obj["Body"].read()), None
        except Exception as e:
            # (a) signals.json read/parse failed for this date — the
            # fallback (trying the prior day, per the WARNING-not-ERROR
            # comment below) is the intended control flow, not a defect.
            # (c) not recorded elsewhere — deliberate carve-out
            # (alpha-engine-config-I10031): an expected-absence probe with
            # a defined fallback, same class as the connection-teardown
            # carve-outs; the caller's own WARNING covers the case where
            # every fallback is exhausted.
            logger.debug("No signals.json for %s (%s) — trying prior day", dt, e)
            continue
    # WARNING not ERROR: EOD reconcile degrades gracefully on absent signals
    # (returns {} + a warning the caller surfaces in the EOD report). The
    # operator-paging "signals.json was not produced" alert is owned by the
    # central artifact-freshness monitor (research_signals), so this consumer
    # must not also page flow-doctor for the same upstream condition.
    logger.warning("No signals found within %d days of %s", max_lookback, run_date)
    return {}, f"Signals unavailable from S3 for {run_date} (checked {max_lookback} days back)"


def _load_constituents_sector_map(bucket: str) -> dict[str, str]:
    """Return the latest S&P 500+400 ticker→GICS sector map from S3.

    Reads ``market_data/weekly/{YYYY-MM-DD}/constituents.json`` written by
    the alpha-engine-data weekly collector. Used as the final sector
    lookup fallback in EOD reconcile — catches legacy/fractional-share
    positions whose ticker isn't in today's research universe or whose
    entry_trade row predates reliable sector population.

    Returns an empty dict on miss so the caller can fall through to the
    "Unknown" sentinel without raising.
    """
    s3 = boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="market_data/weekly/")
        keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith("/constituents.json")
        ]
        if not keys:
            logger.warning("No constituents.json under market_data/weekly/ in %s", bucket)
            return {}
        latest = max(keys)
        obj = s3.get_object(Bucket=bucket, Key=latest)
        data = json.loads(obj["Body"].read())
        sector_map = data.get("sector_map", {}) or {}
        logger.info(
            "Loaded sector_map from %s (%d tickers)", latest, len(sector_map),
        )
        return sector_map
    except Exception as e:
        # WARNING not ERROR: graceful fallthrough to the "Unknown" sentinel
        # (return {}). Upstream constituents.json freshness is owned by the
        # central artifact-freshness monitor, not this best-effort lookup.
        logger.warning("Failed to load constituents sector_map: %s", e)
        return {}


# Broad-market index/ETF core positions are not GICS sector constituents.
# Since the portfolio-optimizer cutover (use_portfolio_optimizer: true,
# 2026-05-13) SPY is held as the enhanced-index core position. SPY has no
# `sector` field in signals.json, no entry_trade.sector, and is not in the
# S&P 500+400 constituents map, so the normal lookup chain misses it and it
# renders as a bare "—"/"Unknown" on the public site — reads as missing data
# rather than "this is the broad-market core." Tag it explicitly so every
# downstream consumer (public site, private console, sector attribution)
# inherits a meaningful label. New core ETFs the optimizer may substitute
# get added to ``reference_rate.INDEX_ETF_TICKERS`` (single source of truth,
# shared with the asset_type classification in the Metron reference-rate
# contract — a ticker added there is simultaneously tagged with this sector
# label AND reported as ETF, not equity).
_INDEX_ETF_SECTOR = "Broad Market / Index"


def _index_etf_sector(ticker: str) -> str | None:
    """Return the broad-market sector label for index/ETF core positions.

    Returns ``"Broad Market / Index"`` for known broad-market ETFs held as
    the enhanced-index core (SPY and S&P 500 trackers the optimizer may
    substitute), else ``None`` so the caller falls through to the normal
    signals.json / entry-trade / S&P-constituents lookup chain.
    """
    return _INDEX_ETF_SECTOR if ticker in reference_rate.INDEX_ETF_TICKERS else None


def _load_predictions_from_s3(bucket: str) -> tuple[dict, str | None]:
    """Load latest predictions from S3. Returns ({ticker: pred_dict}, warning_msg) on failure."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="predictor/predictions/latest.json")
        data = json.loads(obj["Body"].read())
        return {p["ticker"]: p for p in data.get("predictions", []) if "ticker" in p}, None
    except Exception as e:
        # WARNING not ERROR: EOD reconcile degrades gracefully without
        # predictions (returns {} + a warning the caller surfaces). Upstream
        # predictions.json freshness is owned by the central artifact-
        # freshness monitor (predictor_predictions), not this consumer.
        logger.warning("Failed to load predictions from S3: %s", e)
        return {}, "Predictions unavailable from S3"


def _build_position_contexts(
    positions: dict,
    conn,
    signals_bucket: str,
    run_date: str,
) -> tuple[list[dict], list[str]]:
    """Assemble per-position context for rationale synthesis.

    Returns (contexts, data_warnings).
    """
    data_warnings: list[str] = []
    signals_data, sig_warn = _load_signals_from_s3(signals_bucket, run_date)
    predictions, pred_warn = _load_predictions_from_s3(signals_bucket)
    if sig_warn:
        data_warnings.append(sig_warn)
    if pred_warn:
        data_warnings.append(pred_warn)

    # Build signals lookup
    # signals.json::universe read: this is the executor sizing/exit path —
    # the ONE fleet-level exception to "resolve ticker lists from
    # decision_set, not universe" (alpha-engine-config#5809). Formal
    # policy-clause registration tracked separately, not yet landed:
    # alpha-engine-config#6448.
    signals_by_ticker = {}
    for s in (signals_data.get("universe", []) + signals_data.get("buy_candidates", [])):
        t = s.get("ticker")
        if t:
            signals_by_ticker[t] = s

    todays_trades = get_todays_trades(conn, run_date)
    trades_by_ticker = {}
    for t in todays_trades:
        trades_by_ticker.setdefault(t["ticker"], []).append(t)

    contexts = []
    for ticker, pos in sorted(positions.items()):
        entry = get_entry_trade(conn, ticker)
        sig = signals_by_ticker.get(ticker, {})
        pred = predictions.get(ticker, {})
        today_actions = trades_by_ticker.get(ticker, [])

        entry_rationale = None
        if entry and entry.get("rationale_json"):
            try:
                entry_rationale = json.loads(entry["rationale_json"])
            except (json.JSONDecodeError, TypeError):
                logger.debug("Could not parse entry rationale JSON for %s", ticker)

        ctx = {
            "ticker": ticker,
            "shares": pos.get("shares"),
            "market_value": pos.get("market_value"),
            "unrealized_pnl": pos.get("unrealized_pnl"),
            "entry_date": entry["date"] if entry else None,
            "entry_price": entry["price_at_order"] if entry else None,
            "research_score": sig.get("score") or (entry["research_score"] if entry else None),
            "conviction": sig.get("conviction") or (entry["research_conviction"] if entry else None),
            "thesis_summary": sig.get("thesis_summary") or (entry["thesis_summary"] if entry else None),
            "price_target_upside": sig.get("price_target_upside"),
            "sector_rating": sig.get("sector_rating") or (entry["sector_rating"] if entry else None),
            "market_regime": signals_data.get("market_regime"),
            "predicted_direction": pred.get("predicted_direction"),
            "prediction_confidence": pred.get("prediction_confidence"),
            "predicted_alpha": pred.get("predicted_alpha"),
            "today_actions": [
                {"action": t["action"], "shares": t["shares"]}
                for t in today_actions
            ],
            "entry_rationale": entry_rationale,
        }
        contexts.append(ctx)

    return contexts, data_warnings


def _synthesize_rationales(contexts: list[dict]) -> dict[str, str]:
    """Build per-position rationales from the context dict for the EOD email.

    **Zero LLM exposure — hard architectural guardrail.** Per
    ``[[preference_llm_calls_confined_to_research_module]]``, executor
    never invokes an LLM. Trading execution is the most operationally
    critical surface; introducing any external-API dependency (even
    gated) couples trading-day reliability to upstream availability.
    If a future surface genuinely needs LLM-synthesized prose, the call
    goes in research and produces a frozen artifact executor reads.

    Output is mechanical synthesis from the context dict: entry / score
    / GBM prediction / thesis summary / today's actions, joined with
    spaces. Same dict-shape the EOD emailer has always consumed.

    History: an Anthropic-Haiku-backed synthesis path shipped
    2026-03-17 (commit 58dcb9b) and ran for ~10 weeks before being
    nuked outright 2026-05-25 after Brian flagged the unmandated LLM
    exposure. Cost-telemetry wiring + WARN-above-ceiling instrumentation
    + an opt-in kill-switch flag (PRs #210/#211/#212 same session) all
    went with it — the rule is "no LLM in executor, period" rather than
    "LLM with substrate."
    """
    if not contexts:
        return {}

    narratives = {}
    for ctx in contexts:
        parts = []
        ticker = ctx["ticker"]

        if ctx.get("entry_date") and ctx.get("entry_price"):
            parts.append(f"Entered {ctx['entry_date']} at ${ctx['entry_price']:.2f}.")

        if ctx.get("research_score") is not None:
            conv = ctx.get("conviction", "stable")
            parts.append(f"Research score {ctx['research_score']:.0f}/100 ({conv}).")

        if ctx.get("predicted_direction"):
            conf = ctx.get("prediction_confidence")
            conf_str = f" ({conf*100:.0f}% conf)" if conf else ""
            alpha = ctx.get("predicted_alpha")
            alpha_str = f", α={alpha*100:.2f}%" if alpha else ""
            parts.append(f"GBM: {ctx['predicted_direction']}{conf_str}{alpha_str}.")

        if ctx.get("thesis_summary"):
            thesis = ctx["thesis_summary"]
            if len(thesis) > 120:
                thesis = thesis[:117] + "..."
            parts.append(thesis)

        if ctx.get("today_actions"):
            actions = ", ".join(f"{a['action']} {a['shares']} shares" for a in ctx["today_actions"])
            parts.append(f"Today: {actions}.")

        narratives[ticker] = " ".join(parts) if parts else "No rationale data available."

    return narratives


def _resolve_prior_price(
    prior_pos: dict | None,
    pos: dict,
    current_price: float,
) -> float:
    """Pick the right prior-day price for daily return computation.

    Phase 3+ snapshots store an explicit `closing_price` from daily_closes,
    which is the same source today's reconcile uses for current_price —
    eliminating the IB-MV-vs-daily-closes mismatch that was dumping noise
    into the cash residual. Falls back to MV/shares for legacy snapshots
    and to avg_cost for positions opened today.
    """
    if prior_pos:
        cp = prior_pos.get("closing_price")
        if cp is not None:
            return float(cp)
        prior_mv = prior_pos.get("market_value", 0)
        prior_shares = prior_pos.get("shares", 0)
        if prior_shares:
            return prior_mv / prior_shares
    # No prior snapshot — position opened today, use avg_cost
    return pos.get("avg_cost", current_price)


def _compute_daily_return(
    ticker: str,
    pos: dict,
    prior_pos: dict | None,
    current_price: float,
    shares: float,
    prior_close: float | None,
    prior_close_date: date | None,
    expected_prev_td: date,
    add_entry_px: float | None = None,
) -> tuple[float, float, float | None, str | None]:
    """Gap-aware per-position daily return.

    The prior-day baseline for a position HELD on the previous trading day
    must be the close on that previous trading day, read authoritatively
    from ArcticDB (``prior_close`` at ``prior_close_date``) — NOT the stored
    snapshot's ``closing_price``, which can be days stale whenever a
    weekday/EOD Step Function was skipped. In the normal case the two are
    identical (the snapshot persists the same daily_closes-sourced close),
    so behavior is unchanged; the difference only bites across a gap.

    Concrete bug this closes: the 2026-06-24 SF halt left no 06-24
    snapshot, so the 06-25 reconcile picked the 06-23 snapshot as
    "yesterday" and reported RGEN at ``145.23/126.37-1 = +14.92%`` — a
    two-session move mislabeled as one day, which drove the entire
    headline (config#1228).

    Returns ``(daily_return_pct, daily_return_usd, prior_price, na_reason)``:
      * held-through, ArcticDB's prior row IS the previous trading day →
        true one-session return; ``na_reason`` is None.
      * held-through but ArcticDB's latest prior row predates the previous
        trading day (market-data gap not yet healed) → ``(0.0, 0.0, None,
        reason)``: we refuse to compute a return against a stale baseline,
        flag it N/A, and let the position's P&L surface in the NAV
        unattributed bucket rather than fabricate an inflated number.
      * opened since the prior trading day (absent from the prior snapshot)
        → return vs entry ``avg_cost`` (unchanged legacy behavior).
      * held-through with **more shares today than yesterday** AND a recorded
        ENTER fill today → retained shares vs the previous trading day's
        close; added shares vs the share-weighted buy fill. Without a fill,
        fall back to the legacy all-shares prior-close path (share-count drift
        from snapshots/corporate actions must not trigger avg_cost guessing).
        Using the prior close for confirmed intraday adds dumps entry-to-close
        P&L into the NAV ``unattributed`` bucket.
    """
    held_through = prior_pos is not None
    if not held_through:
        # Opened today / during a gap — baseline is the entry price.
        prior_price = pos.get("avg_cost", current_price)
    elif prior_close is not None and prior_close_date == expected_prev_td:
        # Authoritative: the previous trading day's close from ArcticDB.
        prior_price = float(prior_close)
    elif (
        prior_close is not None
        and prior_close_date is not None
        and prior_close_date < expected_prev_td
    ):
        # ArcticDB's latest prior row predates the previous trading day — a
        # weekday/EOD SF was skipped and the market-data gap is not yet
        # healed. Do NOT report a multi-session move as a one-day return.
        reason = (
            f"{ticker}: previous-trading-day ({expected_prev_td}) close unavailable "
            f"in ArcticDB (latest prior row = {prior_close_date}). Daily return marked "
            f"N/A rather than computed against a stale baseline; the position's P&L "
            f"surfaces in the NAV unattributed bucket. Heal the market-data gap "
            f"(config#1228)."
        )
        return 0.0, 0.0, None, reason
    else:
        # No ArcticDB prior close at all (e.g. brand-new listing) — fall back
        # to the legacy snapshot/avg_cost resolution.
        prior_price = _resolve_prior_price(prior_pos, pos, current_price)

    if prior_price and prior_price > 0:
        prior_shares = 0.0
        if prior_pos:
            try:
                prior_shares = float(prior_pos.get("shares", 0) or 0)
            except (TypeError, ValueError):
                prior_shares = 0.0
        added = max(0.0, shares - prior_shares) if prior_shares > 0 else 0.0
        retained = min(prior_shares, shares) if prior_shares > 0 else 0.0
        if (
            held_through
            and added > 0
            and retained > 0
            and add_entry_px is not None
            and add_entry_px > 0
        ):
            daily_usd = (
                (current_price - prior_price) * retained
                + (current_price - add_entry_px) * added
            )
            prior_mv = prior_price * retained + add_entry_px * added
            daily_pct = (daily_usd / prior_mv * 100) if prior_mv else 0.0
            return daily_pct, daily_usd, prior_price, None
        return (
            (current_price / prior_price - 1) * 100,
            (current_price - prior_price) * shares,
            prior_price,
            None,
        )
    return 0.0, 0.0, None, None


def _apply_dividend_delta(
    pos: dict,
    prior_pos: dict | None,
    prior_price: float,
    shares: int,
) -> None:
    """Attribute today's dividend accrual to the position.

    Only positive accrual deltas (ex-dividend earnings) are added to
    daily_return_usd — these represent new economic value earned today.

    Negative deltas (accrual → cash reclassification on payout day) are
    recorded in pos['dividend_paid_usd'] but NOT subtracted from position
    P&L. The dividend was already earned on ex-dividend day; the payout
    is a bookkeeping transfer that raises cash without changing portfolio
    value. The reconciliation bucket uses dividend_paid_usd to explain
    the cash inflow on the payout day.
    """
    today_div = float(pos.get("accrued_dividend", 0.0) or 0.0)
    prior_div = float((prior_pos or {}).get("accrued_dividend", 0.0) or 0.0)
    div_delta = today_div - prior_div
    if div_delta > 0:
        pos["dividend_usd"] = div_delta
        pos["daily_return_usd"] = pos.get("daily_return_usd", 0.0) + div_delta
        prior_mv = prior_price * shares if prior_price else 0
        if prior_mv > 0:
            pos["daily_return_pct"] = (pos["daily_return_usd"] / prior_mv) * 100
    elif div_delta < 0:
        # Accrual dropped — payout to cash. Don't double-count as position loss.
        pos["dividend_paid_usd"] = -div_delta


def run(
    run_date: str | None = None,
    *,
    send_email: bool = True,
    run_audit: bool = True,
) -> None:
    """Reconcile EOD P&L for ``run_date`` against the settled ArcticDB closes.

    ``send_email``: when False, the outbound EOD email is suppressed. A
    re-reconcile / correction pass (``reconcile_audit``) re-runs this for a
    PAST date to fix a value frozen pre-settlement — it must re-emit the
    ``eod_report.json`` artifact (kept) but must NOT resend that day's email.

    ``run_audit``: when True (the live daily run), the trailing-window
    ``reconcile_audit`` self-heal pass fires FIRST, before this run's own NAV
    three-way pricing&timing term reads yesterday's prior-day snapshot
    (config#6349 — a self-heal that lands after that read is one run too
    late). The audit pass itself calls ``run(..., run_audit=False)`` so the
    re-reconcile can't recurse.
    """
    today_trading_day = now_dual().trading_day
    if run_date is None:
        run_date = today_trading_day
        logger.info(
            "EOD reconciliation | date=%s (resolved from now_dual().trading_day)",
            run_date,
        )
    else:
        # Axis guard (config#1610), mirroring snapshot_capturer: the LIVE
        # daily run (run_audit=True) joins today's trades against today's
        # snapshot, so an explicit run_date must equal the just-closed
        # session — post-close, session_date == now_dual().trading_day.
        # A mismatched date (e.g. an SF input carrying a mislabeled daemon
        # run_date) would silently join one session's trades against a
        # different session's snapshot → mis-stated NAV. Historical
        # re-reconciles legitimately pass past dates but arrive via
        # run_audit=False (the reconcile_audit correction pass).
        if run_audit and run_date != today_trading_day:
            raise RuntimeError(
                f"EOD reconcile refusing live run with run_date={run_date!r} "
                f"!= today's trading_day {today_trading_day!r}. A live "
                f"reconcile must join the just-closed session; historical "
                f"correction passes go through reconcile_audit "
                f"(run_audit=False)."
            )
        logger.info("EOD reconciliation | date=%s (explicit)", run_date)

    # ── Session-axis gate: eod_pnl rows are SESSIONS (I9615) ───────────────
    # Every chain-linked measurement over eod_pnl treats adjacent rows as
    # adjacent sessions. A row keyed to a day the market never opened is not a
    # session, and it silently contributes a spurious link to every chained
    # series computed on top.
    #
    # This is not hypothetical. `eod_pnl` carries a row for Good Friday
    # 2026-04-03, written by a live EOD run at 2026-04-03T20:20:21Z. Measured
    # from the live artifact: it is NOT a duplicate of 2026-04-02 — the same
    # three names (CVX/NVT/TER) at the same share counts carry DIFFERENT market
    # values (CVX 69,049.53 -> 69,118.93), so IB holiday quotes moved the book
    # by +$216.77 (+0.0216%), while `spy_close` was carried forward unchanged
    # (655.830017 on both rows) making `spy_return_pct` exactly 0.000000. The
    # row therefore injects +2.16bp of fabricated alpha into every chained
    # series that crosses it.
    #
    # The LIVE path can no longer key a row this way: `now_dual().trading_day`
    # is `last_closed_trading_day(now)` and returns 2026-04-02 on Good Friday
    # (measured), and the axis guard above pins an explicit run_date to it. The
    # hole that remains is this one — the `run_audit=False` correction/backfill
    # path accepts an arbitrary run_date with no session check at all.
    #
    # RAISE rather than warn. There is no legitimate non-trading-day eod_pnl
    # row: the value is definitionally absent, not merely unmeasured, so a
    # producer that writes one is stating something false. `pnl_integrity`'s
    # session-axis gate (I9615 deliverable 3) DETECTS such a row after the
    # fact; this refuses to create one. Detector and producer are the same
    # invariant expressed at two ends, and only this end can prevent it.
    #
    # This gate does NOT touch the existing 2026-04-03 row. Repairing the
    # historical series is a separate, RESERVED decision (I9629 / I9613).
    if not is_trading_day(date.fromisoformat(run_date)):
        raise RuntimeError(
            f"EOD reconcile refusing to write an eod_pnl row for {run_date}: "
            f"it is not an NYSE trading session. An eod_pnl row IS a session — "
            f"every chain-linked series over this table treats adjacent rows as "
            f"adjacent sessions — so a non-trading-day row is a fabricated link, "
            f"not a missing one (alpha-engine-config-I9615). If a real session "
            f"is missing, backfill that session; do not reconcile a holiday."
        )

    # ── T+1 self-heal, run BEFORE today's own reconcile (config#6349) ───────
    # This used to fire at the tail of `run()`, AFTER the NAV three-way
    # pricing&timing term below had already read yesterday's `prior_positions`
    # / `prior_snapshot_nav` from the DB. A correction to yesterday's row
    # landing mid-way through TODAY's run (observed live: the 7/31 SPY
    # correction wrote 2026-08-03T20:56:04Z, inside 8/3's own invocation)
    # therefore arrived one run too late — 8/3 had already computed its
    # mark_basis(t−1) off the stale 7/31 snapshot and paged the hard gate
    # (alpha-engine-config#6349). Running the self-heal first closes the gap:
    # any prior day corrected here is corrected before this run's
    # `prior_positions` read (see the "NAV change reconciliation" block).
    # Cheap in the common case (a few ArcticDB reads, no re-reconcile when
    # clean). Fail-soft: a correction-pass error must never block today's EOD.
    # ``run_audit=False`` on the live run is how the audit's own re-reconciles
    # avoid recursing back into the audit.
    if run_audit:
        from nousergon_lib.logging import get_flow_doctor as _get_flow_doctor_early
        try:
            _fd_early = _get_flow_doctor_early()
        except Exception:  # noqa: BLE001 — flow-doctor optional / not configured
            _fd_early = None
        try:
            from executor.reconcile_audit import audit_window
            _audit = audit_window(exclude_dates={run_date}, send_email=False)
            if _audit.get("corrected"):
                logger.warning(
                    "[reconcile_audit] corrected %d prior day(s) against settled "
                    "ArcticDB before today's reconcile: %s", len(_audit["corrected"]),
                    [c["date"] for c in _audit["corrected"]],
                )
            else:
                logger.info(
                    "[reconcile_audit] clean — %d trailing day(s) checked, all "
                    "stored closes match settled ArcticDB within tolerance.",
                    _audit.get("checked", 0),
                )
        except Exception as _ae:  # noqa: BLE001 — self-heal is secondary; today's EOD must still run
            logger.warning("[reconcile_audit] trailing self-heal FAILED (non-fatal): %s", _ae)
            if _fd_early:
                _fd_early.report(_ae, severity="warning", context={
                    "site": "reconcile_audit_selfheal", "run_date": run_date})

    # Previous NYSE trading day — the baseline every "daily" figure must be
    # measured against. Used to detect skipped-SF gaps (config#1228).
    expected_prev_td = previous_trading_day(date.fromisoformat(run_date))
    # data_warnings is populated through the run (gap flags, NAV residual)
    # and surfaced in the EOD email + report artifact.
    data_warnings: list[str] = []
    _health_start = _time.time()

    config = load_config()
    # Validated at the TOP of the run: an unrecognised nav_basis must refuse
    # before any artifact is written, not after (alpha-engine-config-I9638).
    nav_basis = resolve_nav_basis(config)

    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]

    # Preflight: AWS_REGION + S3 bucket reachable. Fail fast so a
    # misconfigured env surfaces immediately instead of deeper down.
    from executor.preflight import ExecutorPreflight
    ExecutorPreflight(bucket=trades_bucket, mode="eod").run()

    # Flow Doctor: retrieve the shared instance set up at module import
    from nousergon_lib.logging import get_flow_doctor
    fd = get_flow_doctor()

    if not config.get("email_sender") or not config.get("email_recipients"):
        logger.warning(
            "Email not configured (email_sender/email_recipients missing from risk.yaml) "
            "— EOD email will be skipped"
        )

    conn = init_db(db_path)

    # Load EOD state from S3 snapshot keyed by run_date.
    #
    # 2026-04-28 (Phase 2 of EOD-SF cutover): replaced the live IB-read
    # block (`get_account_snapshot` + `get_positions` +
    # `get_accrued_dividends_by_symbol`) with a snapshot read. The
    # snapshot is written by `executor/snapshot_capturer.py` running as
    # the SF's `CaptureSnapshot` step before this step. The snapshot
    # decouples capture from reconciliation — the row keyed by
    # `run_date=X` is now built from observations made at time X, not
    # from now-as-of state at write-time. PR #116's `run_date != today`
    # hard-block is no longer needed: the snapshot-existence check is
    # the new contract, and snapshot existence is what makes the run
    # safe (today, last Tuesday, or any other date with a snapshot).
    #
    # IRREVERSIBILITY (alpha-engine-config-I5569 / I6705): this is the
    # ONE non-re-runnable read in the whole EOD pipeline — the snapshot
    # it loads was itself a live IB capture (see
    # `executor/snapshot_capturer.py::run` docstring). If it is missing
    # for `run_date`, there is no historical source to reconstruct it
    # from; the cost of a missed day was measured in
    # alpha-engine-config-I5325. Mitigations already in place: same-day
    # bounded retry + irreversible-deadline paging inside the EOD SF's
    # `CaptureSnapshot` state (nousergon-data-PR1260), and an independent
    # pre-midnight positive existence check,
    # `alpha-engine-eod-snapshot-existence-check`, scheduled separately
    # so it still fires even if the EOD SF never reaches `CaptureSnapshot`
    # at all (nousergon-data-PR1265).
    from executor.snapshot_capturer import load_snapshot
    snapshot = load_snapshot(
        bucket=trades_bucket,
        run_date=run_date,
        region=config.get("aws_region", "us-east-1"),
    )
    if snapshot is None:
        msg = (
            f"No snapshot at s3://{trades_bucket}/trades/snapshots/{run_date}.json — "
            f"`executor/snapshot_capturer.py` must run before "
            f"`executor/eod_reconcile.py` so the row keyed by run_date={run_date!r} "
            f"sources its inputs from observations made at time {run_date!r} "
            f"(not from now-as-of IB state). The CaptureSnapshot SF step is the "
            f"canonical writer; for manual recovery, run "
            f"`python executor/snapshot_capturer.py --date {run_date}` while IB "
            f"Gateway is up on ae-trading."
        )
        if fd:
            fd.report(
                RuntimeError(msg),
                severity="critical",
                context={"site": "eod_load_snapshot", "run_date": run_date},
            )
        raise RuntimeError(msg)

    account = snapshot["account"]
    nav = account["net_liquidation"]
    positions = snapshot["positions"]
    dividends_by_symbol = snapshot.get("accrued_dividends", {})
    for _tkr, _accrued in dividends_by_symbol.items():
        if _tkr in positions:
            positions[_tkr]["accrued_dividend"] = _accrued
    logger.info(
        "EOD: snapshot loaded | NAV=$%.2f positions=%d dividends=%d captured_at=%s",
        nav,
        len(positions),
        len(dividends_by_symbol),
        snapshot.get("captured_at"),
    )

    # Enrich positions with sector. Lookup chain:
    #   0. index/ETF core (SPY etc.) → "Broad Market / Index" — not a GICS
    #      constituent, so it must short-circuit before the sector lookups
    #      (an index ETF must never be mislabeled with a sector even if it
    #      somehow appears in a lookup table).
    #   1. signals.json today (universe + buy_candidates)
    #   2. trades.db entry_trade.sector
    #   3. S&P 500+400 constituents.json (latest weekly snapshot) — catches
    #      legacy/fractional-share positions whose ticker has fallen out of
    #      today's research universe (e.g. dividend-reinvestment remnants).
    # A missing sector is an observability failure (blank rows in sector
    # attribution), not a hard error — log loudly and continue with "Unknown"
    # only when all sources miss.
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    try:
        sig_data, _ = _load_signals_from_s3(signals_bucket, run_date)
        sector_lookup = {}
        # signals.json::universe read: this is the executor sizing/exit path —
        # the ONE fleet-level exception to "resolve ticker lists from
        # decision_set, not universe" (alpha-engine-config#5809). Formal
        # policy-clause registration tracked separately, not yet landed:
        # alpha-engine-config#6448.
        for s in (sig_data.get("universe", []) + sig_data.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and s.get("sector"):
                sector_lookup[t] = s["sector"]
        constituents_lookup: dict[str, str] | None = None
        for ticker in positions:
            if positions[ticker].get("sector"):
                continue
            etf_sector = _index_etf_sector(ticker)
            if etf_sector:
                positions[ticker]["sector"] = etf_sector
                continue
            if ticker in sector_lookup:
                positions[ticker]["sector"] = sector_lookup[ticker]
                continue
            entry = get_entry_trade(conn, ticker)
            if entry and entry.get("sector"):
                positions[ticker]["sector"] = entry["sector"]
                continue
            if constituents_lookup is None:
                constituents_lookup = _load_constituents_sector_map(signals_bucket)
            if ticker in constituents_lookup:
                positions[ticker]["sector"] = constituents_lookup[ticker]
                continue
            logger.error(
                "Sector unknown for %s — missing from signals.json, entry trade, "
                "and S&P 500+400 constituents. Sector attribution will be incomplete.",
                ticker,
            )
            positions[ticker]["sector"] = "Unknown"
    except Exception as e:
        logger.error(f"Sector enrichment failed: {e}")

    # Prior day's NAV (to compute daily return). Also capture its DATE so we
    # can detect when it is not the previous trading day — i.e. an eod_pnl row
    # is missing because a weekday/EOD SF was skipped, which makes the
    # headline NAV daily return span multiple sessions.
    prior_row = conn.execute(
        "SELECT date, portfolio_nav, nav_basis FROM eod_pnl "
        "WHERE date < ? ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    prior_eod_date = (
        date.fromisoformat(prior_row[0]) if prior_row and prior_row[0] else None
    )
    prior_nav = prior_row[1] if prior_row else None
    # NULL on every row written before alpha-engine-config-I9638, which by
    # definition means the pre-flag basis: IB NetLiquidation.
    prior_nav_basis = (
        (prior_row[2] or NAV_BASIS_IB_NETLIQ) if prior_row else None
    )

    # Headline gap guard: if the prior eod_pnl row is not the previous trading
    # day, the NAV-level daily return / alpha span more than one session. Per-
    # position returns are gap-corrected from ArcticDB, but the NAV baseline
    # stays stale until the missing row is backfilled (config#1229 / Phase 2).
    if prior_nav is not None and prior_eod_date is not None and prior_eod_date != expected_prev_td:
        hdr_warn = (
            f"Headline daily return/alpha span multiple sessions: prior eod_pnl row is "
            f"{prior_eod_date} but the previous trading day is {expected_prev_td} (an "
            f"eod_pnl row is missing — a weekday/EOD SF was skipped). Per-position "
            f"returns are gap-corrected from ArcticDB; the NAV-level baseline is stale "
            f"until the missing row is backfilled (config#1229)."
        )
        logger.warning(hdr_warn)
        data_warnings.append(hdr_warn)

    # SPY return for the day.
    #
    # Both legs are read from the SETTLED ArcticDB macro source — the prior
    # leg is NOT taken from the stored ``eod_pnl.spy_close`` (config#1276).
    # Freezing the prior close meant that any day whose stored close was a
    # pre-settlement value (captured at same-day ~4:20pm ET, before the
    # official close lands in ArcticDB) silently corrupted the NEXT day's
    # spy_return as the denominator — and never self-healed when ArcticDB
    # later corrected. Windowing stays gap-consistent: we still span to the
    # prior *eod_pnl* DATE (the same baseline ``prior_nav`` uses, so SPY and
    # the portfolio measure the same interval across a skipped session), but
    # the close VALUE is always the authoritative settled close for that date.
    # ── Ex-date dividends for the interval (alpha-engine-config-I8188) ──────
    # Fetched ONCE for the held names and SPY together, so both legs of the
    # comparison are on the same return definition and the same interval.
    # The window is (prior eod_pnl date, run_date] — the same interval
    # prior_nav spans — so a skipped session cannot drop a dividend or count
    # one twice.
    prior_date_str = prior_row[0] if prior_row and prior_row[0] else None
    (
        ex_dividends,
        ex_dividend_pay_dates,
        dividend_accrual_available,
        dividend_warning,
    ) = fetch_ex_dividends(
        list(positions.keys()), run_date, prior_date=prior_date_str,
    )
    if dividend_warning:
        data_warnings.append(dividend_warning)
    spy_dividend_per_share = ex_dividends.get(SPY_TICKER, 0.0)

    # SPY TOTAL return, not price return (alpha-engine-config-I8188, defect 3).
    # The portfolio leg is NAV-based and therefore already total return, so
    # leaving the benchmark on price return understated it by ~0.50pp over
    # 2026-03-09 -> 2026-08-21 and overstated reported alpha by the same
    # amount. Distributions going ex in the interval are added to the ending
    # close; spy_close itself is left as the PRICE close so the stored column
    # keeps its meaning and the adjustment stays inspectable
    # (spy_dividend_per_share is persisted beside it).
    spy_price = _spy_close(run_date, config)
    spy_return = None
    if spy_price:
        prior_date_row = conn.execute(
            "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
            (run_date,),
        ).fetchone()
        if prior_date_row:
            prior_spy = _spy_close(prior_date_row[0], config)
            if prior_spy:
                spy_return = spy_total_return_pct(
                    spy_close=spy_price,
                    prior_spy_close=prior_spy,
                    spy_dividend_per_share=spy_dividend_per_share,
                )
                if spy_dividend_per_share:
                    logger.info(
                        "SPY total return for %s includes $%.4f/share of "
                        "distributions going ex in (%s, %s]",
                        run_date, spy_dividend_per_share, prior_date_row[0], run_date,
                    )
            else:
                logger.warning("Could not fetch settled SPY close for prior date %s", prior_date_row[0])
        else:
            logger.warning("No prior eod_pnl row — cannot compute SPY return")

    # Same-day EOD reads run_date's SPY close from ArcticDB at ~4:20pm ET,
    # which can still be pre-settlement. Mark the artifact provisional so the
    # console flags it and the T+1 reconcile_audit pass re-finalizes it from
    # the settled close. A re-reconcile of a PAST date (run_date earlier than
    # today's trading_day) is by definition post-settlement → final.
    spy_close_provisional = run_date == today_trading_day

    # ── Load closing prices from ArcticDB ──────────────────────────────────
    # Hard-fails on any miss: EOD reconcile must reconcile against an
    # authoritative price source, not IB Gateway's delayed intraday data.
    #
    # Macro-routed held positions (sector ETFs / VIX / TNX / etc.) live in
    # the `macro` library, NOT `universe`. The portfolio-optimizer cutover
    # (2026-05-13) made SPY a held core position; its first EOD on
    # 2026-05-14 raised NoSuchVersionException because reconcile was
    # universe-only. SPY-as-held is now read from `universe` directly
    # (alpha-engine-data #245 lifted SPY to a full universe member via
    # `_UNIVERSE_EXTRA`); only the remaining macro-only-Close symbols still
    # need the macro-lib dispatch. Mirror `price_cache.load_price_histories`
    # (executor/price_cache.py, `_MACRO_SYMBOLS`).
    from executor.price_cache import (
        _MACRO_SYMBOLS,
        _open_macro_library,
        _open_universe_library,
    )
    universe_lib = _open_universe_library(trades_bucket)
    macro_lib = None  # lazy-open only if a macro-routed held ticker appears
    target_ts = pd.Timestamp(run_date).normalize()
    closing_prices: dict[str, float] = {}
    # Vendor that produced each frozen close — see _row_close_source above.
    close_sources: dict[str, str] = {}
    # Same-day traded [Low, High] per held ticker — used only to validate the
    # IB portfolio mark (config#6349/#6818), never to price positions.
    day_low: dict[str, float] = {}
    day_high: dict[str, float] = {}
    # Authoritative prior-day baseline: the last ArcticDB row strictly before
    # run_date, with its date, so daily returns are measured against the real
    # previous trading day rather than a possibly-stale snapshot (config#1228).
    prior_closes: dict[str, float] = {}
    prior_close_dates: dict[str, date] = {}
    missing: list[str] = []
    for ticker in positions.keys():
        if ticker in _MACRO_SYMBOLS:
            if macro_lib is None:
                macro_lib = _open_macro_library(trades_bucket)
            lib = macro_lib
        else:
            lib = universe_lib
        try:
            df = lib.read(ticker).data
        except Exception as e:
            missing.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty or "Close" not in df.columns:
            missing.append(f"{ticker} (no Close column)")
            continue
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        match = df[idx == target_ts]
        if match.empty:
            missing.append(f"{ticker} (no row for {run_date})")
            continue
        closing_prices[ticker] = float(match["Close"].iloc[-1])
        close_sources[ticker] = _row_close_source(match)
        if "Low" in match.columns and "High" in match.columns:
            day_low[ticker] = float(match["Low"].iloc[-1])
            day_high[ticker] = float(match["High"].iloc[-1])
        # Capture the previous available close (last row strictly before
        # run_date) + its date — the gap-aware daily-return baseline.
        prior_mask = idx < target_ts
        if prior_mask.any():
            prior_rows = df[prior_mask]
            prior_closes[ticker] = float(prior_rows["Close"].iloc[-1])
            prior_close_dates[ticker] = (
                pd.Timestamp(prior_rows.index[-1]).normalize().date()
            )
    if missing:
        raise RuntimeError(
            f"ArcticDB closing-price lookup failed for {len(missing)} "
            f"held ticker(s) on {run_date}: {missing}. EOD reconcile cannot "
            "proceed without authoritative closes."
        )
    logger.info(
        "[data_source=arcticdb] Loaded closing prices for %d/%d held tickers on %s",
        len(closing_prices), len(positions), run_date,
    )

    # ── Broker-mark repair, BEFORE any NAV-derived figure is computed ───────
    # (alpha-engine-config-I9627.) The IB reference mark is captured here
    # rather than inside the positions loop below, because the correction has
    # to land before `daily_return`, `alpha` and every position weight are
    # derived from NAV — a correction applied after them would leave the
    # headline and the components on different prices, which is the defect
    # being fixed, one layer up.
    for _tkr, _pos in positions.items():
        _pos["ib_market_value"] = _pos.get("market_value")

    # IB mark-outside-range detection (config#6349/#6818) — flags positions,
    # used below to name culprits in the hard-gate alert and classify a
    # fully-explained breach as broker data quality rather than a defect.
    ib_mark_range_flags = _detect_ib_mark_outside_range(
        positions=positions, day_low=day_low, day_high=day_high,
    )
    if ib_mark_range_flags:
        logger.warning(
            "IB mark outside day's traded range for %d ticker(s): %s",
            len(ib_mark_range_flags),
            ", ".join(
                f"{f['ticker']} mark=${f['ib_mark']:.2f} range=[${f['day_low']:.2f},"
                f" ${f['day_high']:.2f}] error=${f['mark_error_usd']:+,.0f}"
                for f in ib_mark_range_flags
            ),
        )

    # Coverage of the check itself (alpha-engine-config-I9637). Computed from
    # the stamps the detector just left, BEFORE the correction, so the number
    # describes what the gate could see rather than what it acted on.
    mark_coverage = check_mark_coverage(positions, nav=nav, run_date=run_date)
    logger.info(
        "Custodian-mark check coverage: %d/%d held position(s) range-checked",
        mark_coverage["checked"], mark_coverage["held"],
    )
    for _w in mark_coverage["warnings"]:
        # An unchecked MATERIAL position means NAV is published on a mark the
        # gate never saw — ERROR. An immaterial one is a WARNING. Neither
        # halts: the cause is a Close-only macro library in another repo.
        if mark_coverage["unchecked_material"]:
            logger.error("MARK CHECK COVERAGE: %s", _w)
        else:
            logger.warning("MARK CHECK COVERAGE: %s", _w)
        data_warnings.append(_w)

    nav_ib_raw = nav
    mark_correction = plan_nav_mark_correction(
        ib_mark_range_flags,
        settled_closes=closing_prices,
        day_low=day_low,
        day_high=day_high,
        nav=nav,
        run_date=run_date,
    )
    if mark_correction["applied"]:
        nav = mark_correction["nav_corrected"]
        # ERROR, not WARNING: the broker sent a provably wrong number and the
        # book was repaired around it. That the pipeline no longer halts does
        # not make it a routine event, and the alert names every ticker.
        logger.error("NAV MARK CORRECTION: %s", mark_correction["message"])
        data_warnings.append(mark_correction["message"])
        # Verify the action worked, rather than assuming it. A settled close
        # lies inside the day's own traded range by construction, so a repaired
        # mark cannot still be out of range; if one is, the correction did not
        # do what it claims and the run must not proceed on it.
        residual_flags = [
            f["ticker"] for f in _detect_ib_mark_outside_range(
                positions={
                    c["ticker"]: {
                        "shares": c["shares"],
                        "ib_market_value": c["shares"] * c["settled_close"],
                    }
                    for c in mark_correction["corrections"]
                },
                day_low=day_low,
                day_high=day_high,
            )
        ]
        if residual_flags:
            raise RuntimeError(
                f"NAV mark correction did not converge for {run_date}: "
                f"{residual_flags} remain outside the day's traded range after "
                "being repriced to their settled closes. The correction is "
                "unsound on this data; NAV is not published."
            )
        # Write the repaired mark onto the positions themselves
        # (alpha-engine-config-I10048). Until this, only the headline NAV
        # moved, so the position rows — and every per-name mark-basis reader
        # downstream, today AND tomorrow via the persisted snapshot — still
        # carried the broker's wrong number. See the block comment above
        # `_apply_mark_correction_to_positions`.
        _corrected_names = _apply_mark_correction_to_positions(
            positions, mark_correction["corrections"],
        )
        logger.info(
            "NAV mark correction written to %d position(s): %s — raw broker "
            "mark preserved as ib_market_value_raw, range flags untouched",
            len(_corrected_names), ", ".join(_corrected_names) or "none",
        )
    elif mark_correction["refused"] or mark_correction["unrepairable"]:
        if mark_correction["message"]:
            logger.error("NAV MARK CORRECTION: %s", mark_correction["message"])
            data_warnings.append(mark_correction["message"])
        for _u in mark_correction["unrepairable"]:
            _msg = (
                f"Broker mark for {_u['ticker']} on {run_date} is NOT repairable: "
                f"{_u['why']}. The custodian gate still holds for this name."
            )
            logger.error("NAV MARK CORRECTION: %s", _msg)
            data_warnings.append(_msg)

    # ── NAV basis: settled-close NAV computed BESIDE IB NetLiquidation ──────
    # Ruled 2026-08-31 (alpha-engine-config#9638), option (b) STAGED. Both
    # figures are computed here on EVERY run, whatever the basis:
    #
    #   nav_ib_usd       IB NetLiquidation after the I9627 mark correction —
    #                    exactly what `nav` carried before this block existed.
    #   nav_settled_usd  total_cash + accrued_interest + Σ shares × settled
    #                    close, i.e. the same rebuild the attribution
    #                    basis-level gate already strikes IB against. Measured
    #                    faithful: `nav_identity_residual_usd` ≤ $64 on 48/48
    #                    sessions.
    #
    # `nav_basis` decides only which one becomes the headline `nav`. IB is
    # retained as the broker cross-check under BOTH bases (I6819 item 3): the
    # three-way hard gate and the attribution basis-level gate are re-based
    # onto `nav_ib_usd` below so neither goes tautological after a cut-over.
    nav_ib_usd = nav
    _basis_cash = account.get("total_cash")
    settled_mv_today_usd, settled_fallback_tickers = _settled_position_value_usd(
        positions, closing_prices,
    )
    if _basis_cash is None:
        # Absent, never a silent zero: a settled NAV rebuilt without the cash
        # leg is not a smaller number, it is a wrong one.
        nav_settled_usd = None
        nav_basis_unavailable_reason = (
            "broker cash balance absent from the snapshot — cash + accrued + "
            "Σ shares·settled close has no cash leg to rebuild from"
        )
    else:
        nav_settled_usd = (
            float(_basis_cash)
            + float(account.get("accrued_interest") or 0.0)
            + settled_mv_today_usd
        )
        nav_basis_unavailable_reason = None
    nav_basis_diff_usd = (
        nav_ib_usd - nav_settled_usd if nav_settled_usd is not None else None
    )
    nav_basis_diff_bps = (
        (nav_basis_diff_usd / nav_ib_usd * 10000.0)
        if (nav_basis_diff_usd is not None and nav_ib_usd) else None
    )

    if nav_basis == NAV_BASIS_SETTLED_CLOSE:
        # Refuse rather than guess. Publishing a NAV labelled `settled_close`
        # that is partly a broker mark — or that has no cash leg at all — is
        # the untraceable outcome the flag exists to remove.
        if nav_settled_usd is None:
            raise RuntimeError(
                f"nav_basis=settled_close but the settled NAV for {run_date} "
                f"cannot be rebuilt: {nav_basis_unavailable_reason}. NAV is "
                "not published. Set nav_basis to ib_netliq in risk.yaml to "
                "fall back to the broker figure for this session."
            )
        if settled_fallback_tickers:
            raise RuntimeError(
                f"nav_basis=settled_close but {len(settled_fallback_tickers)} "
                f"held name(s) carry no settled ArcticDB close for {run_date}: "
                f"{', '.join(sorted(settled_fallback_tickers))}. The headline "
                "NAV would silently contain the broker's mark for them while "
                "claiming a settled-close basis. NAV is not published — "
                "backfill the close(s), or set nav_basis to ib_netliq."
            )
        nav = nav_settled_usd
    logger.info(
        "NAV basis=%s | nav_ib_usd=$%s | nav_settled_usd=$%s | "
        "nav_basis_diff_usd=$%s | nav_basis_diff_bps=%s",
        nav_basis,
        f"{nav_ib_usd:,.2f}" if nav_ib_usd is not None else "ABSENT",
        f"{nav_settled_usd:,.2f}" if nav_settled_usd is not None else "ABSENT",
        f"{nav_basis_diff_usd:+,.2f}" if nav_basis_diff_usd is not None else "ABSENT",
        f"{nav_basis_diff_bps:+.2f}" if nav_basis_diff_bps is not None else "ABSENT",
    )
    # Shadow-stage detection (I9638 deliverable 4). The routine basis gap is an
    # OBSERVATION — it is published on every row and in the report whatever its
    # size. Only a gap past the NAV three-way hard-gate tolerance earns a line
    # in data_warnings, and no new page: the level itself is already paged by
    # the attribution basis-level gate below, and a second page for one fact is
    # the I10049 defect.
    if nav_basis_diff_usd is not None and nav_ib_usd and abs(
        nav_basis_diff_usd
    ) > _nav_hard_gate_tolerance_usd(nav_ib_usd):
        data_warnings.append(
            f"NAV basis divergence for {run_date}: IB NetLiquidation exceeds "
            f"the settled-close rebuild by ${nav_basis_diff_usd:+,.0f} "
            f"({nav_basis_diff_bps:+.1f}bp), past the "
            f"${_nav_hard_gate_tolerance_usd(nav_ib_usd):,.0f} three-way "
            f"tolerance. Headline NAV is on basis {nav_basis!r}."
        )
    # A basis change is a change in what the NAV series MEANS, so the first
    # `daily_return_pct` after a cut-over is struck across two definitions.
    # Named rather than left for a reader to infer from a one-day jump.
    if prior_nav_basis is not None and prior_nav_basis != nav_basis:
        data_warnings.append(
            f"NAV basis changed for {run_date}: the prior eod_pnl row is on "
            f"{prior_nav_basis!r} and this row is on {nav_basis!r}. This "
            f"session's daily_return_pct and daily_alpha_pct span the basis "
            f"change and are not comparable to the surrounding series."
        )

    # ── NAV-derived headline figures (computed on the CORRECTED NAV) ────────
    if prior_nav is None:
        logger.info("First trading day — no prior NAV, daily return unavailable")
        daily_return = None
    else:
        daily_return = ((nav - prior_nav) / prior_nav * 100)

    alpha = (daily_return - spy_return) if (daily_return is not None and spy_return is not None) else None

    logger.info(
        f"NAV=${nav:,.2f} | daily={daily_return:.2f}% | "
        f"SPY={spy_return:.2f}% | alpha={alpha:.2f}%"
        if all(x is not None for x in [daily_return, spy_return, alpha])
        else f"NAV=${nav:,.2f} | prior_nav={prior_nav}"
    )

    # ── Per-position daily return & alpha contribution ──────────────────────
    # Look up prior day's positions_snapshot to get yesterday's price per ticker
    prior_snapshot_row = conn.execute(
        "SELECT positions_snapshot, portfolio_nav, total_cash, accrued_interest, "
        "nav_mark_correction_json, nav_ib_usd "
        "FROM eod_pnl WHERE positions_snapshot IS NOT NULL AND date < ? "
        "ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    prior_positions = {}
    prior_snapshot_loaded = False
    # Prior snapshot's NAV / cash / accrued, used to reconstruct the prior-day
    # settled NAV for the pricing&timing reconciliation term below. Pulled from
    # the SAME row as prior_positions so the settled-MV sum is consistent.
    prior_snapshot_nav = None
    prior_snapshot_cash = None
    prior_snapshot_accrued = None
    if prior_snapshot_row and prior_snapshot_row[0]:
        try:
            prior_positions = json.loads(prior_snapshot_row[0])
            prior_snapshot_loaded = True
            # The mark basis is IB-vs-settled by definition, so the prior leg
            # must be the prior day's IB figure — NOT its headline
            # portfolio_nav, which is the same number today but becomes the
            # SETTLED number after a cut-over, collapsing the basis to a
            # tautological zero (alpha-engine-config-I9638). nav_ib_usd is
            # NULL on every pre-I9638 row, where portfolio_nav IS the IB
            # figure, so the fallback is exact rather than approximate.
            prior_snapshot_nav = (
                prior_snapshot_row[5]
                if prior_snapshot_row[5] is not None
                else prior_snapshot_row[1]
            )
            prior_snapshot_cash = prior_snapshot_row[2]
            prior_snapshot_accrued = prior_snapshot_row[3]
        except (json.JSONDecodeError, TypeError):
            pass
        # Backward compatibility (alpha-engine-config-I10048). A prior row
        # written between PR524 and this fix carries a CORRECTED
        # `portfolio_nav` beside a RAW per-position `ib_market_value`.
        # `prior_snapshot_nav` above is the corrected number, so leaving the
        # positions raw makes the previous day's correction reappear today as
        # `nav_identity_residual_usd` — the +$1,605 HOOD case. The correction
        # plan was persisted whole on that same row, so the corrected prior
        # mark is derived from its own `corrections` list rather than guessed.
        if prior_snapshot_loaded:
            _restated = _restate_prior_positions_for_mark_correction(
                prior_positions, prior_snapshot_row[4],
            )
            if _restated:
                logger.warning(
                    "Restated %d prior-day position(s) to the corrected broker "
                    "mark the prior NAV was struck on: %s (pre-I10048 snapshot "
                    "— raw mark preserved as ib_market_value_raw)",
                    len(_restated), ", ".join(_restated),
                )

    from executor.eod_report import _buy_entry_prices
    from executor.trade_logger import get_todays_trades

    trades_today_rows = get_todays_trades(conn, run_date)
    buy_entry_px = _buy_entry_prices(trades_today_rows)

    for ticker, pos in positions.items():
        shares = pos.get("shares", 0)
        mv = pos.get("market_value", 0)
        current_price = mv / shares if shares else 0

        # ``ib_market_value`` — IB's raw mark-to-market before the settled-close
        # override — was already captured above the mark-correction block, which
        # must run before NAV is consumed. It is deliberately NOT re-read here:
        # the pricing&timing term and the mark flags must both describe the mark
        # the broker actually sent.

        # Prefer closing price from daily_closes over IB Gateway's delayed data
        if ticker in closing_prices:
            current_price = closing_prices[ticker]
            pos["market_value"] = current_price * shares
            mv = pos["market_value"]
        # Persist the canonical close so tomorrow's reconcile reads the same
        # source for prior_price (not derived from possibly-stale IB MV).
        pos["closing_price"] = current_price
        # ...and WHICH VENDOR produced it. The T+1 audit re-reads this cell
        # after the polygon pass has overwritten it, so this is the only
        # surviving record of the provenance of the number we actually froze
        # (alpha-engine-config-I10360).
        pos["close_source"] = close_sources.get(ticker, CLOSE_SOURCE_UNKNOWN)

        # Daily return — gap-aware. Held-through positions price against the
        # previous TRADING day's ArcticDB close (config#1228); a stale
        # snapshot baseline previously inflated returns across a skipped-SF
        # gap (RGEN +14.92% on 2026-06-25 vs the 06-23, not 06-24, close).
        prior_pos = prior_positions.get(ticker)
        daily_pct, daily_usd, prior_price, na_reason = _compute_daily_return(
            ticker, pos, prior_pos, current_price, shares,
            prior_closes.get(ticker), prior_close_dates.get(ticker),
            expected_prev_td,
            add_entry_px=buy_entry_px.get(ticker),
        )
        pos["daily_return_pct"] = daily_pct
        pos["daily_return_usd"] = daily_usd

        # ── Per-ticker price-source traceability (schema 2.1) ─────────────────
        # Expose the lot breakdown + source prices so the report consumer can
        # trace exactly which prices drove each position's daily return:
        #   retained_shares × (close − prior_close) + added_shares × (close − entry_fill)
        prior_shares = 0.0
        if prior_pos:
            try:
                prior_shares = float(prior_pos.get("shares", 0) or 0)
            except (TypeError, ValueError):
                prior_shares = 0.0
        retained = min(prior_shares, shares) if prior_shares > 0 else 0.0
        added = max(0.0, shares - prior_shares) if prior_shares > 0 else 0.0
        pos["prior_shares"] = prior_shares
        pos["retained_shares"] = retained
        pos["added_shares"] = added
        pos["prior_price"] = prior_price
        pos["entry_price"] = buy_entry_px.get(ticker)
        # ── end price-source traceability ─────────────────────────────────────

        if na_reason:
            # Fail loud: the figure is an explicit N/A, not a silent zero.
            pos["daily_return_na"] = True
            pos["daily_return_na_reason"] = na_reason
            logger.warning("Daily-return N/A | %s", na_reason)
            data_warnings.append(na_reason)

        # Dividend attribution: today's accrued dividend for this ticker vs
        # yesterday's snapshot. Delta is the day's dividend income (or its
        # reversal when paid to cash). Flows into position α instead of
        # leaking into the cash residual. Skipped when the baseline is N/A
        # (no valid prior price to express the accrual against).
        if prior_price is not None:
            _apply_dividend_delta(pos, prior_pos, prior_price, shares)

        # Alpha contribution: (weight * position_return) - (weight * SPY_return)
        weight = mv / nav if nav else 0
        pos_spy = spy_return if spy_return is not None else 0
        pos["alpha_contribution_pct"] = weight * (pos["daily_return_pct"] - pos_spy)
        pos["alpha_contribution_usd"] = pos["alpha_contribution_pct"] / 100 * nav if nav else 0

    # Per-position mark-basis diagnostics (alpha-engine-config-I9085). The
    # `[Low, High]` flag is a strict SUBSET of "the IB mark is not the close":
    # a mark can sit inside the day's traded range and still be off the close
    # by a full percent, and it still moves `pricing_timing_usd` dollar for
    # dollar. Persisting the basis and the off-close percentage on every
    # position makes eod_report.json self-diagnosing for the whole class,
    # not just for the boundary-crossing subset (generalizes config#6349
    # deliverable 3).
    for _tkr, _pos in positions.items():
        _basis = _mark_basis_usd(_pos)
        if _basis is None:
            continue
        _pos["mark_basis_usd"] = _basis
        _pos["ib_mark_off_close_pct"] = _off_close_pct(_pos)

    # ── Explicit ex-date dividend accrual (alpha-engine-config-I8188) ───────
    # dividend_usd summed to exactly $0.00 across all 115 live sessions while
    # the book held LMT (34 sessions), CTAS (27), MA, COST, AXP, BRO and FAST
    # — all payers — because the only source was IB's per-symbol accrual,
    # which a paper account never populates (measured: 0 non-zero
    # accrued_dividend values across 114 persisted snapshots) and which the
    # code read as a genuine zero. The dividends were not lost from the RETURN
    # (NAV is IB NetLiquidation, which the cash reaches); they were lost from
    # the ATTRIBUTION and silently credited to the unattributed plug. This
    # accrual moves them into a named line. SPY's own distribution is applied
    # on the benchmark leg, not here.
    dividend_accruals = accrue_position_dividends(
        positions, prior_positions, ex_dividends,
    )
    dividend_accrued_usd = sum(a["amount_usd"] for a in dividend_accruals)
    for accrual in dividend_accruals:
        record_dividend_accrual(
            conn,
            ticker=accrual["ticker"],
            ex_date=run_date,
            pay_date=ex_dividend_pay_dates.get(accrual["ticker"]),
            per_share=accrual["per_share"],
            shares=accrual["shares"],
            amount_usd=accrual["amount_usd"],
        )
    # Release every receivable whose pay date has arrived. NAV is cash-basis on
    # this account, so this is the day the dividend actually reaches NAV.
    dividend_released_usd, dividend_released_rows = settle_due_dividend_accruals(
        conn, run_date,
    )
    # The term that keeps the identity closed on BOTH dates: on the ex-date
    # position P&L gains the accrual while NAV has not moved, on the pay date
    # NAV gains the cash while position P&L has not moved, and this difference
    # cancels each in turn.
    dividend_timing_usd = dividend_accrued_usd - dividend_released_usd
    dividend_receivable = dividend_receivable_usd(conn)
    if dividend_accruals or dividend_released_rows:
        logger.info(
            "Dividends %s: accrued $%.2f across %d name(s) | released $%.2f "
            "across %d accrual(s) | timing $%+.2f | receivable outstanding $%.2f",
            run_date, dividend_accrued_usd, len(dividend_accruals),
            dividend_released_usd, len(dividend_released_rows),
            dividend_timing_usd, dividend_receivable,
        )

    # data_warnings was initialized at the top of run() and accumulates gap
    # flags (per-position N/A, headline multi-session) plus the NAV residual
    # appended below; it is also extended by _build_position_contexts.

    # ── NAV change reconciliation ───────────────────────────────────────────
    # Every dollar of NAV change must be attributable to a source: position
    # MTM, interest, dividends, or (flagged) unattributed. Anything in the
    # unattributed bucket indicates a pricing/snapshot mismatch, fee, FX,
    # corporate action, or similar — surface it loudly instead of burying it
    # in cash return.
    nav_reconciliation: dict = {}
    if prior_nav is not None:
        total_nav_change = nav - prior_nav
        total_day_usd = sum(p.get("daily_return_usd", 0) for p in positions.values())

        # Interest: day-over-day delta in IB's AccruedCash
        prior_accrued_row = conn.execute(
            "SELECT accrued_interest FROM eod_pnl WHERE accrued_interest IS NOT NULL AND date < ? ORDER BY date DESC LIMIT 1",
            (run_date,),
        ).fetchone()
        today_accrued = account.get("accrued_interest")
        if today_accrued is not None and prior_accrued_row and prior_accrued_row[0] is not None:
            interest_usd = float(today_accrued) - float(prior_accrued_row[0])
        else:
            interest_usd = 0.0

        # Dividends earned today (accrual increase) are already added into
        # each position's daily_return_usd, so they flow through total_day_usd.
        # Payout-day cash inflow is exactly offset by accrual drop in IB's
        # NetLiquidation, so NAV doesn't move from the payout itself — no
        # reconciliation term needed. dividend_usd here is informational
        # only, summing positive accrual deltas for the email.
        dividend_usd = sum(p.get("dividend_usd", 0.0) for p in positions.values())

        # dividend_timing_usd is SUBTRACTED because it is the difference
        # between what position P&L has recognised and what NAV has received:
        # on the ex-date it is +dividend and NAV has not moved, on the pay date
        # it is -dividend and position P&L has not moved. Without this term,
        # ex-date accrual against a cash-basis NAV would trade the attribution
        # error for a timing one -- the residual would read -dividend on the
        # ex-date and +dividend on the pay date instead of simply +dividend
        # once (alpha-engine-config-I8188, defect 3).
        unattributed_usd = (
            total_nav_change - total_day_usd - interest_usd + dividend_timing_usd
        )

        # ── Pricing & timing reconciliation ──────────────────────────────────
        # The headline NAV is IB NetLiquidation; per-position P&L is settled
        # close-to-close. Their day-over-day basis difference (IB intraday/
        # unsettled marks vs settled closes — e.g. provisional SPY, config#1276)
        # used to be buried wholesale in `unattributed_usd` (tens of bps even on
        # no-trade days). Isolate it as `mark_basis(t) − mark_basis(t−1)`, where
        #   mark_basis = nav_ib − (cash + accrued + Σ settled_mv).
        # A constant cash/accrued offset cancels in the day-over-day difference.
        # When any prior input is missing the term is 0, the gap stays in
        # unattributed, and a warning fires (fail loud, never silently hide).
        #
        # TWO distinct quantities live here, and they were the same number
        # until alpha-engine-config-I9638 split them:
        #
        #   mark_basis_delta_usd  the IB-vs-settled divergence, day-over-day
        #                         differenced. A property of the BROKER DATA,
        #                         computed on `nav_ib_usd` under either basis,
        #                         and the input the three-way hard gate reads.
        #   pricing_timing_usd    the mark-basis sleeve actually INSIDE the NAV
        #                         series being attributed. Equal to
        #                         mark_basis_delta_usd on the ib_netliq basis;
        #                         ZERO BY CONSTRUCTION on settled_close, where
        #                         the headline NAV *is* the settled rebuild and
        #                         carries no broker mark to attribute.
        #
        # Subtracting a non-zero mark-basis term from a NAV change that never
        # contained it would corrupt `unattributed_true_usd` — which is what
        # the residual bounds gate is measured against — so the split is load
        # bearing, not cosmetic.
        pricing_timing_usd = 0.0
        mark_basis_delta_usd = 0.0
        pricing_timing_available = False
        today_cash = account.get("total_cash")
        if (
            today_cash is not None
            and prior_snapshot_loaded
            and prior_snapshot_nav is not None
            and prior_snapshot_cash is not None
        ):
            settled_mv_today = sum(
                p.get("market_value", 0) or 0 for p in positions.values()
            )
            settled_mv_prior = 0.0
            for pp in prior_positions.values():
                cp = pp.get("closing_price")
                if cp is not None:
                    settled_mv_prior += float(cp) * float(pp.get("shares", 0) or 0)
                else:
                    settled_mv_prior += float(pp.get("market_value", 0) or 0)
            nav_settled_today = (
                float(today_cash) + float(today_accrued or 0.0) + settled_mv_today
            )
            nav_settled_prior = (
                float(prior_snapshot_cash)
                + float(prior_snapshot_accrued or 0.0)
                + settled_mv_prior
            )
            mark_basis_today = nav_ib_usd - nav_settled_today
            mark_basis_prior = float(prior_snapshot_nav) - nav_settled_prior
            mark_basis_delta_usd = mark_basis_today - mark_basis_prior
            pricing_timing_usd = (
                0.0 if nav_basis == NAV_BASIS_SETTLED_CLOSE
                else mark_basis_delta_usd
            )
            pricing_timing_available = True

        # Realized P&L on shares rotated OUT today — also currently inside
        # `unattributed_usd`; the attribution lifts it into its own sleeve.
        from executor.eod_report import compute_rotation_realized
        rotation_realized_usd = compute_rotation_realized(
            positions, prior_positions, trades_today_rows,
        )
        unattributed_true_usd = (
            unattributed_usd - rotation_realized_usd - pricing_timing_usd
        )

        # ── Explicit transaction costs (alpha-engine-config-I8188, defect 2) ──
        # There was no transaction-cost line anywhere in the P&L schema, so
        # gross and net performance were the same number and neither was
        # labelled. Sourced from FILLS: commission from IB's per-execution
        # commissionReport, slippage as implementation shortfall against the
        # arrival price (measured over the live window: 468 fills, +6.4bp
        # of traded notional).
        #
        # NOTE the residual measurement that reshaped this deliverable: the
        # residual's turnover dependence (-$222/turnover-day) is NOT these
        # costs. Reconstructing rotation_realized_usd over the 74 sessions
        # with attribution columns accounts for -$20,815 of the -$20,293 raw
        # plug, leaving +$522. Costs are a real and separate gap; they were
        # never the residual, and the cost lines are therefore reported
        # alongside the reconciliation rather than folded into it.
        costs = session_costs(trades_today_rows)
        returns_split = gross_net_returns(
            nav_change_usd=total_nav_change,
            prior_nav=prior_nav,
            commission_usd=costs["commission_usd"],
            slippage_usd=costs["slippage_usd"],
        )
        if costs.get("n_fills_unclassified_action"):
            data_warnings.append(
                f"Slippage incomplete for {run_date}: "
                f"{costs['n_fills_unclassified_action']} fill(s) carried a trade "
                "action the cost vocabulary does not classify, so their "
                "implementation shortfall is missing while their notional still "
                "counts — slippage_bps is understated, not absent."
            )
        if costs["n_fills"] and not costs["commission_available"]:
            data_warnings.append(
                f"Commission unknown for {run_date}: {costs['n_fills']} fill(s) "
                "executed and IB attached no commissionReport to any of them. "
                "commission_usd is persisted NULL and daily_return_gross_pct is "
                "SUPPRESSED — an ABSENT figure, not a measured $0.00. The "
                "net-of-cost return is unaffected: it comes from NAV, which the "
                "commission already debited whether or not it was reported."
            )

        nav_reconciliation = {
            "nav_change_usd": total_nav_change,
            "position_pnl_usd": total_day_usd,
            "interest_usd": interest_usd,
            "dividend_usd": dividend_usd,
            "dividend_timing_usd": dividend_timing_usd,
            "dividend_receivable_usd": dividend_receivable,
            "unattributed_usd": unattributed_usd,
            "pricing_timing_usd": pricing_timing_usd,
            "pricing_timing_available": pricing_timing_available,
            # The IB-vs-settled divergence itself, day-over-day differenced.
            # Equal to pricing_timing_usd on the ib_netliq basis; the sleeve
            # goes to zero on settled_close while this does not
            # (alpha-engine-config-I9638).
            "mark_basis_delta_usd": mark_basis_delta_usd,
            "rotation_realized_usd": rotation_realized_usd,
            "unattributed_true_usd": unattributed_true_usd,
            "commission_usd": costs["commission_usd"],
            "slippage_usd": costs["slippage_usd"],
            "traded_notional_usd": costs["traded_notional_usd"],
            "commission_available": costs["commission_available"],
            "slippage_bps": costs["slippage_bps"],
            "daily_return_net_pct": returns_split["daily_return_net_pct"],
            "daily_return_gross_pct": returns_split["daily_return_gross_pct"],
            "gross_available": returns_split["gross_available"],
            "gross_unavailable_reason": returns_split["gross_unavailable_reason"],
            "total_cost_usd": returns_split["total_cost_usd"],
        }
        logger.info(
            "Costs: commission=%s (available=%s) | slippage=$%+.0f (%s) | "
            "notional=$%.0f | net=%s | gross=%s",
            # commission_usd is None — never 0.0 — when fills executed and IB
            # attached no commissionReport (I8188 deliverable 2). Formatting it
            # with %.2f would raise here, so it is rendered as the word ABSENT.
            f"${costs['commission_usd']:.2f}"
            if costs["commission_usd"] is not None else "ABSENT",
            costs["commission_available"],
            costs["slippage_usd"],
            f"{costs['slippage_bps']:.1f}bp" if costs["slippage_bps"] is not None else "n/a",
            costs["traded_notional_usd"],
            f"{returns_split['daily_return_net_pct']:.4f}%" if returns_split["daily_return_net_pct"] is not None else "n/a",
            f"{returns_split['daily_return_gross_pct']:.4f}%" if returns_split["daily_return_gross_pct"] is not None else "n/a",
        )
        logger.info(
            "NAV recon: Δ=$%.0f | positions=$%.0f | interest=$%.0f | "
            "dividends=$%.0f | rotation=$%.0f | pricing&timing=$%.0f | "
            "unattributed(raw)=$%.0f | unattributed(true)=$%.0f",
            total_nav_change, total_day_usd, interest_usd, dividend_usd,
            rotation_realized_usd, pricing_timing_usd,
            unattributed_usd, unattributed_true_usd,
        )
        # Honesty warnings — surface each material term in data_warnings (EOD
        # email + console), never silently buried.
        # Fires on `mark_basis_delta_usd`, not on the attribution sleeve
        # (alpha-engine-config-I9638): identical on the default ib_netliq
        # basis, where the two are the same number, but on settled_close the
        # sleeve is zero by construction and this — the fleet's most sensitive
        # IB-vs-settled detector at 5bp — would go permanently silent on the
        # exact divergence it was built to see.
        if pricing_timing_available and nav and abs(mark_basis_delta_usd) > max(
            500.0, 0.0005 * nav
        ):
            data_warnings.append(
                f"Pricing & timing reconciliation: ${mark_basis_delta_usd:+,.0f} "
                f"({mark_basis_delta_usd / nav * 100:+.3f}% of NAV) — IB marks vs "
                "settled closes; "
                + (
                    "outside the headline NAV entirely on the settled_close "
                    "basis, reported as a broker cross-check."
                    if nav_basis == NAV_BASIS_SETTLED_CLOSE
                    else "isolated from Unattributed, not hidden."
                )
            )
        if not pricing_timing_available and prior_nav is not None:
            data_warnings.append(
                "Pricing & timing reconstruction unavailable (missing prior "
                "snapshot cash/NAV) — the IB-mark-vs-settled-close basis gap "
                "stays in Unattributed for this day."
            )

        # ── NAV three-way reconcile — HARD GATE (config#2457) ────────────────
        # Promotes the pricing/timing signal above from observational
        # (data_warnings only) to a paged alarm — the three-way check the
        # parent epic (config#1277) calls "the single most important
        # portfolio control". Fires IN ADDITION TO the soft data_warnings
        # entry above, never instead of it — the email-visible warning stays
        # for the routine/sub-hard-gate band. Decision logic lives in
        # `_check_nav_three_way_hard_gate` (pure function, unit-tested
        # directly); this call site owns dispatch (log + page flow-doctor).
        #
        # Re-based onto `mark_basis_delta_usd` (alpha-engine-config-I9638).
        # Identical to `pricing_timing_usd` on the default ib_netliq basis, so
        # the gate's behaviour is UNCHANGED today; on settled_close the sleeve
        # is zero by construction and passing it here would silence the gate
        # entirely — I6819 item 3 requires it keep firing on IB divergence
        # after the cut-over. The tolerance is untouched.
        #
        # Attribution is computed UNCONDITIONALLY here — not only inside a
        # confirmed raw-term breach — because the gate itself now depends on
        # its output (alpha-engine-config-I9087): the residual the gate reads
        # can only be known once the mark-basis divergence is decomposed.
        # Attribution basis is the FULL BOOK, not the out-of-range subset
        # (alpha-engine-config-I8733) — same per-ticker decomposition
        # `pricing_timing_unattributable_usd` is built from, so the
        # classifier and the artifact can never disagree about how much of
        # the breach the marks explain.
        #
        # Deliberate, narrow deviation from "fail loud, no silent swallows":
        # (a) failure mode swallowed — `compute_pricing_timing_by_ticker` /
        # `_attribute_mark_basis_divergence` raising on this one day's book;
        # (b) the primary EOD deliverable (NAV log + email) survives because
        # this whole reconciliation section already runs inside its own
        # exception boundary at the `run()` call site; (c) recording surface
        # — logged at ERROR here AND, because `attribution_ok` becomes False,
        # the gate below fails CLOSED and pages flow-doctor at error rather
        # than silently passing. Attribution is safety-critical the moment
        # the gate depends on it, so a raise must escalate, never vanish.
        from executor.eod_report import compute_pricing_timing_by_ticker
        _pt_by_ticker: dict = {}
        _pt_uncovered = 0
        _mb = {"contributors": [], "explained_usd": None, "covered_usd": 0.0, "uncovered_names": 0}
        _attribution_exception: Exception | None = None
        try:
            _pt_by_ticker, _pt_uncovered = compute_pricing_timing_by_ticker(
                positions, prior_positions if prior_snapshot_loaded else None,
            )
            # Per-name attribution of the DIFFERENCED mark-basis term
            # (alpha-engine-config-I9085). Supplies both the explanation
            # term the classifier tests against and the ticker detail the
            # alert carries unconditionally.
            _mb = _attribute_mark_basis_divergence(
                positions=positions,
                prior_positions=prior_positions if prior_snapshot_loaded else None,
            )
        except Exception as exc:
            _attribution_exception = exc
            logger.exception(
                "Mark-basis divergence attribution raised for %s — the NAV "
                "hard gate falls back to the raw pricing/timing term and "
                "fails CLOSED (alpha-engine-config-I9087).",
                run_date,
            )
        # Attribution is trustworthy only when it ran clean AND covered the
        # whole book — a partial-coverage or no-attribution-at-all result
        # (I9087's required fail-closed condition) must not be read as "the
        # residual is small", or a defect in this codepath would silently
        # suppress a real breach.
        attribution_ok = (
            _attribution_exception is None
            and prior_snapshot_loaded
            and _mb.get("explained_usd") is not None
            and _mb.get("uncovered_names", 0) == 0
            and _pt_uncovered == 0
        )
        _breach_classification = _classify_nav_breach(
            mark_basis_delta_usd,
            ib_mark_range_flags,
            full_book_mark_basis_usd=(
                sum(_pt_by_ticker.values()) if prior_snapshot_loaded else None
            ),
            full_book_uncovered_names=_pt_uncovered,
            mark_divergence_explained_usd=(
                _mb["explained_usd"] if prior_snapshot_loaded else None
            ),
        )
        nav_hard_gate_breach = _check_nav_three_way_hard_gate(
            pricing_timing_usd=mark_basis_delta_usd,
            pricing_timing_available=pricing_timing_available,
            nav=nav,
            run_date=run_date,
            residual_usd=_breach_classification["residual_usd"],
            attribution_ok=attribution_ok,
        )
        if nav_hard_gate_breach:
            # Fail-closed classification override (alpha-engine-config-I9087):
            # when attribution could not be trusted, the breach is NEVER
            # allowed to read as `broker_data_quality` — `_classify_nav_breach`
            # above was fed a degraded/absent explanation term and must not be
            # taken at face value. Otherwise the classifier's own two-test
            # verdict stands, which already folds in the I9085 NAV-identity
            # cross-check (`nav_identity_holds`).
            if not attribution_ok:
                _breach_classification = dict(
                    _breach_classification,
                    classification="reconcile_defect",
                )
            nav_hard_gate_breach.update(_breach_classification)
            nav_hard_gate_breach["mark_basis_contributors"] = _mb["contributors"]
            nav_hard_gate_breach["attribution_ok"] = attribution_ok
            # Classification is always named — a book-wide mark skew can now
            # classify broker_data_quality with ZERO tickers out of range, and
            # the responder still has to be told which way to go.
            nav_hard_gate_breach["message"] += (
                f" Classification: {_breach_classification['classification']} "
                f"(basis {_breach_classification['attribution_basis']}, "
                f"residual ${_breach_classification['residual_usd']:+,.0f}"
                + (
                    ""
                    if _breach_classification["nav_identity_residual_usd"] is None
                    else f", NAV-identity residual "
                         f"${_breach_classification['nav_identity_residual_usd']:+,.0f}"
                )
                + (
                    ""
                    if attribution_ok
                    else " — ATTRIBUTION UNAVAILABLE, gate failed CLOSED "
                         "(alpha-engine-config-I9087)"
                )
                + ")."
            )
            # ALWAYS name the culprits, not only when a mark crossed a traded
            # range boundary (alpha-engine-config-I9085). On the 2026-08-27
            # breach zero tickers were out of range while MU alone carried
            # 75% of the −$4,393 — the operator got a portfolio total and no
            # name, which is what config#6349 deliverable 2 existed to end.
            if _mb["contributors"]:
                nav_hard_gate_breach["message"] += (
                    " Top mark-basis contributors — "
                    f"{_format_mark_basis_contributors(_mb['contributors'])}."
                )
            if ib_mark_range_flags:
                nav_hard_gate_breach["message"] += (
                    " IB mark outside traded range — "
                    f"{_format_mark_range_detail(ib_mark_range_flags)}."
                )
            _log_paged(
                "NAV three-way reconcile BREACH [%s]: pricing&timing=$%+.0f "
                "(%.3f%% of NAV) exceeds hard-gate tolerance $%.0f "
                "(%.1fbps of NAV) — broker-reported NAV vs settled/system "
                "NAV diverged beyond tolerance for run_date=%s.",
                nav_hard_gate_breach["classification"],
                nav_hard_gate_breach["pricing_timing_usd"],
                nav_hard_gate_breach["pricing_timing_pct_of_nav"],
                nav_hard_gate_breach["tolerance_usd"],
                nav_hard_gate_breach["tolerance_bps"],
                run_date,
            )
            if fd:
                fd.report(
                    RuntimeError(nav_hard_gate_breach["message"]),
                    # A breach fully explained by an out-of-range broker mark
                    # still needs eyes (the feed is serving bad data) but is
                    # not itself evidence of a reconcile code defect — page
                    # at warning, not error, so the two classes triage
                    # differently at a glance.
                    severity=(
                        "warning"
                        if nav_hard_gate_breach["classification"] == "broker_data_quality"
                        else "error"
                    ),
                    context={
                        "site": "nav_three_way_reconcile_hard_gate",
                        "run_date": run_date,
                        "pricing_timing_usd": nav_hard_gate_breach["pricing_timing_usd"],
                        "hard_gate_tolerance_usd": nav_hard_gate_breach["tolerance_usd"],
                        "nav": nav_hard_gate_breach["nav"],
                        "classification": nav_hard_gate_breach["classification"],
                        "residual_usd": nav_hard_gate_breach["residual_usd"],
                        "attribution_ok": nav_hard_gate_breach["attribution_ok"],
                        "ib_mark_outside_range_tickers": [
                            f["ticker"] for f in ib_mark_range_flags
                        ],
                        "nav_identity_residual_usd": nav_hard_gate_breach[
                            "nav_identity_residual_usd"
                        ],
                        "mark_basis_top_contributors": [
                            {
                                "ticker": c["ticker"],
                                "contrib_usd": round(c["contrib_usd"], 2),
                                "reversion": c["reversion"],
                            }
                            for c in _mb["contributors"]
                        ],
                    },
                )
        # The TRUE residual (after rotation + pricing&timing are lifted out) is
        # what should now be small; warn only when IT is material.
        if nav and abs(unattributed_true_usd) > max(100.0, 0.0005 * nav):
            msg = (
                f"NAV reconciliation gap: ${unattributed_true_usd:+,.0f} "
                f"unattributed ({unattributed_true_usd / nav * 100:+.3f}% of NAV, "
                "after rotation + pricing&timing). Likely causes: untracked "
                "corporate action, fees, or FX."
            )
            logger.warning(msg)
            data_warnings.append(msg)

    # ── Daemon-vs-IB reconciliation-integrity audit (config#859) ──
    # Secondary observability hung off the primary EOD path: a failure here
    # must NOT abort EOD reconcile (the NAV log + email are the primary
    # deliverable), and the failure IS recorded — (a) swallowed: audit
    # build/write error; (c) recording surface: the WARN below + the
    # report-card reconciliation_integrity component shows N/A when the
    # artifact is absent. Per the feedback_no_silent_fails secondary-
    # observability carve-out.
    try:
        from executor.reconciliation_audit import (
            build_reconciliation_audit,
            fetch_same_day_split_ratios,
            write_reconciliation_audit,
        )

        # Same-day corporate actions (splits/spinoffs) change IB's share count
        # with no ledger trade and would false-mismatch the anchored parity on
        # the ex-date (config#1682). ONE date-filtered Polygon query resolves
        # the whole day's split set and the held book is intersected against it
        # locally (alpha-engine-config-I9646) — the per-ticker loop could not
        # fit inside the 5-calls/min free tier and rate-limited on every run.
        # Best-effort: no POLYGON_API_KEY or a query failure never aborts the
        # audit — but it no longer reads as clean either. Because one query
        # covers the whole book, unresolved is all-or-nothing.
        _held_tickers = set(positions or {}) | set(prior_positions or {})
        _split_ratios, _split_unresolved = fetch_same_day_split_ratios(
            _held_tickers, run_date,
        )
        if _split_unresolved:
            _msg = (
                f"Same-day split status UNRESOLVED for all "
                f"{len(_split_unresolved)} held ticker(s) on {run_date}: "
                f"{', '.join(sorted(_split_unresolved))}. The single "
                "date-filtered split query failed, so no held name's parity in "
                "the reconciliation audit is verified — the audit carries "
                "split_check_complete=false (alpha-engine-config-I9630)."
            )
            logger.error("[reconciliation_audit] %s", _msg)
            data_warnings.append(_msg)

        _recon_audit = build_reconciliation_audit(
            conn,
            today_positions=positions,
            # Anchor the headline parity on the prior broker snapshot + today's
            # fills (config#1301). Pass None when no prior snapshot genuinely
            # loaded so the metric falls back to cumulative replay rather than
            # anchoring on a phantom empty baseline (which would false-RED).
            prior_positions=prior_positions if prior_snapshot_loaded else None,
            run_date=run_date,
            ib_nav=nav,
            corporate_actions=_split_ratios,
            split_check_unresolved=_split_unresolved,
        )
        _recon_key = write_reconciliation_audit(
            _recon_audit,
            bucket=trades_bucket,
            run_date=run_date,
            region=config.get("aws_region", "us-east-1"),
        )
        logger.info(
            "[reconciliation_audit] match_rate=%.3f status=%s positions=%d "
            "mismatched=%d split_check_complete=%s -> s3://%s/%s",
            _recon_audit["reconciliation_match_rate"], _recon_audit["status"],
            _recon_audit["n_positions"], _recon_audit["n_mismatched"],
            _recon_audit["split_check_complete"], trades_bucket, _recon_key,
        )
    except Exception as _recon_err:  # noqa: BLE001 — secondary observability (see comment above)
        logger.warning(
            "[reconciliation_audit] FAILED to build/write reconciliation "
            "audit for run_date=%s: %s (report card reconciliation_integrity "
            "shows N/A this cycle)", run_date, _recon_err,
        )

    # Persist EOD snapshot AFTER positions are enriched with closing prices,
    # accrued dividends, and per-position returns. Yesterday's reconcile now
    # reads this snapshot via closing_price (same source as today's
    # daily_closes), closing the source-mismatch gap that was causing NAV
    # residuals to land in cash.
    #
    # Phase 2 transparency-inventory: persist the NAV-reconciliation
    # waterfall (nav_change / position_pnl / interest / dividend /
    # unattributed) and the headline residual % as named fields. Closes
    # the *P&L attribution* row in the gate checklist — until now these
    # values existed in logs + the email body but weren't queryable from
    # eod_pnl.csv. nav_reconciliation can be {} when prior_nav is None
    # (first-ever EOD run); .get() defaults to None for those columns.
    # The NAV-basis shadow figures do NOT depend on a prior day, so they are
    # merged in unconditionally — including on the first-ever EOD run, where
    # `nav_reconciliation` is still {} (alpha-engine-config-I9638).
    nav_reconciliation.update({
        "nav_basis": nav_basis,
        "nav_ib_usd": nav_ib_usd,
        "nav_settled_usd": nav_settled_usd,
        "nav_basis_diff_usd": nav_basis_diff_usd,
        "nav_basis_diff_bps": nav_basis_diff_bps,
        "nav_basis_unavailable_reason": nav_basis_unavailable_reason,
        "nav_settled_fallback_tickers": sorted(settled_fallback_tickers),
    })

    unattributed_for_log = nav_reconciliation.get("unattributed_usd")
    unattributed_pct_for_log = _compute_unattributed_residual_pct(
        unattributed_for_log, nav,
    )

    # ── Integrity gates (alpha-engine-config-I8188) ─────────────────────────
    # Both gates are evaluated BEFORE the row is written and their outcome is
    # persisted with it, so a red pipeline never costs us the evidence that
    # made it red. The RAISE is deferred to the end of run() — after the row,
    # the S3 exports, the report artifact and the email — because losing the
    # day's artifacts is strictly worse than a late failure, and the operator
    # needs the artifacts to diagnose the breach.
    integrity_breaches: list[dict] = []
    # OBSERVE-MODE records (alpha-engine-config-I9614 D2). Same shape as
    # `integrity_breaches` and persisted into the same `integrity_breach_json`,
    # but deliberately NOT read by the deferred raise at the end of this
    # function. A gate whose subject already breaches by construction — the
    # published benchmark history, pending the restatement ruling in
    # `alpha-engine-config-I9613` — would fail every postclose for a condition
    # already measured and queued for a human, which is how a fleet acquires a
    # chronic false positive. Visible and recorded, but not yet load-bearing.
    observe_records: list[dict] = []

    # Bound the residual. Prefer the TRUE residual (rotation + pricing&timing
    # lifted out); fall back to the raw plug only when the sleeves could not
    # be computed, and say so — bounding the plug means bounding a number that
    # legitimately moves with turnover.
    residual_bounded = nav_reconciliation.get("unattributed_true_usd")
    residual_basis = "unattributed_true_usd"
    basis_is_true_residual = True
    if residual_bounded is None:
        residual_bounded = unattributed_for_log
        residual_basis = "unattributed_usd (raw plug — sleeves unavailable)"
        basis_is_true_residual = False
        data_warnings.append(
            f"Cumulative P&L residual gate NOT EVALUATED for {run_date}: the "
            "attribution sleeves could not be computed, so today's residual is "
            "the raw plug and summing it into a true-residual window would "
            "compare two different quantities. The per-session bound still "
            "applied."
        )
    # Heal the window before reading it. The three sleeve columns were added
    # as bare ALTER TABLEs with no backfill, so every row written before that
    # migration carries NULL — reconstruct them from what IS persisted rather
    # than letting the window fall back to a different quantity.
    backfill_result = backfill_residual_sleeves(conn)
    # ── Cost columns heal themselves too (alpha-engine-config-I9614 D1) ─────
    # Same argument, one column-family over: commission_usd / slippage_usd /
    # traded_notional_usd / daily_return_gross_pct were reachable only from
    # `python -m executor.pnl_measurement_backfill --apply --costs`, and that
    # CLI was never run — 6 of 120 sessions populated on 2026-08-31, 12 of 126
    # on 2026-09-08, i.e. only the forward path ever moved. A backfill behind
    # an operator step is a page, not a fix. Purely local (the trades ledger in
    # this same sqlite file), no vendor call, and planning skips populated rows,
    # so a converged history costs one query.
    try:
        cost_heal = heal_cost_columns(conn)
    except Exception as exc:  # noqa: BLE001
        # (a) swallowed: a failure inside a HISTORICAL self-heal; (b) the
        # primary deliverable — today's reconciliation, its gates, artifacts
        # and email — is computed from today's inputs and survives intact;
        # (c) recorded: a data_warning on this session's row plus the traceback
        # in the run log. Raising here would abort a good close over a bad
        # historical row, which is the failure mode `backfill_residual_sleeves`
        # was given the same treatment for.
        cost_heal = None
        logger.warning("Cost-column self-heal failed", exc_info=True)
        data_warnings.append(
            f"Cost-column self-heal FAILED on {run_date} ({exc}) — the "
            "historical commission/slippage columns did not heal this run and "
            "are still absent on the sessions they were absent on. Today's own "
            "cost line is unaffected."
        )
    if cost_heal and cost_heal["filled"]:
        logger.info(
            "Cost columns healed on %d historical session(s): "
            "Σslippage $%+.2f, commission ABSENT (NULL, not 0.0) on %d of them",
            cost_heal["filled"], cost_heal["slippage_usd_total"],
            cost_heal["commission_absent"],
        )
    for skipped_date, why in backfill_result["skipped"]:
        logger.warning(
            "Residual sleeves unreconstructible for %s (%s) — that session is "
            "EXCLUDED from the cumulative residual window, not substituted.",
            skipped_date, why,
        )

    # ONE basis, or none. The trailing window must hold the same quantity the
    # bound was derived for (unattributed_true_usd, measured at +$522 over 74
    # sessions). Coalescing onto the raw plug — which carries realized rotation
    # P&L and sums to -$20,293 over that same window BY CONSTRUCTION — mixes
    # two different numbers into one sum and breaches on the first run and
    # every run after it. That is what failed eod-2026-08-24-1787601606:
    # 62 raw-plug rows plus one true residual, judged against a true-residual
    # bound. A short window is the honest outcome when history is missing; a
    # window silently filled with a different quantity is not.
    trailing_residuals = [
        float(r[0])
        for r in conn.execute(
            "SELECT unattributed_true_usd FROM eod_pnl "
            "WHERE date < ? AND unattributed_true_usd IS NOT NULL "
            "ORDER BY date DESC LIMIT ?",
            (run_date, RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1),
        ).fetchall()
    ][::-1]
    residual_window_sessions = len(trailing_residuals) + 1
    if residual_window_sessions < RESIDUAL_CUMULATIVE_WINDOW_SESSIONS:
        # Measurability: a gate running on a partial window is not the same
        # fact as a gate running on a full one, and "no data" is never green.
        # The bound is an absolute dollar bound, so a partial window still
        # fires legitimately — it is only less able to miss.
        logger.warning(
            "Cumulative residual window is %d/%d session(s) deep on the "
            "true-residual basis — the gate is live but shallower than its "
            "derivation assumes.",
            residual_window_sessions, RESIDUAL_CUMULATIVE_WINDOW_SESSIONS,
        )
    residual_breaches = check_residual_bounds(
        unattributed_true_usd=residual_bounded,
        nav=nav,
        trailing_residuals_usd=trailing_residuals,
        run_date=run_date,
        basis_is_true_residual=basis_is_true_residual,
    )
    for breach in residual_breaches:
        breach["basis"] = residual_basis
        breach["window_sessions"] = residual_window_sessions
        breach["window_sessions_expected"] = RESIDUAL_CUMULATIVE_WINDOW_SESSIONS
        _log_paged("P&L INTEGRITY BREACH: %s", breach["message"])
        data_warnings.append(breach["message"])
    integrity_breaches.extend(residual_breaches)

    # ── Attribution closure — the NAV mark-basis LEVEL (I8188 class sweep) ──
    # Replaces the tautological `ties_to_headline` claim that eod_report
    # published and this module logged on. `unattributed_usd` is the plug of
    # the identity that check closed, so it could never fail. The check with
    # power compares broker NetLiquidation against a settled NAV rebuilt from
    # an independent source (broker cash + ArcticDB settled closes) — and it
    # sees the CONSTANT basis error that cancels exactly in the day-over-day
    # `_check_nav_three_way_hard_gate` above and is invisible to it.
    #
    # The left-hand side is `nav_ib_usd`, not the headline `nav`
    # (alpha-engine-config-I9638). They are the same number on the default
    # ib_netliq basis, so this gate is unchanged today; on settled_close the
    # headline NAV *is* the settled rebuild, and comparing it to itself would
    # return exactly $0 every session — the tautology this check was built to
    # replace. The tolerance is untouched and still scales on the headline NAV.
    basis_level = nav_basis_level_usd(
        nav=nav_ib_usd,
        total_cash=account.get("total_cash"),
        accrued_interest=account.get("accrued_interest"),
        positions=positions,
    )
    closure_breaches = check_attribution_closure(
        nav_basis=basis_level,
        nav=nav,
        components=None,  # component arithmetic is checked at the report site
        run_date=run_date,
    )
    for breach in closure_breaches:
        if breach.get("severity") == "unevaluated":
            logger.warning("P&L ATTRIBUTION CLOSURE NOT EVALUATED: %s", breach["message"])
        else:
            _log_paged("P&L ATTRIBUTION CLOSURE BREACH: %s", breach["message"])
        data_warnings.append(breach["message"])
    integrity_breaches.extend(closure_breaches)

    # Custodian marks: promoted from flag to failure. The soft
    # ib_mark_outside_range flag is unchanged and still populates.
    mark_breaches = check_custodian_marks(
        ib_mark_range_flags,
        nav=nav,
        run_date=run_date,
        corrected_tickers=mark_correction["corrected_tickers"],
    )
    for breach in mark_breaches:
        _log_paged("CUSTODIAN MARK BREACH: %s", breach["message"])
        data_warnings.append(breach["message"])
    integrity_breaches.extend(mark_breaches)

    log_eod(conn, {
        "date": run_date,
        "portfolio_nav": nav,
        "daily_return_pct": daily_return,
        "spy_return_pct": spy_return,
        "daily_alpha_pct": alpha,
        "positions_snapshot": positions,
        "spy_close": spy_price,
        "total_cash": account.get("total_cash"),
        "accrued_interest": account.get("accrued_interest"),
        "unrealized_pnl": account.get("unrealized_pnl"),
        "realized_pnl": account.get("realized_pnl"),
        "nav_change_usd": nav_reconciliation.get("nav_change_usd"),
        "position_pnl_usd": nav_reconciliation.get("position_pnl_usd"),
        "interest_usd": nav_reconciliation.get("interest_usd"),
        "dividend_usd": nav_reconciliation.get("dividend_usd"),
        "unattributed_usd": unattributed_for_log,
        "unattributed_residual_pct": unattributed_pct_for_log,
        "pricing_timing_usd": nav_reconciliation.get("pricing_timing_usd"),
        "rotation_realized_usd": nav_reconciliation.get("rotation_realized_usd"),
        "unattributed_true_usd": nav_reconciliation.get("unattributed_true_usd"),
        "commission_usd": nav_reconciliation.get("commission_usd"),
        "slippage_usd": nav_reconciliation.get("slippage_usd"),
        "traded_notional_usd": nav_reconciliation.get("traded_notional_usd"),
        "commission_available": nav_reconciliation.get("commission_available"),
        "daily_return_net_pct": nav_reconciliation.get("daily_return_net_pct"),
        "daily_return_gross_pct": nav_reconciliation.get("daily_return_gross_pct"),
        "dividend_accrual_available": dividend_accrual_available,
        "spy_dividend_per_share": spy_dividend_per_share,
        "dividend_timing_usd": dividend_timing_usd,
        "dividend_receivable_usd": dividend_receivable,
        "integrity_breach_json": (
            json.dumps(integrity_breaches) if integrity_breaches else None
        ),
        # The broker NAV as received, and the repair applied to it. Persisted
        # unconditionally on a correction day so the published NAV can always be
        # traced back to what IB actually sent (alpha-engine-config-I9627).
        "nav_ib_raw_usd": nav_ib_raw,
        # Which basis `portfolio_nav` above is struck on, plus both candidate
        # figures and their gap — the two-week shadow series the ruled
        # cut-over is graded from (alpha-engine-config-I9638).
        "nav_basis": nav_basis,
        "nav_ib_usd": nav_ib_usd,
        "nav_settled_usd": nav_settled_usd,
        "nav_basis_diff_usd": nav_basis_diff_usd,
        "nav_basis_diff_bps": nav_basis_diff_bps,
        "nav_mark_correction_usd": (
            mark_correction["correction_usd"] if mark_correction["applied"] else None
        ),
        "nav_mark_correction_json": (
            json.dumps(mark_correction)
            if (mark_correction["applied"] or mark_correction["refused"]
                or mark_correction["unrepairable"])
            else None
        ),
    })

    # ── TWR closure + self-heal (alpha-engine-config-I8188, defect 4) ────────
    # Runs AFTER today's row is persisted so the check covers the session being
    # written. Chain-linked daily_return_pct and the NAV ratio are identically
    # equal by construction absent external flows — every stored return is
    # (nav - prior_nav)/prior_nav against the immediately preceding persisted
    # row — so any drift is a stored value that no longer agrees with the NAV
    # series it came from. Measured live: 17.4bp of drift, 100% of it from ONE
    # row (2026-04-07, whose 2026-04-06 baseline was corrected after it was
    # written and never recomputed). Recomputing that row moves the chain-link
    # from +3.8261% to +3.6526%, matching the NAV ratio to 3e-5pp.
    #
    # The repair is applied here rather than filed as an operator step, per the
    # automation principle: a detector that stays red until a human runs a
    # backfill is a page, not a fix. It is bounded — a correction larger than
    # TWR_SELF_HEAL_MAX_CORRECTION_PCT is REFUSED and raises instead, because
    # at that size the disagreement is a different NAV series (an external
    # flow, a restated snapshot), not a stale derived value.
    twr_rows = [
        {"date": r[0], "portfolio_nav": r[1], "daily_return_pct": r[2]}
        for r in conn.execute(
            "SELECT date, portfolio_nav, daily_return_pct FROM eod_pnl "
            "WHERE portfolio_nav IS NOT NULL ORDER BY date"
        ).fetchall()
    ]
    heal_plan = plan_twr_self_heal(twr_rows)
    for correction in heal_plan["corrections"]:
        conn.execute(
            "UPDATE eod_pnl SET daily_return_pct = ?, "
            "daily_alpha_pct = CASE WHEN spy_return_pct IS NULL THEN NULL "
            "ELSE ? - spy_return_pct END WHERE date = ?",
            (correction["to_pct"], correction["to_pct"], correction["date"]),
        )
        logger.warning(
            "TWR self-heal: %s daily_return_pct %+.6f%% -> %+.6f%% "
            "(%+.6fpp) — the stored value disagreed with the persisted NAV "
            "series, which is ground truth. daily_alpha_pct recomputed on the "
            "same row.",
            correction["date"], correction["from_pct"],
            correction["to_pct"], correction["delta_pct"],
        )
        data_warnings.append(
            f"TWR self-heal restated {correction['date']} daily_return_pct "
            f"{correction['from_pct']:+.4f}% -> {correction['to_pct']:+.4f}% "
            "(stored value disagreed with the persisted NAV series)."
        )
    if heal_plan["corrections"]:
        conn.commit()
        twr_rows = [
            {"date": r[0], "portfolio_nav": r[1], "daily_return_pct": r[2]}
            for r in conn.execute(
                "SELECT date, portfolio_nav, daily_return_pct FROM eod_pnl "
                "WHERE portfolio_nav IS NOT NULL ORDER BY date"
            ).fetchall()
        ]
    for refusal in heal_plan["refused"]:
        msg = (
            f"TWR self-heal REFUSED for {refusal['date']}: stored "
            f"{refusal['from_pct']:+.4f}% vs NAV-implied {refusal['to_pct']:+.4f}% "
            f"— {refusal['reason']}"
        )
        _log_paged(msg)
        data_warnings.append(msg)
        integrity_breaches.append({"kind": "twr_self_heal_refused", **refusal,
                                   "message": msg})

    twr = verify_twr_closes(twr_rows)
    if twr.get("status") == "ok" and not twr["closes"]:
        _log_paged("TWR CLOSURE BREACH: %s", twr["message"])
        data_warnings.append(twr["message"])
        integrity_breaches.append({"kind": "twr_closure", **{
            k: twr[k] for k in
            ("chain_linked_pct", "nav_ratio_pct", "drift_bps", "tolerance_bps",
             "n_sessions", "message")
        }})
    elif twr.get("status") == "ok":
        logger.info(
            "TWR closes: chain-linked %+.4f%% vs NAV ratio %+.4f%% over %d "
            "sessions (%+.2fbp drift, tolerance %.0fbp)",
            twr["chain_linked_pct"], twr["nav_ratio_pct"], twr["n_sessions"],
            twr["drift_bps"], twr["tolerance_bps"],
        )

    # ── TWR closure, nav_change_usd basis — third arm (alpha-engine-config-
    # I9025) ───────────────────────────────────────────────────────────────
    # A different pair than the NAV-ratio check above: chain-linked
    # daily_return_pct against chain-linked nav_change_usd/prior_nav — the
    # pair return_chain_basis_gap publishes. MEASURED CAUSE: a day-set
    # mismatch (nav_change_usd is NULL on sessions before PR490), not a stale
    # row — see pnl_integrity's module docstring. No self-heal: the fix is
    # coverage (excluding a row missing nav_change_usd from BOTH chains),
    # already built into verify_nav_change_basis_closes.
    nc_rows = [
        {"date": r[0], "portfolio_nav": r[1], "daily_return_pct": r[2],
         "nav_change_usd": r[3]}
        for r in conn.execute(
            "SELECT date, portfolio_nav, daily_return_pct, nav_change_usd "
            "FROM eod_pnl WHERE portfolio_nav IS NOT NULL ORDER BY date"
        ).fetchall()
    ]
    nc_basis = verify_nav_change_basis_closes(nc_rows)
    if nc_basis.get("status") == "ok" and not nc_basis["closes"]:
        _log_paged("TWR CLOSURE BREACH (nav_change_usd basis): %s", nc_basis["message"])
        data_warnings.append(nc_basis["message"])
        integrity_breaches.append({"kind": "twr_closure_nav_change_basis", **{
            k: nc_basis[k] for k in
            ("stored_return_chain_pct", "nav_change_basis_chain_pct",
             "drift_bps", "tolerance_bps", "n_sessions", "message")
        }})
    elif nc_basis.get("status") == "ok":
        logger.info(
            "TWR closes (nav_change_usd basis): chain-linked %+.4f%% vs "
            "nav_change_usd basis %+.4f%% over %d sessions (%+.2fbp drift, "
            "tolerance %.0fbp; %d session(s) excluded for missing "
            "nav_change_usd)",
            nc_basis["stored_return_chain_pct"], nc_basis["nav_change_basis_chain_pct"],
            nc_basis["n_sessions"], nc_basis["drift_bps"], nc_basis["tolerance_bps"],
            len(nc_basis.get("coverage_gap_sessions") or []),
        )
    # ── The benchmark leg's INDEPENDENT side (alpha-engine-config-I9614 D2) ──
    # Every gate above this point is closed inside the system's own numbers.
    # `spy_close` and `spy_return_pct` are written by the same producer on the
    # same run, so no equation built from the two of them can fail for the
    # reason that matters — they can be wrong TOGETHER, and over the live
    # history they were: the published benchmark chain reads +15.5242% against
    # the vendor's +14.0325% total return (149.2bp), and every internal check
    # agreed with itself throughout.
    #
    # `check_benchmark_vendor_anchor` is the one side of this reconciliation
    # the system did not compute — Polygon's own SPY closes and its declared
    # distributions. It landed in PR522 reachable only from a CLI. Running the
    # only non-tautological instrument in the stack when a human remembers to
    # type it is the same defect as not having it.
    #
    # OBSERVE MODE. Breaches go to `data_warnings` and the persisted
    # `integrity_breach_json`, NOT to the pipeline exit code, because the
    # published history breaches by construction until the restatement in
    # `alpha-engine-config-I9613` is ruled. A hard gate today would fail every
    # postclose for a condition already known, measured and queued for a human.
    # Promotion to hard is that issue's closes-when, not this one's.
    try:
        anchor_rows = [
            {"date": r[0], "spy_close": r[1], "spy_return_pct": r[2]}
            for r in conn.execute(
                "SELECT date, spy_close, spy_return_pct FROM eod_pnl "
                "WHERE date <= ? ORDER BY date DESC LIMIT 2",
                (run_date,),
            ).fetchall()
        ][::-1]
        vendor_closes, vendor_dividends = fetch_live_anchor_window(anchor_rows)
        anchor_breaches = check_benchmark_vendor_anchor(
            anchor_rows,
            vendor_closes=vendor_closes,
            vendor_dividends=vendor_dividends,
        )
    except Exception as exc:  # noqa: BLE001
        # A vendor that cannot be reached is a NAMED degradation, never a pass.
        # `executor.dividends.fetch_ex_dividends` is the precedent and the
        # reason it exists: this issue's original defect 3 was an absent
        # measurement (`IBKR.get_accrued_dividends_by_symbol()` returning `{}`)
        # persisted as a measured $0.00. An anchor that silently returned "no
        # breaches" when Polygon was down would be the identical mistake one
        # layer up. (a) swallowed: the third-party fetch; (b) the primary
        # deliverable — the reconciliation, its artifacts and its other gates —
        # survives; (c) recorded: a `benchmark_anchor_unevaluated` record in
        # `integrity_breach_json` and a `data_warnings` line, both persisted.
        logger.warning("Benchmark vendor anchor NOT EVALUATED", exc_info=True)
        unevaluated = (
            f"Benchmark vendor anchor NOT EVALUATED on {run_date}: {exc}. The "
            "benchmark leg of today's alpha carries no independent check — "
            "this is an ABSENT measurement, not a clean one."
        )
        data_warnings.append(unevaluated)
        observe_records.append({
            "kind": "benchmark_anchor_unevaluated",
            "severity": "unevaluated",
            "run_date": run_date,
            "reason": str(exc),
            "message": unevaluated,
        })
    else:
        for breach in anchor_breaches:
            _log_paged("BENCHMARK ANCHOR (observe): %s", breach["message"])
            data_warnings.append(breach["message"])
            observe_records.append({"severity": "observe", **breach})
        if not anchor_breaches:
            logger.info(
                "Benchmark vendor anchor clean over %s: persisted spy_close and "
                "spy_return_pct agree with the vendor's own closes and declared "
                "distributions.",
                " → ".join(r["date"] for r in anchor_rows),
            )

    if integrity_breaches or observe_records:
        # Observe-mode records are persisted BESIDE the hard breaches, in the
        # same column, each carrying its own `severity`. They do not reach the
        # deferred raise. Writing them somewhere else would put the one
        # independent measurement of the benchmark leg on a surface no
        # consumer of `integrity_breach_json` reads.
        conn.execute(
            "UPDATE eod_pnl SET integrity_breach_json = ? WHERE date = ?",
            (json.dumps(integrity_breaches + observe_records), run_date),
        )
        conn.commit()

    # ── Sector attribution ──────────────────────────────────────────────────
    # Daily contribution = today's per-position P&L as % of NAV (not cumulative
    # unrealized, which has no relationship to the day's return).
    sector_attribution = {}
    if positions and nav > 0:
        for _ticker, pos in positions.items():
            sector = pos.get("sector", "Unknown")
            mv = pos.get("market_value", 0)
            weight = mv / nav
            daily_usd = pos.get("daily_return_usd", 0)
            daily_contrib = (daily_usd / nav * 100) if nav else 0
            if sector not in sector_attribution:
                sector_attribution[sector] = {"weight": 0.0, "contribution": 0.0, "positions": 0}
            sector_attribution[sector]["weight"] += weight
            sector_attribution[sector]["contribution"] += daily_contrib
            sector_attribution[sector]["positions"] += 1
        logger.info(f"Sector attribution: {sector_attribution}")

    # Export full history CSVs for dashboard consumption
    trades_df = pd.read_sql("SELECT * FROM trades ORDER BY date, created_at", conn)
    eod_df = pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn)
    shadow_df = pd.read_sql("SELECT * FROM executor_shadow_book ORDER BY date, created_at", conn)
    s3 = boto3.client("s3")
    for df, key in [
        (trades_df, "trades/trades_full.csv"),
        (eod_df, "trades/eod_pnl.csv"),
        (shadow_df, "trades/shadow_book.csv"),
    ]:
        try:
            buf = df.to_csv(index=False).encode()
            s3.put_object(Bucket=trades_bucket, Key=key, Body=buf)
            logger.info(f"Exported {key} ({len(df)} rows) to s3://{trades_bucket}/{key}")
        except Exception as e:
            logger.warning(f"S3 CSV export failed for {key}: {e}")

    # ── Reference-rate showcase artifact (metron/reference_rate.json) ─────────
    # Publish the illustrative-only Reference Rate contract artifact Metron renders
    # as a demo portfolio. Best-effort: it is secondary observability hung off the
    # already-committed eod_pnl + S3 CSV exports (recording surface = this WARN), so
    # a publish failure must never override the EOD run's primary deliverables.
    try:
        ref_payload = reference_rate.build_payload(
            positions=positions,
            nav=nav,
            nav_history=reference_rate.nav_history_from_eod_df(eod_df),
            run_date=run_date,
        )
        reference_rate.publish(s3, trades_bucket, ref_payload)
    except Exception as e:  # noqa: BLE001 — best-effort secondary path; never fatal
        logger.warning("Reference-rate artifact publish failed (non-fatal): %s", e)

    # The durable audit copy. Strict on BOTH counts (alpha-engine-config-I8735):
    # the snapshot must carry today's eod_pnl row, and an upload failure must
    # not be swallowed — this file is the audit record, and a stale or absent
    # one is indistinguishable from a healthy one to every downstream reader.
    backup_to_s3(
        db_path, run_date, trades_bucket,
        require_eod_row=True, fail_loud=True,
    )

    # Backup daemon and executor logs to S3 (before EC2 shuts down at 1:30 PM)
    for log_file, s3_key in [
        ("/var/log/daemon.log", f"trades/logs/{run_date}/daemon.log"),
        ("/var/log/executor.log", f"trades/logs/{run_date}/executor.log"),
    ]:
        try:
            if os.path.exists(log_file):
                s3.upload_file(log_file, trades_bucket, s3_key)
                logger.info("Log backed up to s3://%s/%s", trades_bucket, s3_key)
        except Exception as e:
            # (a) the daemon/executor log backup to S3 failed for this
            # file — the box's own local copy of the log is unaffected,
            # only the S3 durability copy is missing for this run_date.
            # (c) recorded at WARNING (visible at the INFO root level,
            # unlike the DEBUG this replaces) — the app log stream is the
            # recording surface (alpha-engine-config-I10031).
            logger.warning("Log backup failed for %s: %s", log_file, e)

    # Build position rationale narratives — mechanical synthesis from
    # the context dict. No LLM exposure in executor per
    # [[preference_llm_calls_confined_to_research_module]].
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    position_narratives = {}
    try:
        if positions:
            contexts, ctx_warnings = _build_position_contexts(positions, conn, signals_bucket, run_date)
            position_narratives = _synthesize_rationales(contexts)
            logger.info(
                f"Position narratives generated for {len(position_narratives)} tickers"
            )
            data_warnings.extend(ctx_warnings)
    except Exception as e:
        logger.warning(f"Position rationale generation failed: {e}")

    # ── Roundtrip stats (for trades with entry-exit linkage) ──────────────
    roundtrip_stats = None
    try:
        rt_row = conn.execute("""
            SELECT COUNT(*) as n,
                   AVG(realized_return_pct) as avg_ret,
                   AVG(realized_alpha_pct) as avg_alpha,
                   AVG(days_held) as avg_hold,
                   SUM(CASE WHEN realized_alpha_pct > 0 THEN 1 ELSE 0 END) as n_beat_spy
            FROM trades
            WHERE entry_trade_id IS NOT NULL
              AND realized_return_pct IS NOT NULL
        """).fetchone()
        if rt_row and rt_row[0] > 0:
            roundtrip_stats = {
                "n_roundtrips": rt_row[0],
                "avg_return_pct": round(rt_row[1], 2) if rt_row[1] else None,
                "avg_alpha_pct": round(rt_row[2], 2) if rt_row[2] else None,
                "avg_hold_days": round(rt_row[3], 1) if rt_row[3] else None,
                "n_beat_spy": rt_row[4] or 0,
                "win_rate_vs_spy": round(rt_row[4] / rt_row[0] * 100, 1) if rt_row[4] else 0,
            }
            logger.info("Roundtrip stats: %s", roundtrip_stats)
    except Exception as e:
        logger.warning("Roundtrip stats query failed: %s", e)

    # ── Execution quality monitoring ──────────────────────────────────────
    execution_quality = None
    try:
        eq_rows = conn.execute("""
            SELECT trigger_type, slippage_vs_signal, execution_latency_ms,
                   signal_price, fill_price
            FROM trades
            WHERE date = ? AND fill_price IS NOT NULL AND action = 'ENTER'
        """, (run_date,)).fetchall()
        if eq_rows:
            slippage_by_trigger: dict[str, list[float]] = {}
            all_slippage = []
            all_latency = []
            for row in eq_rows:
                trigger = row[0] or "unknown"
                slip = row[1]
                latency = row[2]
                if slip is not None:
                    slippage_by_trigger.setdefault(trigger, []).append(slip)
                    all_slippage.append(slip)
                if latency is not None:
                    all_latency.append(latency)
            execution_quality = {
                "date": run_date,
                "n_entries": len(eq_rows),
                "avg_slippage_pct": round(sum(all_slippage) / len(all_slippage), 4) if all_slippage else None,
                "avg_latency_ms": round(sum(all_latency) / len(all_latency), 0) if all_latency else None,
                "slippage_by_trigger": {
                    t: {"avg": round(sum(v) / len(v), 4), "n": len(v)}
                    for t, v in slippage_by_trigger.items()
                },
            }
            logger.info("Execution quality: %s", execution_quality)
            # Write to S3
            try:
                s3 = boto3.client("s3")
                s3.put_object(
                    Bucket=trades_bucket,
                    Key=f"trades/execution_quality/{run_date}.json",
                    Body=json.dumps(execution_quality, indent=2).encode(),
                    ContentType="application/json",
                )
            except Exception as _eq_s3:
                logger.warning("Execution quality S3 write failed: %s", _eq_s3)
    except Exception as e:
        logger.warning("Execution quality query failed: %s", e)

    # ── Build + write the structured EOD report artifact ──────────────────
    # consolidated/{date}/eod_report.json is the single source of truth for
    # the console EOD Report page. The alpha attribution here is the
    # prior-NAV-basis decomposition that ties to the headline alpha exactly
    # (executor/eod_report.py) — it replaces the old emailer's sign-flipping
    # "α % of Total" column and the positions-table total that never
    # reconciled with the NAV-based headline.
    try:
        report = build_eod_report(
            run_date=run_date,
            nav=nav,
            prior_nav=prior_nav,
            daily_return=daily_return,
            spy_return=spy_return,
            alpha=alpha,
            positions=positions,
            prior_positions=prior_positions,
            conn=conn,
            account_snapshot=account,
            nav_reconciliation=nav_reconciliation,
            integrity_breaches=integrity_breaches,
            nav_mark_correction=mark_correction,
            mark_check_coverage=mark_coverage,
            twr_closure=twr,
            dividend_accrual_available=dividend_accrual_available,
            position_narratives=position_narratives,
            sector_attribution=sector_attribution,
            roundtrip_stats=roundtrip_stats,
            data_warnings=data_warnings,
            generated_at=snapshot.get("captured_at"),
            spy_close_provisional=spy_close_provisional,
        )
        attribution = report.get("alpha_attribution")
        # The old check here tested `ties_to_headline`, which is TRUE BY
        # CONSTRUCTION (alpha-engine-config-I8188 class sweep) — it asserted
        # "investigate before trusting per-position contributions" while being
        # incapable of ever firing. What CAN fire is a non-finite component,
        # which poisons every downstream sum silently. The gate with real power
        # over these sleeves is the NAV mark-basis level check above.
        if attribution is not None:
            for breach in check_attribution_closure(
                nav_basis={"available": False, "reason": "checked at reconcile site"},
                nav=None,
                components=attribution.get("components"),
                run_date=run_date,
            ):
                logger.error("P&L ATTRIBUTION ARITHMETIC BREACH: %s", breach["message"])
                data_warnings.append(breach["message"])
        write_eod_report(report, trades_bucket=trades_bucket, run_date=run_date)
    except Exception as e:
        _log_paged("EOD report artifact build/write failed: %s", e)
        if fd:
            fd.report(e, severity="error", context={
                "site": "eod_report_artifact", "run_date": run_date})

    if not send_email:
        logger.info(
            "send_email=False — skipping EOD email for %s (re-reconcile / "
            "reconcile_audit correction pass; artifact re-emitted, no resend).",
            run_date,
        )
    else:
        try:
            send_eod_email(
                run_date=run_date,
                nav=nav,
                daily_return=daily_return,
                spy_return=spy_return,
                alpha=alpha,
                sender=config["email_sender"],
                recipients=config["email_recipients"],
                account_snapshot=account,
                data_warnings=data_warnings,
                console_base_url=config.get("console_base_url"),
            )
        except Exception as e:
            _log_paged("EOD email failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "eod_email", "run_date": run_date})

    # Write health status
    try:
        from nousergon_lib.health import Deliverable, write_health
        write_health(
            module_name="eod_reconcile",
            deliverables=[
                Deliverable(name="eod_reconcile", required=True, produced=True),
            ],
            run_date=run_date,
            duration_seconds=_time.time() - _health_start,
            summary={
                "nav": round(nav, 2),
                "daily_return": round(daily_return, 4) if daily_return is not None else None,
                "alpha": round(alpha, 4) if alpha is not None else None,
                "n_positions": len(positions),
            },
            bucket=trades_bucket,
        )
    except Exception as _he:
        logger.warning("Health status write failed: %s", _he)

    # ── Data manifest ──────────────────────────────────────────────────────
    try:
        from executor.data_manifest import write_data_manifest
        trades_today_count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE date=?", (run_date,)
        ).fetchone()[0]
        write_data_manifest(
            bucket=trades_bucket,
            module_name="eod_reconcile",
            run_date=run_date,
            manifest={
                "nav": round(nav, 2),
                "n_positions": len(positions),
                "daily_return_pct": round(daily_return, 4) if daily_return is not None else None,
                "spy_return_pct": round(spy_return, 4) if spy_return is not None else None,
                "alpha_pct": round(alpha, 4) if alpha is not None else None,
                "trades_today": trades_today_count,
                "roundtrip_stats": roundtrip_stats,
            },
        )
    except Exception as _me:
        logger.warning("Data manifest write failed: %s", _me)

    # ── Uptime metrics ─────────────────────────────────────────────────────
    try:
        from executor import uptime_tracker
        metrics = uptime_tracker.run(bucket=trades_bucket)
        logger.info(
            "Uptime: active=%d/%d connected=%d crashes=%d uptime=%.1f%%",
            metrics.get("active_minutes", 0),
            metrics.get("market_minutes", 0),
            metrics.get("connected_minutes", 0),
            metrics.get("crashes", 0),
            metrics.get("uptime_pct", 0) * 100,
        )
    except Exception as _ue:
        logger.warning("Uptime tracker failed: %s", _ue)

    # T+1 self-heal now runs at the START of this function, before the NAV
    # three-way pricing&timing term reads yesterday's prior_positions —
    # see the "T+1 self-heal, run BEFORE today's own reconcile" block above.

    if fd:
        fd.log_summary(logger)
    conn.close()

    # ── Deferred integrity RAISE (alpha-engine-config-I8188) ────────────────
    # Every artifact of the session is now durable — the eod_pnl row (with the
    # breach recorded in integrity_breach_json), the S3 CSV exports, the
    # reconciliation audit, the report artifact, the email. Only now does the
    # run fail, so the pipeline goes red WITHOUT costing the operator the
    # evidence needed to diagnose it. Default is RAISE per the fleet fail-loud
    # rule: a residual bound that only warns is the defect this closes.
    if integrity_breaches:
        summary = "; ".join(b.get("message", b.get("kind", "?")) for b in integrity_breaches)
        if fd:
            fd.report(
                RuntimeError(summary),
                severity="error",
                context={
                    "site": "eod_pnl_integrity_gate",
                    "run_date": run_date,
                    "kinds": sorted({b.get("kind", "?") for b in integrity_breaches}),
                    "n_breaches": len(integrity_breaches),
                },
            )
        raise RuntimeError(
            f"EOD P&L integrity gate failed for {run_date} "
            f"({len(integrity_breaches)} breach(es)). All artifacts for the "
            f"session were written before this raise; the breach detail is "
            f"persisted in eod_pnl.integrity_breach_json. {summary}"
        )

    logger.info("EOD reconciliation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "EOD reconciliation. Defaults to today's trading_day "
            "(via nousergon_lib.dates.now_dual). A past --date is SAFE: since "
            "the 2026-04-28 snapshot cutover, the row keyed by run_date sources "
            "its NAV/positions from the durable S3 snapshot for that date (not "
            "now-as-of IB state) and re-prices from settled ArcticDB, so "
            "re-reconciling a past day is the canonical correction path "
            "(config#1276). Requires a snapshot for the date."
        )
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "YYYY-MM-DD; defaults to today's trading_day. A past date "
            "re-reconciles from its snapshot and REQUIRES --no-audit (the "
            "correction path) — the live path refuses non-current dates "
            "(config#1610 axis guard)."
        ),
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Suppress the EOD email (for a manual re-reconcile / correction of a past day).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the trailing reconcile_audit self-heal pass.",
    )
    args = parser.parse_args()
    # Capture an uncaught crash via flow-doctor before re-raising
    # (no-ops when flow-doctor is inactive).
    with guard_entrypoint():
        run(run_date=args.date, send_email=not args.no_email, run_audit=not args.no_audit)
