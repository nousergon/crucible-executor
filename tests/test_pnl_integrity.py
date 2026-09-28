"""Tests for the EOD P&L integrity gates (alpha-engine-config-I8188).

Each test names the live measurement it encodes, so a future threshold change
has to argue with the data rather than with a number.
"""

from __future__ import annotations

import pytest

from executor.pnl_integrity import (
    MARK_HARD_MATERIALITY_NAV_BPS,
    RESIDUAL_HARD_PER_SESSION_NAV_BPS,
    TWR_SELF_HEAL_MAX_CORRECTION_PCT,
    check_custodian_marks,
    check_residual_bounds,
    check_session_axis_coverage,
    gross_net_returns,
    mark_materiality_usd,
    nav_change_implied_returns,
    nav_implied_returns,
    plan_twr_self_heal,
    residual_cumulative_tolerance_usd,
    residual_per_session_tolerance_usd,
    session_costs,
    verify_nav_change_basis_closes,
    verify_twr_closes,
)

NAV = 1_030_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Residual bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualBounds:
    def test_the_identity_can_now_fail(self):
        """The whole defect: the reconciliation identity held on 114 of 114
        sessions because unattributed_usd was the remainder. A bound is the
        thing that makes a tautology falsifiable."""
        breaches = check_residual_bounds(
            unattributed_true_usd=-9_713.0, nav=NAV, run_date="2026-08-04",
        )
        kinds = {b["kind"] for b in breaches}
        assert "per_session" in kinds

    def test_ordinary_session_passes(self):
        """Measured median |residual| after lifting rotation out is $754 —
        7bp of NAV. The gate must not fire on the normal case."""
        assert check_residual_bounds(
            unattributed_true_usd=754.0, nav=NAV, run_date="2026-05-01",
        ) == []

    def test_p95_of_the_measured_distribution_passes(self):
        """p95 of the measured ex-rotation residual is $3,560 (35bp). The
        per-session bound sits at 50bp precisely so the observed band clears
        it — a hard gate set at the soft band's rate trains the operator to
        ignore it."""
        assert check_residual_bounds(
            unattributed_true_usd=3_560.0, nav=NAV, run_date="2026-05-02",
        ) == []

    def test_per_session_tolerance_is_nav_scaled_above_the_floor(self):
        assert residual_per_session_tolerance_usd(500_000.0) == 5_000.0
        assert residual_per_session_tolerance_usd(2_000_000.0) == pytest.approx(
            RESIDUAL_HARD_PER_SESSION_NAV_BPS / 10_000.0 * 2_000_000.0
        )

    def test_cumulative_drift_fires_where_no_single_session_would(self):
        """-$20,293 accumulated over the live window in daily increments that
        no per-session bound could ever see. That is the defect the cumulative
        leg exists for."""
        trailing = [-350.0] * 58  # -$20,300 total, each far inside the daily bound
        breaches = check_residual_bounds(
            unattributed_true_usd=-350.0,
            nav=NAV,
            trailing_residuals_usd=trailing,
            run_date="2026-08-21",
        )
        kinds = {b["kind"] for b in breaches}
        assert kinds == {"cumulative"}
        assert breaches[0]["value_usd"] == pytest.approx(-350.0 * 59)

    def test_measured_true_drift_does_not_fire_cumulatively(self):
        """After the sleeves are lifted, the measured cumulative residual over
        74 sessions is +$522. The bound is ~20x that and must not fire on the
        behaviour of a correctly-attributed book."""
        assert check_residual_bounds(
            unattributed_true_usd=7.0,
            nav=NAV,
            trailing_residuals_usd=[7.0] * 73,
            run_date="2026-08-21",
        ) == []

    def test_cumulative_window_is_bounded_to_a_quarter(self):
        """A residual that has already breached and been dealt with must age
        out, or the gate latches red forever."""
        trailing = [-5_000.0] * 30 + [0.0] * 62
        assert check_residual_bounds(
            unattributed_true_usd=0.0,
            nav=NAV,
            trailing_residuals_usd=trailing,
            run_date="2026-08-21",
        ) == []

    def test_none_residual_is_not_a_pass(self):
        """An absent measurement returns no breach, but the caller records
        dividend/sleeve availability separately — this asserts the gate does
        not invent a zero."""
        assert check_residual_bounds(
            unattributed_true_usd=None, nav=NAV, run_date="x") == []
        assert check_residual_bounds(
            unattributed_true_usd=1.0, nav=None, run_date="x") == []

    def test_cumulative_tolerance_scales(self):
        assert residual_cumulative_tolerance_usd(500_000.0) == 10_000.0
        assert residual_cumulative_tolerance_usd(3_000_000.0) == 30_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction costs
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionCosts:
    def test_absent_commission_is_not_a_measured_zero(self):
        """"Paper-account commissions are trivial" is how the cost line came
        not to exist. An absent figure and a reported $0.00 must be
        distinguishable.

        Tightened in the I8188 second pass: the flag alone was not enough. It
        lived in a ``data_warnings`` string while the PERSISTED column read
        0.0, and it did so on 1 of the first 6 live sessions — so the artifact
        a reader (or a viability threshold) sees rendered the absence as a
        measured zero anyway. ``commission_usd`` is now None on that path.
        """
        absent = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.0,
             "price_at_order": 10.0},
        ])
        assert absent["commission_usd"] is None
        assert absent["commission_available"] is False

        reported = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.0,
             "price_at_order": 10.0, "commission_usd": 0.0},
        ])
        assert reported["commission_usd"] == 0.0
        assert reported["commission_available"] is True

    def test_no_fills_is_available_not_missing(self):
        """A session with no trades has a known, complete cost picture."""
        assert session_costs([])["commission_available"] is True

    def test_slippage_is_signed_by_side(self):
        """Paying above arrival on a buy and selling below arrival are both
        costs. A sign error here would net two costs to zero."""
        out = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.10,
             "price_at_order": 10.00},
            {"action": "SELL", "shares": 100, "fill_price": 9.90,
             "price_at_order": 10.00},
        ])
        assert out["slippage_usd"] == pytest.approx(20.0)
        assert out["n_fills"] == 2

    def test_slippage_bps_matches_the_live_measurement_shape(self):
        """Live window: 468 fills, +6.4bp of traded notional."""
        out = session_costs([
            {"action": "BUY", "shares": 1000, "fill_price": 100.064,
             "price_at_order": 100.0},
        ])
        assert out["slippage_bps"] == pytest.approx(6.4, abs=0.01)

    def test_filled_shares_wins_over_ordered_shares(self):
        out = session_costs([
            {"action": "BUY", "shares": 100, "filled_shares": 40,
             "fill_price": 10.10, "price_at_order": 10.00},
        ])
        assert out["slippage_usd"] == pytest.approx(4.0)

    def test_unfilled_rows_are_skipped(self):
        out = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": None,
             "price_at_order": 10.0},
            {"action": "BUY", "shares": 0, "fill_price": 10.0,
             "price_at_order": 10.0},
        ])
        assert out["n_fills"] == 0
        assert out["traded_notional_usd"] == 0.0

    def test_commission_is_normalised_to_a_positive_cost(self):
        out = session_costs([
            {"action": "BUY", "shares": 10, "fill_price": 10.0,
             "price_at_order": 10.0, "commission_usd": -1.25},
        ])
        assert out["commission_usd"] == pytest.approx(1.25)


class TestGrossNet:
    def test_gross_and_net_are_different_numbers(self):
        """Before this, gross and net performance were the same number and
        neither was labelled."""
        out = gross_net_returns(
            nav_change_usd=1_000.0, prior_nav=1_000_000.0,
            commission_usd=25.0, slippage_usd=200.0,
        )
        assert out["daily_return_net_pct"] == pytest.approx(0.1)
        assert out["daily_return_gross_pct"] == pytest.approx(0.1225)
        assert out["total_cost_usd"] == pytest.approx(225.0)

    def test_zero_cost_day_collapses_them(self):
        out = gross_net_returns(
            nav_change_usd=1_000.0, prior_nav=1_000_000.0,
            commission_usd=0.0, slippage_usd=0.0,
        )
        assert out["daily_return_net_pct"] == out["daily_return_gross_pct"]

    def test_first_session_has_neither(self):
        out = gross_net_returns(
            nav_change_usd=None, prior_nav=None,
            commission_usd=1.0, slippage_usd=2.0,
        )
        assert out["daily_return_net_pct"] is None
        assert out["daily_return_gross_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWR closure
# ─────────────────────────────────────────────────────────────────────────────

def _series(navs, stored=None):
    rows = []
    prior = None
    for i, nav in enumerate(navs):
        pct = None if prior is None else (nav / prior - 1) * 100
        rows.append({
            "date": f"2026-04-{i + 1:02d}",
            "portfolio_nav": nav,
            "daily_return_pct": pct if pct is not None else 0.0,
        })
        prior = nav
    if stored:
        for idx, value in stored.items():
            rows[idx]["daily_return_pct"] = value
    return rows


class TestTwrClosure:
    def test_a_consistent_series_closes_exactly(self):
        result = verify_twr_closes(_series([1_000_000, 1_005_000, 1_002_000, 1_010_000]))
        assert result["closes"] is True
        assert abs(result["drift_bps"]) < 1e-6

    def test_the_live_defect_is_detected(self):
        """2026-04-07 stored +0.026827% where its own NAV series implies
        -0.140311%. That single row is 100% of the 17.4bp live drift."""
        rows = _series([1_001_658.39, 1_002_481.91, 1_002_698.68,
                        1_009_473.08, 1_008_056.68],
                       stored={4: 0.026827})
        result = verify_twr_closes(rows)
        assert result["closes"] is False
        assert [o["date"] for o in result["offenders"]] == ["2026-04-05"]
        assert result["drift_bps"] == pytest.approx(16.72, abs=0.2)

    def test_one_bp_is_the_tolerance(self):
        """Absent external flows the two figures are identically equal by
        construction, so the tolerance is numerical noise, not a band."""
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = 0.5 + 0.02  # 2bp off
        assert verify_twr_closes(rows)["closes"] is False
        rows[1]["daily_return_pct"] = 0.5 + 0.005  # 0.5bp off
        assert verify_twr_closes(rows)["closes"] is True

    def test_short_series_is_na_not_pass(self):
        assert verify_twr_closes([])["status"] == "n/a"
        assert verify_twr_closes(_series([1_000_000]))["status"] == "n/a"

    def test_missing_stored_return_is_na_not_pass(self):
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = None
        assert verify_twr_closes(rows)["status"] == "n/a"

    def test_nav_implied_returns_first_row_has_no_predecessor(self):
        detail = nav_implied_returns(_series([1_000_000, 1_005_000]))
        assert detail[0]["implied_pct"] is None
        assert detail[1]["implied_pct"] == pytest.approx(0.5)


class TestTwrSelfHeal:
    def test_the_live_defect_is_repaired_and_then_closes(self):
        rows = _series([1_001_658.39, 1_002_481.91, 1_002_698.68,
                        1_009_473.08, 1_008_056.68],
                       stored={4: 0.026827})
        plan = plan_twr_self_heal(rows)
        assert [c["date"] for c in plan["corrections"]] == ["2026-04-05"]
        assert plan["refused"] == []
        for correction in plan["corrections"]:
            for row in rows:
                if row["date"] == correction["date"]:
                    row["daily_return_pct"] = correction["to_pct"]
        assert verify_twr_closes(rows)["closes"] is True

    def test_a_clean_series_needs_no_repair(self):
        plan = plan_twr_self_heal(_series([1_000_000, 1_005_000, 1_002_000]))
        assert plan == {"corrections": [], "refused": []}

    def test_a_large_disagreement_is_refused_not_rewritten(self):
        """The self-heal must not be a licence to restate the track record.
        Past the ceiling the disagreement is a different NAV series — an
        external flow, a restated snapshot — and needs a ruling."""
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = 0.5 + TWR_SELF_HEAL_MAX_CORRECTION_PCT + 0.5
        plan = plan_twr_self_heal(rows)
        assert plan["corrections"] == []
        assert len(plan["refused"]) == 1
        assert "external flow" in plan["refused"][0]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. TWR closure, nav_change_usd basis (alpha-engine-config-I9025)
# ─────────────────────────────────────────────────────────────────────────────

def _nc_series(navs, nav_change=None, stored=None, missing_nc_before=0):
    """Like ``_series`` but also carries ``nav_change_usd``.

    ``nav_change_usd`` defaults to the exact NAV delta (so the two bases agree
    unless overridden). ``missing_nc_before`` sets ``nav_change_usd=None`` on
    the first N rows — the I9025 day-set-mismatch shape.
    """
    rows = []
    prior = None
    for i, nav in enumerate(navs):
        pct = None if prior is None else (nav / prior - 1) * 100
        nc = None if prior is None else nav - prior
        rows.append({
            "date": f"2026-04-{i + 1:02d}",
            "portfolio_nav": nav,
            "daily_return_pct": pct if pct is not None else 0.0,
            "nav_change_usd": nc,
        })
        prior = nav
    if nav_change:
        for idx, value in nav_change.items():
            rows[idx]["nav_change_usd"] = value
    if stored:
        for idx, value in stored.items():
            rows[idx]["daily_return_pct"] = value
    for i in range(min(missing_nc_before, len(rows))):
        rows[i]["nav_change_usd"] = None
    return rows


class TestNavChangeBasis:
    def test_a_clean_series_closes_exactly(self):
        result = verify_nav_change_basis_closes(
            _nc_series([1_000_000, 1_005_000, 1_002_000, 1_010_000])
        )
        assert result["closes"] is True
        assert abs(result["drift_bps"]) < 1e-6
        assert result["coverage_gap_sessions"] == []

    def test_the_measured_live_cause_is_day_set_coverage_not_drift(self):
        """I9025: nav_change_usd is NULL on the early sessions (pre-PR490)
        while daily_return_pct is populated throughout. Those rows must be
        excluded from BOTH chains, not counted as drift — and the remaining
        rows, which persist a matching nav_change_usd, close exactly."""
        rows = _nc_series(
            [1_000_000, 1_001_000, 1_003_000, 998_000, 1_004_000],
            missing_nc_before=3,
        )
        result = verify_nav_change_basis_closes(rows)
        assert result["coverage_gap_sessions"] == ["2026-04-02", "2026-04-03"]
        assert result["n_sessions"] == 2
        assert result["closes"] is True
        assert result["offenders"] == []

    def test_a_disagreeing_row_is_an_offender_not_a_coverage_gap(self):
        rows = _nc_series([1_000_000, 1_005_000])
        rows[1]["nav_change_usd"] = 5_000 - 200  # 2bp off from the true delta
        result = verify_nav_change_basis_closes(rows)
        assert result["closes"] is False
        assert [o["date"] for o in result["offenders"]] == ["2026-04-02"]
        assert result["coverage_gap_sessions"] == []

    def test_one_bp_is_the_tolerance(self):
        rows = _nc_series([1_000_000, 1_005_000])
        rows[1]["nav_change_usd"] = 5_000 + 1_000_000 * 0.0002  # 2bp off
        assert verify_nav_change_basis_closes(rows)["closes"] is False
        rows[1]["nav_change_usd"] = 5_000 + 1_000_000 * 0.00005  # 0.5bp off
        assert verify_nav_change_basis_closes(rows)["closes"] is True

    def test_short_series_is_na_not_pass(self):
        assert verify_nav_change_basis_closes([])["status"] == "n/a"
        assert verify_nav_change_basis_closes(_nc_series([1_000_000]))["status"] == "n/a"

    def test_all_rows_missing_nav_change_usd_is_na_not_pass(self):
        """Every row a coverage gap (e.g. a book that has never persisted
        nav_change_usd) must not read as a clean close on an empty chain."""
        rows = _nc_series([1_000_000, 1_005_000, 1_002_000], missing_nc_before=3)
        result = verify_nav_change_basis_closes(rows)
        assert result["status"] == "n/a"
        assert result["coverage_gap_sessions"] == ["2026-04-02", "2026-04-03"]

    def test_nav_change_implied_returns_flags_coverage_gap_not_offender(self):
        rows = _nc_series([1_000_000, 1_005_000], missing_nc_before=2)
        detail = nav_change_implied_returns(rows)
        assert detail[1]["coverage_gap"] is True
        assert detail[1]["implied_pct"] is None
        assert detail[1]["delta_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Custodian marks
# ─────────────────────────────────────────────────────────────────────────────

def _flag(ticker, mark, lo, hi, shares, error):
    return {"ticker": ticker, "ib_mark": mark, "day_low": lo, "day_high": hi,
            "shares": shares, "mark_error_usd": error}


class TestCustodianMarks:
    def test_the_three_material_live_breaches_raise(self):
        """AMD 2026-08-04 (-$5,220, 50.4bp), COIN 2026-07-30 (-$2,999,
        29.7bp), LNTH 2026-06-26 (-$2,532, 25.5bp) — all provably wrong marks,
        all material."""
        for ticker, mark, lo, hi, shares, error in [
            ("AMD", 479.00, 502.20, 530.13, 225, -5_220.00),
            ("COIN", 154.45, 159.31, 164.78, 617, -2_998.62),
            ("LNTH", 105.12, 108.51, 111.46, 747, -2_532.33),
        ]:
            breaches = check_custodian_marks(
                [_flag(ticker, mark, lo, hi, shares, error)],
                nav=NAV, run_date="2026-08-04",
            )
            assert len(breaches) == 1, ticker
            assert breaches[0]["ticker"] == ticker

    def test_the_five_immaterial_live_flags_do_not_raise(self):
        """Edge-of-range rounding: <=$584, <=5.9bp. They stay flags."""
        flags = [
            _flag("SPY", 728.41, 729.10, 742.68, 843, -583.74),
            _flag("COIN", 158.00, 158.68, 169.69, 617, -419.56),
            _flag("DECK", 91.76, 89.06, 91.68, 1273, 101.84),
            _flag("TWLO", 206.00, 206.47, 213.91, 188, -88.36),
            _flag("SPY", 772.42, 772.51, 776.78, 1015, -91.37),
        ]
        assert check_custodian_marks(flags, nav=NAV, run_date="2026-07-29") == []

    def test_materiality_is_nav_scaled_with_a_floor(self):
        assert mark_materiality_usd(1_000.0) == 500.0
        assert mark_materiality_usd(NAV) == pytest.approx(
            MARK_HARD_MATERIALITY_NAV_BPS / 10_000.0 * NAV
        )

    def test_no_flags_means_no_breach(self):
        assert check_custodian_marks([], nav=NAV) == []
        assert check_custodian_marks(None, nav=NAV) == []

    def test_absent_nav_cannot_be_graded(self):
        assert check_custodian_marks(
            [_flag("AMD", 479.0, 502.2, 530.13, 225, -5_220.0)], nav=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# Session-axis coverage — alpha-engine-config-I9615
# ─────────────────────────────────────────────────────────────────────────────
#
# A tiny fake calendar over one real week (2026-03-30 Mon → 2026-04-06 Mon),
# with 2026-04-03 (Good Friday) as its one holiday — the live case the issue
# measured. Independent of krepis so these tests assert the GATE's logic, not
# the calendar's.
import datetime as _dt  # noqa: E402

_FAKE_HOLIDAYS = {"2026-04-03"}


def _is_trading_day(date_str: str) -> bool:
    if date_str in _FAKE_HOLIDAYS:
        return False
    return _dt.date.fromisoformat(date_str).weekday() < 5


def _next_trading_day(date_str: str) -> str:
    d = _dt.date.fromisoformat(date_str) + _dt.timedelta(days=1)
    while not _is_trading_day(d.isoformat()):
        d += _dt.timedelta(days=1)
    return d.isoformat()


def _axis_rows(*dates: str) -> list[dict]:
    return [{"date": d} for d in dates]


class TestSessionAxisCoverage:
    def test_a_contiguous_series_closes_clean(self):
        """2026-04-01, 04-02, 04-06 — Good Friday and the weekend correctly
        absent."""
        out = check_session_axis_coverage(
            _axis_rows("2026-04-01", "2026-04-02", "2026-04-06"),
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out["closes"] is True
        assert out["breaches"] == []

    def test_the_live_good_friday_row_is_flagged(self):
        """The live case: a row exists FOR 2026-04-03, a day the calendar
        says was never a session."""
        out = check_session_axis_coverage(
            _axis_rows("2026-04-02", "2026-04-03", "2026-04-06"),
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out["closes"] is False
        kinds = {(b["kind"], b["date"]) for b in out["breaches"]}
        assert ("non_trading_day_row", "2026-04-03") in kinds

    def test_the_live_missing_session_is_named_by_its_own_date(self):
        """The live case: 2026-03-12 has no row at all. Named by its own
        date, not inferred from the pair either side of it."""
        out = check_session_axis_coverage(
            _axis_rows("2026-03-30", "2026-03-31", "2026-04-02"),  # 04-01 skipped
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out["closes"] is False
        kinds = {(b["kind"], b["date"]) for b in out["breaches"]}
        assert ("missing_session", "2026-04-01") in kinds

    def test_both_defect_classes_together(self):
        """A gap AND a spurious row are the SAME defect class — nothing
        asserts the date axis equals the trading calendar — and both surface
        from one call."""
        out = check_session_axis_coverage(
            _axis_rows("2026-03-30", "2026-04-02", "2026-04-03"),  # 03-31/04-01 missing, 04-03 spurious
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        kinds = {b["kind"] for b in out["breaches"]}
        assert "missing_session" in kinds
        assert "non_trading_day_row" in kinds

    def test_empty_series_is_na_not_pass(self):
        out = check_session_axis_coverage(
            [], is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out["status"] == "n/a"
        assert out["closes"] is None

    def test_a_single_row_needs_no_walk_but_is_still_checked(self):
        out = check_session_axis_coverage(
            _axis_rows("2026-04-01"),
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out["closes"] is True

        out_holiday = check_session_axis_coverage(
            _axis_rows("2026-04-03"),
            is_trading_day=_is_trading_day, next_trading_day=_next_trading_day,
        )
        assert out_holiday["closes"] is False
        assert out_holiday["breaches"][0]["kind"] == "non_trading_day_row"

    def test_a_calendar_lookup_failure_is_its_own_kind_not_a_silent_pass(self):
        def _boom(_date_str: str) -> bool:
            raise ValueError("malformed date")

        out = check_session_axis_coverage(
            _axis_rows("2026-04-01", "2026-04-02"),
            is_trading_day=_boom, next_trading_day=_next_trading_day,
        )
        assert out["closes"] is False
        assert all(b["kind"] == "calendar_lookup_failed" for b in out["breaches"])

# NAV mark correction (alpha-engine-config-I9627)
# ─────────────────────────────────────────────────────────────────────────────

from executor.pnl_integrity import (  # noqa: E402
    mark_correction_bound_usd,
    plan_nav_mark_correction,
)

# The live failure this exists for: eod-2026-08-31-1788206436 halted the
# postclose SF on DUOL. The settled close was independently confirmed against
# two vendors (Finnhub, Alpha Vantage) at $148.36 on a [$144.51, $149.61]
# range, so IB's $152.40 was the wrong number, not ArcticDB's.
_DUOL_FLAG = {
    "ticker": "DUOL",
    "ib_mark": 152.40,
    "day_low": 144.51,
    "day_high": 149.61,
    "shares": 708,
    "mark_error_usd": 708 * (152.40 - 149.61),
}
_DUOL_NAV = 1_020_009.79
_DUOL_CLOSES = {"DUOL": 148.36}
_DUOL_LOW = {"DUOL": 144.51}
_DUOL_HIGH = {"DUOL": 149.61}


def _plan_duol(**over):
    kwargs = {
        "settled_closes": _DUOL_CLOSES,
        "day_low": _DUOL_LOW,
        "day_high": _DUOL_HIGH,
        "nav": _DUOL_NAV,
        "run_date": "2026-08-31",
    }
    kwargs.update(over)
    return plan_nav_mark_correction([_DUOL_FLAG], **kwargs)


def test_live_2026_08_31_duol_mark_is_repaired_off_the_settled_close():
    plan = _plan_duol()
    assert plan["applied"] is True
    assert plan["refused"] is False
    # 708 x (148.36 - 152.40)
    assert plan["correction_usd"] == pytest.approx(-2860.32, abs=0.01)
    assert plan["nav_corrected"] == pytest.approx(_DUOL_NAV - 2860.32, abs=0.01)
    assert plan["corrected_tickers"] == ["DUOL"]
    assert plan["nav_raw"] == _DUOL_NAV
    assert "DUOL" in plan["message"] and "148.36" in plan["message"]


def test_a_repaired_mark_no_longer_halts_the_pipeline():
    plan = _plan_duol()
    assert check_custodian_marks(
        [_DUOL_FLAG], nav=plan["nav_corrected"], run_date="2026-08-31",
        corrected_tickers=plan["corrected_tickers"],
    ) == []


def test_an_unrepaired_mark_still_halts_the_pipeline():
    # Same flag, nothing corrected — the gate is unchanged for every name the
    # repair could not prove.
    breaches = check_custodian_marks(
        [_DUOL_FLAG], nav=_DUOL_NAV, run_date="2026-08-31", corrected_tickers=[],
    )
    assert len(breaches) == 1
    assert breaches[0]["ticker"] == "DUOL"


def test_custodian_gate_default_is_unchanged_when_no_correction_is_passed():
    assert len(check_custodian_marks([_DUOL_FLAG], nav=_DUOL_NAV)) == 1


def test_a_settled_close_outside_its_own_range_is_the_reference_data_being_wrong():
    # THE DISCRIMINATOR: a close is a traded price, so it cannot sit outside
    # the day's own [Low, High]. When it does, ArcticDB is what is wrong and
    # NAV must NOT be moved towards it.
    plan = _plan_duol(settled_closes={"DUOL": 160.00})
    assert plan["applied"] is False
    assert plan["corrected_tickers"] == []
    assert plan["unrepairable"][0]["ticker"] == "DUOL"
    assert "reference data is wrong" in plan["unrepairable"][0]["why"]
    # ... and the gate therefore still halts the run.
    assert len(check_custodian_marks(
        [_DUOL_FLAG], nav=_DUOL_NAV, corrected_tickers=plan["corrected_tickers"],
    )) == 1


def test_a_missing_settled_close_leaves_the_name_uncorrected():
    plan = _plan_duol(settled_closes={})
    assert plan["applied"] is False
    assert plan["corrected_tickers"] == []
    assert "unavailable" in plan["unrepairable"][0]["why"]


def test_a_book_scale_disagreement_is_refused_not_applied():
    nav = 1_000_000.0
    bound = mark_correction_bound_usd(nav)
    assert bound == 10_000.0  # 100bp of NAV, at this NAV equal to the floor
    flag = {
        "ticker": "AAA", "ib_mark": 100.0, "day_low": 60.0, "day_high": 70.0,
        "shares": 1000, "mark_error_usd": 30_000.0,
    }
    plan = plan_nav_mark_correction(
        [flag], settled_closes={"AAA": 65.0}, day_low={"AAA": 60.0},
        day_high={"AAA": 70.0}, nav=nav, run_date="2026-01-02",
    )
    assert plan["refused"] is True
    assert plan["applied"] is False
    assert plan["corrected_tickers"] == []
    assert plan["nav_corrected"] == nav  # NAV is left exactly as the broker sent it
    assert "different book" in plan["message"]
    # The gate still fires, which is the whole point of refusing.
    assert len(check_custodian_marks(
        [flag], nav=nav, corrected_tickers=plan["corrected_tickers"],
    )) == 1


def test_every_observed_historical_instance_sits_inside_the_bound():
    # AMD 2026-08-04 is the largest mark error in the measured window.
    nav = 1_030_000.0
    assert abs(-5_220.0) < mark_correction_bound_usd(nav)
    assert abs(-2_999.0) < mark_correction_bound_usd(nav)   # COIN 2026-07-30
    assert abs(-2_532.0) < mark_correction_bound_usd(nav)   # LNTH 2026-06-26
    assert abs(-2_860.32) < mark_correction_bound_usd(nav)  # DUOL 2026-08-31


def test_no_flags_is_a_no_op():
    plan = plan_nav_mark_correction([], settled_closes={}, day_low={},
                                    day_high={}, nav=1_000_000.0)
    assert plan["applied"] is False and plan["refused"] is False
    assert plan["nav_corrected"] == 1_000_000.0
    assert plan["corrections"] == []


def test_partial_repair_corrects_one_name_and_holds_the_gate_on_the_other():
    ok = {"ticker": "AAA", "ib_mark": 100.0, "day_low": 90.0, "day_high": 95.0,
          "shares": 100, "mark_error_usd": 500.0}
    # Above the 15bp-of-NAV materiality floor, so the gate has something to
    # hold: at NAV $1M that floor is $1,500.
    bad = {"ticker": "BBB", "ib_mark": 200.0, "day_low": 180.0, "day_high": 190.0,
           "shares": 100, "mark_error_usd": 2_000.0}
    plan = plan_nav_mark_correction(
        [ok, bad],
        settled_closes={"AAA": 92.0, "BBB": 250.0},  # BBB's close is out of range
        day_low={"AAA": 90.0, "BBB": 180.0},
        day_high={"AAA": 95.0, "BBB": 190.0},
        nav=1_000_000.0, run_date="2026-01-02",
    )
    assert plan["applied"] is True
    assert plan["corrected_tickers"] == ["AAA"]
    assert [u["ticker"] for u in plan["unrepairable"]] == ["BBB"]
    breaches = check_custodian_marks(
        [ok, bad], nav=plan["nav_corrected"],
        corrected_tickers=plan["corrected_tickers"],
    )
    assert [b["ticker"] for b in breaches] == ["BBB"]


# ─────────────────────────────────────────────────────────────────────────────
# Custodian-mark check COVERAGE (alpha-engine-config-I9637)
# ─────────────────────────────────────────────────────────────────────────────

from executor.pnl_integrity import check_mark_coverage  # noqa: E402


def test_full_coverage_reports_no_warning():
    positions = {
        "AAA": {"ib_mark_range_checked": True, "market_value": 100_000.0},
        "BBB": {"ib_mark_range_checked": True, "market_value": 50_000.0},
    }
    out = check_mark_coverage(positions, nav=1_000_000.0, run_date="2026-09-01")
    assert out["held"] == 2 and out["checked"] == 2 and out["unchecked"] == 0
    assert out["coverage_pct"] == 100.0
    assert out["warnings"] == []
    assert out["unchecked_material"] is False


def test_an_unchecked_material_position_is_named_and_called_material():
    # The measured hole: the ArcticDB macro library is Close-only, so a
    # macro-routed holding has no traded range to check against.
    positions = {
        "AAA": {"ib_mark_range_checked": True, "market_value": 100_000.0},
        "XLK": {"ib_mark_range_checked": False, "market_value": 51_000.0,
                "ib_mark_range_uncheckable_reason": "macro library is Close-only"},
    }
    out = check_mark_coverage(positions, nav=1_000_000.0, run_date="2026-09-01")
    assert out["checked"] == 1 and out["unchecked"] == 1
    assert out["coverage_pct"] == 50.0
    assert out["unchecked_names"][0]["ticker"] == "XLK"
    assert out["unchecked_names"][0]["reason"] == "macro library is Close-only"
    assert out["unchecked_market_value_usd"] == 51_000.0
    assert out["unchecked_material"] is True
    assert "MATERIAL" in out["warnings"][0]
    assert "XLK" in out["warnings"][0]


def test_an_unchecked_immaterial_position_still_warns_but_is_not_material():
    positions = {
        "AAA": {"ib_mark_range_checked": True, "market_value": 100_000.0},
        "TINY": {"ib_mark_range_checked": False, "market_value": 200.0,
                 "ib_mark_range_uncheckable_reason": "no share count"},
    }
    out = check_mark_coverage(positions, nav=1_000_000.0, run_date="2026-09-01")
    assert out["unchecked_material"] is False
    assert len(out["warnings"]) == 1
    assert "MATERIAL" not in out["warnings"][0]


def test_a_position_missing_the_stamp_counts_as_unchecked_not_checked():
    # Fail-safe direction: an absent stamp is not evidence the check ran.
    out = check_mark_coverage(
        {"AAA": {"market_value": 10_000.0}}, nav=1_000_000.0, run_date="2026-09-01",
    )
    assert out["unchecked"] == 1 and out["checked"] == 0


def test_empty_book_is_not_an_error():
    out = check_mark_coverage({}, nav=1_000_000.0, run_date="2026-09-01")
    assert out["held"] == 0 and out["coverage_pct"] is None and out["warnings"] == []


def test_the_live_2026_08_31_book_was_fully_covered():
    # Twelve universe-routed names, zero macro-routed — coverage was 12/12,
    # but that was a property of the day's holdings, not of the gate.
    positions = {
        t: {"ib_mark_range_checked": True, "market_value": 100_000.0}
        for t in ("ANF CRUS DECK DOCS DUOL HL HOOD MU NBIX PBF QLYS UAL".split())
    }
    out = check_mark_coverage(positions, nav=1_017_149.47, run_date="2026-08-31")
    assert out["held"] == 12 and out["checked"] == 12
    assert out["coverage_pct"] == 100.0 and out["warnings"] == []


class TestMacroRoutedHoldingIsDeclaredUncheckable:
    """alpha-engine-config-I9637 — the exclusion is a DECLARED property of the
    data plane, not an accident of the writer.

    I9637's closes-when asks for a test asserting the mark check EVALUATES a
    macro-routed holding. That cannot be written truthfully today: the ArcticDB
    `macro` library is Close-only (measured 2026-08-31 — XLK/SPY/GLD/VIX all
    return cols == ['Close']), so there is no [Low, High] to evaluate against.
    What is written instead is the honest inverse — the gap is NAMED on the
    coverage surface rather than rendered as a pass — plus a lock on the
    declared holdable set. When the data-plane fix lands (moving the holdable
    symbols into `universe`, per alpha-engine-data #245's SPY precedent), the
    first test here is what must be inverted, and its failure is the signal
    that I9637 is genuinely closed.
    """

    def test_a_macro_routed_holding_is_reported_unchecked_and_named(self):
        from executor.eod_reconcile import _detect_ib_mark_outside_range
        from executor.pnl_integrity import check_mark_coverage

        positions = {
            # universe-routed: has a traded range, gets checked.
            "AAPL": {"shares": 100, "ib_market_value": 20_000.0,
                     "market_value": 20_000.0},
            # macro-routed sector ETF: Close-only, no [Low, High] exists.
            "XLK": {"shares": 500, "ib_market_value": 100_000.0,
                    "market_value": 100_000.0},
        }
        _detect_ib_mark_outside_range(
            positions=positions,
            day_low={"AAPL": 195.0},
            day_high={"AAPL": 205.0},
        )
        cov = check_mark_coverage(positions, nav=1_000_000.0,
                                  run_date="2026-08-31")

        assert cov["held"] == 2
        assert cov["checked"] == 1
        assert cov["unchecked"] == 1
        names = [u["ticker"] for u in cov["unchecked_names"]]
        assert names == ["XLK"]
        # The reason travels with it — "unchecked" must never be silent.
        assert "Close-only" in cov["unchecked_names"][0]["reason"]
        # $100k against a $1M NAV is far past the 15bp materiality floor, so
        # this is an ERROR-grade gap, not a footnote.
        assert cov["unchecked_material"] is True
        assert cov["warnings"]

    def test_holdable_macro_set_is_a_declared_subset(self):
        from executor.price_cache import (
            _MACRO_SYMBOLS,
            MACRO_HOLDABLE_SYMBOLS,
        )

        assert MACRO_HOLDABLE_SYMBOLS <= _MACRO_SYMBOLS
        # Index LEVELS are not holdable and need no mark check. If one of these
        # ever becomes holdable, it joins the unverified-mark set and this
        # assertion is the thing that says so.
        assert MACRO_HOLDABLE_SYMBOLS.isdisjoint({"VIX", "VIX3M", "TNX", "IRX"})
        # Every sector ETF the optimizer can hold is in the declared set.
        assert {"XLK", "XLE", "XLF", "GLD", "USO"} <= MACRO_HOLDABLE_SYMBOLS


# ── Paging level for an applied correction (2026-09-28 HOOD) ─────────────────

import logging as _logging  # noqa: E402

from executor.pnl_integrity import (  # noqa: E402
    MARK_CORRECTION_PAGE_NAV_BPS,
    mark_correction_log_level,
)


def _hood_plan():
    # 875 × ($116.46 − $116.01) on a $1,025,587 NAV: 3.8bp, the 09-28 instance.
    return plan_nav_mark_correction(
        [{"ticker": "HOOD", "ib_mark": 116.01, "day_low": 116.06, "day_high": 120.0,
          "shares": 875, "mark_error_usd": -43.75}],
        settled_closes={"HOOD": 116.46}, day_low={"HOOD": 116.06},
        day_high={"HOOD": 120.0}, nav=1_025_587.36, run_date="2026-09-28",
    )


def test_a_small_applied_correction_is_tracked_not_paged():
    plan = _hood_plan()
    assert plan["applied"]
    assert mark_correction_log_level(plan) == _logging.WARNING


def test_a_material_applied_correction_still_pages():
    plan = _plan_duol()  # DUOL, 28bp
    assert plan["applied"]
    assert abs(plan["correction_usd"]) / plan["nav_raw"] * 1e4 >= MARK_CORRECTION_PAGE_NAV_BPS
    assert mark_correction_log_level(plan) == _logging.ERROR


def test_a_repeat_report_of_the_same_snapshot_is_info_whatever_the_size():
    assert mark_correction_log_level(_plan_duol(), announce=False) == _logging.INFO
    assert mark_correction_log_level(_hood_plan(), announce=False) == _logging.INFO
