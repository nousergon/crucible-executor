"""Tests for executor/eod_reconcile.py — testable logic without IB Gateway."""

import io
import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from executor.eod_reconcile import (
    IB_MARK_OFF_CLOSE_PCT_FLOOR,
    NAV_BREACH_RESIDUAL_FLOOR_USD,
    NAV_HARD_GATE_TOLERANCE_NAV_BPS,
    NAV_HARD_GATE_TOLERANCE_USD_FLOOR,
    _apply_dividend_delta,
    _apply_mark_correction_to_positions,
    _attribute_mark_basis_divergence,
    _check_nav_three_way_hard_gate,
    _classify_nav_breach,
    _compute_daily_return,
    _compute_unattributed_residual_pct,
    _detect_ib_mark_outside_range,
    _format_mark_basis_contributors,
    _format_mark_range_detail,
    _load_constituents_sector_map,
    _mark_basis_usd,
    _nav_hard_gate_tolerance_usd,
    _resolve_prior_price,
    _restate_prior_positions_for_mark_correction,
    _synthesize_rationales,
)


class TestComputeDailyReturn:
    """Gap-aware per-position daily return (config#1228).

    The held-through baseline is the previous TRADING day's ArcticDB close,
    not a possibly-stale snapshot — so a skipped weekday/EOD SF can no longer
    inflate a multi-session move into a one-day return.
    """

    PREV_TD = date(2026, 6, 24)  # the trading day before run_date 2026-06-25

    def test_held_through_uses_prev_trading_day_close(self):
        # Held yesterday; ArcticDB prior row IS the previous trading day.
        pct, usd, prior_price, na = _compute_daily_return(
            "AAA", {"avg_cost": 90.0}, prior_pos={"shares": 10},
            current_price=110.0, shares=10,
            prior_close=100.0, prior_close_date=self.PREV_TD,
            expected_prev_td=self.PREV_TD,
        )
        assert pct == pytest.approx(10.0)       # 110/100 - 1
        assert usd == pytest.approx(100.0)      # (110-100)*10
        assert prior_price == pytest.approx(100.0)
        assert na is None

    def test_rgen_regression_healed_gap_is_not_inflated(self):
        # RGEN: held through; once 06-24 ($145.41) is healed into ArcticDB,
        # the 06-25 close ($145.23) is ~flat — NOT the +14.92% the stale
        # 06-23 close ($126.37) produced.
        pct, usd, prior_price, na = _compute_daily_return(
            "RGEN", {"avg_cost": 129.43}, prior_pos={"shares": 607},
            current_price=145.23, shares=715,
            prior_close=145.41, prior_close_date=self.PREV_TD,
            expected_prev_td=self.PREV_TD,
        )
        assert na is None
        assert pct == pytest.approx((145.23 / 145.41 - 1) * 100)
        assert abs(pct) < 1.0  # ~flat, decisively not +14.92%

    def test_unhealed_gap_marks_na_not_a_stale_number(self):
        # ArcticDB's latest prior row (06-23) predates the previous trading
        # day (06-24) — gap not healed. Refuse to compute against the stale
        # baseline; return an explicit N/A with a reason.
        pct, usd, prior_price, na = _compute_daily_return(
            "RGEN", {"avg_cost": 129.43}, prior_pos={"shares": 607},
            current_price=145.23, shares=715,
            prior_close=126.37, prior_close_date=date(2026, 6, 23),
            expected_prev_td=self.PREV_TD,
        )
        assert (pct, usd, prior_price) == (0.0, 0.0, None)
        assert na is not None and "RGEN" in na
        # The bogus +14.92% must never be produced.
        assert pct != pytest.approx(14.92, abs=0.5)

    def test_opened_today_prices_against_avg_cost(self):
        # No prior snapshot entry → opened today; baseline is entry avg_cost.
        pct, usd, prior_price, na = _compute_daily_return(
            "BBB", {"avg_cost": 50.0}, prior_pos=None,
            current_price=55.0, shares=4,
            prior_close=48.0, prior_close_date=self.PREV_TD,
            expected_prev_td=self.PREV_TD,
        )
        assert pct == pytest.approx(10.0)       # 55/50 - 1, uses avg_cost
        assert prior_price == pytest.approx(50.0)
        assert na is None

    def test_held_through_no_arctic_prior_falls_back(self):
        # Held yesterday but no ArcticDB prior close (e.g. brand-new listing)
        # → legacy snapshot/avg_cost resolution.
        pct, usd, prior_price, na = _compute_daily_return(
            "CCC", {"avg_cost": 20.0}, prior_pos={"closing_price": 25.0},
            current_price=30.0, shares=2,
            prior_close=None, prior_close_date=None,
            expected_prev_td=self.PREV_TD,
        )
        assert prior_price == pytest.approx(25.0)  # snapshot closing_price
        assert pct == pytest.approx((30.0 / 25.0 - 1) * 100)
        assert na is None


class TestComputeUnattributedResidualPct:
    """Phase 2 transparency-inventory headline metric: residual P&L
    not attributable to position MTM, interest, or dividends, expressed
    as % of NAV. Inventory gate is ≤1%."""

    def test_typical_small_residual(self):
        # $50 unattributed on $100,000 NAV → 0.05%
        assert _compute_unattributed_residual_pct(50.0, 100_000.0) == pytest.approx(0.05)

    def test_breaches_one_percent_gate(self):
        # $1,500 unattributed on $100,000 NAV → 1.5% > 1% gate
        result = _compute_unattributed_residual_pct(1_500.0, 100_000.0)
        assert result == pytest.approx(1.5)
        assert abs(result) > 1.0  # the alarm condition

    def test_zero_residual_returns_zero(self):
        assert _compute_unattributed_residual_pct(0.0, 100_000.0) == 0.0

    def test_negative_residual_preserves_sign(self):
        # Position pnl + interest exceeded actual NAV change (unaccounted fee)
        assert _compute_unattributed_residual_pct(-105.0, 100_000.0) == pytest.approx(-0.105)

    def test_none_unattributed_returns_none(self):
        """First-ever EOD run has no prior_nav → nav_reconciliation is {}
        → unattributed_usd is None. Persist NULL, not 0 — they mean
        different things."""
        assert _compute_unattributed_residual_pct(None, 100_000.0) is None

    def test_zero_nav_returns_none_not_div_by_zero(self):
        assert _compute_unattributed_residual_pct(50.0, 0.0) is None

    def test_none_nav_returns_none(self):
        assert _compute_unattributed_residual_pct(50.0, None) is None


class TestNavThreeWayHardGate:
    """config#2457 — NAV three-way reconcile promoted from observational
    (data_warnings only) to a hard gate that pages flow-doctor.

    `_check_nav_three_way_hard_gate` is the pure decision function `run()`
    calls; these tests exercise it directly rather than driving all of
    `run()`'s IO (snapshot/DB/S3 mocking is covered by test_eod_reconcile.py
    for the parts that need it)."""

    NAV = 1_000_000.0  # tolerance floor is bps-of-NAV dominant at this size
    # 15bps of $1,000,000 = $1,500, which is below the $2,500 floor, so the
    # floor governs at this NAV — pick divergences relative to the floor.

    def test_no_breach_within_tolerance_is_silent(self):
        """A small divergence well inside tolerance returns None — no gate
        fires, no page. (The pre-existing soft data_warnings entry, appended
        separately in run(), is unaffected by this function.)"""
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=200.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-07-13",
        )
        assert result is None

    def test_no_breach_exactly_at_tolerance_is_silent(self):
        """Boundary: exactly at tolerance does not breach (strict >)."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=tolerance,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-07-13",
        )
        assert result is None

    def test_breach_above_tolerance_fires(self):
        """A divergence beyond the hard-gate tolerance returns a breach dict
        with the fields run() needs to log + page flow-doctor."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        breach_amount = tolerance + 500.0
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=breach_amount,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-07-13",
        )
        assert result is not None
        assert result["run_date"] == "2026-07-13"
        assert result["pricing_timing_usd"] == pytest.approx(breach_amount)
        assert result["tolerance_usd"] == pytest.approx(tolerance)
        assert result["nav"] == self.NAV
        assert "2026-07-13" in result["message"]
        assert "NAV three-way reconcile breach" in result["message"]

    def test_breach_fires_on_negative_divergence_too(self):
        """Sign-agnostic: the broker NAV can be BELOW the settled/system NAV
        just as easily as above it — abs() comparison, not one-sided."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=-(tolerance + 1000.0),
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-07-13",
        )
        assert result is not None
        assert result["pricing_timing_usd"] < 0

    def test_pricing_timing_unavailable_does_not_fire(self):
        """The pricing_timing_available=False fallback path (missing prior
        snapshot) must NOT page — that's a data-availability gap, not a
        confirmed divergence, and already gets its own honesty warning in
        data_warnings (asserted separately in run()'s email-warnings path).
        A huge pricing_timing_usd value is passed here specifically to prove
        `available=False` short-circuits before the magnitude check."""
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=1_000_000.0,
            pricing_timing_available=False,
            nav=self.NAV,
            run_date="2026-07-13",
        )
        assert result is None

    def test_zero_nav_does_not_fire(self):
        """Divide-by-zero / degenerate-NAV protection, mirroring
        _compute_unattributed_residual_pct's zero-nav guard."""
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=10_000.0,
            pricing_timing_available=True,
            nav=0.0,
            run_date="2026-07-13",
        )
        assert result is None

    def test_none_nav_does_not_fire(self):
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=10_000.0,
            pricing_timing_available=True,
            nav=None,
            run_date="2026-07-13",
        )
        assert result is None

    def test_tolerance_uses_floor_for_small_nav(self):
        """Below the crossover NAV, the $ floor governs (not bps-of-NAV)."""
        small_nav = 100_000.0  # 15bps = $150, well under the $2,500 floor
        assert _nav_hard_gate_tolerance_usd(small_nav) == pytest.approx(
            NAV_HARD_GATE_TOLERANCE_USD_FLOOR
        )

    def test_tolerance_uses_bps_for_large_nav(self):
        """Above the crossover NAV, bps-of-NAV governs (not the $ floor)."""
        large_nav = 100_000_000.0  # 15bps = $150,000, well over the $2,500 floor
        expected = NAV_HARD_GATE_TOLERANCE_NAV_BPS / 10000.0 * large_nav
        assert _nav_hard_gate_tolerance_usd(large_nav) == pytest.approx(expected)
        assert _nav_hard_gate_tolerance_usd(large_nav) > NAV_HARD_GATE_TOLERANCE_USD_FLOOR

    def test_hard_gate_tolerance_wider_than_soft_warning_threshold(self):
        """Deliberate design invariant: the hard-gate (paged) tolerance must
        be wider than the existing soft data_warnings threshold
        (max($500, 5bps of NAV)) at every NAV level, or the hard gate pages
        exactly as often as the email already warns — training the operator
        to ignore it (the config#2145 lesson reconcile_audit.py's
        PAGE_THRESHOLD_BPS also encodes)."""
        for nav in (50_000.0, 1_000_000.0, 50_000_000.0):
            soft_threshold = max(500.0, 0.0005 * nav)
            hard_threshold = _nav_hard_gate_tolerance_usd(nav)
            assert hard_threshold > soft_threshold, (
                f"hard gate tolerance ${hard_threshold} must exceed soft "
                f"warning threshold ${soft_threshold} at nav=${nav}"
            )


class TestNavThreeWayHardGateResidualBased:
    """alpha-engine-config-I9087 — the gate re-based onto the UNEXPLAINED
    RESIDUAL rather than the raw pricing/timing term, with a required
    fail-closed guard.

    Brian's ruling (config#9087, 2026-09-08): the raw term breached 8 of 48
    sessions (16.7%) — almost all routine delayed-feed mark staleness — while
    the unexplained residual on the two fully-attributed sessions was three
    orders of magnitude smaller (−$30.7 / +$8.4). The gate now fires on
    EITHER the residual crossing `NAV_BREACH_RESIDUAL_FLOOR_USD` (when
    attribution is trustworthy) OR the raw term still crossing its own
    (unwidened) tolerance — the raw-term signal is retained, never removed,
    for the `broker_data_quality` warning-tier classification the call site
    derives from `_classify_nav_breach`.
    """

    NAV = 1_000_000.0  # tolerance floor ($2,500) governs at this NAV level

    def test_small_raw_term_does_not_fire_when_residual_is_also_small(self):
        """The quiet-day case this issue exists to produce: raw term and
        residual both within their respective bands — no page at all, where
        the pre-I9087 gate would already have been silent too (raw < old
        tolerance)."""
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=1_000.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
            residual_usd=50.0,
            attribution_ok=True,
        )
        assert result is None

    def test_unexplained_residual_fires_even_when_raw_term_is_within_tolerance(self):
        """THE REGRESSION GUARD for I9087. A raw term that never crosses the
        old $2,500/15bp tolerance must still page when the ATTRIBUTED
        residual — the portion `_attribute_mark_basis_divergence` cannot
        explain — exceeds `NAV_BREACH_RESIDUAL_FLOOR_USD`. This is the one
        behavior the raw-term-only gate could never produce."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        raw = tolerance - 100.0  # comfortably inside the raw-term tolerance
        residual = NAV_BREACH_RESIDUAL_FLOOR_USD + 100.0  # breaches the residual floor
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=raw,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
            residual_usd=residual,
            attribution_ok=True,
        )
        assert result is not None
        assert result["residual_breach"] is True
        assert result["raw_breach"] is False
        assert result["residual_usd"] == pytest.approx(residual)

    def test_raw_term_breach_still_fires_when_residual_is_explained(self):
        """The raw term is RETAINED, never removed (I9087 requirement 2):
        a raw-term breach with a small (explained) residual still returns a
        breach dict — the call site classifies it `broker_data_quality` and
        pages at the warning tier, not silence."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=tolerance + 500.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
            residual_usd=30.0,
            attribution_ok=True,
        )
        assert result is not None
        assert result["raw_breach"] is True
        assert result["residual_breach"] is False

    def test_attribution_failure_fails_closed_and_still_pages(self):
        """MERGE BLOCKER (alpha-engine-config-I9087). When attribution is
        untrustworthy (`attribution_ok=False` — raised, partial coverage, or
        no attribution at all), the gate must NOT read a small/absent
        residual as "no breach". It falls back to the raw term against its
        own tolerance and pages — attribution failure can only make the gate
        MORE likely to fire, never less. Here the residual argument passed
        in is deliberately tiny (as a broken attribution codepath might
        wrongly report) to prove the gate does not trust it."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=tolerance + 1_000.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
            residual_usd=1.0,  # would pass under a trusted residual test
            attribution_ok=False,
        )
        assert result is not None
        assert result["attribution_ok"] is False
        assert result["residual_breach"] is True  # fell back to the raw test
        assert result["raw_breach"] is True

    def test_attribution_failure_with_raw_term_within_tolerance_does_not_fire(self):
        """Fail-closed does not mean "always page" — with no divergence at
        all (raw within tolerance), a broken attribution codepath still has
        nothing to report on."""
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=200.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
            residual_usd=None,
            attribution_ok=False,
        )
        assert result is None

    def test_no_attribution_data_supplied_behaves_as_pre_i9087_raw_gate(self):
        """Call sites (or tests) that don't pass residual/attribution
        arguments at all default to `attribution_ok=False`, which reduces to
        the exact pre-I9087 raw-term-only gate — no silent behavior change
        for a caller that hasn't wired attribution through yet."""
        tolerance = _nav_hard_gate_tolerance_usd(self.NAV)
        result = _check_nav_three_way_hard_gate(
            pricing_timing_usd=tolerance + 500.0,
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-09-08",
        )
        assert result is not None
        assert result["raw_breach"] is True


class TestDetectIbMarkOutsideRange:
    """config#6349/#6818 — flag a held ticker whose IB portfolio mark lands
    outside the day's own ArcticDB [Low, High], the root cause behind six-
    plus historical NAV hard-gate breaches (AMD 2026-08-04 is the reference
    instance: IB mark $479.00 vs day range [$502.20, $530.13])."""

    def test_mark_below_day_low_flags_and_prices_the_error(self):
        positions = {"AMD": {"shares": 225, "ib_market_value": 225 * 479.00}}
        flags = _detect_ib_mark_outside_range(
            positions=positions,
            day_low={"AMD": 502.20},
            day_high={"AMD": 530.13},
        )
        assert len(flags) == 1
        assert flags[0]["ticker"] == "AMD"
        assert flags[0]["ib_mark"] == pytest.approx(479.00)
        assert flags[0]["mark_error_usd"] == pytest.approx(225 * (479.00 - 502.20))
        # The position itself is mutated so the flag reaches eod_report.json.
        assert positions["AMD"]["ib_mark_outside_range"] is True
        assert positions["AMD"]["ib_mark_range_error_usd"] == pytest.approx(
            225 * (479.00 - 502.20)
        )

    def test_mark_above_day_high_flags_too(self):
        positions = {"XYZ": {"shares": 100, "ib_market_value": 100 * 55.0}}
        flags = _detect_ib_mark_outside_range(
            positions=positions, day_low={"XYZ": 40.0}, day_high={"XYZ": 50.0},
        )
        assert len(flags) == 1
        assert flags[0]["mark_error_usd"] == pytest.approx(100 * (55.0 - 50.0))

    def test_mark_inside_range_does_not_flag(self):
        positions = {"SPY": {"shares": 10, "ib_market_value": 10 * 450.0}}
        flags = _detect_ib_mark_outside_range(
            positions=positions, day_low={"SPY": 440.0}, day_high={"SPY": 460.0},
        )
        assert flags == []
        # alpha-engine-config-I9637: an in-range mark now records EXPLICIT
        # negative evidence. Absence used to mean both "checked and fine" and
        # "never looked at", which is the hole this closes.
        assert positions["SPY"]["ib_mark_outside_range"] is False
        assert positions["SPY"]["ib_mark_range_checked"] is True

    def test_missing_day_range_or_shares_is_silently_skipped(self):
        """No day range (e.g. macro symbol not in ArcticDB universe) or a
        zero-share/legacy position without ib_market_value must not raise
        or false-flag — absence of data is not evidence of a bad mark."""
        positions = {
            "NO_RANGE": {"shares": 5, "ib_market_value": 500.0},
            "NO_SHARES": {"shares": 0, "ib_market_value": 0.0},
            "NO_IB_MV": {"shares": 5},
        }
        flags = _detect_ib_mark_outside_range(
            positions=positions,
            day_low={"NO_SHARES": 1.0, "NO_IB_MV": 1.0},
            day_high={"NO_SHARES": 2.0, "NO_IB_MV": 2.0},
        )
        assert flags == []


class TestClassifyNavBreach:
    """config#6349 deliverable 4 — a breach fully explained by out-of-range
    IB marks is a broker data-quality event, not a reconcile code defect."""

    def test_fully_explained_by_single_flagged_ticker_classifies_as_data_quality(self):
        flags = [{"ticker": "AMD", "mark_error_usd": -9560.36}]
        result = _classify_nav_breach(pricing_timing_usd=-9560.36, mark_range_flags=flags)
        assert result["classification"] == "broker_data_quality"
        assert result["residual_usd"] == pytest.approx(0.0)

    def test_residual_within_floor_still_classifies_as_data_quality(self):
        flags = [{"ticker": "AMD", "mark_error_usd": -9560.36}]
        result = _classify_nav_breach(
            pricing_timing_usd=-9560.36 - (NAV_BREACH_RESIDUAL_FLOOR_USD - 1),
            mark_range_flags=flags,
        )
        assert result["classification"] == "broker_data_quality"

    def test_large_residual_beyond_flagged_marks_stays_reconcile_defect(self):
        """Flagged marks explain only part of the divergence — the leftover
        is a real code-path question, not resolved by naming the ticker."""
        flags = [{"ticker": "AMD", "mark_error_usd": -1000.0}]
        result = _classify_nav_breach(pricing_timing_usd=-9560.36, mark_range_flags=flags)
        assert result["classification"] == "reconcile_defect"
        assert result["residual_usd"] == pytest.approx(-8560.36)

    def test_no_flags_is_always_reconcile_defect(self):
        result = _classify_nav_breach(pricing_timing_usd=-9560.36, mark_range_flags=[])
        assert result["classification"] == "reconcile_defect"


class TestClassifyNavBreachTieOut:
    """alpha-engine-config-I9085 — the full-book term is the NAV IDENTITY
    tie-out, not an explanation of the breach.

    `full_book_mark_basis_usd` is `Σ Δ(ib_market_value − market_value)`;
    `pricing_timing_usd` is `Δ(nav_ib − cash − accrued − Σ market_value)`.
    Differenced, cash/accrued/settled all cancel and what is left is whether
    IB's own NetLiquidation equals the sum of IB's own components — ~$0 on
    any ordinary equities book. Measured over all 48 sessions carrying an
    artifact (2026-06-22 → 2026-08-28): |residual| <= $63.54 every day and
    <= $10 on 45 of 48, against a $500 floor. Shipped as of I8733 that term
    WAS the explanation test, which made `reconcile_defect` unreachable from
    the call site — the same tautological-tie-out class as I8188.
    """

    # Live figures, run_date=2026-08-26 (alpha-engine-config#8722/#8733).
    PRICING_TIMING_USD = 3_933.92
    FLAGGED = [
        {"ticker": "MU", "mark_error_usd": 1_169.00},
        {"ticker": "SPY", "mark_error_usd": 32.00},
    ]
    FULL_BOOK_USD = 3_933.91

    def test_perfect_identity_tie_out_alone_does_not_earn_data_quality(self):
        """THE REGRESSION GUARD. The full book tying out exactly is the
        normal, always-true state; on its own it must no longer explain
        anything. With no name measurably off its settled close, the honest
        answer is `reconcile_defect`."""
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD,
            [],
            full_book_mark_basis_usd=self.PRICING_TIMING_USD,
            mark_divergence_explained_usd=0.0,
        )
        assert result["classification"] == "reconcile_defect"
        assert result["nav_identity_residual_usd"] == pytest.approx(0.0)
        assert result["nav_identity_holds"] is True
        assert result["residual_usd"] == pytest.approx(self.PRICING_TIMING_USD)

    def test_off_close_marks_explaining_the_breach_classify_data_quality(self):
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD,
            self.FLAGGED,
            full_book_mark_basis_usd=self.FULL_BOOK_USD,
            mark_divergence_explained_usd=self.PRICING_TIMING_USD - 30.0,
        )
        assert result["classification"] == "broker_data_quality"
        assert result["attribution_basis"] == "off_close_marks"
        assert result["residual_usd"] == pytest.approx(30.0)

    def test_partially_explained_by_off_close_marks_stays_reconcile_defect(self):
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD,
            self.FLAGGED,
            full_book_mark_basis_usd=self.FULL_BOOK_USD,
            mark_divergence_explained_usd=1_200.00,
        )
        assert result["classification"] == "reconcile_defect"
        assert result["residual_usd"] == pytest.approx(2_733.92)

    def test_broken_nav_identity_forces_reconcile_defect(self):
        """IB's NetLiquidation not equalling the sum of its own components is
        a real control — an unmodelled sleeve. It overrides a clean
        off-close explanation."""
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD,
            self.FLAGGED,
            full_book_mark_basis_usd=self.PRICING_TIMING_USD - 5_000.0,
            mark_divergence_explained_usd=self.PRICING_TIMING_USD,
        )
        assert result["classification"] == "reconcile_defect"
        assert result["nav_identity_holds"] is False
        assert result["nav_identity_residual_usd"] == pytest.approx(5_000.0)

    def test_classification_is_sign_symmetric(self):
        """The term is a day-over-day difference and its recorded exceedances
        split roughly even high/low (premise correction on config#6819), so
        the guard must behave identically on a negative breach."""
        hi = _classify_nav_breach(
            self.PRICING_TIMING_USD, [],
            full_book_mark_basis_usd=self.PRICING_TIMING_USD,
            mark_divergence_explained_usd=self.PRICING_TIMING_USD,
        )
        lo = _classify_nav_breach(
            -self.PRICING_TIMING_USD, [],
            full_book_mark_basis_usd=-self.PRICING_TIMING_USD,
            mark_divergence_explained_usd=-self.PRICING_TIMING_USD,
        )
        assert hi["classification"] == lo["classification"] == "broker_data_quality"
        hi_bad = _classify_nav_breach(
            self.PRICING_TIMING_USD, [],
            full_book_mark_basis_usd=self.PRICING_TIMING_USD,
            mark_divergence_explained_usd=0.0,
        )
        lo_bad = _classify_nav_breach(
            -self.PRICING_TIMING_USD, [],
            full_book_mark_basis_usd=-self.PRICING_TIMING_USD,
            mark_divergence_explained_usd=0.0,
        )
        assert hi_bad["classification"] == lo_bad["classification"] == "reconcile_defect"

    def test_missing_explanation_term_falls_back_to_flagged_subset(self):
        """No schema-2.1 ib_market_value on either day — the pre-#8733
        behaviour, which never over-explains."""
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD, self.FLAGGED,
            full_book_mark_basis_usd=None,
            mark_divergence_explained_usd=None,
        )
        assert result["attribution_basis"] == "flagged_subset"
        assert result["classification"] == "reconcile_defect"

    def test_uncovered_names_are_reported_and_do_not_soften_the_test(self):
        result = _classify_nav_breach(
            self.PRICING_TIMING_USD, [],
            full_book_mark_basis_usd=self.PRICING_TIMING_USD,
            full_book_uncovered_names=3,
            mark_divergence_explained_usd=1_000.0,
        )
        assert result["full_book_uncovered_names"] == 3
        assert result["classification"] == "reconcile_defect"


def _pos(shares, ib_mark, close):
    return {
        "shares": shares,
        "ib_market_value": shares * ib_mark,
        "market_value": shares * close,
    }


class TestAttributeMarkBasisDivergence:
    """alpha-engine-config-I9085 — `pricing_timing_usd` is a day-over-day
    DIFFERENCE of the mark basis, but `_detect_ib_mark_outside_range` is a
    POINT test on today's book. Every bad mark therefore produces TWO
    breaches — the day it lands and the day it reverts — and the point
    detector is structurally blind to the second.

    Reference instance, run_date=2026-08-27 (the third occurrence of this
    alert class): MU 108 shares, IB mark $923.07 vs settled close $935.39
    (−1.32%, INSIDE the day's traded range so nothing flagged), prior-day
    basis +$1,962.36 from a mark that WAS out of range on 08-26. The
    resulting −$3,292.92 is 75% of a −$4,393 breach on which the alert named
    not one ticker.
    """

    # 2026-08-26 → 2026-08-27, live figures.
    PRIOR = {"MU": _pos(108, 956.57, 938.40), "CRUS": _pos(1008, 111.11, 110.34)}
    TODAY = {"MU": _pos(108, 923.07, 935.39), "CRUS": _pos(1008, 110.90, 110.88)}

    def test_mu_reversion_is_named_and_priced(self):
        out = _attribute_mark_basis_divergence(
            positions=self.TODAY, prior_positions=self.PRIOR,
        )
        top = out["contributors"][0]
        assert top["ticker"] == "MU"
        assert top["contrib_usd"] == pytest.approx(-3_292.92, abs=0.5)
        assert top["basis_prior_usd"] == pytest.approx(1_962.36, abs=0.5)
        assert top["basis_today_usd"] == pytest.approx(-1_330.56, abs=0.5)
        # The mechanism the point detector cannot see.
        assert top["reversion"] is True
        # In range on 08-27 — so this is NOT reachable via the range flag.
        assert top["off_close_pct_today"] == pytest.approx(1.32, abs=0.02)

    def test_contributors_are_ranked_by_absolute_dollars(self):
        out = _attribute_mark_basis_divergence(
            positions=self.TODAY, prior_positions=self.PRIOR,
        )
        dollars = [abs(c["contrib_usd"]) for c in out["contributors"]]
        assert dollars == sorted(dollars, reverse=True)

    def test_explained_excludes_names_whose_mark_IS_the_close(self):
        """The explanation term is a FILTERED subset keyed on a property of
        the DATA — that is what makes it non-tautological. A name whose IB
        mark equals its settled close on both days contributes nothing to
        it, so a breach moved by something else cannot be explained away."""
        prior = {"AAA": _pos(100, 50.0, 50.0)}
        today = {"AAA": _pos(100, 50.0, 50.0), "BBB": _pos(100, 20.0, 20.0)}
        out = _attribute_mark_basis_divergence(
            positions=today, prior_positions=prior,
        )
        assert out["explained_usd"] == pytest.approx(0.0)

    def test_off_close_floor_is_the_discriminator(self):
        """Just over the floor counts; just under does not."""
        over = 1.0 + (IB_MARK_OFF_CLOSE_PCT_FLOOR * 2) / 100.0
        under = 1.0 + (IB_MARK_OFF_CLOSE_PCT_FLOOR / 2) / 100.0
        prior = {"OVR": _pos(100, 50.0, 50.0), "UND": _pos(100, 50.0, 50.0)}
        today = {
            "OVR": _pos(100, 50.0 * over, 50.0),
            "UND": _pos(100, 50.0 * under, 50.0),
        }
        out = _attribute_mark_basis_divergence(
            positions=today, prior_positions=prior,
        )
        by = {c["ticker"]: c for c in out["contributors"]}
        assert by["OVR"]["mark_is_not_close"] is True
        assert by["UND"]["mark_is_not_close"] is False
        assert out["explained_usd"] == pytest.approx(by["OVR"]["contrib_usd"])

    def test_pre_schema_2_1_names_are_uncovered_not_guessed(self):
        prior = {"AAA": {"shares": 10, "market_value": 100.0}}
        today = {"AAA": _pos(10, 11.0, 10.0)}
        out = _attribute_mark_basis_divergence(
            positions=today, prior_positions=prior,
        )
        assert out["uncovered_names"] == 1
        assert out["contributors"] == []
        assert out["explained_usd"] is None

    def test_no_prior_snapshot_is_not_an_error(self):
        out = _attribute_mark_basis_divergence(
            positions=self.TODAY, prior_positions=None,
        )
        assert {c["ticker"] for c in out["contributors"]} == {"MU", "CRUS"}


class TestFormatMarkBasisContributors:
    """alpha-engine-config-I9085 deliverable — the alert names culprits on
    EVERY breach, not only when a mark crossed a traded-range boundary. The
    2026-08-27 page carried a portfolio total and zero tickers, which is the
    operator failure config#6349 deliverable 2 was filed to end."""

    def test_names_ticker_dollars_and_the_reversion(self):
        contributors = _attribute_mark_basis_divergence(
            positions=TestAttributeMarkBasisDivergence.TODAY,
            prior_positions=TestAttributeMarkBasisDivergence.PRIOR,
        )["contributors"]
        detail = _format_mark_basis_contributors(contributors)
        assert "MU" in detail
        assert "-3,292" in detail or "-3,293" in detail
        assert "REVERSION" in detail

    def test_empty_contributors_render_empty(self):
        assert _format_mark_basis_contributors([]) == ""


class TestFormatMarkRangeDetail:
    def test_names_ticker_and_dollar_error_in_alert_text(self):
        flags = [{
            "ticker": "AMD", "ib_mark": 479.00, "day_low": 502.20,
            "day_high": 530.13, "shares": 225, "mark_error_usd": -5224.50,
        }]
        detail = _format_mark_range_detail(flags)
        assert "AMD" in detail
        assert "479.00" in detail
        assert "-5,224" in detail or "-5224" in detail

    def test_multiple_tickers_are_all_named(self):
        flags = [
            {"ticker": "AMD", "ib_mark": 1, "day_low": 2, "day_high": 3, "shares": 1, "mark_error_usd": -1},
            {"ticker": "COIN", "ib_mark": 1, "day_low": 2, "day_high": 3, "shares": 1, "mark_error_usd": -1},
        ]
        detail = _format_mark_range_detail(flags)
        assert "AMD" in detail
        assert "COIN" in detail


class TestResolvePriorPrice:
    """Phase 3: prior-day price source resolution."""

    def test_prefers_explicit_closing_price(self):
        prior = {"closing_price": 105.0, "market_value": 500.0, "shares": 10}
        pos = {"avg_cost": 100.0}
        # closing_price wins even though MV/shares would give 50
        assert _resolve_prior_price(prior, pos, current_price=110.0) == 105.0

    def test_falls_back_to_mv_over_shares_for_legacy_snapshot(self):
        # Pre-Phase-3 snapshots have no closing_price
        prior = {"market_value": 1050.0, "shares": 10}
        pos = {"avg_cost": 100.0}
        assert _resolve_prior_price(prior, pos, current_price=110.0) == 105.0

    def test_uses_avg_cost_when_no_prior_snapshot(self):
        # Position opened today — no prior snapshot
        pos = {"avg_cost": 99.50}
        assert _resolve_prior_price(None, pos, current_price=101.0) == 99.50

    def test_falls_back_to_current_price_when_no_avg_cost(self):
        # Degenerate case — position has no avg_cost either
        assert _resolve_prior_price(None, {}, current_price=110.0) == 110.0


class TestApplyDividendDelta:
    """Day-over-day dividend accrual delta is attributed to the position."""

    def test_no_accrual_is_noop(self):
        pos = {"daily_return_usd": 1.5, "daily_return_pct": 0.1}
        _apply_dividend_delta(pos, {"accrued_dividend": 0.0}, prior_price=150.0, shares=10)
        assert pos["daily_return_usd"] == 1.5
        assert "dividend_usd" not in pos

    def test_new_accrual_added(self):
        pos = {"accrued_dividend": 5.0, "daily_return_usd": 2.0, "daily_return_pct": 0.1}
        _apply_dividend_delta(pos, {"accrued_dividend": 0.0}, prior_price=100.0, shares=10)
        assert pos["dividend_usd"] == 5.0
        assert pos["daily_return_usd"] == 7.0
        # prior_mv = 1000, daily_usd = 7 → pct = 0.7%
        assert pos["daily_return_pct"] == pytest.approx(0.7)

    def test_dividend_payout_does_not_touch_position_pnl(self):
        """On payout day, accrual drops to 0 and cash rises by the same amount.

        IB's NetLiquidation is invariant to the payout (accrual↓ = cash↑), so
        position P&L must NOT be reduced. The dividend was already earned on
        the ex-dividend day. Record it in dividend_paid_usd for visibility.
        """
        pos = {"accrued_dividend": 0.0, "daily_return_usd": 2.0, "daily_return_pct": 0.2}
        _apply_dividend_delta(pos, {"accrued_dividend": 5.0}, prior_price=100.0, shares=10)
        assert "dividend_usd" not in pos
        # daily_return_usd unchanged — payout is not a loss
        assert pos["daily_return_usd"] == 2.0
        assert pos["dividend_paid_usd"] == 5.0

    def test_no_prior_snapshot_treats_accrual_as_new(self):
        pos = {"accrued_dividend": 3.0, "daily_return_usd": 1.0}
        _apply_dividend_delta(pos, None, prior_price=100.0, shares=5)
        assert pos["dividend_usd"] == 3.0
        assert pos["daily_return_usd"] == 4.0


class TestSynthesizeRationales:
    """Mechanical (non-LLM) synthesis of per-position narratives.

    Executor has zero LLM exposure per
    ``[[preference_llm_calls_confined_to_research_module]]`` — the only
    path is template-derived synthesis from the context dict. The
    earlier Haiku-backed path + opt-in flag + cost-telemetry substrate
    were deleted outright 2026-05-25.
    """

    def test_empty_contexts(self):
        assert _synthesize_rationales([]) == {}

    def test_template_basic(self):
        contexts = [{
            "ticker": "AAPL",
            "entry_date": "2026-04-01",
            "entry_price": 150.0,
            "research_score": 82.0,
            "conviction": "rising",
        }]
        result = _synthesize_rationales(contexts)
        assert "AAPL" in result
        assert "150.00" in result["AAPL"]
        assert "82" in result["AAPL"]
        assert "rising" in result["AAPL"]

    def test_template_with_predictor(self):
        contexts = [{
            "ticker": "MSFT",
            "predicted_direction": "UP",
            "prediction_confidence": 0.75,
            "predicted_alpha": 0.025,
        }]
        result = _synthesize_rationales(contexts)
        assert "UP" in result["MSFT"]
        assert "75%" in result["MSFT"]

    def test_template_with_thesis(self):
        contexts = [{
            "ticker": "GOOG",
            "thesis_summary": "Strong AI momentum driving cloud revenue growth across enterprise segment.",
        }]
        result = _synthesize_rationales(contexts)
        assert "AI momentum" in result["GOOG"]

    def test_template_long_thesis_truncated(self):
        contexts = [{
            "ticker": "AMZN",
            "thesis_summary": "x" * 200,
        }]
        result = _synthesize_rationales(contexts)
        assert len(result["AMZN"]) < 200
        assert result["AMZN"].endswith("...")

    def test_template_with_today_actions(self):
        contexts = [{
            "ticker": "NVDA",
            "today_actions": [{"action": "BUY", "shares": 10}],
        }]
        result = _synthesize_rationales(contexts)
        assert "BUY" in result["NVDA"]
        assert "10 shares" in result["NVDA"]

    def test_template_no_data(self):
        contexts = [{"ticker": "TSLA"}]
        result = _synthesize_rationales(contexts)
        assert "No rationale" in result["TSLA"]

    def test_multiple_tickers(self):
        contexts = [
            {"ticker": "AAPL", "research_score": 85},
            {"ticker": "MSFT", "research_score": 72},
        ]
        result = _synthesize_rationales(contexts)
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result


class TestNoLlmExposure:
    """Guardrail test: ``executor/eod_reconcile.py`` must never import
    anthropic. Source-level pin so a future PR can't quietly re-add an
    LLM call (or even the SDK dep). Per
    ``[[preference_llm_calls_confined_to_research_module]]`` — executor
    is hard-guardrail zero-LLM."""

    def test_eod_reconcile_does_not_import_anthropic(self):
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "executor" / "eod_reconcile.py"
        ).read_text()
        assert "import anthropic" not in src, (
            "executor/eod_reconcile.py must not import anthropic — "
            "executor is hard-guardrail zero-LLM per "
            "[[preference_llm_calls_confined_to_research_module]]. "
            "If a future surface genuinely needs LLM-synthesized output, "
            "the call goes in alpha-engine-research and produces a "
            "frozen artifact executor reads."
        )
        assert "anthropic.Anthropic" not in src
        assert "from anthropic" not in src


class TestLoadConstituentsSectorMap:
    """Sector enrichment fallback reads latest weekly constituents.json."""

    def _mock_s3(self, keys: list[str], sector_map: dict | None):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": k} for k in keys],
        }
        body = {"sector_map": sector_map} if sector_map is not None else {}
        s3.get_object.return_value = {
            "Body": io.BytesIO(json.dumps(body).encode()),
        }
        return s3

    @patch("executor.eod_reconcile.boto3")
    def test_picks_latest_weekly_snapshot(self, mock_boto3):
        # Lexicographic max of ISO dates == chronological latest
        s3 = self._mock_s3(
            keys=[
                "market_data/weekly/2026-04-04/constituents.json",
                "market_data/weekly/2026-04-18/constituents.json",
                "market_data/weekly/2026-04-11/constituents.json",
            ],
            sector_map={"VRTX": "Health Care", "MSFT": "Information Technology"},
        )
        mock_boto3.client.return_value = s3

        result = _load_constituents_sector_map("alpha-engine-research")

        assert result == {"VRTX": "Health Care", "MSFT": "Information Technology"}
        # Confirms the most recent key was the one fetched
        s3.get_object.assert_called_once_with(
            Bucket="alpha-engine-research",
            Key="market_data/weekly/2026-04-18/constituents.json",
        )

    @patch("executor.eod_reconcile.boto3")
    def test_empty_when_no_snapshots_listed(self, mock_boto3):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": []}
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}
        s3.get_object.assert_not_called()

    @patch("executor.eod_reconcile.boto3")
    def test_empty_on_s3_exception(self, mock_boto3):
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = RuntimeError("boom")
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}

    @patch("executor.eod_reconcile.boto3")
    def test_empty_when_sector_map_missing_from_payload(self, mock_boto3):
        s3 = self._mock_s3(
            keys=["market_data/weekly/2026-04-18/constituents.json"],
            sector_map=None,  # body has no sector_map key
        )
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}


class TestHardGateWiringIsNotTautological:
    """alpha-engine-config-I9085 — the defect the unit tests could not see.

    `_classify_nav_breach`'s `reconcile_defect` branch was exercised only by
    tests that hand-fed a `full_book_mark_basis_usd` the CALL SITE cannot
    produce: `run()` passes `sum(compute_pricing_timing_by_ticker(...))`,
    which is the same quantity `pricing_timing_usd` is built from. These
    tests assert on the composition `run()` actually performs.
    """

    def _wire(self, positions, prior_positions, pricing_timing_usd):
        """Exactly the composition at the `run()` hard-gate call site."""
        from executor.eod_report import compute_pricing_timing_by_ticker

        by_ticker, uncovered = compute_pricing_timing_by_ticker(
            positions, prior_positions,
        )
        mb = _attribute_mark_basis_divergence(
            positions=positions, prior_positions=prior_positions,
        )
        return mb, _classify_nav_breach(
            pricing_timing_usd,
            _detect_ib_mark_outside_range(
                positions=positions, day_low={}, day_high={},
            ),
            full_book_mark_basis_usd=sum(by_ticker.values()),
            full_book_uncovered_names=uncovered,
            mark_divergence_explained_usd=mb["explained_usd"],
        )

    def test_marks_that_are_the_close_cannot_explain_a_breach(self):
        """A large divergence with every IB mark equal to its settled close
        is NOT a broker data-quality event — it is unexplained, and pages at
        error. Under the shipped (I8733) wiring this returned
        `broker_data_quality` at severity `warning`."""
        prior = {"AAA": _pos(1000, 100.0, 100.0)}
        today = {"AAA": _pos(1000, 100.0, 100.0)}
        _mb, result = self._wire(today, prior, pricing_timing_usd=-6_000.0)
        assert result["classification"] == "reconcile_defect"
        assert result["attribution_basis"] == "off_close_marks"

    def test_2026_08_27_reversion_day_classifies_and_names_mu(self):
        """The live breach this was filed on: −$4,393, ZERO tickers out of
        range, MU carrying 75% of it via a mark $12.32 below the settled
        close and INSIDE the day's traded range."""
        prior = TestAttributeMarkBasisDivergence.PRIOR
        today = TestAttributeMarkBasisDivergence.TODAY
        mb, result = self._wire(today, prior, pricing_timing_usd=-4_051.62)
        assert result["classification"] == "broker_data_quality"
        assert mb["contributors"][0]["ticker"] == "MU"
        assert mb["contributors"][0]["reversion"] is True
        # Deliverable: the alert names a ticker even though nothing was
        # flagged out of range.
        assert result["total_mark_error_usd"] == 0.0
        assert "MU" in _format_mark_basis_contributors(mb["contributors"])

    def test_explanation_term_is_strictly_narrower_than_the_full_book(self):
        """The structural guard: if these two are ever equal by
        construction, the classifier is tautological again."""
        prior = {
            "OFF": _pos(100, 50.0, 50.0),
            "ON": _pos(100, 50.0, 50.0),
        }
        today = {
            "OFF": _pos(100, 55.0, 50.0),   # 10% off close
            "ON": _pos(100, 50.001, 50.0),  # 0.002% — the mark IS the close
        }
        mb, _result = self._wire(today, prior, pricing_timing_usd=500.1)
        assert mb["explained_usd"] != pytest.approx(mb["covered_usd"])
        assert mb["explained_usd"] == pytest.approx(500.0)


class TestEodPnlSessionAxisGate:
    """alpha-engine-config-I9615 — an eod_pnl row IS a session.

    `eod_pnl` carries a row for Good Friday 2026-04-03. Measured from the live
    artifact, it is NOT a duplicate of 2026-04-02: the same three names
    (CVX/NVT/TER) at the same share counts carry different market values
    (CVX 69,049.53 -> 69,118.93), so IB holiday quotes moved the book by
    +$216.77 (+0.0216%) while `spy_close` was carried forward unchanged
    (655.830017 on both rows), making `spy_return_pct` exactly 0.000000 —
    +2.16bp of fabricated alpha in every chained series crossing it.

    `pnl_integrity`'s session-axis gate DETECTS such a row after the fact.
    This asserts the producer REFUSES to create one.
    """

    def test_refuses_to_reconcile_a_market_holiday(self):
        import pytest

        from executor.eod_reconcile import run

        # Good Friday 2026. run_audit=False is the correction/backfill path —
        # the one that accepts an arbitrary run_date, and the only remaining
        # way a non-trading-day row could be produced.
        with pytest.raises(RuntimeError, match="not an NYSE trading session"):
            run(run_date="2026-04-03", send_email=False, run_audit=False)

    def test_the_refusal_names_the_date_and_the_reason(self):
        import pytest

        from executor.eod_reconcile import run

        with pytest.raises(RuntimeError) as exc:
            run(run_date="2026-12-25", send_email=False, run_audit=False)
        msg = str(exc.value)
        assert "2026-12-25" in msg
        # A silent refusal is as bad as a silent write: the message has to say
        # what to do instead, or an operator "fixes" it by removing the gate.
        assert "backfill that session" in msg

    def test_a_real_trading_day_is_not_refused_by_this_gate(self):
        """The gate must not become a reason a legitimate correction pass
        cannot run — it fails PAST this check for a real session."""
        import pytest

        from executor.eod_reconcile import run

        # 2026-03-12 IS an NYSE session (it is one of I9615's two genuinely
        # missing rows). Whatever this raises, it must not be the session gate.
        with pytest.raises(Exception) as exc:  # noqa: B017 — any downstream failure is fine
            run(run_date="2026-03-12", send_email=False, run_audit=False)
        assert "not an NYSE trading session" not in str(exc.value)


class TestCorrectedMarkIsWrittenToThePosition:
    """alpha-engine-config-I10048 — the correction reached NAV and stopped.

    `plan_nav_mark_correction` (I9627 / PR524) removed HOOD's provably-wrong
    2026-09-02 broker mark from the headline NAV ($1,027,254.75 ->
    $1,025,649.99, −$1,604.76) and left the position row carrying the RAW
    mark: `ib_market_value: 101426.43` against `market_value: 99821.67`.

    Share count is not in the artifact; 909 is chosen so `shares × price`
    reproduces both published market values to the cent
    ($109.814818 settled, $111.579791 broker).
    """

    SHARES = 909
    SETTLED_CLOSE = 99_821.67 / 909
    IB_MARK = 101_426.43 / 909

    def _raw_position_book(self):
        # The book as `run()` holds it at the correction call site: the
        # settled-close override has NOT run yet, so `market_value` is still
        # IB's mark-to-market and `ib_market_value` was copied from it.
        return {
            "HOOD": {
                "shares": self.SHARES,
                "market_value": 101_426.43,
                "ib_market_value": 101_426.43,
            }
        }

    def test_the_repair_lands_on_the_position_not_only_on_nav(self):
        from executor.pnl_integrity import plan_nav_mark_correction

        positions = self._raw_position_book()
        # The detector runs FIRST and stamps its evidence; the mark is above
        # the day's high, the settled close is inside it.
        flags = _detect_ib_mark_outside_range(
            positions=positions,
            day_low={"HOOD": 108.50},
            day_high={"HOOD": 110.50},
        )
        assert flags and flags[0]["ticker"] == "HOOD"
        plan = plan_nav_mark_correction(
            flags,
            settled_closes={"HOOD": self.SETTLED_CLOSE},
            day_low={"HOOD": 108.50},
            day_high={"HOOD": 110.50},
            nav=1_027_254.75,
            run_date="2026-09-02",
        )
        assert plan["applied"] is True
        assert plan["correction_usd"] == pytest.approx(-1_604.76, abs=0.01)
        assert plan["nav_corrected"] == pytest.approx(1_025_649.99, abs=0.01)

        written = _apply_mark_correction_to_positions(positions, plan["corrections"])
        assert written == ["HOOD"]
        pos = positions["HOOD"]
        assert pos["ib_market_value"] == pytest.approx(99_821.67, abs=0.01)
        assert pos["ib_market_value_raw"] == pytest.approx(101_426.43)
        assert pos["ib_mark_correction_usd"] == pytest.approx(-1_604.76, abs=0.01)
        assert pos["ib_mark_corrected"] is True
        # Detector evidence is UNTOUCHED — a repaired row must never be
        # readable as one that was never wrong.
        assert pos["ib_mark_outside_range"] is True
        assert pos["ib_mark_range_error_usd"] == pytest.approx(
            self.SHARES * (self.IB_MARK - 110.50), abs=0.01,
        )
        # After the settled-close override the basis is $0 by construction —
        # the position is now on the same price the headline NAV is.
        pos["market_value"] = self.SHARES * self.SETTLED_CLOSE
        assert _mark_basis_usd(pos) == pytest.approx(0.0, abs=0.01)

    def test_a_correction_naming_an_absent_position_raises(self):
        """Fail loud: the plan is built from this very dict one call earlier,
        so a name it carries that the book does not is a contract violation,
        not a skippable row."""
        with pytest.raises(RuntimeError, match="absent from the positions book"):
            _apply_mark_correction_to_positions(
                {}, [{"ticker": "HOOD", "shares": 909, "settled_close": 109.81}],
            )


class TestPriorDayCorrectionIsNotTodaysResidual:
    """alpha-engine-config-I10048 — the classification defect, reproduced.

    2026-09-03 measured: `pricing_timing_usd` differences the NAV-level mark
    basis and therefore used the CORRECTED prior NAV (−$2,517), while
    `compute_pricing_timing_by_ticker` and `_attribute_mark_basis_divergence`
    differenced the RAW prior POSITION mark (full book −$4,122, explained
    −$4,004). The gap is the previous day's correction to the dollar:
    `nav_identity_residual_usd` = +$1,605, `residual_usd` = +$1,487, and the
    breach was labelled `reconcile_defect` and paged at ERROR.

    The book below is a three-name reduction that reproduces every one of
    those figures: HOOD carries the correction, one name carries the genuine
    off-close divergence (−$2,399) and one carries an on-close remainder
    (−$118) that must stay OUT of the explanation term.
    """

    PRICING_TIMING_USD = -2_517.0  # struck from the CORRECTED prior NAV

    # 2026-09-02, as PR524 persisted it: NAV corrected, HOOD's mark raw.
    PRIOR_RAW = {
        "HOOD": {"shares": 909, "ib_market_value": 101_426.43,
                 "market_value": 99_821.67, "ib_mark_outside_range": True,
                 "ib_mark_range_error_usd": 1_604.76},
        "OFFCLOSE": _pos(1000, 101.20, 100.00),   # 1.20% off close
        "ONCLOSE": _pos(2000, 100.00, 100.00),    # the mark IS the close
    }
    # 2026-09-03.
    TODAY = {
        "HOOD": _pos(909, 110.0110, 110.0110),    # back on the close
        "OFFCLOSE": _pos(1000, 98.801, 100.00),   # −1.199%
        "ONCLOSE": _pos(2000, 99.941, 100.00),    # −0.059%, below the 10bp floor
    }

    @staticmethod
    def _corrected_prior():
        """The same snapshot with HOOD restated to the mark NAV was struck on."""
        prior = {t: dict(p) for t, p in
                 TestPriorDayCorrectionIsNotTodaysResidual.PRIOR_RAW.items()}
        prior["HOOD"]["ib_market_value_raw"] = 101_426.43
        prior["HOOD"]["ib_market_value"] = 99_821.67
        prior["HOOD"]["ib_mark_corrected"] = True
        return prior

    def _wire(self, prior):
        """Exactly the composition `run()` performs at the hard-gate call site."""
        from executor.eod_report import compute_pricing_timing_by_ticker

        by_ticker, uncovered = compute_pricing_timing_by_ticker(self.TODAY, prior)
        mb = _attribute_mark_basis_divergence(
            positions=self.TODAY, prior_positions=prior,
        )
        return mb, _classify_nav_breach(
            self.PRICING_TIMING_USD,
            [],  # 2026-09-03 flagged nothing out of range
            full_book_mark_basis_usd=sum(by_ticker.values()),
            full_book_uncovered_names=uncovered,
            mark_divergence_explained_usd=mb["explained_usd"],
        )

    def test_raw_prior_mark_reproduces_the_reconcile_defect_misclassification(self):
        """The BUG, pinned to the live figures. This is what shipped."""
        mb, result = self._wire(self.PRIOR_RAW)
        assert result["full_book_mark_basis_usd"] == pytest.approx(-4_121.76, abs=1.0)
        assert mb["explained_usd"] == pytest.approx(-4_003.76, abs=1.0)
        assert result["nav_identity_residual_usd"] == pytest.approx(1_604.76, abs=1.0)
        assert result["residual_usd"] == pytest.approx(1_486.76, abs=1.0)
        assert result["classification"] == "reconcile_defect"

    def test_corrected_prior_mark_ties_the_identity_and_classifies_broker_data(self):
        """The FIX. Same day, same breach, consistent bookkeeping."""
        mb, result = self._wire(self._corrected_prior())
        assert result["full_book_mark_basis_usd"] == pytest.approx(-2_517.0, abs=1.0)
        assert mb["explained_usd"] == pytest.approx(-2_399.0, abs=1.0)
        assert abs(result["nav_identity_residual_usd"]) < 1.0
        assert result["residual_usd"] == pytest.approx(-118.0, abs=1.0)
        assert result["classification"] == "broker_data_quality"

    def test_restating_a_pre_fix_snapshot_derives_the_corrected_prior_mark(self):
        """Backward compatibility: a prior row written between PR524 and this
        fix carries a corrected `portfolio_nav` and a raw position mark. The
        correction plan was persisted whole on that same row, so the corrected
        prior mark is DERIVED from its own `corrections` list — not guessed
        from the NAV delta and not hand-edited in S3."""
        prior = {t: dict(p) for t, p in self.PRIOR_RAW.items()}
        blob = json.dumps({
            "applied": True,
            "correction_usd": -1_604.76,
            "corrections": [{
                "ticker": "HOOD", "shares": 909,
                "ib_mark": 101_426.43 / 909, "settled_close": 99_821.67 / 909,
                "correction_usd": -1_604.76,
            }],
        })
        assert _restate_prior_positions_for_mark_correction(prior, blob) == ["HOOD"]
        assert prior["HOOD"]["ib_market_value"] == pytest.approx(99_821.67, abs=0.01)
        assert prior["HOOD"]["ib_market_value_raw"] == pytest.approx(101_426.43)
        assert prior["HOOD"]["mark_basis_usd"] == pytest.approx(0.0, abs=0.01)
        assert prior["HOOD"]["ib_mark_corrected"] is True
        assert prior["HOOD"]["ib_mark_corrected_source"] == (
            "prior_row_nav_mark_correction_json"
        )
        # And the restated snapshot classifies the way the fix requires.
        _mb, result = self._wire(prior)
        assert result["classification"] == "broker_data_quality"
        assert abs(result["nav_identity_residual_usd"]) < 1.0

    def test_a_row_with_no_correction_is_left_exactly_as_persisted(self):
        prior = {t: dict(p) for t, p in self.PRIOR_RAW.items()}
        assert _restate_prior_positions_for_mark_correction(prior, None) == []
        assert _restate_prior_positions_for_mark_correction(
            prior, json.dumps({"applied": False, "corrections": []}),
        ) == []
        assert prior["HOOD"]["ib_market_value"] == pytest.approx(101_426.43)
        assert "ib_mark_corrected" not in prior["HOOD"]

    def test_a_row_already_corrected_forward_is_not_restated_twice(self):
        prior = self._corrected_prior()
        blob = json.dumps({
            "applied": True,
            "corrections": [{"ticker": "HOOD", "shares": 909,
                             "settled_close": 1.0, "correction_usd": -1.0}],
        })
        assert _restate_prior_positions_for_mark_correction(prior, blob) == []
        assert prior["HOOD"]["ib_market_value"] == pytest.approx(99_821.67)
