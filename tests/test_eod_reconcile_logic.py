"""Tests for executor/eod_reconcile.py — testable logic without IB Gateway."""

import io
import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from executor.eod_reconcile import (
    NAV_HARD_GATE_TOLERANCE_NAV_BPS,
    NAV_HARD_GATE_TOLERANCE_USD_FLOOR,
    _apply_dividend_delta,
    _check_nav_three_way_hard_gate,
    _compute_daily_return,
    _compute_unattributed_residual_pct,
    _load_constituents_sector_map,
    _nav_hard_gate_tolerance_usd,
    _resolve_prior_price,
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
