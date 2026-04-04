"""Unit tests for executor.risk_guard — pure logic, no IBKR/S3 calls."""
import pytest
from unittest.mock import patch

from executor.risk_guard import (
    compute_drawdown_multiplier,
    check_order,
    check_correlation,
    _pearson_correlation,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _base_config(**overrides):
    """Minimal config dict for risk_guard functions."""
    cfg = {
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "bear_block_underweight": True,
        # Strategy sub-config for graduated drawdown
        "strategy": {
            "graduated_drawdown": {
                "enabled": True,
                "tiers": [
                    (-0.02, 1.00, "0% to -2%: full sizing"),
                    (-0.04, 0.50, "-2% to -4%: half sizing"),
                    (-0.06, 0.25, "-4% to -6%: quarter sizing"),
                ],
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _base_signal(**overrides):
    """Minimal signal dict."""
    sig = {
        "score": 80,
        "conviction": "stable",
        "price_target_upside": 0.15,
        "sector_rating": "market_weight",
        "signal": "ENTER",
    }
    sig.update(overrides)
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# compute_drawdown_multiplier
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDrawdownMultiplier:
    """Tests for the graduated drawdown multiplier logic."""

    def test_no_drawdown_returns_full_multiplier(self):
        config = _base_config()
        mult, desc = compute_drawdown_multiplier(100_000, 100_000, config)
        assert mult == 1.0

    def test_small_drawdown_within_first_tier_returns_full(self):
        """Drawdown of -1% is above the -2% tier threshold."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(99_000, 100_000, config)
        assert mult == 1.0

    def test_drawdown_at_tier1_boundary_triggers_tier1(self):
        """-2% drawdown should hit the first tier (multiplier 1.0 for the
        tier definition, but the tier at -0.02 has multiplier 1.00)."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(98_000, 100_000, config)
        # At exactly -2%, the tier (-0.02, 1.00) is breached
        assert mult == 1.0

    def test_drawdown_at_tier2_boundary_returns_half(self):
        """-4% drawdown should hit the second tier (multiplier 0.50)."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(96_000, 100_000, config)
        assert mult == 0.50

    def test_drawdown_at_tier3_boundary_returns_quarter(self):
        """-6% drawdown should hit the third tier (multiplier 0.25)."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(94_000, 100_000, config)
        assert mult == 0.25

    def test_drawdown_beyond_circuit_breaker_returns_zero(self):
        """-8% (or worse) drawdown triggers the circuit breaker → 0.0."""
        config = _base_config()
        mult, desc = compute_drawdown_multiplier(92_000, 100_000, config)
        assert mult == 0.0
        assert "circuit breaker" in desc.lower()

    def test_drawdown_well_beyond_circuit_breaker_returns_zero(self):
        """-15% drawdown → still 0.0."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(85_000, 100_000, config)
        assert mult == 0.0

    def test_graduated_drawdown_disabled_uses_binary_breaker(self):
        """When graduated drawdown is disabled, only the binary circuit
        breaker matters — any drawdown above -8% returns 1.0."""
        config = _base_config()
        config["strategy"]["graduated_drawdown"]["enabled"] = False
        # -5% drawdown, no graduated tiers → should still be 1.0
        mult, _ = compute_drawdown_multiplier(95_000, 100_000, config)
        assert mult == 1.0

    def test_graduated_drawdown_disabled_breaker_fires(self):
        """Disabled graduated + at circuit breaker → 0.0."""
        config = _base_config()
        config["strategy"]["graduated_drawdown"]["enabled"] = False
        mult, _ = compute_drawdown_multiplier(92_000, 100_000, config)
        assert mult == 0.0

    def test_zero_peak_nav_returns_full_multiplier(self):
        """Edge case: peak_nav=0 → no drawdown computed, return 1.0."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(100_000, 0, config)
        assert mult == 1.0

    def test_between_tier2_and_tier3(self):
        """-5% drawdown falls between tier2 (-4%) and tier3 (-6%).
        The last breached tier is tier2 → 0.50."""
        config = _base_config()
        mult, _ = compute_drawdown_multiplier(95_000, 100_000, config)
        assert mult == 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# check_order
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckOrder:
    """Tests for the order validation logic."""

    def _call(self, **kwargs):
        """Call check_order with sensible defaults, allowing overrides."""
        defaults = dict(
            ticker="AAPL",
            action="ENTER",
            dollar_size=4000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="neutral",
            signal=_base_signal(),
            config=_base_config(),
            price_histories=None,
        )
        defaults.update(kwargs)
        return check_order(**defaults)

    # ── Score gate ──

    def test_score_below_minimum_rejected(self):
        approved, reason = self._call(signal=_base_signal(score=50))
        assert not approved
        assert "Score" in reason

    def test_score_at_minimum_approved(self):
        approved, _ = self._call(signal=_base_signal(score=70))
        assert approved

    # ── Conviction (no hard gate — sizing adjustment only) ──

    def test_declining_conviction_approved(self):
        """Declining conviction should pass risk guard (sized down by position_sizer)."""
        approved, _ = self._call(signal=_base_signal(conviction="declining"))
        assert approved

    def test_rising_conviction_approved(self):
        approved, _ = self._call(signal=_base_signal(conviction="rising"))
        assert approved

    def test_stable_conviction_approved(self):
        approved, _ = self._call(signal=_base_signal(conviction="stable"))
        assert approved

    # ── Position size gate ──

    def test_position_exceeding_max_position_pct_rejected(self):
        """dollar_size / NAV > 5% → rejected."""
        approved, reason = self._call(dollar_size=6000)  # 6% of 100k
        assert not approved
        assert "Position size" in reason

    def test_position_within_max_position_pct_approved(self):
        approved, _ = self._call(dollar_size=4000)  # 4%
        assert approved

    # ── Bear regime uses lower max_position_pct ──

    def test_bear_regime_tighter_position_limit(self):
        """In bear regime, max_position_pct drops to 2.5%."""
        approved, reason = self._call(
            dollar_size=3000,  # 3% — under 5% but over 2.5%
            market_regime="bear",
            signal=_base_signal(sector_rating="overweight"),  # avoid underweight block
        )
        assert not approved
        assert "Position size" in reason

    # ── Sector exposure gate ──

    def test_sector_exceeding_max_sector_pct_rejected(self):
        positions = {
            "MSFT": {"market_value": 20_000, "sector": "Technology"},
            "GOOG": {"market_value": 5_000, "sector": "Technology"},
        }
        # 25k existing + 1k new = 26k / 100k = 26% > 25%
        approved, reason = self._call(
            dollar_size=1000,
            current_positions=positions,
        )
        assert not approved
        assert "Sector exposure" in reason

    def test_sector_within_limit_approved(self):
        positions = {
            "MSFT": {"market_value": 10_000, "sector": "Technology"},
        }
        approved, _ = self._call(dollar_size=4000, current_positions=positions)
        assert approved

    # ── Total equity exposure gate ──

    def test_total_equity_exceeding_max_equity_pct_rejected(self):
        positions = {
            f"STOCK{i}": {"market_value": 10_000, "sector": f"Sector{i}"}
            for i in range(9)
        }
        # 90k existing + 4k new = 94k / 100k = 94% > 90%
        approved, reason = self._call(
            dollar_size=4000,
            current_positions=positions,
        )
        assert not approved
        assert "Total equity" in reason

    # ── EXIT and REDUCE bypass all rules ──

    def test_exit_always_passes(self):
        """EXIT bypasses all rules, even with terrible signal scores."""
        approved, reason = self._call(
            action="EXIT",
            signal=_base_signal(score=10, conviction="declining"),
        )
        assert approved
        assert "EXIT" in reason

    def test_reduce_always_passes(self):
        approved, reason = self._call(
            action="REDUCE",
            signal=_base_signal(score=10, conviction="declining"),
        )
        assert approved
        assert "REDUCE" in reason

    # ── Valid ENTER ──

    def test_valid_enter_approved(self):
        approved, reason = self._call()
        assert approved
        assert "ENTER approved" in reason

    # ── Portfolio NAV zero ──

    def test_zero_nav_rejected(self):
        approved, reason = self._call(portfolio_nav=0)
        assert not approved

    # ── Drawdown halt blocks ENTER ──

    def test_drawdown_halt_blocks_enter(self):
        """When drawdown exceeds circuit breaker, ENTER is blocked."""
        approved, reason = self._call(
            portfolio_nav=91_000,
            peak_nav=100_000,
        )
        assert not approved
        assert "Drawdown halt" in reason

    # ── Bear regime blocks underweight sectors ──

    def test_bear_blocks_underweight_sector(self):
        approved, reason = self._call(
            market_regime="bear",
            dollar_size=1000,  # small enough to pass size checks
            signal=_base_signal(sector_rating="underweight"),
        )
        assert not approved
        assert "underweight" in reason.lower()

    # ── Correlation gate ──

    def test_high_correlation_blocks_entry(self):
        """Highly correlated same-sector ticker should be blocked."""
        # Build price histories with perfectly correlated returns
        history = [{"close": 100 + i} for i in range(70)]
        positions = {"MSFT": {"market_value": 5000, "sector": "Technology"}}
        approved, reason = self._call(
            ticker="AAPL",
            dollar_size=4000,
            current_positions=positions,
            price_histories={
                "AAPL": history,
                "MSFT": history,  # identical → correlation = 1.0
            },
            config=_base_config(
                correlation_block_enabled=True,
                correlation_block_threshold=0.80,
            ),
        )
        assert not approved
        assert "correlation" in reason.lower()

    def test_low_correlation_allows_entry(self):
        """Uncorrelated tickers should pass."""
        import random
        random.seed(42)
        history_a = [{"close": 100 + i} for i in range(70)]
        history_b = [{"close": 100 + random.uniform(-5, 5)} for _ in range(70)]
        positions = {"MSFT": {"market_value": 5000, "sector": "Technology"}}
        approved, _ = self._call(
            ticker="AAPL",
            dollar_size=4000,
            current_positions=positions,
            price_histories={
                "AAPL": history_a,
                "MSFT": history_b,
            },
            config=_base_config(
                correlation_block_enabled=True,
                correlation_block_threshold=0.80,
            ),
        )
        assert approved

    def test_correlation_check_disabled_always_passes(self):
        history = [{"close": 100 + i} for i in range(70)]
        positions = {"MSFT": {"market_value": 5000, "sector": "Technology"}}
        approved, _ = self._call(
            ticker="AAPL",
            dollar_size=4000,
            current_positions=positions,
            price_histories={"AAPL": history, "MSFT": history},
            config=_base_config(correlation_block_enabled=False),
        )
        assert approved

    def test_correlation_no_price_histories_passes(self):
        """When price_histories is None, correlation check is skipped."""
        approved, _ = self._call(price_histories=None)
        assert approved


# ═══════════════════════════════════════════════════════════════════════════════
# _pearson_correlation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(_pearson_correlation(x, y) - 1.0) < 0.001

    def test_perfect_negative_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(_pearson_correlation(x, y) - (-1.0)) < 0.001

    def test_uncorrelated(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 1, 4, 2, 3]
        corr = _pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr) < 0.5

    def test_single_element_returns_none(self):
        assert _pearson_correlation([1], [1]) is None

    def test_constant_series_returns_none(self):
        assert _pearson_correlation([5, 5, 5], [1, 2, 3]) is None


# ═══════════════════════════════════════════════════════════════════════════════
# check_correlation (standalone)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckCorrelation:
    def test_insufficient_history_passes(self):
        """Short history → skip correlation check."""
        approved, reason = check_correlation(
            "AAPL",
            {"AAPL": {"sector": "Tech"}, "MSFT": {"sector": "Tech"}},
            {"AAPL": [{"close": 100}] * 10, "MSFT": [{"close": 100}] * 10},
            {"correlation_block_enabled": True, "correlation_lookback_days": 60},
        )
        assert approved
        assert "insufficient" in reason

    def test_no_same_sector_positions_passes(self):
        """Different sectors → no correlation computed."""
        history = [{"close": 100 + i} for i in range(70)]
        approved, reason = check_correlation(
            "AAPL",
            {"AAPL": {"sector": "Tech"}, "JPM": {"sector": "Financial"}},
            {"AAPL": history, "JPM": history},
            {"correlation_block_enabled": True, "correlation_lookback_days": 60,
             "correlation_block_threshold": 0.80},
        )
        assert approved
        assert "no same-sector" in reason
