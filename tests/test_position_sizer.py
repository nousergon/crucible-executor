"""Unit tests for executor.position_sizer — pure sizing math, no external calls."""
import pytest

from executor.position_sizer import compute_position_size


# ── Helpers ──────────────────────────────────────────────────────────────────


def _base_config(**overrides):
    """Minimal config dict for position sizer."""
    cfg = {
        "max_position_pct": 0.05,
        "conviction_decline_adj": 0.70,
        "min_price_target_upside": 0.05,
        "upside_fail_adj": 0.70,
        "min_position_dollar": 500,
        "sector_adj": {
            "overweight": 1.05,
            "market_weight": 1.00,
            "underweight": 0.85,
        },
        # Disable optional adjustments by default for focused tests
        "atr_sizing_enabled": False,
        "confidence_sizing_enabled": False,
        "staleness_discount_enabled": False,
        "earnings_sizing_enabled": False,
    }
    cfg.update(overrides)
    return cfg


def _base_signal(**overrides):
    """Minimal signal dict."""
    sig = {
        "score": 82,
        "conviction": "stable",
        "price_target_upside": 0.15,
    }
    sig.update(overrides)
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# compute_position_size
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputePositionSize:

    def test_base_weight_equals_one_over_n(self):
        """With 4 enter signals, base weight = 0.25."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(4)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        # 1/4 = 0.25, capped at max_position_pct=0.05
        assert result["position_pct"] == 0.05

    def test_base_weight_single_entry(self):
        """With 1 entry, base weight = 1.0, capped at max_position_pct=0.05."""
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "AAPL"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["position_pct"] == 0.05
        assert result["dollar_size"] == 5000.0

    def test_base_weight_many_entries_below_cap(self):
        """With 25 entries, base weight = 0.04 < 0.05 cap."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["position_pct"] == 0.04

    def test_overweight_sector_adjustment(self):
        """Overweight sector should increase weight by 1.05x."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="overweight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["sector_adj"] == 1.05
        # 0.04 * 1.05 = 0.042
        assert result["position_pct"] == 0.042

    def test_underweight_sector_adjustment(self):
        """Underweight sector should decrease weight by 0.85x."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="underweight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["sector_adj"] == 0.85
        # 0.04 * 0.85 = 0.034
        assert result["position_pct"] == 0.034

    def test_declining_conviction_reduces_weight(self):
        """Declining conviction should apply 0.70 multiplier."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(conviction="declining"),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["conviction_adj"] == 0.70
        # 0.04 * 0.70 = 0.028
        assert result["position_pct"] == 0.028

    def test_stable_conviction_no_reduction(self):
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": f"T{i}"} for i in range(25)],
            signal=_base_signal(conviction="stable"),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["conviction_adj"] == 1.0

    def test_rising_conviction_no_reduction(self):
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": f"T{i}"} for i in range(25)],
            signal=_base_signal(conviction="rising"),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["conviction_adj"] == 1.0

    def test_cap_at_max_position_pct(self):
        """Even with all multipliers > 1, position weight should be capped."""
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "AAPL"}],  # base = 1.0
            signal=_base_signal(conviction="rising"),
            sector_rating="overweight",
            current_price=150.0,
            config=_base_config(),
        )
        # 1.0 * 1.05 * 1.0 * 1.0 = 1.05, capped at 0.05
        assert result["position_pct"] == 0.05

    def test_zero_entries_handled_gracefully(self):
        """Empty enter_signals list should not crash (max(0,1) = 1)."""
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        # base = 1/max(0,1) = 1.0, capped at 0.05
        assert result["position_pct"] == 0.05
        assert result["shares"] > 0

    def test_zero_price_returns_zero_shares(self):
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "AAPL"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=0,
            config=_base_config(),
        )
        assert result["shares"] == 0

    def test_drawdown_multiplier_reduces_sizing(self):
        """Passing drawdown_multiplier=0.50 should halve the position."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
            drawdown_multiplier=0.50,
        )
        # 0.04 * 0.50 = 0.02
        assert result["position_pct"] == 0.02
        assert result["dd_multiplier"] == 0.50

    def test_shares_floor_division(self):
        """Shares should be floor(dollar_size / price)."""
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "AAPL"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        # 5000 / 150 = 33.33 → 33 shares
        assert result["shares"] == 33
        assert result["dollar_size"] == 5000.0

    def test_upside_below_minimum_reduces_weight(self):
        """If price_target_upside < min_price_target_upside, apply penalty."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(price_target_upside=0.02),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(),
        )
        assert result["upside_adj"] == 0.70
        # 0.04 * 0.70 = 0.028
        assert result["position_pct"] == 0.028

    # ── ATR sizing ──

    def test_atr_sizing_reduces_volatile_stocks(self):
        """High ATR% → smaller position (target_risk / atr_pct)."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(atr_sizing_enabled=True, atr_sizing_target_risk=0.02),
            atr_pct=0.04,  # 4% daily ATR → adj = 0.02/0.04 = 0.5
        )
        assert result["atr_adj"] == 0.5
        # 0.04 * 0.5 = 0.02
        assert result["position_pct"] == 0.02

    def test_atr_sizing_capped_at_1_5(self):
        """Very low ATR should be capped at 1.5x, not unlimited."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(atr_sizing_enabled=True, atr_sizing_target_risk=0.02),
            atr_pct=0.005,  # 0.5% ATR → 0.02/0.005 = 4.0 → capped at 1.5
        )
        assert result["atr_adj"] == 1.5

    def test_atr_none_defaults_to_1(self):
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": f"T{i}"} for i in range(25)],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(atr_sizing_enabled=True),
            atr_pct=None,
        )
        assert result["atr_adj"] == 1.0

    # ── Confidence sizing ──

    def test_confidence_sizing_high_confidence(self):
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(confidence_sizing_enabled=True,
                                confidence_sizing_min=0.7, confidence_sizing_range=0.6),
            prediction_confidence=1.0,  # max confidence → adj = 0.7 + 0.6*1.0 = 1.3
        )
        assert abs(result["confidence_adj"] - 1.3) < 0.01

    def test_confidence_sizing_low_confidence(self):
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(confidence_sizing_enabled=True,
                                confidence_sizing_min=0.7, confidence_sizing_range=0.6),
            prediction_confidence=0.0,  # min confidence → adj = 0.7
        )
        assert abs(result["confidence_adj"] - 0.7) < 0.01

    def test_confidence_clamped_to_0_1(self):
        """Confidence > 1 should be clamped."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(confidence_sizing_enabled=True,
                                confidence_sizing_min=0.7, confidence_sizing_range=0.6),
            prediction_confidence=5.0,  # way above 1 → clamped to 1.0
        )
        assert abs(result["confidence_adj"] - 1.3) < 0.01

    # ── Staleness discount ──

    def test_staleness_within_cadence_no_decay(self):
        """Weekly research read mid-week (age < cadence) should NOT be decayed."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(staleness_discount_enabled=True,
                                signal_cadence_days=7,
                                staleness_decay_per_day=0.03, staleness_floor=0.70),
            signal_age_days=4,  # mid-week read of Saturday signals — within 7-day cadence
        )
        assert result["staleness_adj"] == 1.0

    def test_staleness_at_cadence_boundary_no_decay(self):
        """Age exactly equal to cadence should still be fresh (boundary inclusive)."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(staleness_discount_enabled=True,
                                signal_cadence_days=7,
                                staleness_decay_per_day=0.03, staleness_floor=0.70),
            signal_age_days=7,
        )
        assert result["staleness_adj"] == 1.0

    def test_staleness_decays_past_cadence(self):
        """Once age exceeds cadence, decay applies to the EXCESS only."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(staleness_discount_enabled=True,
                                signal_cadence_days=7,
                                staleness_decay_per_day=0.03, staleness_floor=0.70),
            signal_age_days=12,  # effective_age = 5 → 1 - 0.03*5 = 0.85
        )
        assert abs(result["staleness_adj"] - 0.85) < 0.01

    def test_staleness_daily_cadence_decays_from_day_one(self):
        """Setting cadence=0 restores the legacy daily-signal decay behavior."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(staleness_discount_enabled=True,
                                signal_cadence_days=0,
                                staleness_decay_per_day=0.03, staleness_floor=0.70),
            signal_age_days=5,
        )
        assert abs(result["staleness_adj"] - 0.85) < 0.01

    def test_staleness_floor(self):
        """Staleness discount should not go below floor even far past cadence."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(staleness_discount_enabled=True,
                                signal_cadence_days=7,
                                staleness_decay_per_day=0.03, staleness_floor=0.70),
            signal_age_days=27,  # effective_age = 20 → 1 - 0.03*20 = 0.40 → floored at 0.70
        )
        assert abs(result["staleness_adj"] - 0.70) < 0.01

    # ── Earnings proximity ──

    def test_earnings_proximity_reduces_sizing(self):
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(earnings_sizing_enabled=True,
                                earnings_proximity_days=5, earnings_sizing_reduction=0.50),
            days_to_earnings=3,  # within 5-day window → adj = 0.50
        )
        assert abs(result["earnings_adj"] - 0.50) < 0.01

    def test_earnings_far_away_no_reduction(self):
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_base_config(earnings_sizing_enabled=True,
                                earnings_proximity_days=5, earnings_sizing_reduction=0.50),
            days_to_earnings=30,  # outside window → adj = 1.0
        )
        assert result["earnings_adj"] == 1.0

    def test_min_position_dollar_filters_tiny_orders(self):
        """If dollar_size < min_position_dollar, shares should be 0."""
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=5_000,  # small portfolio
            enter_signals=[{"ticker": f"T{i}"} for i in range(25)],
            signal=_base_signal(),
            sector_rating="underweight",
            current_price=150.0,
            config=_base_config(),
            drawdown_multiplier=0.25,
        )
        # base=0.04, sector=0.85, dd=0.25 → 0.04*0.85*0.25=0.0085
        # dollar = 5000 * 0.0085 = 42.50 < 500 min
        assert result["shares"] == 0
