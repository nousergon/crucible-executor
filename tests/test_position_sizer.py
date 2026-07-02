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


# ═══════════════════════════════════════════════════════════════════════════════
# Stance-conditional sizing (stance taxonomy arc PR 4 follow-up, 2026-05-11)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStanceConditionalSizing:
    """Per-stance position-size multipliers.

    Defaults (executor falls back to these if config doesn't override):
      momentum 1.0× — baseline thesis (trend-following)
      value    0.7× — contrarian thesis carries higher uncertainty
      quality  0.8× — defensive names: smaller stake, longer hold
      catalyst 0.6× — event-driven: highest variance, smallest stake

    Backtester-tunable from day 1 — multipliers in
    config/executor_params.json. Auto-tuned weekly once 4+ weeks of
    stance-tagged history accumulates.
    """

    def _sizing(self, stance, **cfg_overrides):
        """Single-entry helper — returns the full sizing dict."""
        cfg_kwargs = {
            "max_position_pct": 1.0,        # disable cap so the multiplier shows through
            "stance_sizing_enabled": True,
        }
        cfg_kwargs.update(cfg_overrides)
        cfg = _base_config(**cfg_kwargs)
        return compute_position_size(
            ticker="X",
            portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "X"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=100.0,
            config=cfg,
            stance=stance,
        )

    def test_momentum_stance_default_multiplier_is_1_0(self):
        r = self._sizing("momentum")
        assert r["stance_adj"] == pytest.approx(1.0)

    def test_value_stance_default_multiplier_is_0_7(self):
        r = self._sizing("value")
        assert r["stance_adj"] == pytest.approx(0.7)

    def test_quality_stance_default_multiplier_is_0_8(self):
        r = self._sizing("quality")
        assert r["stance_adj"] == pytest.approx(0.8)

    def test_catalyst_stance_default_multiplier_is_0_6(self):
        r = self._sizing("catalyst")
        assert r["stance_adj"] == pytest.approx(0.6)

    def test_unknown_stance_falls_back_to_1_0(self):
        """A stance label not in the canonical set falls back to 1.0×
        (no adjustment). Defensive — protects against a future
        vocabulary change that ships before this PR is updated."""
        r = self._sizing("growth")  # not in canonical 4
        assert r["stance_adj"] == pytest.approx(1.0)

    def test_stance_none_no_adjustment(self):
        """stance=None (pre-stance-arc artifacts) → no multiplier
        applied. Legacy behavior preserved."""
        r = self._sizing(None)
        assert r["stance_adj"] == pytest.approx(1.0)

    def test_stance_sizing_disabled_via_config_flag(self):
        """``stance_sizing_enabled=False`` break-glass flag disables
        the multiplier even when stance is provided. Used for A/B
        comparison during the rollout window."""
        r = self._sizing("catalyst", stance_sizing_enabled=False)
        assert r["stance_adj"] == pytest.approx(1.0)

    def test_stance_multipliers_overridable_via_config(self):
        """Backtester-tunable: each multiplier reads from config first,
        falls back to the canonical default. Pinned so the
        executor_optimizer auto-tune path can move them."""
        r = self._sizing("value", stance_size_value=0.5)
        assert r["stance_adj"] == pytest.approx(0.5)

    def test_stance_multiplier_propagates_through_to_position_pct(self):
        """End-to-end: a catalyst pick should size at 0.6× of a momentum
        pick with otherwise-identical inputs."""
        r_momentum = self._sizing("momentum")
        r_catalyst = self._sizing("catalyst")
        ratio = r_catalyst["position_pct"] / r_momentum["position_pct"]
        assert ratio == pytest.approx(0.6, rel=1e-3)

    def test_value_stance_smaller_dollar_size_than_momentum(self):
        """A value pick gets ~70% of a momentum pick's dollars on the
        same setup — the institutional contrarian-stance discipline."""
        r_momentum = self._sizing("momentum")
        r_value = self._sizing("value")
        assert r_value["dollar_size"] < r_momentum["dollar_size"]
        assert r_value["dollar_size"] / r_momentum["dollar_size"] == pytest.approx(0.7, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier-win-probability sizing (Task B2 — meta-labeling consumer, dormant)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarrierWinProbSizing:

    def _adj(self, *, enabled, bwp, **cfg_overrides):
        cfg = _base_config(barrier_win_prob_sizing_enabled=enabled, **cfg_overrides)
        return compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "AAPL"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=cfg,
            barrier_win_prob=bwp,
        )["barrier_win_prob_adj"]

    def test_dormant_by_default(self):
        """Flag absent → no adjustment even when bwp is provided."""
        assert self._adj(enabled=False, bwp=0.9) == 1.0

    def test_coinflip_is_neutral(self):
        """bwp=0.5 → 1.0× (min 0.70 + range 0.60 * 0.5)."""
        assert self._adj(enabled=True, bwp=0.5) == pytest.approx(1.0)

    def test_certain_win_max_multiplier(self):
        assert self._adj(enabled=True, bwp=1.0) == pytest.approx(1.30)

    def test_certain_loss_min_multiplier(self):
        assert self._adj(enabled=True, bwp=0.0) == pytest.approx(0.70)

    def test_missing_field_graceful_degrade(self):
        """Enabled but no bwp on the prediction (pre-B1) → 1.0×."""
        assert self._adj(enabled=True, bwp=None) == 1.0

    def test_clamps_out_of_range(self):
        # bwp > 1 clamps to 1 → 1.30; bwp < 0 clamps to 0 → 0.70
        assert self._adj(enabled=True, bwp=5.0) == pytest.approx(1.30)
        assert self._adj(enabled=True, bwp=-3.0) == pytest.approx(0.70)

    def test_custom_min_and_range(self):
        # min=0.5, range=1.0 → bwp=1.0 → 1.5×
        assert self._adj(
            enabled=True, bwp=1.0,
            barrier_win_prob_sizing_min=0.5,
            barrier_win_prob_sizing_range=1.0,
        ) == pytest.approx(1.5)

    def test_downsizes_position_when_low_prob(self):
        """End-to-end: a low-barrier-prob pick sizes smaller than a high one."""
        enter = [{"ticker": f"T{i}"} for i in range(40)]  # base 0.025 < cap

        def _pp(bwp):
            return compute_position_size(
                ticker="AAPL", portfolio_nav=100_000, enter_signals=enter,
                signal=_base_signal(), sector_rating="market_weight",
                current_price=150.0,
                config=_base_config(barrier_win_prob_sizing_enabled=True),
                barrier_win_prob=bwp,
            )["position_pct"]

        assert _pp(0.1) < _pp(0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# ADV-based size cap (tradeability arc, config#1401)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdvSizeCap:
    """Per-name ADV size cap: a single new position may consume at most
    ``adv_size_cap_pct_adv`` of the name's average daily dollar volume."""

    def _size(self, *, adv_usd, **cfg_over):
        # Single entry → base_weight 1.0, capped at max_position_pct=0.05 →
        # $5,000 target on a $100k book (before any ADV cap).
        return compute_position_size(
            ticker="THINCO",
            portfolio_nav=100_000,
            enter_signals=[{"ticker": "THINCO"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=100.0,
            config=_base_config(**cfg_over),
            adv_usd=adv_usd,
        )

    def test_adv_cap_binds_on_illiquid_name(self):
        """ADV so thin that 10% of it < the $5k target → dollar_size clipped to
        10% of ADV and adv_cap_applied flagged."""
        # ADV = $20,000 → 10% cap = $2,000 < $5,000 target.
        r = self._size(adv_usd=20_000.0, adv_size_cap_pct_adv=0.10)
        assert r["adv_cap_applied"] is True
        assert r["dollar_size"] == pytest.approx(2_000.0)
        assert r["shares"] == 20  # floor(2000 / 100)
        # position_pct reflects the capped notional.
        assert r["position_pct"] == pytest.approx(0.02, abs=1e-6)

    def test_adv_cap_does_not_bind_on_liquid_name(self):
        """Ample ADV → 10% cap far exceeds the $5k target → no clip, legacy size."""
        # ADV = $50M → 10% cap = $5M ≫ $5,000 target.
        r = self._size(adv_usd=50_000_000.0, adv_size_cap_pct_adv=0.10)
        assert r["adv_cap_applied"] is False
        assert r["dollar_size"] == pytest.approx(5_000.0)
        assert r["position_pct"] == pytest.approx(0.05)

    def test_adv_cap_scales_with_pct_config(self):
        """Tightening adv_size_cap_pct_adv tightens the notional monotonically."""
        loose = self._size(adv_usd=100_000.0, adv_size_cap_pct_adv=0.20)["dollar_size"]
        tight = self._size(adv_usd=100_000.0, adv_size_cap_pct_adv=0.02)["dollar_size"]
        assert tight < loose
        assert tight == pytest.approx(2_000.0)   # 2% of 100k
        assert loose == pytest.approx(5_000.0)   # 20% of 100k = 20k, but target 5k wins

    def test_adv_cap_failsoft_when_adv_missing(self):
        """No ADV coverage (adv_usd=None) → cap skipped, legacy sizing preserved."""
        r = self._size(adv_usd=None, adv_size_cap_pct_adv=0.10)
        assert r["adv_cap_applied"] is False
        assert r["dollar_size"] == pytest.approx(5_000.0)

    def test_adv_cap_failsoft_on_nonpositive_or_nan_adv(self):
        """ADV ≤0 / NaN is a coverage gap, not a floor — cap skipped."""
        for bad in (0.0, -1.0, float("nan")):
            r = self._size(adv_usd=bad, adv_size_cap_pct_adv=0.10)
            assert r["adv_cap_applied"] is False
            assert r["dollar_size"] == pytest.approx(5_000.0)

    def test_adv_cap_disabled_via_flag(self):
        """adv_size_cap_enabled=False → cap never applies even on a thin name."""
        r = self._size(
            adv_usd=20_000.0, adv_size_cap_pct_adv=0.10, adv_size_cap_enabled=False,
        )
        assert r["adv_cap_applied"] is False
        assert r["dollar_size"] == pytest.approx(5_000.0)

    def test_adv_cap_can_zero_out_below_min_dollar(self):
        """When the ADV cap drops the notional below min_position_dollar, the
        min-size gate zeroes the order (never places a sub-min ticket)."""
        # ADV = $2,000 → 10% = $200 < min_position_dollar ($500) → shares 0.
        r = self._size(adv_usd=2_000.0, adv_size_cap_pct_adv=0.10)
        assert r["adv_cap_applied"] is True
        assert r["dollar_size"] == pytest.approx(200.0)
        assert r["shares"] == 0
