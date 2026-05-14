"""Stage D' Wire 5 — regime-conditional entry score threshold.

Pins:

1. ``regime_conditional_min_score`` math: ``base + (-z*scale)`` clamped to
   ``[base-cap, base+cap]`` then to ``[floor, ceil]``; ``None`` → base.
2. ``check_order`` honors ``regime_min_score_enabled`` flag; default OFF
   preserves legacy ``min_score_to_enter`` behavior.
3. Risk-off (z<0) RAISES threshold (more selective — fewer entries);
   risk-on (z>0) LOWERS (more inclusive — more entries).
4. min_score veto event context carries ``base_min_score`` +
   ``regime_intensity_z`` for downstream observability.

See ``~/Development/alpha-engine-docs/private/regime-v3-260514.md``
§6 Stage D' Wire 5 for the architectural framing.
"""
from __future__ import annotations

import pytest

from executor.risk_guard import (
    check_order,
    regime_conditional_min_score,
)


# ─────────────────────────────────────────────────────────────────────
# regime_conditional_min_score — pure math
# ─────────────────────────────────────────────────────────────────────


class TestRegimeConditionalMinScore:
    def test_none_returns_base(self):
        """Substrate-unavailable path: None → base (legacy preserved)."""
        assert regime_conditional_min_score(None, base_min_score=70) == 70.0

    def test_zero_intensity_returns_base(self):
        """Neutral regime (z=0) → base (no adjustment)."""
        assert regime_conditional_min_score(0.0, base_min_score=70) == 70.0

    def test_negative_z_raises_threshold(self):
        """Risk-off (z<0) → threshold goes UP (more selective)."""
        # z=-1, scale=2.0 → adjustment = +2.0; base 70 → 72
        assert regime_conditional_min_score(-1.0, base_min_score=70) == 72.0

    def test_positive_z_lowers_threshold(self):
        """Risk-on (z>0) → threshold goes DOWN (more inclusive)."""
        # z=+1, scale=2.0 → adjustment = -2.0; base 70 → 68
        assert regime_conditional_min_score(1.0, base_min_score=70) == 68.0

    def test_capped_to_max_adjustment(self):
        """Extreme negative z → adjustment clamped to +cap (default 10.0)."""
        # z=-100 → raw adj +200, capped to +10; base 70 → 80
        assert regime_conditional_min_score(-100.0, base_min_score=70) == 80.0

    def test_capped_to_min_adjustment(self):
        """Extreme positive z → adjustment clamped to -cap (default -10.0)."""
        # z=+100 → raw adj -200, capped to -10; base 70 → 60
        assert regime_conditional_min_score(100.0, base_min_score=70) == 60.0

    def test_clamped_to_floor(self):
        """Even within cap range, hard floor=50.0 prevents disabling gate."""
        # Cap doesn't bite here — but floor does
        # base=55, z=-100 → cap +10 → 65 (no floor issue)
        # base=45, z=-100 → cap +10 → 55 (no floor)
        # base=45, z=+0 → adj 0 → 45 → clamped UP to 50
        assert regime_conditional_min_score(0.0, base_min_score=45) == 50.0

    def test_clamped_to_ceil(self):
        """Hard ceiling=90.0 prevents unreachable threshold."""
        # base=95, z=0 → 95 → clamped DOWN to 90
        assert regime_conditional_min_score(0.0, base_min_score=95) == 90.0

    def test_custom_scale(self):
        """``scale`` parameter controls per-σ sensitivity."""
        # z=-1, scale=5.0 → adjustment +5; base 70 → 75
        assert regime_conditional_min_score(
            -1.0, base_min_score=70, scale=5.0,
        ) == 75.0

    def test_custom_cap(self):
        """``cap`` parameter tightens adjustment clamp."""
        # z=-100 → raw +200; cap=3 → +3; base 70 → 73
        assert regime_conditional_min_score(
            -100.0, base_min_score=70, cap=3.0,
        ) == 73.0

    def test_custom_floor_and_ceil(self):
        """``floor`` and ``ceil`` parameters override the default range."""
        # base=70, z=0, floor=72 → clamped UP to 72
        assert regime_conditional_min_score(
            0.0, base_min_score=70, floor=72.0,
        ) == 72.0
        # base=70, z=0, ceil=68 → clamped DOWN to 68
        assert regime_conditional_min_score(
            0.0, base_min_score=70, ceil=68.0,
        ) == 68.0


# ─────────────────────────────────────────────────────────────────────
# check_order integration — Wire 5 gate behavior
# ─────────────────────────────────────────────────────────────────────


def _base_config(**overrides):
    """Minimal config that lets all rules below the score gate pass for
    the synthetic ticker — they're not the subject of these tests."""
    cfg = {
        "min_score_to_enter": 70,
        "max_position_pct": 0.10,
        "bear_max_position_pct": 0.10,
        "max_sector_pct": 0.50,
        "max_equity_pct": 0.95,
        "drawdown_circuit_breaker": 0.50,
        "correlation_block_enabled": False,
        "strategy": {
            "graduated_drawdown": {
                "enabled": False,
                "tiers": [],
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _enter_signal(score=80):
    return {
        "score": score,
        "conviction": "stable",
        "price_target_upside": 0.15,
        "signal": "ENTER",
    }


class TestWire5DisabledByDefault:
    """Without ``regime_min_score_enabled``, intensity_z has zero effect."""

    def test_score_at_base_threshold_passes(self):
        config = _base_config()  # min_score 70, wire OFF
        approved, _ = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="neutral",
            signal=_enter_signal(score=70),
            config=config,
            regime_intensity_z=-3.0,  # would raise threshold if enabled
        )
        # Wire off: 70 >= 70 passes
        assert approved is True

    def test_negative_z_does_not_tighten_when_flag_off(self):
        config = _base_config()
        approved, reason = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=71),  # would FAIL if z raised threshold above 71
            config=config,
            regime_intensity_z=-3.0,
        )
        assert approved is True


class TestWire5EnabledRiskOffRaisesThreshold:
    """Flag ON + risk-off → score gate fires at HIGHER threshold."""

    def test_risk_off_raises_threshold_above_base(self):
        """Score 71 passes base 70 but fails regime-adjusted 76 (z=-3)."""
        config = _base_config(regime_min_score_enabled=True)
        approved, reason = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=71),
            config=config,
            regime_intensity_z=-3.0,
        )
        # z=-3, scale=2.0 → +6; base 70 → 76. 71 < 76 → veto.
        assert approved is False
        assert "minimum 76" in reason

    def test_risk_off_score_clears_higher_threshold(self):
        """Score 80 passes even the risk-off-tightened threshold."""
        config = _base_config(regime_min_score_enabled=True)
        approved, _ = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=80),
            config=config,
            regime_intensity_z=-3.0,
        )
        # 80 >= 76 (adjusted) → passes
        assert approved is True


class TestWire5EnabledRiskOnLowersThreshold:
    """Flag ON + risk-on → score gate fires at LOWER threshold."""

    def test_risk_on_admits_score_below_base(self):
        """Score 67 fails base 70 but passes regime-adjusted 64 (z=+3)."""
        config = _base_config(regime_min_score_enabled=True)
        approved, _ = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bull",
            signal=_enter_signal(score=67),
            config=config,
            regime_intensity_z=3.0,
        )
        # z=+3, scale=2.0 → -6; base 70 → 64. 67 >= 64 → passes.
        assert approved is True

    def test_risk_on_still_rejects_far_below(self):
        """Risk-on doesn't open the floodgates — score 50 still fails."""
        config = _base_config(regime_min_score_enabled=True)
        approved, reason = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bull",
            signal=_enter_signal(score=50),
            config=config,
            regime_intensity_z=3.0,
        )
        # 50 < 64 (adjusted) → veto
        assert approved is False


class TestWire5NoneFallsBackToBase:
    """Wire enabled but substrate read returned None → use base unchanged."""

    def test_z_none_uses_base_threshold(self):
        config = _base_config(regime_min_score_enabled=True)
        # Score 70 should pass base 70 when z=None falls through
        approved, _ = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="neutral",
            signal=_enter_signal(score=70),
            config=config,
            regime_intensity_z=None,
        )
        assert approved is True


class TestWire5StructuredEventContext:
    """min_score veto events carry the base + regime context."""

    def test_event_includes_base_and_intensity_z(self):
        config = _base_config(regime_min_score_enabled=True)
        events: list[dict] = []
        check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=71),
            config=config,
            events=events,
            regime_intensity_z=-3.0,
        )
        vetoes = [e for e in events if e.get("rule") == "min_score"]
        assert len(vetoes) == 1
        ctx = vetoes[0]["context"]
        assert ctx["base_min_score"] == 70.0
        assert ctx["regime_intensity_z"] == -3.0
        # Threshold stamped on event is the SCALED value, not the raw base
        assert vetoes[0]["threshold"] == 76.0

    def test_event_z_none_when_substrate_missing(self):
        """Event context records z=None when substrate is unavailable."""
        config = _base_config(regime_min_score_enabled=True)
        events: list[dict] = []
        check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="neutral",
            signal=_enter_signal(score=50),  # fails base 70
            config=config,
            events=events,
            regime_intensity_z=None,
        )
        vetoes = [e for e in events if e.get("rule") == "min_score"]
        assert len(vetoes) == 1
        assert vetoes[0]["context"]["regime_intensity_z"] is None
        assert vetoes[0]["threshold"] == 70.0  # base, unchanged


class TestWire5ConfigParamsFlow:
    """Custom scale/cap/floor/ceil from config flow through to helper."""

    def test_custom_scale_amplifies_adjustment(self):
        """At z=-2 with scale=5.0, adjustment = +10 (base 70 → 80)."""
        config = _base_config(
            regime_min_score_enabled=True,
            regime_min_score_scale=5.0,
        )
        approved, reason = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=79),
            config=config,
            regime_intensity_z=-2.0,
        )
        # Adjusted threshold = 80; 79 < 80 → veto
        assert approved is False
        assert "minimum 80" in reason

    def test_custom_cap_limits_adjustment(self):
        """At z=-100 with cap=3, adjustment is +3 (base 70 → 73)."""
        config = _base_config(
            regime_min_score_enabled=True,
            regime_min_score_cap=3.0,
        )
        approved, _ = check_order(
            ticker="AAPL",
            action="ENTER",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=74),
            config=config,
            regime_intensity_z=-100.0,
        )
        # Adjusted threshold = 73; 74 >= 73 → passes
        assert approved is True


class TestWire5ExitsAndReducesBypassGate:
    """EXIT and REDUCE actions skip ALL risk rules including score gate."""

    def test_exit_bypasses_regime_score_gate(self):
        config = _base_config(regime_min_score_enabled=True)
        approved, _ = check_order(
            ticker="AAPL",
            action="EXIT",
            dollar_size=1000,
            portfolio_nav=100_000,
            peak_nav=100_000,
            current_positions={},
            sector="Technology",
            market_regime="bear",
            signal=_enter_signal(score=10),  # absurdly low
            config=config,
            regime_intensity_z=-3.0,
        )
        assert approved is True
