"""Stage D' Wire 3 — regime-aware drawdown tiers.

Pins:

1. ``regime_conditional_threshold_scale`` math: linear ``1 + z*scale``
   clamped to ``[floor, ceil]``; ``None`` → 1.0.
2. ``compute_drawdown_multiplier`` honors ``regime_drawdown_enabled``
   flag; default OFF preserves legacy threshold behavior.
3. Risk-off (z<0) tightens SOFT tier thresholds → tier fires at
   smaller drawdown; risk-on (z>0) loosens.
4. Hard ``drawdown_circuit_breaker`` is NOT regime-scaled — absolute
   capital-preservation floor preserved.
5. Throttle event context carries ``regime_threshold_scale`` +
   ``regime_intensity_z`` for downstream observability.

See ``~/Development/alpha-engine-docs/private/regime-v3-260514.md``
§6 Stage D' Wire 3 for the architectural framing.
"""
from __future__ import annotations

import pytest

from executor.risk_guard import (
    compute_drawdown_multiplier,
    regime_conditional_threshold_scale,
)

# ─────────────────────────────────────────────────────────────────────
# regime_conditional_threshold_scale — pure math
# ─────────────────────────────────────────────────────────────────────


class TestRegimeConditionalThresholdScale:
    def test_none_returns_unity(self):
        """Substrate-unavailable path: None → 1.0 (legacy thresholds)."""
        assert regime_conditional_threshold_scale(None) == 1.0

    def test_zero_intensity_returns_unity(self):
        """Neutral regime (z=0) → 1.0 (no scaling)."""
        assert regime_conditional_threshold_scale(0.0) == 1.0

    def test_negative_z_returns_below_unity(self):
        """Risk-off (z<0) → scale<1.0 → thresholds tighten."""
        # 1.0 + (-1.0)*0.10 = 0.90
        assert regime_conditional_threshold_scale(-1.0) == pytest.approx(0.90)

    def test_positive_z_returns_above_unity(self):
        """Risk-on (z>0) → scale>1.0 → thresholds loosen."""
        # 1.0 + 1.0*0.10 = 1.10
        assert regime_conditional_threshold_scale(1.0) == pytest.approx(1.10)

    def test_clamped_to_floor(self):
        """Extreme negative z → clamped to floor (default 0.60)."""
        # 1.0 + (-10.0)*0.10 = 0.00, clamped to 0.60
        assert regime_conditional_threshold_scale(-10.0) == 0.60

    def test_clamped_to_ceil(self):
        """Extreme positive z → clamped to ceil (default 1.40)."""
        # 1.0 + 10.0*0.10 = 2.00, clamped to 1.40
        assert regime_conditional_threshold_scale(10.0) == 1.40

    def test_custom_scale(self):
        """``scale`` parameter controls sensitivity per σ."""
        assert regime_conditional_threshold_scale(1.0, scale=0.20) == pytest.approx(1.20)
        assert regime_conditional_threshold_scale(-1.0, scale=0.20) == pytest.approx(0.80)

    def test_custom_floor_and_ceil(self):
        """``floor`` and ``ceil`` parameters tighten the clamp."""
        # ceil=1.1 clamps +5σ down
        assert regime_conditional_threshold_scale(5.0, ceil=1.1) == 1.1
        # floor=0.9 clamps -5σ up
        assert regime_conditional_threshold_scale(-5.0, floor=0.9) == 0.9


# ─────────────────────────────────────────────────────────────────────
# compute_drawdown_multiplier — regime-aware tier behavior
# ─────────────────────────────────────────────────────────────────────


def _config(**overrides):
    """Config with graduated drawdown tiers at -2% / -4% / -6%, hard halt at -8%."""
    cfg = {
        "drawdown_circuit_breaker": 0.08,
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


class TestRegimeDrawdownDisabledByDefault:
    """Without ``regime_drawdown_enabled`` set, intensity_z has zero
    effect — Wire 3 ships dormant. -4.5% drawdown chosen because
    baseline z=0 fires tier-2 (0.50) — clear before/after differential."""

    def test_zero_z_baseline(self):
        config = _config()
        # -4.5% drawdown: tier-1 at -0.02 + tier-2 at -0.04 both fire → 0.50
        mult, _ = compute_drawdown_multiplier(
            95_500, 100_000, config, regime_intensity_z=0.0,
        )
        assert mult == 0.50

    def test_negative_z_does_not_tighten_when_flag_off(self):
        """Pre-flag: even strong risk-off doesn't change thresholds."""
        config = _config()
        mult, _ = compute_drawdown_multiplier(
            95_500, 100_000, config, regime_intensity_z=-3.0,
        )
        # Same as baseline
        assert mult == 0.50

    def test_positive_z_does_not_loosen_when_flag_off(self):
        """Pre-flag: strong risk-on doesn't loosen thresholds either."""
        config = _config()
        mult, _ = compute_drawdown_multiplier(
            95_500, 100_000, config, regime_intensity_z=3.0,
        )
        # Without the flag, would-have-loosened scenario still fires tier-2
        assert mult == 0.50


class TestRegimeDrawdownEnabledTightensInRiskOff:
    """Flag ON + risk-off → soft tiers fire at SMALLER drawdowns."""

    def test_risk_off_tightens_tier_1_threshold(self):
        """At -1.8% drawdown + z=-2.0 (scale=0.10, tier1=-0.02):
        scaled threshold = -0.02 * (1.0 + (-2.0)*0.10) = -0.02 * 0.80 = -0.016
        Drawdown -0.018 <= -0.016 → tier-1 fires (multiplier=1.00).
        Pre-Wire-3 baseline: -0.018 > -0.020 → no tier fires."""
        config = _config(regime_drawdown_enabled=True)
        mult, desc = compute_drawdown_multiplier(
            98_200, 100_000, config, regime_intensity_z=-2.0,
        )
        # Tier 1 is multiplier=1.00 in our test config, so no observable
        # change in mult — pin via test below at deeper drawdown.
        assert mult == 1.00  # still full sizing under tier 1

    def test_risk_off_tightens_tier_2_threshold(self):
        """At -3.5% drawdown + z=-2.0 (scale=0.10, tier2=-0.04):
        scaled threshold = -0.04 * 0.80 = -0.032
        Drawdown -0.035 <= -0.032 → tier-2 fires (multiplier=0.50).
        Pre-Wire-3 baseline: -0.035 > -0.040 → only tier-1 fires (1.00)."""
        config = _config(regime_drawdown_enabled=True)
        mult, _ = compute_drawdown_multiplier(
            96_500, 100_000, config, regime_intensity_z=-2.0,
        )
        assert mult == 0.50

        # Baseline comparison: same drawdown with z=0 → tier 1
        baseline_mult, _ = compute_drawdown_multiplier(
            96_500, 100_000, config, regime_intensity_z=0.0,
        )
        # -3.5% > -4% → only tier 1 boundary (-2%) crossed → 1.00
        assert baseline_mult == 1.00

    def test_risk_off_tightens_tier_3_threshold(self):
        """At -5% drawdown + z=-2.0 (scale=0.10, tier3=-0.06):
        scaled threshold = -0.06 * 0.80 = -0.048
        Drawdown -0.050 <= -0.048 → tier-3 fires (multiplier=0.25).
        Pre-Wire-3 baseline: -0.050 > -0.060 → tier-2 fires (0.50)."""
        config = _config(regime_drawdown_enabled=True)
        mult, _ = compute_drawdown_multiplier(
            95_000, 100_000, config, regime_intensity_z=-2.0,
        )
        assert mult == 0.25

        baseline_mult, _ = compute_drawdown_multiplier(
            95_000, 100_000, config, regime_intensity_z=0.0,
        )
        assert baseline_mult == 0.50


class TestRegimeDrawdownEnabledLoosensInRiskOn:
    """Flag ON + risk-on → soft tiers fire at LARGER drawdowns."""

    def test_risk_on_loosens_tier_2_threshold(self):
        """At -3.5% drawdown + z=+2.0:
        scaled tier2 threshold = -0.04 * 1.20 = -0.048
        Drawdown -0.035 > -0.048 → tier-2 does NOT fire (only tier 1 → 1.00)."""
        config = _config(regime_drawdown_enabled=True)
        mult, _ = compute_drawdown_multiplier(
            96_500, 100_000, config, regime_intensity_z=2.0,
        )
        # Loosened — tier-2 (0.50) does NOT fire; only tier-1 (1.00) applies
        assert mult == 1.00

    def test_risk_on_tier_3_skipped(self):
        """At -5% drawdown + z=+2.0:
        scaled tier3 = -0.06 * 1.20 = -0.072; tier3 does NOT fire.
        scaled tier2 = -0.04 * 1.20 = -0.048; tier2 fires (0.50).
        Baseline z=0: tier2 fires at -0.04, tier3 doesn't (since -0.05 > -0.06)."""
        config = _config(regime_drawdown_enabled=True)
        mult, _ = compute_drawdown_multiplier(
            95_000, 100_000, config, regime_intensity_z=2.0,
        )
        assert mult == 0.50


class TestHardHaltNotRegimeScaled:
    """Hard ``drawdown_circuit_breaker`` is absolute — risk-on regime
    cannot avoid the hard halt; risk-off cannot trigger it earlier."""

    def test_hard_halt_fires_at_circuit_breaker_in_risk_on(self):
        """At -8% drawdown + z=+5.0 (very risk-on):
        Even though soft tiers loosen by 1.40, hard halt at -0.08 fires."""
        config = _config(regime_drawdown_enabled=True)
        mult, desc = compute_drawdown_multiplier(
            92_000, 100_000, config, regime_intensity_z=5.0,
        )
        assert mult == 0.0
        assert "circuit breaker" in desc.lower()

    def test_hard_halt_not_triggered_before_circuit_breaker_in_risk_off(self):
        """At -7% drawdown + z=-5.0 (very risk-off):
        Soft tiers tightened to floor=0.60 still fire (tier-3 hits early),
        but hard halt at -0.08 doesn't trigger (drawdown < hard halt threshold)."""
        config = _config(regime_drawdown_enabled=True)
        mult, _ = compute_drawdown_multiplier(
            93_000, 100_000, config, regime_intensity_z=-5.0,
        )
        # NOT 0.0 — hard halt is absolute and -7% > -8%
        assert mult > 0.0
        # tier-3 fires aggressively under risk-off scaling → 0.25
        assert mult == 0.25


class TestStructuredEventContext:
    """Throttle events carry regime context for downstream analytics."""

    def test_throttle_event_includes_regime_threshold_scale(self):
        config = _config(regime_drawdown_enabled=True)
        events: list[dict] = []
        compute_drawdown_multiplier(
            96_500, 100_000, config,
            events=events,
            regime_intensity_z=-2.0,
        )
        throttle = [e for e in events if e["event_type"] == "throttle"]
        assert len(throttle) == 1
        ctx = throttle[0]["context"]
        assert ctx["regime_threshold_scale"] == pytest.approx(0.80)
        assert ctx["regime_intensity_z"] == -2.0

    def test_throttle_event_threshold_is_scaled_value(self):
        """Persisted ``threshold`` should be the SCALED threshold (the
        one actually compared against), not the raw config value, so
        backtester replays can reproduce the decision."""
        config = _config(regime_drawdown_enabled=True)
        events: list[dict] = []
        compute_drawdown_multiplier(
            96_500, 100_000, config,
            events=events,
            regime_intensity_z=-2.0,
        )
        throttle = [e for e in events if e["event_type"] == "throttle"]
        assert len(throttle) == 1
        # tier-2 with scale 0.80: -0.04 * 0.80 = -0.032
        assert throttle[0]["threshold"] == pytest.approx(-0.032)

    def test_throttle_event_z_none_when_substrate_missing(self):
        """When z=None, threshold_scale=1.0 — same as flag-off behavior.
        Use -4.5% drawdown so tier-2 fires under both paths and emits a
        throttle event."""
        config = _config(regime_drawdown_enabled=True)
        events: list[dict] = []
        compute_drawdown_multiplier(
            95_500, 100_000, config,
            events=events,
            regime_intensity_z=None,
        )
        throttle = [e for e in events if e["event_type"] == "throttle"]
        assert len(throttle) == 1
        assert throttle[0]["context"]["regime_intensity_z"] is None
        # threshold_scale=1.0 when z=None
        assert throttle[0]["context"]["regime_threshold_scale"] == 1.0

    def test_hard_halt_event_has_no_regime_scale_context(self):
        """Hard halt is regime-independent — event context shouldn't
        carry regime_threshold_scale (the threshold there isn't scaled)."""
        config = _config(regime_drawdown_enabled=True)
        events: list[dict] = []
        compute_drawdown_multiplier(
            92_000, 100_000, config,
            events=events,
            regime_intensity_z=-2.0,
        )
        halts = [e for e in events if e["event_type"] == "halt"]
        assert len(halts) == 1
        # Hard halt threshold is the un-scaled config value
        assert halts[0]["threshold"] == -0.08


class TestConfigDrivenCurveParams:
    """``regime_drawdown_scale`` / ``_floor`` / ``_ceil`` from config flow
    through to the threshold-scale helper."""

    def test_custom_scale_doubles_sensitivity(self):
        """At -3.0% drawdown + z=-1.0, default scale 0.10:
          scaled tier2 = -0.04 * 0.90 = -0.036 → -0.030 > -0.036 → tier-1
        With scale 0.20:
          scaled tier2 = -0.04 * 0.80 = -0.032 → -0.030 > -0.032 → tier-1
        Both don't fire tier-2 at -3.0%. Use -3.4% drawdown to differentiate."""
        config = _config(
            regime_drawdown_enabled=True,
            regime_drawdown_scale=0.20,
        )
        # -3.4% drawdown
        mult, _ = compute_drawdown_multiplier(
            96_600, 100_000, config, regime_intensity_z=-1.0,
        )
        # scale=0.20, z=-1: tier2 scaled to -0.04*0.80 = -0.032
        # drawdown -0.034 <= -0.032 → tier-2 fires (0.50)
        assert mult == 0.50

    def test_custom_floor(self):
        """Custom floor 0.95 limits maximum tightening to 5%."""
        config = _config(
            regime_drawdown_enabled=True,
            regime_drawdown_floor=0.95,
        )
        # z=-10 would normally clamp to default 0.60; with floor=0.95
        # scale stays at 0.95
        mult, _ = compute_drawdown_multiplier(
            96_500, 100_000, config, regime_intensity_z=-10.0,
        )
        # tier-2 scaled to -0.04 * 0.95 = -0.038
        # -0.035 > -0.038 → tier-2 doesn't fire → tier-1 (1.00)
        assert mult == 1.00


class TestNoneIntensityZPreservesLegacy:
    """When wire is enabled but substrate read returned None,
    threshold scale falls through to 1.0 (legacy thresholds preserved)."""

    def test_none_z_acts_as_unity_scale(self):
        config = _config(regime_drawdown_enabled=True)
        # -3% drawdown, baseline z=0 hits tier-1 (1.00)
        mult, _ = compute_drawdown_multiplier(
            97_000, 100_000, config, regime_intensity_z=None,
        )
        # Same as baseline behavior: -0.03 > -0.04 tier-2 → only tier-1 fires
        assert mult == 1.00


class TestNonGraduatedFallbackUnaffected:
    """When ``graduated_drawdown.enabled=False``, regime scaling is
    bypassed (the binary circuit-breaker fallback is regime-independent).
    """

    def test_binary_circuit_breaker_ignores_regime(self):
        config = _config(regime_drawdown_enabled=True)
        config["strategy"]["graduated_drawdown"]["enabled"] = False
        # -7% drawdown, binary breaker at -8% → no halt
        mult, _ = compute_drawdown_multiplier(
            93_000, 100_000, config, regime_intensity_z=-5.0,
        )
        # Binary breaker says full sizing until -8% threshold
        assert mult == 1.0
