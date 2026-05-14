"""Stage D' Wire 2 — position sizing regime multiplier.

Pins three contracts:

1. ``regime_conditional_size_multiplier`` math: linear ``1 + z*scale``
   clamped to ``[floor, ceil]``; ``None`` → 1.0.
2. ``compute_position_size`` honors ``regime_sizing_enabled`` config
   flag; default OFF (1.0× regime_adj when omitted).
3. ``read_regime_substrate`` + ``extract_intensity_z`` round-trip on
   the canonical eval-artifact payload shape; tolerate None / malformed.

See ``~/Development/alpha-engine-docs/private/regime-v3-260514.md``
§6 Stage D' Wire 2 for the architectural framing.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from executor.position_sizer import (
    compute_position_size,
    regime_conditional_size_multiplier,
)
from executor.signal_reader import (
    REGIME_SUBSTRATE_PREFIX,
    extract_intensity_z,
    read_regime_substrate,
)


# ─────────────────────────────────────────────────────────────────────
# regime_conditional_size_multiplier — pure math
# ─────────────────────────────────────────────────────────────────────


class TestRegimeConditionalSizeMultiplier:
    def test_none_returns_unity(self):
        """Substrate-unavailable path: regime_intensity_z=None → 1.0."""
        assert regime_conditional_size_multiplier(None) == 1.0

    def test_zero_intensity_returns_unity(self):
        """Neutral regime (z=0) → 1.0 (centered at neutral)."""
        assert regime_conditional_size_multiplier(0.0) == 1.0

    def test_positive_intensity_upweights(self):
        """Risk-on (z>0) → multiplier > 1.0."""
        # 1.0 + 1.0 * 0.05 = 1.05
        assert regime_conditional_size_multiplier(1.0) == pytest.approx(1.05)

    def test_negative_intensity_downweights(self):
        """Risk-off (z<0) → multiplier < 1.0."""
        # 1.0 + (-1.0) * 0.05 = 0.95
        assert regime_conditional_size_multiplier(-1.0) == pytest.approx(0.95)

    def test_clamped_to_ceil(self):
        """Extreme positive z → clamped to ceil (default 1.30)."""
        # 1.0 + 10.0 * 0.05 = 1.50, clamped to 1.30
        assert regime_conditional_size_multiplier(10.0) == 1.30

    def test_clamped_to_floor(self):
        """Extreme negative z → clamped to floor (default 0.70)."""
        # 1.0 + (-10.0) * 0.05 = 0.50, clamped to 0.70
        assert regime_conditional_size_multiplier(-10.0) == 0.70

    def test_custom_scale(self):
        """``scale`` parameter controls sensitivity per σ."""
        assert regime_conditional_size_multiplier(1.0, scale=0.10) == pytest.approx(1.10)
        assert regime_conditional_size_multiplier(-1.0, scale=0.10) == pytest.approx(0.90)

    def test_custom_floor_and_ceil(self):
        """``floor`` and ``ceil`` parameters tighten the clamp."""
        # ceil=1.1 clamps the +5σ upside down
        assert regime_conditional_size_multiplier(5.0, ceil=1.1) == 1.1
        # floor=0.9 clamps the -5σ downside up
        assert regime_conditional_size_multiplier(-5.0, floor=0.9) == 0.9

    def test_int_intensity_z_coerced_to_float(self):
        """Integer intensity_z values still work (defensive type coercion)."""
        # 1.0 + 2 * 0.05 = 1.10
        assert regime_conditional_size_multiplier(2) == pytest.approx(1.10)


# ─────────────────────────────────────────────────────────────────────
# compute_position_size integration — regime multiplier stacks in
# ─────────────────────────────────────────────────────────────────────


def _baseline_config(**overrides):
    """Minimal config that isolates regime_adj — disables all the other
    optional adjustments so test arithmetic reads cleanly."""
    cfg = {
        "max_position_pct": 0.10,
        "conviction_decline_adj": 0.70,
        "min_price_target_upside": 0.05,
        "upside_fail_adj": 0.70,
        "min_position_dollar": 0,
        "sector_adj": {
            "overweight": 1.05,
            "market_weight": 1.00,
            "underweight": 0.85,
        },
        "atr_sizing_enabled": False,
        "confidence_sizing_enabled": False,
        "staleness_discount_enabled": False,
        "earnings_sizing_enabled": False,
        "coverage_sizing_enabled": False,
        "stance_sizing_enabled": False,
    }
    cfg.update(overrides)
    return cfg


def _baseline_signal(**overrides):
    sig = {"score": 80, "conviction": "stable", "price_target_upside": 0.15}
    sig.update(overrides)
    return sig


class TestComputePositionSizeRegimeAdj:
    def test_regime_disabled_by_default(self):
        """Without ``regime_sizing_enabled`` set, regime_adj is 1.0 even
        when a non-trivial intensity_z is passed — gate is OFF by default."""
        # 25 entries → base 0.04, no other adjustments, max=0.10
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(),
            regime_intensity_z=2.0,  # would multiply by 1.10 if enabled
        )
        assert result["regime_adj"] == 1.0
        assert result["position_pct"] == pytest.approx(0.04)

    def test_regime_enabled_with_positive_z_upweights(self):
        """Risk-on regime + flag ON → position grows by scale fraction."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=1.0,
        )
        # 0.04 * 1.05 = 0.042
        assert result["regime_adj"] == pytest.approx(1.05)
        assert result["position_pct"] == pytest.approx(0.042)

    def test_regime_enabled_with_negative_z_downweights(self):
        """Risk-off regime + flag ON → position shrinks."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=-2.0,
        )
        # 1.0 + -2.0 * 0.05 = 0.90; 0.04 * 0.90 = 0.036
        assert result["regime_adj"] == pytest.approx(0.90)
        assert result["position_pct"] == pytest.approx(0.036)

    def test_regime_none_intensity_falls_back_to_unity(self):
        """Substrate unavailable (intensity_z=None) → regime_adj=1.0
        even with flag ON. Preserves legacy behavior when substrate
        loader returns None (sidecar miss, fresh deploy, etc.)."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=None,
        )
        assert result["regime_adj"] == 1.0
        assert result["position_pct"] == pytest.approx(0.04)

    def test_regime_clamped_at_ceiling_under_extreme_z(self):
        """Extreme positive intensity_z still respects the ceil clamp."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=20.0,  # would give 2.0× sans clamp
        )
        # Clamped to ceil=1.30; 0.04 * 1.30 = 0.052
        assert result["regime_adj"] == 1.30
        assert result["position_pct"] == pytest.approx(0.052)

    def test_regime_clamped_at_floor_under_extreme_negative_z(self):
        """Extreme negative intensity_z still respects the floor clamp."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=-20.0,  # would give 0.0× sans clamp
        )
        # Clamped to floor=0.70; 0.04 * 0.70 = 0.028
        assert result["regime_adj"] == 0.70
        assert result["position_pct"] == pytest.approx(0.028)

    def test_regime_stacks_multiplicatively_with_sector_adj(self):
        """regime_adj composes with sector_adj — both apply in product."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="overweight",  # 1.05x
            current_price=150.0,
            config=_baseline_config(regime_sizing_enabled=True),
            regime_intensity_z=1.0,  # 1.05x
        )
        # 0.04 * 1.05 (sector) * 1.05 (regime) = 0.0441
        assert result["sector_adj"] == 1.05
        assert result["regime_adj"] == pytest.approx(1.05)
        assert result["position_pct"] == pytest.approx(0.0441)

    def test_regime_adj_in_returned_payload(self):
        """``regime_adj`` is always emitted in the result dict — schema
        contract for downstream observability (decision_capture, etc.)."""
        enter_signals = [{"ticker": "AAPL"}]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(),
        )
        # Flag off, intensity_z not provided — schema key still present.
        assert "regime_adj" in result
        assert result["regime_adj"] == 1.0

    def test_custom_regime_curve_params_from_config(self):
        """``regime_sizing_scale`` / ``_floor`` / ``_ceil`` config keys
        flow through to the multiplier helper."""
        enter_signals = [{"ticker": f"T{i}"} for i in range(25)]
        result = compute_position_size(
            ticker="AAPL",
            portfolio_nav=100_000,
            enter_signals=enter_signals,
            signal=_baseline_signal(),
            sector_rating="market_weight",
            current_price=150.0,
            config=_baseline_config(
                regime_sizing_enabled=True,
                regime_sizing_scale=0.10,  # 2x default sensitivity
            ),
            regime_intensity_z=1.0,
        )
        # 1.0 + 1.0 * 0.10 = 1.10; 0.04 * 1.10 = 0.044
        assert result["regime_adj"] == pytest.approx(1.10)
        assert result["position_pct"] == pytest.approx(0.044)


# ─────────────────────────────────────────────────────────────────────
# extract_intensity_z — schema-defense
# ─────────────────────────────────────────────────────────────────────


class TestExtractIntensityZ:
    def test_valid_substrate_returns_float(self):
        sub = {"composite": {"intensity_z": 0.75}, "hmm": {}, "bocpd": {}}
        assert extract_intensity_z(sub) == 0.75

    def test_int_intensity_z_coerced_to_float(self):
        sub = {"composite": {"intensity_z": 1}}
        result = extract_intensity_z(sub)
        assert isinstance(result, float)
        assert result == 1.0

    def test_none_substrate_returns_none(self):
        assert extract_intensity_z(None) is None

    def test_non_dict_substrate_returns_none(self):
        assert extract_intensity_z("not a dict") is None
        assert extract_intensity_z(42) is None

    def test_missing_composite_returns_none(self):
        assert extract_intensity_z({"hmm": {}}) is None

    def test_non_dict_composite_returns_none(self):
        assert extract_intensity_z({"composite": "stringified"}) is None

    def test_missing_intensity_z_returns_none(self):
        assert extract_intensity_z({"composite": {"posterior": []}}) is None

    def test_non_numeric_intensity_z_returns_none(self):
        """Defensive — schema drift / corrupt artifact body."""
        assert extract_intensity_z({"composite": {"intensity_z": "high"}}) is None
        assert extract_intensity_z({"composite": {"intensity_z": None}}) is None

    def test_bool_intensity_z_rejected(self):
        """Python's ``isinstance(True, int)`` is True — guard against it."""
        assert extract_intensity_z({"composite": {"intensity_z": True}}) is None


# ─────────────────────────────────────────────────────────────────────
# read_regime_substrate — boto3 mocking, sidecar resolution
# ─────────────────────────────────────────────────────────────────────


class _FakeBody:
    def __init__(self, payload: dict):
        self._buf = BytesIO(json.dumps(payload).encode())

    def read(self):
        return self._buf.read()


class _FakeS3:
    def __init__(self, objects: dict[str, dict]):
        # objects: {key: payload-dict-or-Exception}
        self._objects = objects

    def get_object(self, *, Bucket, Key):  # noqa: N803 boto3 keyword
        v = self._objects.get(Key)
        if v is None:
            raise FileNotFoundError(Key)
        if isinstance(v, Exception):
            raise v
        return {"Body": _FakeBody(v)}


class TestReadRegimeSubstrate:
    def test_canonical_sidecar_resolution(self):
        """Happy path: sidecar → artifact body."""
        fake_s3 = _FakeS3({
            "regime/latest.json": {
                "artifact_key": "regime/2605120930/substrate.json"
            },
            "regime/2605120930/substrate.json": {
                "composite": {"intensity_z": 1.23},
                "hmm": {"posterior": [0.1, 0.2, 0.7]},
            },
        })
        with patch("executor.signal_reader.boto3") as mock_boto3:
            mock_boto3.client.return_value = fake_s3
            substrate = read_regime_substrate("alpha-engine-research")

        assert substrate is not None
        assert substrate["composite"]["intensity_z"] == 1.23
        # Round-trip with the helper
        assert extract_intensity_z(substrate) == 1.23

    def test_missing_sidecar_returns_none(self):
        """No latest.json → None, no exception."""
        fake_s3 = _FakeS3({})  # nothing at all
        with patch("executor.signal_reader.boto3") as mock_boto3:
            mock_boto3.client.return_value = fake_s3
            substrate = read_regime_substrate("alpha-engine-research")
        assert substrate is None

    def test_sidecar_without_artifact_key_returns_none(self):
        """Malformed sidecar → None."""
        fake_s3 = _FakeS3({
            "regime/latest.json": {"some_other_field": "x"},
        })
        with patch("executor.signal_reader.boto3") as mock_boto3:
            mock_boto3.client.return_value = fake_s3
            substrate = read_regime_substrate("alpha-engine-research")
        assert substrate is None

    def test_prefix_constant_is_regime(self):
        """Pin the S3 prefix — must match the predictor regime
        Lambda's writer location."""
        assert REGIME_SUBSTRATE_PREFIX == "regime"
