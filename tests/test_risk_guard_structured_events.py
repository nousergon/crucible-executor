"""Structured-event emission tests for risk_guard.check_order +
compute_drawdown_multiplier.

Phase 2 transparency-inventory: closes the *risk decisions* row.
Each rule appends one structured dict (rule + value + threshold + reason)
to the caller's `events` list. Default `events=None` preserves the
existing 2-tuple return contract — covered by test_risk_guard.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from executor.risk_guard import (
    check_correlation,
    check_order,
    compute_drawdown_multiplier,
)


def _df_from_closes(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _base_config(**overrides):
    cfg = {
        "min_score_to_enter": 70,
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "bear_block_underweight": True,
        "strategy": {
            "graduated_drawdown": {
                "enabled": True,
                "tiers": [
                    (-0.02, 1.00, "0% to -2%"),
                    (-0.04, 0.50, "-2% to -4%"),
                    (-0.06, 0.25, "-4% to -6%"),
                ],
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _base_signal(**overrides):
    sig = {
        "score": 80,
        "conviction": "stable",
        "price_target_upside": 0.15,
        "sector_rating": "market_weight",
        "signal": "ENTER",
    }
    sig.update(overrides)
    return sig


def _call(events=None, **kwargs):
    defaults = {
        "ticker": "AAPL",
        "action": "ENTER",
        "dollar_size": 4000,
        "portfolio_nav": 100_000,
        "peak_nav": 100_000,
        "current_positions": {},
        "sector": "Technology",
        "market_regime": "neutral",
        "signal": _base_signal(),
        "config": _base_config(),
        "price_histories": None,
    }
    defaults.update(kwargs)
    return check_order(**defaults, events=events)


# ── compute_drawdown_multiplier ──────────────────────────────────────────────


class TestDrawdownEventEmission:
    def test_no_drawdown_emits_no_event(self):
        events: list[dict] = []
        compute_drawdown_multiplier(100_000, 100_000, _base_config(), events=events)
        assert events == []

    def test_throttle_event_on_active_tier(self):
        """-4% drawdown hits tier 2 (mult 0.50) — emits a throttle event."""
        events: list[dict] = []
        mult, _ = compute_drawdown_multiplier(96_000, 100_000, _base_config(), events=events)
        assert mult == 0.50
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "throttle"
        assert ev["rule"] == "drawdown_tier_throttle"
        assert ev["value"] == pytest.approx(-0.04)
        assert ev["threshold"] == pytest.approx(-0.04)
        assert ev["context"]["multiplier"] == 0.50

    def test_halt_event_on_circuit_breaker(self):
        """-9% drawdown trips the circuit breaker — emits a halt event."""
        events: list[dict] = []
        mult, _ = compute_drawdown_multiplier(91_000, 100_000, _base_config(), events=events)
        assert mult == 0.0
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "halt"
        assert ev["rule"] == "drawdown_halt"
        assert ev["value"] == pytest.approx(-0.09)
        assert ev["threshold"] == pytest.approx(-0.08)

    def test_no_event_at_full_sizing_under_tier_floor(self):
        """-1% drawdown — no tier breached, no event."""
        events: list[dict] = []
        mult, _ = compute_drawdown_multiplier(99_000, 100_000, _base_config(), events=events)
        assert mult == 1.0
        assert events == []

    def test_default_events_none_preserves_legacy_contract(self):
        """No events kwarg → 2-tuple return unchanged."""
        result = compute_drawdown_multiplier(91_000, 100_000, _base_config())
        assert isinstance(result, tuple) and len(result) == 2
        mult, reason = result
        assert mult == 0.0
        assert "circuit breaker" in reason.lower()


# ── check_order rule-by-rule ─────────────────────────────────────────────────


class TestCheckOrderEventEmission:
    def test_min_score_emits_veto(self):
        events: list[dict] = []
        approved, _ = _call(events=events, signal=_base_signal(score=50))
        assert not approved
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "veto"
        assert ev["rule"] == "min_score"
        assert ev["ticker"] == "AAPL"
        assert ev["value"] == 50
        assert ev["threshold"] == 70

    def test_max_position_emits_veto(self):
        events: list[dict] = []
        approved, _ = _call(events=events, dollar_size=6000)  # 6% of 100k > 5%
        assert not approved
        ev = events[0]
        assert ev["rule"] == "max_position"
        assert ev["value"] == pytest.approx(0.06)
        assert ev["threshold"] == pytest.approx(0.05)
        assert "dollar_size" in ev["context"]

    def test_max_sector_emits_veto(self):
        events: list[dict] = []
        positions = {
            "MSFT": {"market_value": 20_000, "sector": "Technology"},
            "GOOG": {"market_value": 5_000, "sector": "Technology"},
        }
        approved, _ = _call(events=events, dollar_size=1000, current_positions=positions)
        assert not approved
        ev = events[0]
        assert ev["rule"] == "max_sector"
        assert ev["value"] == pytest.approx(0.26)
        assert ev["threshold"] == pytest.approx(0.25)
        assert ev["sector"] == "Technology"
        assert ev["context"]["existing_sector_exposure"] == 25_000

    def test_max_equity_emits_veto(self):
        events: list[dict] = []
        positions = {
            f"STOCK{i}": {"market_value": 10_000, "sector": f"Sector{i}"}
            for i in range(9)
        }
        approved, _ = _call(events=events, dollar_size=4000, current_positions=positions)
        assert not approved
        ev = events[0]
        assert ev["rule"] == "max_equity"
        assert ev["value"] == pytest.approx(0.94)
        assert ev["threshold"] == pytest.approx(0.90)

    def test_bear_underweight_emits_veto(self):
        events: list[dict] = []
        approved, _ = _call(
            events=events,
            dollar_size=1000,
            market_regime="bear",
            signal=_base_signal(sector_rating="underweight"),
        )
        assert not approved
        ev = events[0]
        assert ev["rule"] == "bear_underweight"
        assert ev["context"]["sector_rating"] == "underweight"

    def test_correlation_breach_emits_veto(self):
        history = _df_from_closes([100 + i for i in range(70)])
        positions = {"MSFT": {"market_value": 5000, "sector": "Technology"}}
        events: list[dict] = []
        approved, _ = _call(
            events=events,
            ticker="AAPL",
            dollar_size=4000,
            current_positions=positions,
            price_histories={"AAPL": history, "MSFT": history},
            config=_base_config(
                correlation_block_enabled=True,
                correlation_block_threshold=0.80,
            ),
        )
        assert not approved
        ev = events[0]
        assert ev["rule"] == "correlation"
        assert ev["value"] >= 0.80
        assert ev["threshold"] == pytest.approx(0.80)

    def test_drawdown_halt_does_not_double_emit_per_ticker(self):
        """check_order's inner compute_drawdown_multiplier call deliberately
        does NOT propagate `events` — the portfolio-level halt is logged
        once at the planner top, not N times per ticker."""
        events: list[dict] = []
        approved, _ = _call(events=events, portfolio_nav=91_000, peak_nav=100_000)
        assert not approved
        # No drawdown_halt event from the per-ticker call
        rules = [ev["rule"] for ev in events]
        assert "drawdown_halt" not in rules

    def test_approved_enter_emits_no_events(self):
        events: list[dict] = []
        approved, _ = _call(events=events)
        assert approved
        assert events == []

    def test_exit_action_emits_no_events(self):
        events: list[dict] = []
        approved, _ = _call(events=events, action="EXIT")
        assert approved
        assert events == []

    def test_first_failure_short_circuits(self):
        """Score gate fires before max_position even when both would fail."""
        events: list[dict] = []
        approved, _ = _call(
            events=events,
            signal=_base_signal(score=50),
            dollar_size=6000,  # would also breach max_position
        )
        assert not approved
        assert len(events) == 1
        assert events[0]["rule"] == "min_score"

    def test_default_events_none_preserves_legacy_contract(self):
        """No events kwarg → 2-tuple return unchanged."""
        result = check_order(
            ticker="AAPL", action="ENTER", dollar_size=4000,
            portfolio_nav=100_000, peak_nav=100_000, current_positions={},
            sector="Technology", market_regime="neutral",
            signal=_base_signal(score=50), config=_base_config(),
        )
        assert isinstance(result, tuple) and len(result) == 2
        approved, reason = result
        assert not approved
        assert "Score" in reason


# ── check_correlation event emission (standalone) ────────────────────────────


class TestCheckCorrelationEvents:
    def test_breach_emits_event_with_per_ticker_context(self):
        history = _df_from_closes([100 + i for i in range(70)])
        positions = {
            "AAPL": {"sector": "Technology"},
            "MSFT": {"market_value": 5000, "sector": "Technology"},
            "GOOG": {"market_value": 5000, "sector": "Technology"},
        }
        events: list[dict] = []
        approved, _ = check_correlation(
            "AAPL", positions,
            {"AAPL": history, "MSFT": history, "GOOG": history},
            {"correlation_block_enabled": True, "correlation_lookback_days": 60,
             "correlation_block_threshold": 0.80},
            events=events,
        )
        assert not approved
        ev = events[0]
        assert ev["rule"] == "correlation"
        assert ev["sector"] == "Technology"
        # Per-ticker breakdown is preserved
        assert isinstance(ev["context"]["per_ticker"], list)
        per_t = dict(ev["context"]["per_ticker"])
        assert "MSFT" in per_t and "GOOG" in per_t

    def test_no_event_when_passing(self):
        history_a = _df_from_closes([100 + i for i in range(70)])
        history_b = _df_from_closes([100 + (-1) ** i for i in range(70)])
        events: list[dict] = []
        approved, _ = check_correlation(
            "AAPL",
            {"AAPL": {"sector": "Tech"}, "MSFT": {"market_value": 5000, "sector": "Tech"}},
            {"AAPL": history_a, "MSFT": history_b},
            {"correlation_block_enabled": True, "correlation_lookback_days": 60,
             "correlation_block_threshold": 0.80},
            events=events,
        )
        assert approved
        assert events == []
