"""Tests that decide_entries accumulates structured risk events on
plan.risk_events for momentum-gate vetoes, GBM-veto overrides, and
risk_guard rule vetoes. The live shell (main._plan_entries) returns
this list to the caller for persistence via trade_logger.log_risk_event.

Phase 2 transparency-inventory — closes the *risk decisions* row.
"""
from __future__ import annotations

import pandas as pd
import pytest

from executor.deciders import decide_entries


def _df_history(n_bars: int = 100, base: float = 100.0,
                trend: float = 0.1) -> pd.DataFrame:
    """Construct a synthetic OHLCV history with configurable trend.

    `trend` of 0.1 = +0.1/bar (uptrend, momentum gate passes).
    `trend` of -0.5 = aggressive downtrend (momentum gate fails).
    """
    return pd.DataFrame(
        {
            "open":  [base + i * trend for i in range(n_bars)],
            "high":  [base + i * trend + 0.5 for i in range(n_bars)],
            "low":   [base + i * trend - 0.5 for i in range(n_bars)],
            "close": [base + i * trend + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
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
        "earnings_proximity_warning_days": 2,
        "momentum_gate_enabled": False,
        "momentum_gate_threshold": -5.0,
        "atr_sizing_enabled": True,
        "correlation_block_enabled": False,
        "coverage_sizing_enabled": False,
        "reduce_fraction": 0.50,
        "strategy": {
            "graduated_drawdown": {
                "enabled": True,
                "tiers": [(-0.02, 1.00, "tier1"), (-0.04, 0.50, "tier2")],
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _strategy_config():
    return {
        "intraday_pullback_atr_multiple": 1.0,
        "intraday_vwap_discount_pct": 0.005,
        "intraday_support_lookback_days": 20,
        "drawdown_forced_exit_enabled": False,
    }


def _make_inputs(*, signals_date: str, run_date: str, enter_signals: list[dict],
                 predictions_by_ticker: dict | None = None,
                 config_overrides: dict | None = None,
                 price_histories: dict | None = None):
    sig_payload = {
        "date": signals_date,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "overweight"},
                           "Defensives": {"rating": "underweight"}},
        "enter": enter_signals,
        "exit": [], "reduce": [], "hold": [],
        "universe": enter_signals,
        "buy_candidates": enter_signals,
    }
    tickers = sorted({s["ticker"] for s in enter_signals} | {
        "SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE",
        "XLU", "XLRE", "XLB", "XLI", "XLC",
    })
    return {
        "enter_signals": enter_signals,
        "signals_raw": sig_payload,
        "predictions_by_ticker": predictions_by_ticker or {},
        "config": _base_config(**(config_overrides or {})),
        "strategy_config": _strategy_config(),
        "market_regime": "neutral",
        "sector_ratings": sig_payload["sector_ratings"],
        "portfolio_nav": 1_000_000.0,
        "peak_nav": 1_000_000.0,
        "current_positions": {},
        "prices_now": {t: 100.0 for t in tickers},
        "price_histories": price_histories or {
            t: _df_history(base=100 + i) for i, t in enumerate(tickers)
        },
        "atr_map": {t: 0.02 for t in tickers},
        "vwap_map": {t: 100.0 for t in tickers},
        "coverage_map": {t: 1.0 for t in tickers},
        "dd_multiplier": 1.0,
        "signal_age_days": 0,
        "earnings_by_ticker": {},
        "run_date": run_date,
    }


# ── Approval path ────────────────────────────────────────────────────────────


def test_approved_enter_emits_no_risk_events():
    enter = [{"ticker": "NVDA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 1
    assert plan.risk_events == []


# ── Momentum-gate veto ───────────────────────────────────────────────────────


def test_momentum_gate_emits_veto_event():
    enter = [{"ticker": "NVDA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    # Aggressive downtrend pushes 20d momentum well below the -5% gate.
    bad_history = _df_history(trend=-0.5)
    histories = {"NVDA": bad_history}
    # Pad in flat histories for ETFs so price_histories isn't sparse.
    for t in ("SPY", "XLK", "XLV", "XLF", "XLY"):
        histories[t] = _df_history()
    inputs = _make_inputs(
        signals_date="2026-05-06", run_date="2026-05-06",
        enter_signals=enter,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
        price_histories=histories,
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 0
    assert len(plan.risk_events) == 1
    ev = plan.risk_events[0]
    assert ev["event_type"] == "veto"
    assert ev["rule"] == "momentum_gate"
    assert ev["ticker"] == "NVDA"
    assert ev["signal_date"] == "2026-05-06"
    assert ev["prediction_date"] == "2026-05-06"


# ── Predictor-emitted momentum_veto (PR alpha-engine-predictor#136) ─────────


def test_predictor_momentum_veto_blocks_entry_and_records_source():
    """When the predictor emits ``momentum_veto=True`` on a ticker,
    the executor must skip the entry and tag the risk event with
    ``veto_source='predictor'``. Replaces the executor's inline
    momentum_gate computation (which becomes the fallback path)."""
    enter = [{"ticker": "WING", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Consumer Discretionary",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"WING": {
        # Predictor saw -27.9% over 20d → emits the veto flag.
        # Executor consumes the boolean; the inline price-history path
        # is bypassed even if price_histories has the same ticker.
        "momentum_veto": True,
        "momentum_20d": -0.279,
        "predicted_alpha": 0.01,
        "predicted_direction": "UP",  # alpha-positive but momentum says no
        "prediction_confidence": 0.55,
        "combined_rank": 5,
        "gbm_veto": False,
    }}
    # Flat history that on its own would NOT trip the executor inline
    # gate — pins that the predictor's flag is authoritative.
    histories = {"WING": _df_history(trend=0.1)}
    for t in ("SPY", "XLK", "XLV", "XLF", "XLY"):
        histories[t] = _df_history(trend=0.1)
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
        price_histories=histories,
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    assert plan.n_entered == 0, "predictor momentum_veto must block entry"
    veto_events = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(veto_events) == 1
    ev = veto_events[0]
    assert ev["event_type"] == "veto"
    assert ev["ticker"] == "WING"
    assert ev["veto_source"] == "predictor", (
        "Risk event must record that the predictor authored the veto — "
        "backtester attribution depends on this field to split"
        " predictor-vs-executor contribution to the veto distribution."
    )
    assert ev["value"] == pytest.approx(-0.279)


def test_predictor_momentum_veto_false_does_not_block_even_on_bad_history():
    """If the predictor emits ``momentum_veto=False``, the executor must
    trust the predictor and skip the inline-history fallback. Pinned so
    a future refactor doesn't accidentally re-introduce the fallback
    when the predictor has already weighed in."""
    enter = [{"ticker": "AAPL", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"AAPL": {
        # Predictor saw the 20d return + macro context and decided NOT
        # to veto (perhaps because sector ETF is up while individual
        # ticker is mildly down — relative strength).
        "momentum_veto": False,
        "momentum_20d": -0.07,  # would trip inline -5% gate
        "predicted_alpha": 0.01,
        "predicted_direction": "UP",
        "prediction_confidence": 0.65,
        "combined_rank": 5,
        "gbm_veto": False,
    }}
    # Bad history that WOULD trip the inline gate if we ran it.
    histories = {"AAPL": _df_history(trend=-0.5)}
    for t in ("SPY", "XLK", "XLV", "XLF", "XLY"):
        histories[t] = _df_history(trend=0.1)
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
        price_histories=histories,
    )
    plan = decide_entries(predictions_date="2026-05-02", **inputs)
    veto_events = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert veto_events == [], (
        "Predictor said momentum_veto=False — executor must NOT run the "
        "inline fallback as a second-opinion override"
    )


def test_executor_inline_fallback_when_predictor_field_absent():
    """During the rollout transition (predictor PR shipped, executor PR
    being deployed), predictions may not yet carry ``momentum_veto``.
    Pin the fallback to the executor's inline computation so today's
    entries aren't silently un-gated.

    After ~1 trading week of predictor PR being live (every prediction
    carries the field), this fallback can be removed."""
    enter = [{"ticker": "TSLA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    # No predictions_by_ticker entry → predictor field absent.
    histories = {"TSLA": _df_history(trend=-0.5)}
    for t in ("SPY", "XLK", "XLV", "XLF", "XLY"):
        histories[t] = _df_history(trend=0.1)
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
        price_histories=histories,
    )
    plan = decide_entries(predictions_date="2026-05-02", **inputs)
    assert plan.n_entered == 0
    veto_events = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(veto_events) == 1
    ev = veto_events[0]
    assert ev["veto_source"] == "executor_fallback", (
        "When predictor lacks momentum_veto field, executor must fall "
        "through to inline computation and tag the source for attribution"
    )


def test_predictor_momentum_veto_with_missing_momentum_20d_logs_unknown():
    """Defensive: predictor may emit ``momentum_veto=True`` but omit
    ``momentum_20d`` (unlikely but tolerated). The veto still fires;
    the diagnostic value is unset rather than crashing."""
    enter = [{"ticker": "MDT", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Health Care",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"MDT": {
        "momentum_veto": True,
        # momentum_20d omitted
        "predicted_alpha": 0.01,
        "predicted_direction": "UP",
        "prediction_confidence": 0.55,
        "combined_rank": 5,
        "gbm_veto": False,
    }}
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
    )
    plan = decide_entries(predictions_date="2026-05-02", **inputs)
    veto_events = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(veto_events) == 1
    ev = veto_events[0]
    assert ev["veto_source"] == "predictor"
    assert ev["value"] is None  # no diagnostic value to record


# ── Stance-conditional gating (stance arc PR 3) ─────────────────────────────


def _pred_with_stance(
    stance: str,
    momentum_20d: float | None = None,
    catalyst_date: str | None = None,
    **extra,
) -> dict:
    """Build a minimal prediction payload with stance fields for testing."""
    base = {
        "stance": stance,
        "stance_loadings": {
            "momentum": 0.25, "value": 0.25,
            "quality": 0.25, "catalyst": 0.25,
        },
        "catalyst_date": catalyst_date,
        "predicted_alpha": 0.01,
        "predicted_direction": "UP",
        "prediction_confidence": 0.65,
        "combined_rank": 5,
        "gbm_veto": False,
        "momentum_veto": False,
    }
    if momentum_20d is not None:
        base["momentum_20d"] = momentum_20d
    base.update(extra)
    return base


# ── Uniform momentum gating (de-stanced — L4565) ──
# value & quality are NO LONGER special-cased: the predictor's momentum_veto
# applies uniformly, and a reversal-confirmation gate blocks names whose
# momentum is still deteriorating. "Don't treat value differently."


def test_value_stance_blocked_uniformly_when_momentum_veto():
    """DE-STANCED (L4565): a value pick the predictor flags
    momentum_veto=True (a falling knife, e.g. COIN) is now BLOCKED by the
    uniform momentum gate. The prior value bypass is retired — there is no
    longer a stance_gate rule."""
    enter = [{"ticker": "COIN", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Financials",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"COIN": _pred_with_stance(
        "value", momentum_20d=-0.15, momentum_veto=True)}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    momentum_vetos = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(momentum_vetos) == 1
    assert momentum_vetos[0]["veto_source"] == "predictor"
    # stance_gate rule is retired
    assert [e for e in plan.risk_events if e["rule"] == "stance_gate"] == []
    assert plan.n_entered == 0


def test_value_stance_passes_when_no_veto_and_momentum_confirmed():
    """A value pick the predictor does NOT veto (momentum_veto=False) with
    non-bearish momentum_confirmation passes uniformly — value isn't
    penalized, it's just not given a bypass."""
    enter = [{"ticker": "AAPL", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"AAPL": _pred_with_stance(
        "value", momentum_20d=0.02, momentum_veto=False,
        momentum_confirmation=0.10)}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    gated = [e for e in plan.risk_events
             if e["rule"] in ("stance_gate", "momentum_gate", "reversal_confirmation")]
    assert gated == [], f"value should pass uniformly; got {gated}"


def test_quality_stance_blocked_uniformly_when_momentum_veto():
    """DE-STANCED (L4565): quality no longer gets a relaxed threshold — a
    quality name the predictor vetoes is blocked uniformly via the
    momentum gate, not a stance_gate."""
    enter = [{"ticker": "PG", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Consumer Staples",
              "rating": "BUY", "price_target_upside": 0.10}]
    predictions = {"PG": _pred_with_stance(
        "quality", momentum_20d=-0.10, momentum_veto=True)}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    momentum_vetos = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(momentum_vetos) == 1
    assert [e for e in plan.risk_events if e["rule"] == "stance_gate"] == []


def test_reversal_confirmation_blocks_deteriorating_momentum():
    """L4565: even when momentum_veto is False, a name whose
    momentum_confirmation is still bearish (< threshold) is blocked by the
    uniform reversal-confirmation gate — 'wait for the bounce.'"""
    enter = [{"ticker": "XYZ", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"XYZ": _pred_with_stance(
        "value", momentum_20d=-0.03, momentum_veto=False,
        momentum_confirmation=-0.12)}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True,
                          "reversal_confirmation_enabled": True,
                          "reversal_confirmation_threshold": 0.0},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    rc_vetos = [e for e in plan.risk_events if e["rule"] == "reversal_confirmation"]
    assert len(rc_vetos) == 1
    assert rc_vetos[0]["value"] == pytest.approx(-0.12)
    assert plan.n_entered == 0


# ── Catalyst stance: skip momentum, require date ──


def test_catalyst_stance_requires_catalyst_date():
    """catalyst stance is event-driven. Without catalyst_date the
    position has no exit boundary — the executor's future catalyst
    gate hard-exits at catalyst_date + 3 trading days. No date = no
    exit boundary = block."""
    enter = [{"ticker": "NVDA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"NVDA": _pred_with_stance("catalyst", catalyst_date=None)}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    stance_vetos = [e for e in plan.risk_events if e["rule"] == "stance_gate"]
    assert len(stance_vetos) == 1
    ev = stance_vetos[0]
    assert ev["stance"] == "catalyst"
    assert "catalyst_date missing" in ev["reason"]


def test_catalyst_stance_with_date_skips_momentum_gate():
    """catalyst stance with a valid date bypasses the momentum gate
    entirely — even severe drawdowns are acceptable if there's an
    event-driven thesis with a defined exit boundary."""
    enter = [{"ticker": "MRNA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Health Care",
              "rating": "BUY", "price_target_upside": 0.30}]
    # Down 25% — would trip ANY non-catalyst stance gate
    predictions = {"MRNA": _pred_with_stance(
        "catalyst", momentum_20d=-0.25, catalyst_date="2026-06-15",
    )}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True,
                          "momentum_gate_threshold": -5.0},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    stance_vetos = [e for e in plan.risk_events if e["rule"] == "stance_gate"]
    momentum_vetos = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert stance_vetos == [], (
        f"catalyst stance with valid date should pass; got {stance_vetos}"
    )
    assert momentum_vetos == [], (
        f"catalyst stance must skip the momentum gate; got {momentum_vetos}"
    )


# ── Momentum stance + None / legacy ──


def test_momentum_stance_uses_standard_gate():
    """momentum stance pins down the trend-following branch — same
    behavior as the legacy non-stance path. Predictor's momentum_veto
    (if True) blocks; otherwise passes."""
    enter = [{"ticker": "TSLA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    # momentum_veto=True from predictor → existing gate fires
    predictions = {"TSLA": _pred_with_stance(
        "momentum", momentum_20d=-0.10, momentum_veto=True,
    )}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
        config_overrides={"momentum_gate_enabled": True},
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    # momentum stance falls through to the existing momentum_gate
    # rule path (not the stance_gate rule path)
    momentum_vetos = [e for e in plan.risk_events if e["rule"] == "momentum_gate"]
    assert len(momentum_vetos) == 1
    assert momentum_vetos[0]["veto_source"] == "predictor"


def test_stance_none_is_legacy_behavior():
    """stance=None (pre-predictor#137 artifacts) must NOT trip the
    stance-conditional branches — falls through to the existing
    momentum_veto path. Pinned so the rollout transition doesn't
    block legacy predictions."""
    enter = [{"ticker": "AAPL", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    # No stance field at all — legacy prediction
    predictions = {"AAPL": {
        "predicted_alpha": 0.02,
        "predicted_direction": "UP",
        "prediction_confidence": 0.65,
        "combined_rank": 5,
        "gbm_veto": False,
        "momentum_veto": False,
        "momentum_20d": 0.08,
    }}
    inputs = _make_inputs(
        signals_date="2026-05-08", run_date="2026-05-11",
        enter_signals=enter,
        predictions_by_ticker=predictions,
    )
    plan = decide_entries(predictions_date="2026-05-11", **inputs)
    stance_vetos = [e for e in plan.risk_events if e["rule"] == "stance_gate"]
    assert stance_vetos == [], (
        f"stance=None must NOT trip the stance_gate branches; got {stance_vetos}"
    )


# ── Predictor GBM veto override ─────────────────────────────────────────────


def test_gbm_veto_emits_override_event():
    enter = [{"ticker": "TSLA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    predictions = {"TSLA": {
        "gbm_veto": True,
        "predicted_alpha": -0.018,
        "predicted_direction": "DOWN",
        "prediction_confidence": 0.72,
        "combined_rank": 412,
    }}
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
        predictions_by_ticker=predictions,
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 0
    assert len(plan.risk_events) == 1
    ev = plan.risk_events[0]
    assert ev["event_type"] == "override"
    assert ev["rule"] == "predictor_gbm_veto"
    assert ev["ticker"] == "TSLA"
    assert ev["value"] == -0.018
    assert ev["context"]["predicted_direction"] == "DOWN"
    assert ev["context"]["combined_rank"] == 412
    assert ev["signal_date"] == "2026-05-02"
    assert ev["prediction_date"] == "2026-05-06"


# ── Risk-guard rule veto threaded through ───────────────────────────────────


def test_min_score_veto_threads_through_to_plan_events():
    enter = [{"ticker": "KO", "signal": "ENTER", "score": 60,  # below 70
              "conviction": "stable", "sector": "Defensives",
              "rating": "BUY", "price_target_upside": 0.10}]
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 0
    assert len(plan.risk_events) == 1
    ev = plan.risk_events[0]
    assert ev["event_type"] == "veto"
    assert ev["rule"] == "min_score"
    assert ev["ticker"] == "KO"
    assert ev["value"] == 60
    assert ev["threshold"] == 70
    # Lineage stamped by deciders even though risk_guard didn't know.
    assert ev["signal_date"] == "2026-05-02"
    assert ev["prediction_date"] == "2026-05-06"


def test_max_sector_veto_threads_through():
    enter = [{"ticker": "NVDA", "signal": "ENTER", "score": 85,
              "conviction": "rising", "sector": "Technology",
              "rating": "BUY", "price_target_upside": 0.20}]
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
    )
    # Push existing Technology exposure over the cap so the order breaches.
    inputs["current_positions"] = {
        "MSFT": {"market_value": 200_000, "sector": "Technology"},
        "GOOG": {"market_value": 50_000, "sector": "Technology"},
    }
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 0
    rules = [ev["rule"] for ev in plan.risk_events]
    assert "max_sector" in rules
    sector_ev = next(ev for ev in plan.risk_events if ev["rule"] == "max_sector")
    assert sector_ev["sector"] == "Technology"
    assert sector_ev["signal_date"] == "2026-05-02"


# ── Multi-ticker accumulation ───────────────────────────────────────────────


def test_multiple_blocked_tickers_all_emit_events():
    enter = [
        {"ticker": "KO",  "signal": "ENTER", "score": 50,  # min_score veto
         "conviction": "stable", "sector": "Defensives",
         "rating": "BUY", "price_target_upside": 0.10},
        {"ticker": "TSLA", "signal": "ENTER", "score": 85,  # gbm_veto override
         "conviction": "rising", "sector": "Technology",
         "rating": "BUY", "price_target_upside": 0.20},
    ]
    predictions = {"TSLA": {"gbm_veto": True, "predicted_alpha": -0.02,
                            "predicted_direction": "DOWN",
                            "prediction_confidence": 0.7,
                            "combined_rank": 500}}
    inputs = _make_inputs(
        signals_date="2026-05-02", run_date="2026-05-02",
        enter_signals=enter,
        predictions_by_ticker=predictions,
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.n_entered == 0
    assert len(plan.risk_events) == 2
    by_ticker = {ev["ticker"]: ev for ev in plan.risk_events}
    assert by_ticker["KO"]["rule"] == "min_score"
    assert by_ticker["KO"]["event_type"] == "veto"
    assert by_ticker["TSLA"]["rule"] == "predictor_gbm_veto"
    assert by_ticker["TSLA"]["event_type"] == "override"
