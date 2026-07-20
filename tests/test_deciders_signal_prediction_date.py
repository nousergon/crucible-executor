"""Tests that deciders.entries_with_meta + urgent_exits_with_meta carry
artifact-filename lineage (signal_date + prediction_date).

These fields are the producer side of the Phase 2 transparency-inventory
lineage column on the trades table — daemon.py reads them off the
order book record and threads them into log_trade(). If the producer
stops emitting them, the daemon's log_trade call silently writes NULL,
which would breach the going-forward coverage gate without surfacing
as a test failure on log_trade alone.

Coverage:
  1. ENTER path — entries_with_meta carries both fields when supplied
  2. ENTER path — signal_date sources from signals_raw["date"], not run_date
     (the stale-signals fallback case where Research's Saturday SF was
     missed and signal_reader.read_signals_with_fallback walks back)
  3. ENTER path — prediction_date = None when caller doesn't supply
  4. EXIT path — urgent_exits_with_meta carries both fields
  5. REDUCE path — urgent_exits_with_meta carries both fields
"""
from __future__ import annotations

import pandas as pd

from executor.deciders import decide_entries, decide_exits_and_reduces


def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open":  [base + i * 0.1 for i in range(n_bars)],
            "high":  [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low":   [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


def _base_config():
    return {
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "bear_block_underweight": True,
        "earnings_proximity_warning_days": 2,
        "momentum_gate_enabled": False,
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


def _base_strategy_config():
    return {
        "intraday_pullback_atr_multiple": 1.0,
        "intraday_vwap_discount_pct": 0.005,
        "intraday_support_lookback_days": 20,
        "drawdown_forced_exit_enabled": False,
    }


def _build_entry_inputs(signals_filename_date: str, run_date: str):
    enter = [{
        "ticker": "NVDA",
        "signal": "ENTER",
        "score": 85,
        "conviction": "rising",
        "sector": "Technology",
        "rating": "BUY",
        "price_target_upside": 0.20,
    }]
    signals_raw = {
        "date": signals_filename_date,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "overweight"}},
        "enter": enter,
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": enter,
        "buy_candidates": enter,
    }
    tickers = ["NVDA", "SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE",
               "XLU", "XLRE", "XLB", "XLI", "XLC"]
    return {
        "enter_signals": enter,
        "signals_raw": signals_raw,
        "predictions_by_ticker": {},
        "config": _base_config(),
        "strategy_config": _base_strategy_config(),
        "market_regime": "neutral",
        "sector_ratings": signals_raw["sector_ratings"],
        "portfolio_nav": 1_000_000.0,
        "peak_nav": 1_000_000.0,
        "current_positions": {},
        "prices_now": dict.fromkeys(tickers, 100.0),
        "price_histories": {t: _df_history(base=100 + i) for i, t in enumerate(tickers)},
        "atr_map": dict.fromkeys(tickers, 0.02),
        "vwap_map": dict.fromkeys(tickers, 100.0),
        "coverage_map": dict.fromkeys(tickers, 1.0),
        "dd_multiplier": 1.0,
        "signal_age_days": 0,
        "earnings_by_ticker": {},
        "run_date": run_date,
    }


def test_entries_with_meta_carries_signal_and_prediction_date():
    inputs = _build_entry_inputs(
        signals_filename_date="2026-05-02", run_date="2026-05-02",
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.entries_with_meta, "expected at least one entry to pass gates"
    meta = plan.entries_with_meta[0]
    assert meta["signal_date"] == "2026-05-02"
    assert meta["prediction_date"] == "2026-05-06"


def test_entries_with_meta_signal_date_sources_from_signals_raw_not_run_date():
    """The stale-signals fallback case: signal_reader.read_signals_with_fallback
    walked back to a prior Saturday's signals.json when this Saturday's run
    didn't fire. signals_raw["date"] must win over run_date — otherwise the
    lineage column points at today and the artifact-traceability claim breaks.
    """
    inputs = _build_entry_inputs(
        signals_filename_date="2026-04-25",  # 11 days back — fallback hit
        run_date="2026-05-06",                # today
    )
    plan = decide_entries(predictions_date="2026-05-06", **inputs)
    assert plan.entries_with_meta
    meta = plan.entries_with_meta[0]
    assert meta["signal_date"] == "2026-04-25", (
        "signal_date must come from signals_raw['date'] (the artifact filename), "
        "not run_date — otherwise stale-fallback rows misattribute lineage."
    )


def test_entries_with_meta_prediction_date_defaults_to_none():
    """Caller path that doesn't load predictions (simulate mode, S3 miss)
    must not crash and must persist None — which log_trade then writes
    as NULL per the back-compat policy.
    """
    inputs = _build_entry_inputs(
        signals_filename_date="2026-05-02", run_date="2026-05-02",
    )
    plan = decide_entries(**inputs)  # predictions_date omitted
    assert plan.entries_with_meta
    meta = plan.entries_with_meta[0]
    assert meta["signal_date"] == "2026-05-02"
    assert meta["prediction_date"] is None


def test_urgent_exits_with_meta_carries_both_dates_for_research_exit():
    """Research-driven EXIT — both dates threaded through from the
    decide_exits_and_reduces caller. Distinct from strategy-driven
    intraday exits which fire from daemon.py without these dates.
    """
    signals = {
        "enter": [],
        "exit": [{"ticker": "AAPL", "score": 50, "conviction": "declining",
                  "rating": "SELL", "reason": "thesis_violation"}],
        "reduce": [],
        "hold": [],
        "market_regime": "neutral",
        "sector_ratings": {},
    }
    current_positions = {
        "AAPL": {"shares": 100, "avg_cost": 150.0,
                 "market_value": 15000, "sector": "Technology"},
    }
    plan = decide_exits_and_reduces(
        signals=signals,
        strategy_exits=[],
        current_positions=current_positions,
        prices_now={"AAPL": 145.0},
        predictions_by_ticker={},
        config=_base_config(),
        market_regime="neutral",
        portfolio_nav=1_000_000.0,
        run_date="2026-05-06",
        signals_date="2026-05-02",
        predictions_date="2026-05-06",
    )
    assert len(plan.urgent_exits_with_meta) == 1
    ue = plan.urgent_exits_with_meta[0]
    assert ue["signal_date"] == "2026-05-02"
    assert ue["prediction_date"] == "2026-05-06"
    assert ue["signal"] == "EXIT"


def test_urgent_exits_with_meta_carries_both_dates_for_research_reduce():
    signals = {
        "enter": [],
        "exit": [],
        "reduce": [{"ticker": "TSLA", "score": 60, "conviction": "stable",
                    "rating": "HOLD", "reason": "drawdown_tier_reduction"}],
        "hold": [],
        "market_regime": "neutral",
        "sector_ratings": {},
    }
    current_positions = {
        "TSLA": {"shares": 200, "avg_cost": 200.0,
                 "market_value": 40000, "sector": "Consumer Cyclical"},
    }
    plan = decide_exits_and_reduces(
        signals=signals,
        strategy_exits=[],
        current_positions=current_positions,
        prices_now={"TSLA": 195.0},
        predictions_by_ticker={},
        config=_base_config(),
        market_regime="neutral",
        portfolio_nav=1_000_000.0,
        run_date="2026-05-06",
        signals_date="2026-05-02",
        predictions_date="2026-05-06",
    )
    assert len(plan.urgent_exits_with_meta) == 1
    ue = plan.urgent_exits_with_meta[0]
    assert ue["signal_date"] == "2026-05-02"
    assert ue["prediction_date"] == "2026-05-06"
    assert ue["signal"] == "REDUCE"
