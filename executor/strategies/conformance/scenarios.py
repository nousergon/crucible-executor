"""Golden scenario fixtures for the Slot S conformance kit.

A small battery of ExitContext scenarios every conforming rule must
survive WITHOUT RAISING — rich paths (trending/crashing/split-shaped
histories) and the degenerate-but-valid inputs the live executor
produces on bad data days (no price, no history, no avg_cost, one bar).
Mirrors the spirit of the L4593 golden battery (backtester #318): known
inputs, mechanically reproducible, no network.

Scenario prices are deterministic (no randomness — conformance must be
bit-stable run to run).
"""

from __future__ import annotations

import pandas as pd

from executor.strategies.config import load_strategy_config
from executor.strategies.contract import ExitContext

RUN_DATE = "2026-06-05"
ENTRY_DATE = "2026-05-01"


def _ohlc(closes: list[float], start: str = "2026-04-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(closes))
    df = pd.DataFrame(index=idx)
    df["close"] = closes
    df["open"] = [c * 0.995 for c in closes]
    df["high"] = [c * 1.01 for c in closes]
    df["low"] = [c * 0.99 for c in closes]
    return df


def _ctx(**overrides) -> ExitContext:
    base = dict(
        ticker="TEST",
        position={"avg_cost": 100.0, "sector": "Technology", "shares": 10},
        research_action="HOLD",
        current_price=105.0,
        price_history=_ohlc([100 + 0.2 * i for i in range(45)]),
        sector_etf_histories={"XLK": _ohlc([200 + 0.1 * i for i in range(45)]),
                              "SPY": _ohlc([400 + 0.1 * i for i in range(45)])},
        config=load_strategy_config({}),  # stock defaults, no risk.yaml needed
        catalyst_date=None,
        entry_date=ENTRY_DATE,
        run_date=RUN_DATE,
        feature_lookup=None,
    )
    base.update(overrides)
    return ExitContext(**base)


def golden_scenarios() -> dict[str, ExitContext]:
    """name -> ExitContext. Every conforming rule must return a valid
    RuleOutcome (decision or none) on each, never raise."""
    steady = [100 + 0.2 * i for i in range(45)]
    crash = steady[:40] + [95.0, 88.0, 80.0, 72.0, 65.0]
    melt_up = steady[:40] + [115.0, 122.0, 130.0, 138.0, 146.0]
    split_shaped = steady[:42] + [25.4, 25.5, 25.6]  # 4:1-split-like cliff, unadjusted

    return {
        # rich paths
        "steady_uptrend": _ctx(),
        "crash_into_run_date": _ctx(price_history=_ohlc(crash), current_price=63.0),
        "melt_up_profit_zone": _ctx(price_history=_ohlc(melt_up), current_price=148.0),
        "split_shaped_history": _ctx(price_history=_ohlc(split_shaped), current_price=25.5),
        "deep_loss_vs_cost": _ctx(current_price=70.0),
        "catalyst_expired": _ctx(catalyst_date="2026-05-20",
                                 position={"avg_cost": 100.0, "sector": "Healthcare",
                                           "shares": 10, "stance": "catalyst"}),
        "research_reaffirms": _ctx(research_action="ENTER"),
        "stale_position_hold": _ctx(entry_date="2026-01-05"),
        "unknown_sector": _ctx(position={"avg_cost": 100.0, "sector": "Cryptids", "shares": 10}),
        # degenerate-but-valid (live data-gap shapes — decline, don't crash)
        "no_current_price": _ctx(current_price=None),
        "no_price_history": _ctx(price_history=None),
        "one_bar_history": _ctx(price_history=_ohlc([100.0])),
        "no_avg_cost": _ctx(position={"avg_cost": None, "sector": "Technology", "shares": 10}),
        "no_entry_date": _ctx(entry_date=None),
        "no_etf_histories": _ctx(sector_etf_histories=None),
        "empty_config_uses_rule_defaults": _ctx(config={}),
    }
