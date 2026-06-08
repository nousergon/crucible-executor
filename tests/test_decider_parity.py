"""Parity tests for executor/deciders.py (Tier 2 refactor, 2026-04-27).

Pins the invariant that calling the pure decider directly produces
byte-equal output to invoking it through the live shell wrappers
(_plan_entries / _plan_exits_and_reduces). Both code paths must stay
in lock-step — any drift between them invalidates the backtester's
parity-against-live guarantee.

If a future PR adds a side effect to the live shell that the decider
doesn't model (or vice versa), these tests catch the drift before it
reaches production.
"""
from __future__ import annotations

import pandas as pd
import pytest

from executor.deciders import (
    decide_entries,
    decide_exits_and_reduces,
)
from executor.ibkr import SimulatedIBKRClient
from executor.main import _plan_entries, _plan_exits_and_reduces
from executor.order_book import OrderBook


def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV with mild upward drift so momentum gate passes."""
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
        "momentum_gate_enabled": True,
        "momentum_gate_threshold": -50.0,  # disable so test fixture passes
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
        "drawdown_forced_exit_enabled": True,
        "drawdown_forced_exit_tier3_count": 2,
        "drawdown_forced_exit_tier2_count": 1,
    }


def _enter_signals(n: int = 3) -> list[dict]:
    return [
        {
            "ticker": f"TKR{i:03d}",
            "signal": "ENTER",
            "score": 80,
            "conviction": "rising",
            "sector": "Technology",
            "rating": "BUY",
            "price_target_upside": 0.15,
            "thesis_summary": "test",
        }
        for i in range(n)
    ]


def _build_inputs(n_signals: int = 3):
    """Construct shared inputs for both parity paths."""
    enter = _enter_signals(n_signals)
    universe = enter + [
        {"ticker": "ETHER", "signal": "HOLD", "sector": "Technology", "score": 50}
    ]
    signals_raw = {
        "date": "2026-04-25",
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "market_weight"}},
        "enter": enter,
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": universe,
        "buy_candidates": enter,
    }
    tickers = [s["ticker"] for s in enter]
    sector_etfs = ["SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
                   "XLRE", "XLB", "XLI", "XLC"]
    all_tickers = list(set(tickers + sector_etfs))
    price_histories = {t: _df_history(base=100 + i) for i, t in enumerate(all_tickers)}
    atr_map = {t: 0.02 for t in all_tickers}
    vwap_map = {t: 100.0 for t in all_tickers}
    coverage_map = {t: 1.0 for t in all_tickers}

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
        "price_histories": price_histories,
        "atr_map": atr_map,
        "vwap_map": vwap_map,
        "coverage_map": coverage_map,
        "dd_multiplier": 1.0,
        "signal_age_days": 0,
        "earnings_by_ticker": {},
        "run_date": "2026-04-25",
        "all_tickers": all_tickers,
    }


class TestDecideEntriesParity:
    """Direct decide_entries call vs _plan_entries(simulate=True) wrapper.

    The wrapper resolves prices_now from the IB client, then delegates
    to the decider. Parity here means: orders + blocked + n_entered
    are byte-equal regardless of which entry point the caller uses.
    """

    def test_orders_match_under_simulate(self):
        inputs = _build_inputs(n_signals=3)
        prices_now = {t: 100.0 for t in inputs["all_tickers"]}

        # Path A — direct decider call
        plan_a = decide_entries(
            enter_signals=inputs["enter_signals"],
            signals_raw=inputs["signals_raw"],
            predictions_by_ticker=inputs["predictions_by_ticker"],
            config=inputs["config"],
            strategy_config=inputs["strategy_config"],
            market_regime=inputs["market_regime"],
            sector_ratings=inputs["sector_ratings"],
            portfolio_nav=inputs["portfolio_nav"],
            peak_nav=inputs["peak_nav"],
            current_positions=inputs["current_positions"],
            prices_now=prices_now,
            price_histories=inputs["price_histories"],
            atr_map=inputs["atr_map"],
            vwap_map=inputs["vwap_map"],
            coverage_map=inputs["coverage_map"],
            dd_multiplier=inputs["dd_multiplier"],
            signal_age_days=inputs["signal_age_days"],
            earnings_by_ticker=inputs["earnings_by_ticker"],
            run_date=inputs["run_date"],
        )

        # Path B — through _plan_entries(simulate=True). Need a sim client
        # seeded with prices so ibkr.get_current_price returns the same
        # values the direct call used.
        sim_client = SimulatedIBKRClient(prices=dict(prices_now), nav=inputs["portfolio_nav"])
        n_b, orders_b, blocked_b, _events_b = _plan_entries(
            enter_signals=inputs["enter_signals"],
            signals_raw=inputs["signals_raw"],
            predictions_by_ticker=inputs["predictions_by_ticker"],
            config=inputs["config"],
            strategy_config=inputs["strategy_config"],
            market_regime=inputs["market_regime"],
            sector_ratings=inputs["sector_ratings"],
            ibkr=sim_client,
            portfolio_nav=inputs["portfolio_nav"],
            peak_nav=inputs["peak_nav"],
            current_positions=inputs["current_positions"],
            price_histories=inputs["price_histories"],
            atr_map=inputs["atr_map"],
            dd_multiplier=inputs["dd_multiplier"],
            signal_age_days=inputs["signal_age_days"],
            earnings_by_ticker=inputs["earnings_by_ticker"],
            vwap_map=inputs["vwap_map"],
            coverage_map=inputs["coverage_map"],
            ob=OrderBook.load(),  # not used in simulate mode
            run_date=inputs["run_date"],
            dry_run=False,
            simulate=True,
        )

        # Byte-equal orders + blocked + n_entered
        assert orders_b == plan_a.orders, (
            f"Orders drift between deciders direct vs shell wrapper:\n"
            f"  direct: {plan_a.orders}\n"
            f"  shell:  {orders_b}"
        )
        assert blocked_b == plan_a.blocked, (
            f"Blocked drift:\n  direct: {plan_a.blocked}\n  shell: {blocked_b}"
        )
        assert n_b == plan_a.n_entered, f"n_entered drift: {plan_a.n_entered} vs {n_b}"

    def test_already_held_skipped_consistently(self):
        inputs = _build_inputs(n_signals=3)
        prices_now = {t: 100.0 for t in inputs["all_tickers"]}
        # Hold one of the enter tickers
        held_ticker = inputs["enter_signals"][0]["ticker"]
        current_positions = {
            held_ticker: {"shares": 100, "avg_cost": 100.0, "market_value": 10000,
                          "sector": "Technology", "entry_date": "2026-04-20"},
        }

        plan = decide_entries(
            enter_signals=inputs["enter_signals"],
            signals_raw=inputs["signals_raw"],
            predictions_by_ticker=inputs["predictions_by_ticker"],
            config=inputs["config"],
            strategy_config=inputs["strategy_config"],
            market_regime=inputs["market_regime"],
            sector_ratings=inputs["sector_ratings"],
            portfolio_nav=inputs["portfolio_nav"],
            peak_nav=inputs["peak_nav"],
            current_positions=current_positions,
            prices_now=prices_now,
            price_histories=inputs["price_histories"],
            atr_map=inputs["atr_map"],
            vwap_map=inputs["vwap_map"],
            coverage_map=inputs["coverage_map"],
            dd_multiplier=inputs["dd_multiplier"],
            signal_age_days=inputs["signal_age_days"],
            earnings_by_ticker=inputs["earnings_by_ticker"],
            run_date=inputs["run_date"],
        )

        held_blocked = [b for b in plan.blocked if b["ticker"] == held_ticker]
        assert len(held_blocked) == 1, (
            f"Held ticker should be blocked exactly once; got {len(held_blocked)}"
        )
        assert held_blocked[0]["block_reason"] == "already in portfolio"
        # And no order generated for it
        assert held_ticker not in [o["ticker"] for o in plan.orders]


class TestEnrichPositionsUniverseSectorsOverride:
    """Tier 3 Part A (2026-04-27): backtester precomputes
    ``universe_sectors`` once per signal date, shared across 60 combos
    in a ``predictor_param_sweep``. Verify the optional override
    produces byte-equal output to the per-call rebuild path.
    """

    def test_universe_sectors_override_matches_internal_rebuild(self):
        from executor.deciders import enrich_positions

        signals_raw = {
            "universe": [
                {"ticker": "AAPL", "sector": "Technology"},
                {"ticker": "JPM", "sector": "Financial"},
                {"ticker": "JNJ", "sector": "Healthcare"},
            ],
            "buy_candidates": [
                {"ticker": "MSFT", "sector": "Technology"},
            ],
        }
        positions = {
            "AAPL": {"shares": 100, "avg_cost": 150.0},
            "MSFT": {"shares": 50, "avg_cost": 300.0},
        }
        entry_dates = {"AAPL": "2026-04-20", "MSFT": "2026-04-22"}

        # Path A — internal rebuild (live behavior, override=None)
        result_a = enrich_positions(positions, signals_raw, entry_dates)

        # Path B — explicit override (backtester behavior)
        precomputed = {
            "AAPL": "Technology",
            "JPM": "Financial",
            "JNJ": "Healthcare",
            "MSFT": "Technology",
        }
        result_b = enrich_positions(
            positions, signals_raw, entry_dates,
            universe_sectors=precomputed,
        )

        assert result_a == result_b, (
            "enrich_positions output drift between internal-rebuild "
            "(override=None) and explicit override paths"
        )

    def test_override_takes_precedence_over_signals_raw(self):
        """If caller passes override, signals_raw's sectors are ignored
        — caller has authoritative pre-built mapping. (Backtester filters
        signals against universe at simulation-loop bootstrap; the
        precomputed map reflects that filter.)"""
        from executor.deciders import enrich_positions

        # signals_raw says AAPL is Technology
        signals_raw = {
            "universe": [{"ticker": "AAPL", "sector": "Technology"}],
            "buy_candidates": [],
        }
        # Override says AAPL is Financial (intentionally divergent for the test)
        override = {"AAPL": "Financial"}
        positions = {"AAPL": {"shares": 100, "avg_cost": 150.0}}

        result = enrich_positions(
            positions, signals_raw, entry_dates_lookup=None,
            universe_sectors=override,
        )

        assert result["AAPL"]["sector"] == "Financial", (
            "Override map must take precedence over signals_raw's sectors"
        )


class TestDecideExitsAndReducesParity:
    """Direct decide_exits_and_reduces vs _plan_exits_and_reduces wrapper.

    Smoke-level coverage: empty signals + empty positions returns empty
    plan. Real behavior is exercised by the existing test_exit_manager
    suite (which feeds evaluate_strategy_exits).
    """

    def test_empty_signals_empty_plan(self):
        signals = {"exit": [], "reduce": [], "enter": [], "hold": [], "universe": []}
        plan = decide_exits_and_reduces(
            signals=signals,
            strategy_exits=[],
            current_positions={},
            prices_now={},
            predictions_by_ticker={},
            config={"reduce_fraction": 0.50},
            market_regime="neutral",
            portfolio_nav=1_000_000.0,
            run_date="2026-04-25",
        )
        assert plan.orders == []
        assert plan.urgent_exits_with_meta == []

    def test_exit_signal_produces_matching_order(self):
        positions = {
            "AAPL": {"shares": 100, "avg_cost": 95.0, "market_value": 10000,
                     "sector": "Technology"},
        }
        signals = {
            "exit": [{"ticker": "AAPL", "reason": "research_signal", "score": 30}],
            "reduce": [],
            "enter": [], "hold": [], "universe": [],
        }
        prices_now = {"AAPL": 110.0}

        plan = decide_exits_and_reduces(
            signals=signals,
            strategy_exits=[],
            current_positions=positions,
            prices_now=prices_now,
            predictions_by_ticker={},
            config={"reduce_fraction": 0.50},
            market_regime="neutral",
            portfolio_nav=1_000_000.0,
            run_date="2026-04-25",
        )

        assert len(plan.orders) == 1
        order = plan.orders[0]
        assert order["ticker"] == "AAPL"
        assert order["action"] == "EXIT"
        assert order["shares"] == 100
        assert order["price_at_order"] == 110.0
        assert order["sector_rating"] == "Technology"

        # And one urgent_exit metadata entry for the order book
        assert len(plan.urgent_exits_with_meta) == 1
        ue = plan.urgent_exits_with_meta[0]
        assert ue["ticker"] == "AAPL"
        assert ue["signal"] == "EXIT"
        assert ue["reason"] == "research_signal"

    def test_position_loss_floor_strategy_exit_produces_coin_sell(self):
        """L4549a Monday-path proof: a position_loss_floor EXIT carried in
        strategy_exits (the hard-risk override appended by main.py §2f',
        which survives the optimizer the same way drawdown_forced_exit does)
        flows through the decider to an executable EXIT order + an order-book
        urgent exit. This is the exact path that exits COIN at Monday open."""
        positions = {
            "COIN": {"shares": 454, "avg_cost": 187.38, "market_value": 69189.6,
                     "sector": "Financials"},
        }
        # Research says nothing / HOLD — COIN is exited purely by the floor.
        signals = {"exit": [], "reduce": [], "enter": [], "hold": [], "universe": []}
        strategy_exits = [{
            "ticker": "COIN", "action": "EXIT", "reason": "position_loss_floor",
            "detail": "MAE floor breached: -18.7% from avg cost (floor -15.0%)",
        }]
        prices_now = {"COIN": 152.40}

        plan = decide_exits_and_reduces(
            signals=signals,
            strategy_exits=strategy_exits,
            current_positions=positions,
            prices_now=prices_now,
            predictions_by_ticker={},
            config={"reduce_fraction": 0.50},
            market_regime="neutral",
            portfolio_nav=1_000_000.0,
            run_date="2026-06-08",
        )

        coin_orders = [o for o in plan.orders if o["ticker"] == "COIN"]
        assert len(coin_orders) == 1
        assert coin_orders[0]["action"] == "EXIT"
        assert coin_orders[0]["shares"] == 454
        coin_urgent = [u for u in plan.urgent_exits_with_meta if u["ticker"] == "COIN"]
        assert len(coin_urgent) == 1
        assert coin_urgent[0]["signal"] == "EXIT"
        assert coin_urgent[0]["reason"] == "position_loss_floor"
