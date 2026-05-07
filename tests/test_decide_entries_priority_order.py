"""Verify decide_entries processes ENTER candidates in priority order.

Per the 2026-05-07 predictor audit's Phase 1: candidates are processed
in (research composite score DESC, predicted_alpha DESC) order. The
risk_guard's max_total_equity / max_sector caps bind on the running
tally of approved entries, so processing higher-priority candidates
first means cap-binding cases surrender lower-priority candidates.

Tests target the ``_entry_priority_key`` helper directly (pure-function
ordering) plus the integrated ``decide_entries`` behavior under
cap-binding conditions.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.deciders import _entry_priority_key, decide_entries


# ── _entry_priority_key (pure ordering) ──────────────────────────────────────

class TestEntryPriorityKey:
    """Pure-function tests for the sort key."""

    def test_higher_score_sorts_first(self):
        sig_low = {"ticker": "LOW", "score": 70}
        sig_high = {"ticker": "HIGH", "score": 90}
        items = [sig_low, sig_high]
        items.sort(key=lambda s: _entry_priority_key(s, {}))
        assert [s["ticker"] for s in items] == ["HIGH", "LOW"]

    def test_predicted_alpha_breaks_score_ties(self):
        sig_a = {"ticker": "AAA", "score": 80}
        sig_b = {"ticker": "BBB", "score": 80}
        preds = {
            "AAA": {"predicted_alpha": 0.001},
            "BBB": {"predicted_alpha": 0.005},
        }
        items = [sig_a, sig_b]
        items.sort(key=lambda s: _entry_priority_key(s, preds))
        assert [s["ticker"] for s in items] == ["BBB", "AAA"]

    def test_score_dominates_predicted_alpha(self):
        # Even when predicted_alpha is much higher, lower score sorts last.
        sig_high_score = {"ticker": "HS", "score": 90}
        sig_high_alpha = {"ticker": "HA", "score": 70}
        preds = {
            "HS": {"predicted_alpha": 0.001},
            "HA": {"predicted_alpha": 0.020},
        }
        items = [sig_high_alpha, sig_high_score]
        items.sort(key=lambda s: _entry_priority_key(s, preds))
        assert [s["ticker"] for s in items] == ["HS", "HA"]

    def test_missing_score_defaults_to_zero_sorts_last(self):
        sig_with = {"ticker": "WITH", "score": 75}
        sig_without = {"ticker": "WITHOUT"}  # no score field
        items = [sig_without, sig_with]
        items.sort(key=lambda s: _entry_priority_key(s, {}))
        assert [s["ticker"] for s in items] == ["WITH", "WITHOUT"]

    def test_none_score_treated_as_zero(self):
        sig_none = {"ticker": "NONE", "score": None}
        sig_real = {"ticker": "REAL", "score": 60}
        items = [sig_none, sig_real]
        items.sort(key=lambda s: _entry_priority_key(s, {}))
        assert [s["ticker"] for s in items] == ["REAL", "NONE"]

    def test_missing_predicted_alpha_defaults_to_zero(self):
        # Same score; one has predictions, one doesn't. The one with a
        # positive predicted_alpha should sort first.
        sig_a = {"ticker": "AAA", "score": 80}
        sig_b = {"ticker": "BBB", "score": 80}
        preds = {"AAA": {"predicted_alpha": 0.005}}  # BBB missing
        items = [sig_a, sig_b]
        items.sort(key=lambda s: _entry_priority_key(s, preds))
        assert [s["ticker"] for s in items] == ["AAA", "BBB"]

    def test_negative_predicted_alpha_sorts_after_zero(self):
        # Same score; predicted_alpha < 0 should sort after predicted_alpha = 0
        # (descending sort means higher alpha first).
        sig_neg = {"ticker": "NEG", "score": 80}
        sig_zero = {"ticker": "ZERO", "score": 80}
        preds = {
            "NEG": {"predicted_alpha": -0.002},
            "ZERO": {},  # no predicted_alpha → defaults to 0.0
        }
        items = [sig_neg, sig_zero]
        items.sort(key=lambda s: _entry_priority_key(s, preds))
        assert [s["ticker"] for s in items] == ["ZERO", "NEG"]

    def test_empty_predictions_dict_safe(self):
        sig = {"ticker": "AAA", "score": 80}
        # Should not raise — empty preds dict is the cold-start case.
        key = _entry_priority_key(sig, {})
        assert key == (-80.0, 0.0)

    def test_none_predictions_dict_safe(self):
        sig = {"ticker": "AAA", "score": 80}
        # ``predictions_by_ticker`` is typed as ``dict`` in the signature
        # but defensively None should not crash — covers the case where
        # predictions failed to load and decide_entries was still called.
        key = _entry_priority_key(sig, None)
        assert key == (-80.0, 0.0)


# ── decide_entries integration with priority ordering ───────────────────────

def _df_history(base: float = 100.0):
    """Minimal price history for momentum_gate (always passes)."""
    import pandas as pd
    closes = [base * (1 + 0.001 * i) for i in range(40)]
    return pd.DataFrame(
        {"close": closes, "high": [c * 1.01 for c in closes],
         "low": [c * 0.99 for c in closes], "volume": [1_000_000] * 40},
        index=pd.date_range("2026-03-01", periods=40, freq="B"),
    )


def _config():
    return {
        "min_score_to_enter": 50,
        "max_position_pct": 0.05,
        "max_sector_pct": 0.30,
        "max_total_equity_pct": 0.95,
        "drawdown_halt_pct": 0.20,
        "atr_sizing_enabled": False,
        "momentum_gate_enabled": False,
        "earnings_proximity_warning_days": 0,
    }


def _strategy_config():
    return {
        "vwap_threshold_pct": 1.0,
        "intraday_support_lookback_days": 20,
        "drawdown_forced_exit_enabled": False,
    }


class TestDecideEntriesProcessesInPriorityOrder:
    """Integration: confirm decide_entries sorts before iterating.

    These tests construct an input list in *reverse* priority order
    (lower-score first) and verify the entries are approved in score-
    descending order. The shadow log (`plan.blocked` + `plan.orders`)
    captures processing order via the order they were appended.
    """

    def _signal(self, ticker: str, score: int, sector: str = "Technology"):
        return {
            "ticker": ticker,
            "signal": "ENTER",
            "score": score,
            "conviction": "rising",
            "sector": sector,
            "rating": "BUY",
            "price_target_upside": 0.15,
            "thesis_summary": "test",
        }

    def test_higher_score_appears_in_orders_first(self):
        # Three signals fed in ascending-score order; expect descending in
        # orders list.
        enter = [
            self._signal("LOW", 60),
            self._signal("MID", 75),
            self._signal("HIGH", 90),
        ]
        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": enter,
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": enter,
            "buy_candidates": enter,
        }
        all_tickers = ["LOW", "MID", "HIGH"]
        plan = decide_entries(
            enter_signals=enter,
            signals_raw=signals_raw,
            predictions_by_ticker={},
            config=_config(),
            strategy_config=_strategy_config(),
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={t: 100.0 for t in all_tickers},
            price_histories={t: _df_history() for t in all_tickers},
            atr_map={t: 0.02 for t in all_tickers},
            vwap_map={t: 100.0 for t in all_tickers},
            coverage_map={t: 1.0 for t in all_tickers},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )

        # All three should be approved (cap doesn't bind here); the order
        # in plan.orders is the processing order, which should be
        # descending by score.
        order_tickers = [o["ticker"] for o in plan.orders]
        assert order_tickers == ["HIGH", "MID", "LOW"]

    def test_predicted_alpha_breaks_ties_in_processing_order(self):
        # Two equal-score candidates with different predicted_alpha; the
        # one with higher predicted_alpha should appear first in orders.
        enter = [
            self._signal("AAA", 80),
            self._signal("BBB", 80),
        ]
        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": enter,
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": enter,
            "buy_candidates": enter,
        }
        all_tickers = ["AAA", "BBB"]
        predictions = {
            "AAA": {"predicted_alpha": 0.001, "gbm_veto": False},
            "BBB": {"predicted_alpha": 0.008, "gbm_veto": False},
        }
        plan = decide_entries(
            enter_signals=enter,
            signals_raw=signals_raw,
            predictions_by_ticker=predictions,
            config=_config(),
            strategy_config=_strategy_config(),
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={t: 100.0 for t in all_tickers},
            price_histories={t: _df_history() for t in all_tickers},
            atr_map={t: 0.02 for t in all_tickers},
            vwap_map={t: 100.0 for t in all_tickers},
            coverage_map={t: 1.0 for t in all_tickers},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )

        order_tickers = [o["ticker"] for o in plan.orders]
        assert order_tickers == ["BBB", "AAA"]

    def test_does_not_mutate_input_list(self):
        # decide_entries reassigns enter_signals locally via sorted();
        # caller's list should still be in original order.
        enter = [
            self._signal("LOW", 60),
            self._signal("HIGH", 90),
        ]
        original_order = [s["ticker"] for s in enter]
        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": enter,
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": enter,
            "buy_candidates": enter,
        }
        all_tickers = ["LOW", "HIGH"]
        decide_entries(
            enter_signals=enter,
            signals_raw=signals_raw,
            predictions_by_ticker={},
            config=_config(),
            strategy_config=_strategy_config(),
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={t: 100.0 for t in all_tickers},
            price_histories={t: _df_history() for t in all_tickers},
            atr_map={t: 0.02 for t in all_tickers},
            vwap_map={t: 100.0 for t in all_tickers},
            coverage_map={t: 1.0 for t in all_tickers},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )
        # Caller's input must remain unmodified.
        assert [s["ticker"] for s in enter] == original_order
