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

from executor.deciders import _days_until, _entry_priority_key, _entry_urgency_score, decide_entries

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


# ── urgency-weighted entry ranking (config#676 Phase 1, opt-in) ──────────────


class TestUrgencyWeightedEntryRanking:
    """The opt-in Gârleanu–Pedersen Phase-1 urgency ranking (config#676).

    Flag defaults OFF → ordering byte-identical to the flat-priority baseline.
    Flag ON → urgency (magnitude × conviction × boost) is the primary key, with
    score / predicted_alpha as deterministic tie-breakers.
    """

    URGENCY_ON = {"urgency_weighted_entry_ranking_enabled": True}

    def test_flag_off_is_byte_identical_baseline(self):
        # No config and explicit-flag-off must both yield the 2-tuple baseline.
        sig = {"ticker": "AAA", "score": 80}
        preds = {"AAA": {"predicted_alpha": 0.01, "expected_move": 0.2, "prediction_confidence": 0.9}}
        assert _entry_priority_key(sig, preds) == (-80.0, -0.01)
        assert _entry_priority_key(sig, preds, {"urgency_weighted_entry_ranking_enabled": False}) == (-80.0, -0.01)

    def test_urgency_outranks_higher_score_when_enabled(self):
        # LOWSCORE has a much larger expected_move × confidence → more urgent →
        # sorts ahead of the higher-composite-score name when the flag is on.
        sig_high_score = {"ticker": "HS", "score": 90}
        sig_urgent = {"ticker": "URG", "score": 70}
        preds = {
            "HS": {"expected_move": 0.01, "prediction_confidence": 0.5},
            "URG": {"expected_move": 0.12, "prediction_confidence": 0.9},
        }
        items = [sig_high_score, sig_urgent]
        items.sort(key=lambda s: _entry_priority_key(s, preds, self.URGENCY_ON))
        assert [s["ticker"] for s in items] == ["URG", "HS"]
        # ...and with the flag OFF the higher score wins (baseline preserved).
        items.sort(key=lambda s: _entry_priority_key(s, preds))
        assert [s["ticker"] for s in items] == ["HS", "URG"]

    def test_equal_urgency_falls_back_to_score(self):
        # Identical urgency inputs → score breaks the tie deterministically.
        preds = {
            "A": {"expected_move": 0.05, "prediction_confidence": 0.8},
            "B": {"expected_move": 0.05, "prediction_confidence": 0.8},
        }
        a = {"ticker": "A", "score": 60}
        b = {"ticker": "B", "score": 85}
        items = [a, b]
        items.sort(key=lambda s: _entry_priority_key(s, preds, self.URGENCY_ON))
        assert [s["ticker"] for s in items] == ["B", "A"]

    def test_catalyst_proximity_boost(self):
        # Same magnitude × conviction; CAT has a catalyst inside the window →
        # boosted urgency → sorts first.
        preds = {
            "CAT": {"expected_move": 0.05, "prediction_confidence": 0.8, "catalyst_date": "2026-06-30"},
            "NOCAT": {"expected_move": 0.05, "prediction_confidence": 0.8},
        }
        cat = {"ticker": "CAT", "score": 70}
        nocat = {"ticker": "NOCAT", "score": 70}
        items = [nocat, cat]
        items.sort(key=lambda s: _entry_priority_key(s, preds, self.URGENCY_ON, run_date="2026-06-28"))
        assert [s["ticker"] for s in items] == ["CAT", "NOCAT"]

    def test_catalyst_outside_window_no_boost(self):
        # A catalyst far in the future (beyond the window) earns no boost.
        score = _entry_urgency_score(
            {"ticker": "X"},
            {"expected_move": 0.05, "prediction_confidence": 0.8, "catalyst_date": "2026-09-01"},
            self.URGENCY_ON,
            run_date="2026-06-28",
        )
        assert score == 0.05 * 0.8  # no catalyst boost applied

    def test_confirmed_momentum_boost(self):
        boosted = _entry_urgency_score(
            {"ticker": "M"},
            {"expected_move": 0.05, "prediction_confidence": 0.8, "momentum_20d": 0.04},
            self.URGENCY_ON,
        )
        flat = _entry_urgency_score(
            {"ticker": "F"},
            {"expected_move": 0.05, "prediction_confidence": 0.8, "momentum_20d": -0.04},
            self.URGENCY_ON,
        )
        assert boosted > flat

    def test_magnitude_prefers_expected_move_else_predicted_alpha(self):
        with_em = _entry_urgency_score({"ticker": "E"}, {"expected_move": 0.07, "prediction_confidence": 1.0}, self.URGENCY_ON)
        assert with_em == 0.07
        # No expected_move → |predicted_alpha| is the magnitude.
        with_pa = _entry_urgency_score({"ticker": "P"}, {"predicted_alpha": -0.03, "prediction_confidence": 1.0}, self.URGENCY_ON)
        assert with_pa == 0.03

    def test_missing_fields_neutral_no_raise(self):
        # Empty prediction → magnitude 0, conviction neutral 1.0, no boost → 0.0.
        assert _entry_urgency_score({"ticker": "Z"}, {}, self.URGENCY_ON) == 0.0
        # Key path never raises on a totally empty prediction set.
        key = _entry_priority_key({"ticker": "Z", "score": 50}, {}, self.URGENCY_ON)
        assert key == (0.0, -50.0, 0.0)

    def test_days_until_helper(self):
        assert _days_until("2026-06-30", "2026-06-28") == 2
        assert _days_until("2026-06-26", "2026-06-28") == -2
        assert _days_until(None, "2026-06-28") is None
        assert _days_until("not-a-date", "2026-06-28") is None
        assert _days_until("2026-06-30", None) is None


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
            prices_now=dict.fromkeys(all_tickers, 100.0),
            price_histories={t: _df_history() for t in all_tickers},
            atr_map=dict.fromkeys(all_tickers, 0.02),
            vwap_map=dict.fromkeys(all_tickers, 100.0),
            coverage_map=dict.fromkeys(all_tickers, 1.0),
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
            prices_now=dict.fromkeys(all_tickers, 100.0),
            price_histories={t: _df_history() for t in all_tickers},
            atr_map=dict.fromkeys(all_tickers, 0.02),
            vwap_map=dict.fromkeys(all_tickers, 100.0),
            coverage_map=dict.fromkeys(all_tickers, 1.0),
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
            prices_now=dict.fromkeys(all_tickers, 100.0),
            price_histories={t: _df_history() for t in all_tickers},
            atr_map=dict.fromkeys(all_tickers, 0.02),
            vwap_map=dict.fromkeys(all_tickers, 100.0),
            coverage_map=dict.fromkeys(all_tickers, 1.0),
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )
        # Caller's input must remain unmodified.
        assert [s["ticker"] for s in enter] == original_order
