"""
Unit tests for executor/optimizer_cutover.py — PR 5 of portfolio-optimizer arc.

Covers:
  - is_log_usable: happy + every failure mode (no log, sentinel, infeasible).
    An optimal/ok solve with EMPTY would_be_trades is a valid HOLD, not a
    failure — it is usable (apply → safe no-op). Regression lock for the
    2026-05-19 conflation bug.
  - apply_optimizer_targets_to_orderbook:
      * BUY → entry record with optimizer-derived shares / dollars / triggers
      * SELL @ target=0 → urgent_exit EXIT for full held shares
      * SELL @ target>0 → urgent_exit REDUCE for partial shares
      * SELL with no current position → skipped
      * BUY where delta_dollars < price → skipped (shares=0)
      * ibkr returns None → skipped
      * Unknown action → skipped
      * OrderBook dedup (already-pending entry / urgent_exit) — handled by OB
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from executor.optimizer_cutover import (
    apply_optimizer_targets_to_orderbook,
    is_log_usable,
)
from executor.order_book import OrderBook


# ── is_log_usable ──────────────────────────────────────────────────────────────


def test_is_log_usable_happy_path():
    log = {
        "shadow_status": "ok",
        "diagnostics": {"status": "optimal"},
        "would_be_trades": [{"ticker": "SPY", "action": "BUY"}],
    }
    assert is_log_usable(log) is True


def test_is_log_usable_optimal_inaccurate_also_ok():
    log = {
        "shadow_status": "ok",
        "diagnostics": {"status": "optimal_inaccurate"},
        "would_be_trades": [{"ticker": "SPY", "action": "BUY"}],
    }
    assert is_log_usable(log) is True


def test_is_log_usable_none_log():
    assert is_log_usable(None) is False


def test_is_log_usable_sentinel_failed():
    log = {"shadow_status": "failed", "error": "TypeError"}
    assert is_log_usable(log) is False


def test_is_log_usable_infeasible_diag():
    log = {
        "shadow_status": "ok",
        "diagnostics": {"status": "infeasible_fallback"},
        "would_be_trades": [{"ticker": "SPY"}],
    }
    assert is_log_usable(log) is False


def test_is_log_usable_optimal_with_no_trades_is_usable():
    """2026-05-19 regression: an optimal/ok solve with EMPTY
    would_be_trades is the optimizer's valid 'current ≈ target, hold'
    verdict — usable, NOT a failure. Previously conflated with genuine
    failures, producing a false 'operator must investigate' ERROR."""
    log = {
        "shadow_status": "ok",
        "diagnostics": {"status": "optimal", "turnover_one_way": 0.0017},
        "would_be_trades": [],
    }
    assert is_log_usable(log) is True


def test_is_log_usable_missing_would_be_trades_key_still_usable():
    """Absent (not just empty) would_be_trades on a clean solve is also a
    valid hold — usability depends only on the solve status."""
    log = {"shadow_status": "ok", "diagnostics": {"status": "optimal"}}
    assert is_log_usable(log) is True


def test_apply_optimizer_empty_trades_is_safe_noop():
    """The caller path for a hold day: usable log + empty would_be_trades
    → zero entries/exits, no exception (planner logs INFO, not ERROR)."""
    ob = OrderBook({"date": "2026-05-19", "entries": [], "urgent_exits": [],
                     "stops": [], "executed_today": []})
    entries, exits = apply_optimizer_targets_to_orderbook(
        log={"shadow_status": "ok",
             "diagnostics": {"status": "optimal"},
             "would_be_trades": []},
        ob=ob,
        ibkr=MagicMock(),
        current_positions={},
        price_histories={},
        atr_map={},
        strategy_config={},
        vwap_map={},
        signals_raw={"date": "2026-05-19", "signals": {}},
        predictions_by_ticker={},
        market_regime="neutral",
        run_date="2026-05-19",
        predictions_date="2026-05-19",
    )
    assert entries == []
    assert exits == []


# ── apply_optimizer_targets_to_orderbook ───────────────────────────────────────


def _make_ibkr(prices: dict[str, float]) -> MagicMock:
    ibkr = MagicMock()
    ibkr.get_current_price.side_effect = lambda t: prices.get(t)
    return ibkr


def _baseline_kwargs(ob: OrderBook, ibkr, **overrides):
    base = {
        "ob": ob,
        "ibkr": ibkr,
        "current_positions": {},
        "price_histories": None,
        "atr_map": {},
        "strategy_config": {
            "intraday_pullback_atr_multiple": 1.5,
            "intraday_vwap_discount_pct": 0.005,
        },
        "vwap_map": {},
        "signals_raw": {"date": "2026-05-13", "signals": {}},
        "predictions_by_ticker": {},
        "market_regime": "neutral",
        "run_date": "2026-05-13",
        "predictions_date": "2026-05-13",
    }
    base.update(overrides)
    return base


def test_buy_produces_entry_with_optimizer_sizing():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"SPY": 500.0})
    log = {
        "would_be_trades": [{
            "ticker": "SPY",
            "action": "BUY",
            "delta_dollars": 856_069.23,
            "delta_weight": 0.83,
            "target_weight": 0.83,
            "current_weight": 0.0,
        }],
    }
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr, atr_map={"SPY": 0.012}),
    )
    assert len(entries) == 1
    assert len(exits) == 0
    e = entries[0]
    assert e["ticker"] == "SPY"
    assert e["signal"] == "ENTER"
    assert e["shares"] == 1712  # floor(856069.23 / 500)
    assert e["dollar_size"] == pytest.approx(856_069.23, rel=1e-6)
    assert e["position_pct"] == pytest.approx(0.83)
    assert e["sizing_source"] == "portfolio_optimizer"
    assert e["sizing_factors"]["optimizer_target_weight"] == 0.83
    assert e["atr_pct"] == 0.012
    assert e["triggers"]["pullback_atr_multiple"] == 1.5


def test_sell_target_zero_produces_full_exit():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"CME": 200.0})
    log = {
        "would_be_trades": [{
            "ticker": "CME",
            "action": "SELL",
            "delta_dollars": -30_877.45,
            "delta_weight": -0.02994,
            "target_weight": 0.0,
            "current_weight": 0.02994,
        }],
    }
    current_positions = {"CME": {"shares": 154, "sector": "Financials"}}
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr, current_positions=current_positions),
    )
    assert len(entries) == 0
    assert len(exits) == 1
    x = exits[0]
    assert x["ticker"] == "CME"
    assert x["signal"] == "EXIT"
    assert x["shares"] == 154  # full held
    assert x["reason"] == "optimizer_target_zero"
    assert x["sizing_source"] == "portfolio_optimizer"


def test_sell_partial_produces_reduce():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"NVDA": 100.0})
    # Held 1000 shares @ $100 = $100k; scale down to $60k (sell $40k = 400 shares).
    log = {
        "would_be_trades": [{
            "ticker": "NVDA",
            "action": "SELL",
            "delta_dollars": -40_000.0,
            "delta_weight": -0.04,
            "target_weight": 0.06,
            "current_weight": 0.10,
        }],
    }
    current_positions = {"NVDA": {"shares": 1000, "sector": "Information Technology"}}
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr, current_positions=current_positions),
    )
    assert len(exits) == 1
    x = exits[0]
    assert x["signal"] == "REDUCE"
    assert x["shares"] == 400
    assert x["reason"] == "optimizer_scale_down"


def test_sell_no_position_is_skipped():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"AAA": 50.0})
    log = {
        "would_be_trades": [{
            "ticker": "AAA",
            "action": "SELL",
            "delta_dollars": -5_000.0,
            "target_weight": 0.0,
            "current_weight": 0.0,
        }],
    }
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr),
    )
    assert entries == []
    assert exits == []


def test_buy_with_delta_below_price_is_skipped():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"BRK.A": 600_000.0})
    log = {
        "would_be_trades": [{
            "ticker": "BRK.A",
            "action": "BUY",
            "delta_dollars": 100.0,  # < 1 share
            "target_weight": 0.001,
            "current_weight": 0.0,
        }],
    }
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr),
    )
    assert entries == []
    assert exits == []


def test_ibkr_returns_none_skips_ticker():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({})  # all prices unknown
    log = {
        "would_be_trades": [
            {"ticker": "FOO", "action": "BUY", "delta_dollars": 1000, "target_weight": 0.01, "current_weight": 0},
            {"ticker": "BAR", "action": "SELL", "delta_dollars": -1000, "target_weight": 0, "current_weight": 0.01},
        ],
    }
    current_positions = {"BAR": {"shares": 10, "sector": "X"}}
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr, current_positions=current_positions),
    )
    assert entries == []
    assert exits == []


def test_unknown_action_is_skipped():
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"FOO": 50.0})
    log = {
        "would_be_trades": [{
            "ticker": "FOO", "action": "HOLD",
            "delta_dollars": 0, "target_weight": 0.01, "current_weight": 0.01,
        }],
    }
    entries, exits = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(ob, ibkr),
    )
    assert entries == []
    assert exits == []


def test_orderbook_dedup_on_duplicate_buy():
    """OrderBook dedups entries by ticker — a re-call with same ticker no-ops."""
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"SPY": 500.0})
    log = {
        "would_be_trades": [{
            "ticker": "SPY", "action": "BUY",
            "delta_dollars": 100_000.0, "target_weight": 0.10, "current_weight": 0.0,
        }],
    }
    # First call adds.
    entries1, _ = apply_optimizer_targets_to_orderbook(log=log, **_baseline_kwargs(ob, ibkr))
    assert len(entries1) == 1
    # Second call: our returned list still has the record (we don't read OB
    # back), but OrderBook.add_entry skips the duplicate. The OB state has 1.
    apply_optimizer_targets_to_orderbook(log=log, **_baseline_kwargs(ob, ibkr))
    assert len(ob._data.get("approved_entries", [])) == 1


def test_research_metadata_propagated_to_entry():
    """Signals + predictions metadata should flow through to the entry record."""
    ob = OrderBook(data={"date": "2026-05-13"})
    ibkr = _make_ibkr({"CAH": 100.0})
    log = {
        "would_be_trades": [{
            "ticker": "CAH", "action": "BUY",
            "delta_dollars": 56_693.12, "target_weight": 0.07, "current_weight": 0.015,
        }],
    }
    signals_raw = {
        "date": "2026-05-13",
        "signals": {
            "CAH": {
                "score": 78.5, "conviction": "rising", "rating": "BUY",
                "sector": "Health Care", "sector_rating": "overweight",
                "price_target_upside": 0.18,
            },
        },
    }
    predictions_by_ticker = {
        "CAH": {
            "predicted_direction": "UP", "prediction_confidence": 0.62,
            "predicted_alpha": 0.005, "stance": "quality",
        },
    }
    entries, _ = apply_optimizer_targets_to_orderbook(
        log=log,
        **_baseline_kwargs(
            ob, ibkr,
            signals_raw=signals_raw,
            predictions_by_ticker=predictions_by_ticker,
        ),
    )
    e = entries[0]
    assert e["research_score"] == 78.5
    assert e["research_conviction"] == "rising"
    assert e["sector"] == "Health Care"
    assert e["predicted_direction"] == "UP"
    assert e["stance"] == "quality"
