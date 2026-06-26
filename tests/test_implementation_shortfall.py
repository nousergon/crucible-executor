"""Tests for implementation-shortfall TCA by entry trigger (L4583 · #656, G11)."""

from __future__ import annotations

import sqlite3

import pytest

from executor.implementation_shortfall import (
    aggregate_by_trigger,
    build_tca_summary,
    load_entry_orders,
    order_shortfall,
)


# ── Per-order math ─────────────────────────────────────────────────────────


def test_buy_paying_more_than_decision_is_a_positive_cost():
    o = order_shortfall(
        ticker="AAPL",
        side="BUY",
        entry_trigger="pullback",
        decision_price=100.0,
        arrival_price=100.5,  # drifted up 50 bps before order
        fill_price=100.6,  # +10 bps execution
    )
    assert o is not None
    assert o.delay_bps == pytest.approx(50.0)
    assert o.execution_bps == pytest.approx((100.6 - 100.5) / 100.5 * 10_000.0)
    assert o.total_is_bps == pytest.approx(o.delay_bps + o.execution_bps)
    assert o.total_is_bps > 0  # net cost


def test_sell_receiving_less_than_decision_is_a_positive_cost():
    o = order_shortfall(
        ticker="MSFT",
        side="SELL",
        entry_trigger="expiry",
        decision_price=200.0,
        arrival_price=199.0,  # drifted down -> cost for a sell
        fill_price=198.0,
    )
    assert o.delay_bps > 0  # received less than decision -> cost
    assert o.execution_bps > 0
    assert o.total_is_bps > 0


def test_pullback_that_gets_a_better_price_shows_negative_delay():
    # A BUY pullback that waited and bought BELOW the decision price -> the
    # delay leg is a negative cost (a saving) — the property G11 wants to see.
    o = order_shortfall(
        ticker="NVDA",
        side="BUY",
        entry_trigger="pullback",
        decision_price=100.0,
        arrival_price=99.0,
        fill_price=99.0,
    )
    assert o.delay_bps == pytest.approx(-100.0)


def test_missing_price_returns_none_not_crash():
    assert (
        order_shortfall(
            ticker="X",
            side="BUY",
            entry_trigger="vwap",
            decision_price=None,
            arrival_price=10.0,
            fill_price=10.0,
        )
        is None
    )


def test_zero_reference_raises():
    with pytest.raises(ValueError):
        order_shortfall(
            ticker="X",
            side="BUY",
            entry_trigger="vwap",
            decision_price=0.0,
            arrival_price=10.0,
            fill_price=10.0,
        )


def test_unrecognised_side_raises():
    with pytest.raises(ValueError):
        order_shortfall(
            ticker="X",
            side="HOLD",
            entry_trigger="vwap",
            decision_price=10.0,
            arrival_price=10.0,
            fill_price=10.0,
        )


def test_unlabelled_trigger_bucketed_not_dropped():
    o = order_shortfall(
        ticker="X",
        side="BUY",
        entry_trigger=None,
        decision_price=10.0,
        arrival_price=10.0,
        fill_price=10.0,
    )
    assert o.entry_trigger == "unlabelled"


# ── Aggregation ────────────────────────────────────────────────────────────


def _o(trigger, total_driver):
    # Build an order whose total IS ≈ total_driver bps (all delay) for AAPL BUY.
    return order_shortfall(
        ticker="AAPL",
        side="BUY",
        entry_trigger=trigger,
        decision_price=100.0,
        arrival_price=100.0 * (1 + total_driver / 10_000.0),
        fill_price=100.0 * (1 + total_driver / 10_000.0),
    )


def test_aggregate_groups_and_sorts_worst_first():
    orders = [
        _o("pullback", -20.0),
        _o("pullback", -10.0),
        _o("vwap", 30.0),
        _o("expiry", 5.0),
    ]
    agg = aggregate_by_trigger(orders)
    assert [t.entry_trigger for t in agg] == ["vwap", "expiry", "pullback"]
    pullback = next(t for t in agg if t.entry_trigger == "pullback")
    assert pullback.n_orders == 2
    assert pullback.mean_total_is_bps == pytest.approx(-15.0)
    assert pullback.mean_total_is_bps < 0  # the trigger earned its delay


def test_build_summary_overall_and_by_trigger():
    orders = [_o("vwap", 30.0), _o("pullback", -10.0)]
    summary = build_tca_summary(orders, since_date="2026-06-01")
    assert summary["n_orders"] == 2
    assert summary["overall_mean_total_is_bps"] == pytest.approx(10.0)
    assert summary["since_date"] == "2026-06-01"
    assert len(summary["by_trigger"]) == 2


def test_build_summary_empty_is_zeroed_not_crash():
    summary = build_tca_summary([])
    assert summary["n_orders"] == 0
    assert summary["overall_mean_total_is_bps"] == 0.0
    assert summary["by_trigger"] == []


# ── DB read ────────────────────────────────────────────────────────────────


def _trades_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trades ("
        "ticker TEXT, action TEXT, entry_trigger TEXT, trigger_type TEXT, "
        "signal_price REAL, price_at_order REAL, fill_price REAL, "
        "entry_trade_id TEXT, date TEXT)"
    )
    rows = [
        # entry BUY, pullback, full data -> included
        ("AAPL", "BUY", "pullback", None, 100.0, 100.5, 100.6, None, "2026-06-10"),
        # entry SELL, falls back to trigger_type when entry_trigger NULL
        ("MSFT", "SELL", None, "vwap", 200.0, 199.0, 198.5, None, "2026-06-11"),
        # exit row (entry_trade_id set) -> excluded
        ("AAPL", "SELL", None, "stop_loss", 100.0, 99.0, 99.0, "t1", "2026-06-12"),
        # entry missing signal_price -> excluded by WHERE clause
        ("NVDA", "BUY", "support", None, None, 50.0, 50.1, None, "2026-06-13"),
        # older row before the window
        ("TSLA", "BUY", "pullback", None, 10.0, 10.0, 10.0, None, "2026-05-01"),
    ]
    conn.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return conn


def test_load_entry_orders_filters_and_computes():
    conn = _trades_conn()
    orders = load_entry_orders(conn, since_date="2026-06-01")
    tickers = {o.ticker for o in orders}
    # AAPL + MSFT entries inside the window; exit, missing-price, and the
    # pre-window TSLA row are all excluded.
    assert tickers == {"AAPL", "MSFT"}
    msft = next(o for o in orders if o.ticker == "MSFT")
    assert msft.entry_trigger == "vwap"  # trigger_type fallback


def test_load_entry_orders_no_window_includes_older_rows():
    conn = _trades_conn()
    orders = load_entry_orders(conn)
    assert "TSLA" in {o.ticker for o in orders}
