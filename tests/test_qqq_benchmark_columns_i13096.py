"""Record-only QQQ benchmark columns on eod_pnl (alpha-engine-config-I13096).

``qqq_close`` / ``qqq_dividend_per_share`` are sourced the way ``spy_close`` /
``spy_dividend_per_share`` are (ArcticDB ``macro`` close, one Polygon
ex-dividend window query) and persisted beside them, so the perf tracker's
vs_qqq leg can run on the EOD source instead of a dividend-adjusted yfinance
history. Unlike SPY, a missing QQQ close is a NULL column, never a failed EOD.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from executor.dividends import (
    BENCHMARK_TICKERS,
    QQQ_TICKER,
    SPY_TICKER,
    accrue_position_dividends,
    fetch_ex_dividends,
)
from executor.eod_reconcile import _benchmark_close_or_none
from executor.trade_logger import init_db, log_eod


def _macro(symbols: dict) -> MagicMock:
    lib = MagicMock()

    def _read(sym):
        if sym not in symbols:
            raise KeyError(f"no such symbol: {sym}")
        return SimpleNamespace(data=symbols[sym])

    lib.read.side_effect = _read
    return lib


class _Client:
    def __init__(self, events):
        self.events = events

    def get_dividends_for_window(self, start, end):
        return self.events


@pytest.fixture
def conn():
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    c = init_db(db_path)
    yield c
    c.close()
    os.unlink(db_path)


def test_qqq_is_a_benchmark_beside_spy():
    assert BENCHMARK_TICKERS == (SPY_TICKER, QQQ_TICKER)


def test_benchmark_close_reads_the_macro_close():
    df = pd.DataFrame({"Close": [500.0, 503.25]},
                      index=pd.to_datetime(["2026-10-29", "2026-10-30"]))
    with patch("executor.price_cache._open_macro_library",
               return_value=_macro({"QQQ": df})):
        assert _benchmark_close_or_none("QQQ", "2026-10-30") == pytest.approx(503.25)


@pytest.mark.parametrize("symbols", [
    {},  # symbol absent from macro (before the collector writes it)
    {"QQQ": pd.DataFrame({"Close": [500.0]}, index=pd.to_datetime(["2026-10-29"]))},
    {"QQQ": pd.DataFrame({"Open": [500.0]}, index=pd.to_datetime(["2026-10-30"]))},
])
def test_benchmark_close_is_none_never_a_raise(symbols):
    with patch("executor.price_cache._open_macro_library",
               return_value=_macro(symbols)):
        assert _benchmark_close_or_none("QQQ", "2026-10-30") is None


def test_ex_dividend_window_includes_qqq_without_holding_it():
    client = _Client([
        {"ticker": QQQ_TICKER, "ex_dividend_date": "2026-09-22", "cash_amount": 0.71},
        {"ticker": SPY_TICKER, "ex_dividend_date": "2026-09-19", "cash_amount": 1.88},
    ])
    out, _pay, available, _warn = fetch_ex_dividends(
        ["LMT"], "2026-09-22", prior_date="2026-09-18", client=client,
    )
    assert available
    assert out == {QQQ_TICKER: 0.71, SPY_TICKER: 1.88}


def test_an_unheld_qqq_distribution_accrues_nothing():
    """The benchmark distribution must never reach the book's attribution."""
    positions = {"LMT": {"shares": 10, "prior_price": 480.0}}
    accruals = accrue_position_dividends(positions, {"LMT": {"shares": 10}},
                                         {QQQ_TICKER: 0.71})
    assert accruals == []


def test_log_eod_persists_the_qqq_columns(conn):
    log_eod(conn, {"date": "2026-10-30", "portfolio_nav": 1_000_000.0,
                   "spy_close": 680.0, "spy_dividend_per_share": 0.0,
                   "qqq_close": 503.25, "qqq_dividend_per_share": 0.71})
    row = conn.execute(
        "SELECT qqq_close, qqq_dividend_per_share FROM eod_pnl WHERE date='2026-10-30'"
    ).fetchone()
    assert row == (pytest.approx(503.25), pytest.approx(0.71))


def test_log_eod_without_qqq_writes_null(conn):
    log_eod(conn, {"date": "2026-10-30", "portfolio_nav": 1_000_000.0})
    row = conn.execute(
        "SELECT qqq_close, qqq_dividend_per_share FROM eod_pnl WHERE date='2026-10-30'"
    ).fetchone()
    assert row == (None, None)


def test_columns_are_added_to_an_existing_table():
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    try:
        init_db(db_path).close()
        c = init_db(db_path)
        cols = {r[1] for r in c.execute("PRAGMA table_info(eod_pnl)")}
        c.close()
        assert {"qqq_close", "qqq_dividend_per_share"} <= cols
    finally:
        os.unlink(db_path)
