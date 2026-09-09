"""Tests for main._next_earnings_date_from_calendar.

Root cause fixed here: modern yfinance (1.6.0, 1.7.0 — the version this
repo pins in requirements.txt) returns ``Ticker.calendar`` as a ``dict``
(e.g. ``{"Earnings Date": [date(...), ...], ...}``), NOT a pandas
DataFrame. The old code called ``cal.empty`` unconditionally, which raises
``AttributeError: 'dict' object has no attribute 'empty'`` for every ENTER
candidate — swallowed by the surrounding ``except Exception`` into a daily
ERROR log + ops alert. The REAL damage: earnings_by_ticker was empty every
day, so earnings-proximity sizing/gating silently never applied.

Verified against yfinance/scrapers/quote.py::_fetch_calendar (1.6.0,
installed via /opt/homebrew's site-packages; 1.7.0 pinned in
requirements.txt) — ``calendar`` is typed ``-> dict`` and always
constructed as a plain dict (``{}`` when Yahoo has no data).
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from executor.main import _next_earnings_date_from_calendar


def test_dict_shape_with_earnings_date():
    cal = {
        "Earnings Date": [date(2026, 9, 20), date(2026, 9, 21)],
        "Earnings High": 1.5,
    }
    assert _next_earnings_date_from_calendar(cal) == date(2026, 9, 20)


def test_dict_shape_empty_dict():
    assert _next_earnings_date_from_calendar({}) is None


def test_dict_shape_missing_earnings_date_key():
    assert _next_earnings_date_from_calendar({"Dividend Date": date(2026, 9, 1)}) is None


def test_dict_shape_earnings_date_empty_list():
    assert _next_earnings_date_from_calendar({"Earnings Date": []}) is None


def test_dict_shape_datetime_entries():
    # Defensive: some yfinance builds have returned datetime instead of date.
    cal = {"Earnings Date": [datetime(2026, 9, 20, 8, 0, 0)]}
    assert _next_earnings_date_from_calendar(cal) == date(2026, 9, 20)


def test_none_calendar():
    assert _next_earnings_date_from_calendar(None) is None


def test_legacy_dataframe_shape_kept_as_fallback():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Earnings Date": [pd.Timestamp("2026-09-20")]})
    assert _next_earnings_date_from_calendar(df) == date(2026, 9, 20)


def test_legacy_dataframe_shape_empty():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame()
    assert _next_earnings_date_from_calendar(df) is None


def test_unrecognized_shape_returns_none_not_raise():
    assert _next_earnings_date_from_calendar(object()) is None
    assert _next_earnings_date_from_calendar("not a calendar") is None
