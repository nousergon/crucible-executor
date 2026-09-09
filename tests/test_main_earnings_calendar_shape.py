"""Tests for main._next_earnings_date_from_dates_df.

Root cause fixed here: `Ticker.calendar`'s return shape changed from a
pandas DataFrame to a dict across yfinance releases. The old code called
`cal.empty` unconditionally, which raises
`AttributeError: 'dict' object has no attribute 'empty'` for every ENTER
candidate under this repo's pinned yfinance==1.7.0 — swallowed by the
surrounding `except Exception` into a daily ERROR log + ops alert. The
real damage: earnings_by_ticker was empty every day, so
earnings-proximity sizing/gating silently never applied.

Fix mirrors the SOTA pattern already in
nousergon-data/collectors/metron_market_data.py::_yfinance_earnings:
`Ticker.get_earnings_dates(limit=8)`, typed `-> pandas.DataFrame | None`
and shape-stable across the fleet's divergent yfinance pins
(crucible-backtester ~=1.5.2, crucible-research ~=1.6.0,
crucible-dashboard/crucible-executor ==1.7.0,
crucible-predictor/nousergon-data >=1.7.0 — filed alpha-engine-config-I10305).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from executor.main import _next_earnings_date_from_dates_df


def test_none_df():
    assert _next_earnings_date_from_dates_df(None, date(2026, 9, 9)) is None


def test_empty_df():
    df = pd.DataFrame()
    assert _next_earnings_date_from_dates_df(df, date(2026, 9, 9)) is None


def test_picks_earliest_upcoming_date_tz_naive_index():
    idx = pd.to_datetime(["2026-09-20", "2026-09-25", "2026-06-01"])
    df = pd.DataFrame({"EPS Estimate": [1.0, 1.1, 0.9]}, index=idx)
    result = _next_earnings_date_from_dates_df(df, date(2026, 9, 9))
    assert result == date(2026, 9, 20)


def test_picks_earliest_upcoming_date_tz_aware_index():
    idx = pd.to_datetime(["2026-09-20", "2026-09-25"]).tz_localize("America/New_York")
    df = pd.DataFrame({"EPS Estimate": [1.0, 1.1]}, index=idx)
    result = _next_earnings_date_from_dates_df(df, date(2026, 9, 9))
    assert result == date(2026, 9, 20)


def test_all_dates_in_past_returns_none():
    idx = pd.to_datetime(["2026-01-01", "2026-02-01"])
    df = pd.DataFrame({"EPS Estimate": [1.0, 1.1]}, index=idx)
    assert _next_earnings_date_from_dates_df(df, date(2026, 9, 9)) is None


def test_ref_date_itself_is_included():
    idx = pd.to_datetime(["2026-09-09"])
    df = pd.DataFrame({"EPS Estimate": [1.0]}, index=idx)
    assert _next_earnings_date_from_dates_df(df, date(2026, 9, 9)) == date(2026, 9, 9)


def test_dict_shape_raises_attributeerror_reproducing_the_live_defect():
    # get_earnings_dates() never returns a dict (typed -> DataFrame | None)
    # — this pins that a dict input is out of contract for this function,
    # unlike the old Ticker.calendar code path it replaces.
    with pytest.raises(AttributeError):
        _next_earnings_date_from_dates_df({"Earnings Date": [date(2026, 9, 20)]}, date(2026, 9, 9))
