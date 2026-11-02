"""Tests for explicit ex-date dividend accrual and total-return SPY
(alpha-engine-config-I8188, defect 3).

The load-bearing test is
``TestAccrual::test_a_dividend_paying_holding_produces_a_non_zero_accrual`` —
``dividend_usd`` summed to exactly $0.00 across all 115 live sessions while the
book held seven payers, and nothing in the suite could have noticed.
"""

from __future__ import annotations

import pytest

from executor.dividends import (
    SPY_TICKER,
    accrue_position_dividends,
    fetch_ex_dividends,
    spy_total_return_pct,
)


class _FakeClient:
    """Minimal PolygonClient stand-in: ``get_dividends_for_window(start, end)``.

    ONE method, taking a window and no ticker — the shape the fix moved to
    (alpha-engine-config-I10047). ``calls`` records every transport call, so a
    test can assert the whole book costs exactly one.
    """

    def __init__(self, rows=(), fail=False):
        self._rows = list(rows)
        self._fail = fail
        self.calls = []

    def get_dividends_for_window(self, start, end, limit=1000):
        self.calls.append((start, end))
        if self._fail:
            raise RuntimeError("polygon 429")
        return list(self._rows)


class TestFetch:
    def test_an_ex_date_in_the_interval_is_picked_up(self):
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30},
        ])
        out, pay_dates, available, warning = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {"LMT": 3.30}
        assert pay_dates == {"LMT": None}
        assert available is True
        assert warning is None

    def test_the_window_query_bounds_are_the_reconciliation_interval(self):
        client = _FakeClient()
        fetch_ex_dividends(["LMT"], "2026-08-21", prior_date="2026-08-20",
                           client=client)
        assert client.calls == [("2026-08-20", "2026-08-21")]

    def test_no_prior_date_queries_the_run_date_alone(self):
        client = _FakeClient()
        fetch_ex_dividends(["LMT"], "2026-08-21", client=client)
        assert client.calls == [("2026-08-21", "2026-08-21")]

    def test_the_pay_date_is_carried_for_the_receivable(self):
        """NAV is cash-basis on this account, so the accrual ledger has to know
        when the cash actually lands or the receivable never releases."""
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30, "pay_date": "2026-09-26"},
        ])
        _out, pay_dates, _a, _w = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert pay_dates["LMT"] == "2026-09-26"

    def test_two_distributions_in_one_interval_take_the_later_pay_date(self):
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30, "pay_date": "2026-09-26"},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 1.00, "pay_date": "2026-10-02"},
        ])
        out, pay_dates, _a, _w = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out["LMT"] == pytest.approx(4.30)
        assert pay_dates["LMT"] == "2026-10-02"

    def test_a_dividend_outside_the_interval_is_ignored(self):
        """The endpoint's `.gte` bound is INCLUSIVE, so a dividend going ex on
        `prior_date` — already counted in the prior session — comes back in the
        payload and must be dropped by the local half-open filter."""
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-08-20",
             "cash_amount": 3.30},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-25",
             "cash_amount": 3.30},
        ])
        out, _pay, available, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {}
        assert available is True

    def test_the_interval_spans_a_skipped_session(self):
        """The NAV baseline spans from the prior persisted eod_pnl row, so the
        dividend leg must span the same window or the two sides of the
        reconciliation measure different intervals."""
        client = _FakeClient([
            {"ticker": "CTAS", "ex_dividend_date": "2026-06-24",
             "cash_amount": 1.56},
        ])
        out, _pay, _, _ = fetch_ex_dividends(
            ["CTAS"], "2026-06-25", prior_date="2026-06-23", client=client,
        )
        assert out == {"CTAS": 1.56}
        assert client.calls == [("2026-06-23", "2026-06-25")]

    def test_a_ticker_not_held_is_not_accrued(self):
        """The window query returns EVERY ticker going ex that day; the book is
        intersected locally."""
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30},
            {"ticker": "XOM", "ex_dividend_date": "2026-08-21",
             "cash_amount": 0.99},
        ])
        out, _pay, _, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {"LMT": 3.30}

    def test_spy_is_included_for_the_benchmark_leg(self):
        client = _FakeClient([
            {"ticker": SPY_TICKER, "ex_dividend_date": "2026-08-21",
             "cash_amount": 1.80},
        ])
        out, _pay, _, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {SPY_TICKER: 1.80}

    def test_a_twelve_name_book_costs_exactly_one_transport_call(self):
        """The defect (alpha-engine-config-I10047): thirteen calls per run date
        against a 5-calls/min budget, twice per postclose."""
        book = ["AMD", "AXP", "BRO", "COST", "CRUS", "CTAS", "FAST", "LMT",
                "MA", "MU", "NVDA", "ORCL"]
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": "2026-09-04",
             "cash_amount": 3.30, "pay_date": "2026-09-26"},
            {"ticker": SPY_TICKER, "ex_dividend_date": "2026-09-04",
             "cash_amount": 1.80, "pay_date": "2026-09-30"},
        ])
        out, pay_dates, available, warning = fetch_ex_dividends(
            book, "2026-09-04", prior_date="2026-09-03", client=client,
        )
        assert len(client.calls) == 1
        assert out == {"LMT": 3.30, SPY_TICKER: 1.80}
        assert pay_dates == {"LMT": "2026-09-26", SPY_TICKER: "2026-09-30"}
        assert available is True
        assert warning is None

    def test_no_dividends_today_is_available_true(self):
        """An empty result and an unavailable feed are DIFFERENT facts —
        collapsing them is exactly the defect being fixed."""
        out, _pay, available, warning = fetch_ex_dividends(
            ["AMD"], "2026-08-21", prior_date="2026-08-20", client=_FakeClient(),
        )
        assert out == {}
        assert available is True
        assert warning is None

    def test_query_failure_is_all_or_nothing_and_names_the_date(self):
        """No per-ticker "treating as no ex-dividend": one query covers the
        whole book, so its failure leaves the whole book unmeasured."""
        book = ["AMD", "CRUS", "MU"]
        client = _FakeClient(fail=True)
        out, pay_dates, available, warning = fetch_ex_dividends(
            book, "2026-09-04", prior_date="2026-09-03", client=client,
        )
        assert out == {}
        assert pay_dates == {}
        assert available is False
        assert "2026-09-04" in warning
        assert "NO held ticker" in warning
        assert "4 name(s)" in warning  # the book plus SPY
        assert "unattributed residual" in warning

    def test_malformed_rows_are_skipped(self):
        client = _FakeClient([
            {"ticker": "LMT", "ex_dividend_date": None, "cash_amount": 3.30},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": None},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": "x"},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": -1.0},
            {"ticker": None, "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30},
            {"ex_dividend_date": "2026-08-21", "cash_amount": 3.30},
            {"ticker": "LMT", "ex_dividend_date": "2026-08-21",
             "cash_amount": 3.30},
        ])
        out, _pay, _, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {"LMT": 3.30}


class TestAccrual:
    def test_a_dividend_paying_holding_produces_a_non_zero_accrual(self):
        """The regression that could not previously be caught: LMT was held
        for 34 sessions and contributed exactly $0.00 of dividend for all of
        them."""
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                             {"LMT": 3.30})
        assert [a["ticker"] for a in accruals] == ["LMT"]
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["dividend_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["daily_return_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["daily_return_pct"] == pytest.approx(0.66)

    def test_entitlement_uses_the_prior_close_share_count(self):
        """A name bought ON the ex-date does not receive that dividend; a name
        sold on the ex-date still does."""
        positions = {"LMT": {"shares": 500, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                             {"LMT": 3.30})
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)
        assert accruals[0]["shares"] == pytest.approx(200)

    def test_no_prior_snapshot_falls_back_to_todays_shares(self):
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {}, {"LMT": 3.30})
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)

    def test_broker_accrual_wins_and_is_not_double_counted(self):
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "dividend_usd": 660.0,
                             "daily_return_usd": 660.0}}
        assert accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                         {"LMT": 3.30}) == []
        assert positions["LMT"]["dividend_usd"] == pytest.approx(660.0)

    def test_spy_is_accrued_like_any_other_holding(self):
        """SPY is the enhanced-index core position. Accruing its dividend on
        the BOOK's shares is not a double count against the BENCHMARK's total
        return — they are the two sides of the comparison."""
        positions = {SPY_TICKER: {"shares": 160, "prior_price": 762.60,
                                  "daily_return_usd": 0.0,
                                  "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(
            positions, {SPY_TICKER: {"shares": 160}}, {SPY_TICKER: 1.80})
        assert accruals[0]["amount_usd"] == pytest.approx(288.0)

    def test_a_name_with_no_dividend_is_untouched(self):
        positions = {"AMD": {"shares": 100, "prior_price": 100.0,
                             "daily_return_usd": 50.0}}
        assert accrue_position_dividends(positions, {}, {"LMT": 3.30}) == []
        assert "dividend_usd" not in positions["AMD"]

    def test_zero_share_position_accrues_nothing(self):
        positions = {"LMT": {"shares": 0, "prior_price": 500.0,
                             "daily_return_usd": 0.0}}
        assert accrue_position_dividends(positions, {"LMT": {"shares": 0}},
                                         {"LMT": 3.30}) == []


class TestSpyTotalReturn:
    def test_the_benchmark_is_total_return_not_price_return(self):
        price_only = spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60, spy_dividend_per_share=0.0)
        total = spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60, spy_dividend_per_share=1.80)
        assert total > price_only
        assert total - price_only == pytest.approx(1.80 / 762.60 * 100)

    def test_no_distribution_day_matches_price_return(self):
        assert spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60
        ) == pytest.approx((765.72 / 762.60 - 1) * 100)

    def test_missing_leg_is_none_not_zero(self):
        assert spy_total_return_pct(spy_close=None, prior_spy_close=762.60) is None
        assert spy_total_return_pct(spy_close=765.72, prior_spy_close=None) is None
