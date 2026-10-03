"""The process-wide Polygon rate limiter and the dividend window query
(alpha-engine-config-I10047).

WHAT THESE GUARD. On 2026-09-04 the postclose EOD process ran two run dates in
one process and issued 28 Polygon requests in ~5 minutes against a
``calls_per_min=5`` free-tier budget: 13 per-ticker ``get_dividends`` calls per
date (12 held names + SPY) plus one split query. The log carried 16 ``Rate
limited (429)`` lines, CRUS and MU were recorded as having no ex-dividend when
in fact they were never measured, and the split query then failed leaving all
12 held names ``split_check_unresolved``.

Three compounding causes, one test class each:

* the fan-out — :class:`TestDividendWindowQuery` and the contract test below;
* the per-instance limiter — :class:`TestSharedWindow`: four call sites each
  built their own ``PolygonClient`` with its own timestamp deque, so each began
  with an empty window while the server's per-key budget was already spent;
* the 429 handler — :class:`TestRateLimitedRetry`: it slept ``Retry-After``
  (15s) and then CLEARED the window, retrying into a 60s server window it had
  just forgotten.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import polygon_client
from polygon_client import (
    PolygonAccessError,
    PolygonClient,
    PolygonRateLimitError,
    reset_shared_windows,
)

_KEY = "test-key-i10047"


class _Clock:
    """A fake monotonic clock that only advances when something sleeps."""

    def __init__(self, start: float = 1000.0):
        self.t = start
        self.sleeps: list[float] = []
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    reset_shared_windows()
    c = _Clock()
    monkeypatch.setattr(polygon_client.time, "monotonic", c.monotonic)
    monkeypatch.setattr(polygon_client.time, "sleep", c.sleep)
    yield c
    reset_shared_windows()


class _Response:
    def __init__(self, payload=None, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"results": []}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _CountingSession:
    """Stands in for ``requests.Session``; counts every transport call."""

    def __init__(self, responses=None):
        self.params: dict = {}
        self.requests: list[tuple] = []
        self._responses = list(responses or [])

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, dict(params or {})))
        if self._responses:
            return self._responses.pop(0)
        return _Response()


def _client(calls_per_min: int = 5, session: _CountingSession | None = None):
    c = PolygonClient(api_key=_KEY, calls_per_min=calls_per_min)
    c._session = session or _CountingSession()
    return c


class TestSharedWindow:
    """Cause 2: the limiter must see every call made on the API KEY."""

    def test_two_clients_share_one_window(self, clock):
        a = _client()
        b = _client()
        assert a._window is b._window

    def test_a_different_key_gets_its_own_window(self, clock):
        a = PolygonClient(api_key=_KEY, calls_per_min=5)
        b = PolygonClient(api_key="other-key", calls_per_min=5)
        assert a._window is not b._window

    def test_the_third_call_across_two_instances_waits(self, clock):
        """With ``calls_per_min=2``, two calls fill the minute — whichever
        instance makes the third one blocks. Before the fix the second client
        started with an empty deque and made it immediately."""
        a = _client(calls_per_min=2)
        b = _client(calls_per_min=2)

        a._wait_for_slot()
        b._wait_for_slot()
        assert clock.sleeps == []

        b._wait_for_slot()
        assert clock.sleeps == [pytest.approx(60.5)]

    def test_a_slot_frees_once_the_oldest_call_ages_out(self, clock):
        a = _client(calls_per_min=2)
        a._wait_for_slot()
        clock.sleep(30.0)
        b = _client(calls_per_min=2)
        b._wait_for_slot()
        clock.sleeps.clear()
        # The window holds calls at t and t+30; the next slot frees 30.5s on.
        a._wait_for_slot()
        assert clock.sleeps == [pytest.approx(30.5)]

    def test_every_executor_call_site_routes_through_the_shared_window(self, clock):
        """The issue's closes-when: the number of ``PolygonClient()``
        constructions under ``executor/`` equals the number routed through the
        shared limiter. That holds by construction — ``__init__`` binds the
        process-wide window for its key — provided every site really is this
        class and no module has grown a second, unpaced client."""
        root = Path(__file__).resolve().parents[1]
        sites: list[str] = []
        for path in sorted((root / "executor").rglob("*.py")):
            text = path.read_text()
            if "PolygonClient(" not in text:
                continue
            assert "from polygon_client import" in text, (
                f"{path.relative_to(root)} constructs a PolygonClient it does "
                "not import from polygon_client — a second, unpaced client"
            )
            for lineno, line in enumerate(text.splitlines(), 1):
                if "PolygonClient(" in line:
                    sites.append(f"{path.relative_to(root)}:{lineno}")
        assert sites, "the call sites this shared limiter exists for have moved"
        assert PolygonClient(api_key=_KEY)._window is polygon_client.shared_window(_KEY)

    def test_a_full_eod_shaped_burst_is_paced_not_dropped(self, clock):
        a = _client(calls_per_min=5)
        for _ in range(5):
            a._wait_for_slot()
        assert clock.sleeps == []
        b = _client(calls_per_min=5)
        b._wait_for_slot()
        assert len(clock.sleeps) == 1


class TestRateLimitedRetry:
    """Cause 3: the 429 handler must not shrink the window it is pacing on."""

    def test_429_does_not_clear_the_window_and_waits_the_longer_bound(self, clock):
        session = _CountingSession([
            _Response(status_code=429, headers={"Retry-After": "15"}),
            _Response({"results": [{"ticker": "LMT"}]}),
        ])
        c = _client(calls_per_min=3, session=session)
        c._wait_for_slot()
        c._wait_for_slot()

        depths: list[int] = []
        clock.on_sleep = lambda _s: depths.append(len(c._window._call_times))

        out = c._get("/v3/reference/dividends", params={})

        assert out == {"results": [{"ticker": "LMT"}]}
        # The window was FULL (3 calls) at the moment of the backoff — the old
        # code called `_call_times.clear()` here, which would show depth 0.
        assert depths == [3]
        # Retry-After was 15s but the shared window does not free a slot for
        # 60.5s; the larger bound is the one that is honoured.
        assert clock.sleeps == [pytest.approx(60.5)]

    def test_retry_after_wins_when_it_is_the_larger_bound(self, clock):
        session = _CountingSession([
            _Response(status_code=429, headers={"Retry-After": "42"}),
            _Response({"results": []}),
        ])
        c = _client(calls_per_min=5, session=session)
        c._get("/v3/reference/dividends", params={})
        # Window depth 1 of 5 → it frees a slot immediately, so Retry-After
        # is the binding constraint.
        assert clock.sleeps == [pytest.approx(42.0)]

    def test_the_retry_is_itself_metered(self, clock):
        session = _CountingSession([
            _Response(status_code=429, headers={"Retry-After": "42"}),
            _Response({"results": []}),
        ])
        c = _client(calls_per_min=5, session=session)
        c._get("/v3/reference/dividends", params={})
        # One slot for the original request, one for the retry.
        assert c._window.depth() == 2

    def test_four_consecutive_429s_still_raise(self, clock):
        session = _CountingSession([
            _Response(status_code=429) for _ in range(4)
        ])
        c = _client(calls_per_min=5, session=session)
        with pytest.raises(PolygonRateLimitError):
            c._get("/v3/reference/dividends", params={})
        assert len(session.requests) == 4
        # Exponential floor with no sleep after the final attempt.
        assert clock.sleeps == [pytest.approx(15.0), pytest.approx(30.0),
                                pytest.approx(60.0)]


class TestPaginationPage:
    """2026-10-02 postclose: page two of the dividend window 429'd one second
    after page one, ``_get_raw_url`` had no retry, the whole window query
    failed, and the HTTPError carried the API key (twice) into the alert."""

    _NEXT = ("https://api.polygon.io/v3/reference/dividends?cursor=YXA9MTAw"
             "&apiKey=SECRET")

    def test_a_429_on_a_next_url_page_is_retried(self, clock):
        session = _CountingSession([
            _Response(status_code=429),
            _Response({"results": [{"ticker": "B"}]}),
        ])
        c = _client(session=session)
        assert c._get_raw_url(self._NEXT) == {"results": [{"ticker": "B"}]}
        assert len(session.requests) == 2

    def test_the_page_url_carries_no_api_key_of_its_own(self, clock):
        session = _CountingSession()
        c = _client(session=session)
        c._get_raw_url("https://api.polygon.io/v3/reference/dividends?cursor=abc")
        c._get_raw_url(self._NEXT)
        for url, params in session.requests:
            assert "apiKey" not in url and "apiKey" not in params
            assert "cursor=" in url

    def test_http_errors_never_carry_the_key(self, clock):
        import requests

        class _Err(_Response):
            def raise_for_status(self):
                raise requests.HTTPError(
                    "500 Server Error for url: https://api.polygon.io/x?"
                    f"cursor=a&apiKey={_KEY}", response=self)

        session = _CountingSession([_Err(status_code=500)])
        c = _client(session=session)
        with pytest.raises(requests.HTTPError) as info:
            c._get_raw_url(self._NEXT)
        assert _KEY not in str(info.value)
        assert "apiKey=***" in str(info.value)
        assert info.value.__cause__ is None and info.value.__suppress_context__

    def test_transport_errors_never_carry_the_key(self, clock):
        import requests

        class _Boom(_CountingSession):
            def get(self, url, params=None, timeout=None):
                raise requests.ConnectionError(f"Max retries exceeded with url: /x?apiKey={_KEY}")

        c = _client(session=_Boom())
        with pytest.raises(requests.ConnectionError) as info:
            c._get("/x")
        assert _KEY not in str(info.value)


class TestDividendWindowQuery:
    """Cause 1: one window query replaces the per-ticker fan-out."""

    def _bare(self):
        return PolygonClient.__new__(PolygonClient)  # bypass __init__ (needs a key)

    def test_the_params_are_a_date_range_with_no_ticker_filter(self):
        c = self._bare()
        c._get = MagicMock(return_value={"results": []})
        c.get_dividends_for_window("2026-09-03", "2026-09-04")
        path, kwargs = c._get.call_args[0][0], c._get.call_args[1]
        assert path == "/v3/reference/dividends"
        params = kwargs["params"]
        assert params["ex_dividend_date.gte"] == "2026-09-03"
        assert params["ex_dividend_date.lte"] == "2026-09-04"
        assert params["sort"] == "ex_dividend_date"
        assert params["limit"] == 1000
        assert "ticker" not in params

    def test_paginates_on_next_url(self):
        c = self._bare()
        c._get = MagicMock(return_value={
            "results": [{"ticker": "A"}], "next_url": "https://x/page2",
        })
        c._get_raw_url = MagicMock(side_effect=[
            {"results": [{"ticker": "B"}], "next_url": "https://x/page3"},
            {"results": [{"ticker": "C"}]},
        ])
        rows = c.get_dividends_for_window("2026-09-03", "2026-09-04")
        assert [r["ticker"] for r in rows] == ["A", "B", "C"]
        assert c._get_raw_url.call_count == 2

    def test_raises_rather_than_reporting_an_unmeasured_window_as_empty(self):
        c = self._bare()
        c._get = MagicMock(side_effect=PolygonRateLimitError("rate limited"))
        with pytest.raises(PolygonRateLimitError):
            c.get_dividends_for_window("2026-09-03", "2026-09-04")

    def test_a_403_is_an_error_not_an_empty_day(self):
        """``_get`` renders a 403 as an empty result set, which on this path is
        indistinguishable from 'nothing went ex' — the silent-clean shape."""
        c = self._bare()
        c._get = MagicMock(return_value={
            "results": [], "resultsCount": 0, "status": "FORBIDDEN",
        })
        with pytest.raises(PolygonAccessError):
            c.get_dividends_for_window("2026-09-03", "2026-09-04")


class TestEodRequestBudget:
    """The contract test from the issue: a two-date postclose over a 12-name
    book issues at most FOUR Polygon requests, counted at the transport."""

    def test_two_dates_cost_at_most_four_requests(self, clock):
        from executor.dividends import fetch_ex_dividends
        from executor.reconciliation_audit import fetch_same_day_split_ratios

        book = ["AMD", "AXP", "BRO", "COST", "CRUS", "CTAS", "FAST", "LMT",
                "MA", "MU", "NVDA", "ORCL"]
        session = _CountingSession()
        # Both call sites in one process, exactly as postclose runs them: two
        # separate PolygonClient objects sharing one API key and one window.
        div_client = _client(calls_per_min=5, session=session)
        split_client = _client(calls_per_min=5, session=session)

        for run_date, prior_date in (
            ("2026-09-03", "2026-09-02"),
            ("2026-09-04", "2026-09-03"),
        ):
            _out, _pay, available, warning = fetch_ex_dividends(
                book, run_date, prior_date=prior_date, client=div_client,
            )
            assert available is True
            assert warning is None
            _ratios, unresolved = fetch_same_day_split_ratios(
                book, run_date, client=split_client,
            )
            assert unresolved == []

        assert len(session.requests) == 4
        # Four calls in one minute is inside the 5/min budget, so the shared
        # limiter never had to stall the run.
        assert clock.sleeps == []
