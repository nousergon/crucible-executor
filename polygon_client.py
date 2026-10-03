"""
Polygon.io (Massive) market data client with rate limiting and dividend adjustment.

Replaces yfinance as primary price data source. Free tier: 5 API calls/min,
~2 years historical depth, EOD data only. Index tickers (VIX/TNX/IRX) are
not available on free tier — use FRED or yfinance for those.

Usage:
    from polygon_client import PolygonClient, polygon_client

    # Singleton (reads POLYGON_API_KEY from env):
    client = polygon_client()
    bars = client.get_daily_bars("AAPL", "2025-01-01", "2026-03-28")

    # Dividend-adjusted (matches yfinance auto_adjust=True):
    bars = client.get_daily_bars_dividend_adjusted("XOM", "2025-01-01", "2026-03-28")

    # All US stocks for a single date:
    prices = client.get_grouped_daily("2026-03-28")
    # -> {"AAPL": {"open": 253.9, "high": 255.5, ...}, ...}
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from nousergon_lib.secrets import get_secret

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_MAX_BARS_PER_REQUEST = 50_000  # polygon limit param max
_WINDOW_SECONDS = 60.0  # Polygon's free-tier budget is per rolling minute


# 429 backoff for every request, paginated pages included. See _request.
_MAX_429_ATTEMPTS = 4
_BACKOFF_429_BASE = 15.0
_BACKOFF_429_CAP = 60.0

# Polygon authenticates via the ``apiKey`` querystring, so the str() of a
# requests error embeds the live key, and it reached the ERROR alert and the
# box logs on 2026-10-02. Every error raised from this client is scrubbed.
_API_KEY_RE = re.compile(r"(?:apiKey|api_key)=[^&\s]+")


def _scrub_api_key(msg: object) -> str:
    """Mask the ``apiKey=...`` (or ``api_key=...``) querystring value."""
    return _API_KEY_RE.sub(lambda m: m.group(0).split("=", 1)[0] + "=***", str(msg))


def _strip_api_key(url: str) -> str:
    """Drop every ``apiKey`` query parameter from ``url``."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k not in ("apiKey", "api_key")]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _scrubbed(exc: requests.RequestException) -> requests.RequestException:
    """A same-type copy of ``exc`` whose message carries no API key."""
    response = getattr(exc, "response", None)
    try:
        return type(exc)(_scrub_api_key(exc), response=response)
    except TypeError:
        return requests.RequestException(_scrub_api_key(exc), response=response)


class PolygonRateLimitError(Exception):
    """Raised when rate limit is exhausted and caller should backoff."""


class PolygonAccessError(Exception):
    """Raised when Polygon refuses the request (403) on a path whose caller
    cannot distinguish "not authorized" from "nothing to report"."""


# ── Shared per-API-key rate limiter ───────────────────────────────────────
#
# WHY THIS IS MODULE-LEVEL (alpha-engine-config-I10047). The budget Polygon
# meters is per API KEY, not per client object. Every call site in this repo
# constructs its own ``PolygonClient()`` — ``executor/dividends.py``,
# ``executor/reconciliation_audit.py`` and three sites in
# ``executor/pnl_measurement_backfill.py`` — and each carried its own
# ``deque`` of call timestamps, so the split client began with an empty window
# immediately after the dividend client had spent the whole minute's budget.
# Measured on 2026-09-04: a two-date postclose issued 28 requests in ~5 minutes
# against ``calls_per_min=5`` and logged 16 ``Rate limited (429)`` lines, after
# which two dividend fetches and the whole split query failed.
#
# One window per key, shared by every instance in the process, is what makes
# the limiter's view match the server's. It is keyed rather than a plain
# singleton so a second key (a paid tier, a test key) is metered separately.


class _SharedWindow:
    """A sliding 60-second call window shared by every client on one API key.

    Thread-safe: :meth:`acquire` reserves its slot under the lock, so two
    threads cannot both observe the same free slot and take it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_times: deque[float] = deque()

    def _purge(self, now: float) -> None:
        while self._call_times and now - self._call_times[0] > _WINDOW_SECONDS:
            self._call_times.popleft()

    def seconds_until_slot(self, limit: int) -> float:
        """Seconds until the window has room for one more call (0.0 if now)."""
        with self._lock:
            now = time.monotonic()
            self._purge(now)
            if len(self._call_times) < limit:
                return 0.0
            return _WINDOW_SECONDS - (now - self._call_times[0]) + 0.5

    def acquire(self, limit: int) -> None:
        """Block until a slot is free, then record this call in the window."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._purge(now)
                if len(self._call_times) < limit:
                    self._call_times.append(now)
                    return
                wait = _WINDOW_SECONDS - (now - self._call_times[0]) + 0.5
            logger.debug("Rate limit: waiting %.1fs for a shared slot", wait)
            time.sleep(wait)

    def depth(self) -> int:
        """Number of calls currently inside the window (diagnostics/tests)."""
        with self._lock:
            self._purge(time.monotonic())
            return len(self._call_times)


_windows: dict[str, _SharedWindow] = {}
_windows_lock = threading.Lock()


def shared_window(api_key: str) -> _SharedWindow:
    """The process-wide call window for ``api_key`` (created on first use)."""
    with _windows_lock:
        window = _windows.get(api_key)
        if window is None:
            window = _SharedWindow()
            _windows[api_key] = window
        return window


def reset_shared_windows() -> None:
    """Drop every shared window. For test isolation only."""
    with _windows_lock:
        _windows.clear()


class PolygonClient:
    """Rate-limited polygon.io REST client with dividend adjustment."""

    def __init__(self, api_key: str | None = None, calls_per_min: int = 5):
        self._api_key = api_key or get_secret("POLYGON_API_KEY", required=False, default="")
        if not self._api_key:
            raise ValueError("POLYGON_API_KEY not set")
        self._calls_per_min = calls_per_min
        # Shared with every other client on this key in this process — see
        # the _SharedWindow comment above (alpha-engine-config-I10047).
        self._window = shared_window(self._api_key)
        self._session = requests.Session()
        self._session.params = {"apiKey": self._api_key}  # type: ignore[assignment]

    # ── Rate limiter ──────────────────────────────────────────────────────

    def _wait_for_slot(self) -> None:
        """Block until a rate limit slot is available in the SHARED window.

        ``calls_per_min`` keeps its per-instance meaning — it is the ceiling
        this client will let the shared window reach — but the timestamps it
        counts are every call made on this API key in this process.
        """
        self._window.acquire(self._calls_per_min)

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a rate-limited GET request. Handles 429 with retry."""
        return self._request(f"{_BASE_URL}{path}", params or {}, label=path)

    def _get_raw_url(self, url: str) -> dict:
        """GET a full URL (for pagination ``next_url``).

        Goes through the same 429 retry loop as :meth:`_get`. It used to call
        ``raise_for_status`` directly, so a 429 on page two of a paginated
        query failed the whole query on the first try (2026-10-02 postclose:
        the dividend window's second page 429'd one second after the first
        page and the day's dividend accrual went unmeasured).

        Any ``apiKey`` already on the URL is dropped: the session's params add
        it, and the old code appended a second copy on top of that.
        """
        url = _strip_api_key(url)
        return self._request(url, {}, label=_scrub_api_key(url.split("?", 1)[0]))

    def _request(self, url: str, params: dict, *, label: str) -> dict:
        """One metered GET with bounded 429 backoff; errors are key-scrubbed.

        The key is shared fleet-wide, so a 429 can arrive while this process's
        own window is nearly empty (the data collector spends the same budget
        minutes earlier). The wait after the ``n``-th 429 is the longest of the
        server's ``Retry-After``, an exponential floor (15s, 30s, 60s) and the
        time until our own window frees a slot; there is no sleep after the
        final attempt, because nothing follows it. Worst case ~105s, which
        spans a full rate-limit window before ``PolygonRateLimitError``.
        """
        self._wait_for_slot()
        for attempt in range(_MAX_429_ATTEMPTS):
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                raise _scrubbed(exc) from None
            if resp.status_code == 429:
                if attempt == _MAX_429_ATTEMPTS - 1:
                    break
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                backoff = min(_BACKOFF_429_BASE * (2 ** attempt), _BACKOFF_429_CAP)
                # Do NOT clear the window: the old code cleared it and retried
                # into a 60s server window it had just forgotten
                # (alpha-engine-config-I10047).
                window_wait = self._window.seconds_until_slot(self._calls_per_min)
                wait = max(retry_after, backoff, window_wait)
                logger.warning(
                    "Rate limited (429) on %s, waiting %.1fs (attempt %d/%d, "
                    "Retry-After=%.0fs, shared window frees a slot in %.1fs)",
                    label, wait, attempt + 1, _MAX_429_ATTEMPTS, retry_after,
                    window_wait,
                )
                time.sleep(wait)
                # The retry is another metered call — book it in the window.
                self._wait_for_slot()
                continue
            if resp.status_code == 403:
                data = resp.json()
                msg = data.get("message", "Not authorized")
                logger.warning("Polygon 403: %s (path=%s)", msg, label)
                return {"results": [], "resultsCount": 0, "status": "FORBIDDEN"}
            try:
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise _scrubbed(exc) from None
            return resp.json()
        raise PolygonRateLimitError(
            f"Rate limited (429) on {label} after {_MAX_429_ATTEMPTS} attempts"
        )

    # ── Core endpoints ────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a single ticker.

        Returns DataFrame with DatetimeIndex and columns:
        [Open, High, Low, Close, Volume]

        Prices are split-adjusted (adjusted=True) but NOT dividend-adjusted.
        Use get_daily_bars_dividend_adjusted() for fully-adjusted prices.
        """
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": _MAX_BARS_PER_REQUEST,
        }
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            params=params,
        )
        results = data.get("results", [])
        if not results:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
        df = df.sort_index()
        return df

    def get_grouped_daily(self, date_str: str) -> dict[str, dict]:
        """Fetch OHLCV for ALL US stocks on a single date.

        Returns {ticker: {"open": float, "high": float, "low": float,
                          "close": float, "volume": float}}
        """
        data = self._get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            params={"adjusted": "true"},
        )
        results = data.get("results", [])
        return {
            r["T"]: {
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            }
            for r in results
            if "T" in r
        }

    def get_dividends(
        self,
        ticker: str,
        start: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch dividend history for a ticker.

        Returns list of dicts with keys:
        ex_dividend_date, cash_amount, frequency, declaration_date, pay_date, etc.
        """
        params: dict = {"ticker": ticker, "limit": limit, "sort": "ex_dividend_date"}
        if start:
            params["ex_dividend_date.gte"] = start
        all_dividends: list[dict] = []
        next_url: str | None = None

        # First page
        data = self._get("/v3/reference/dividends", params=params)
        all_dividends.extend(data.get("results", []))
        next_url = data.get("next_url")

        # Paginate
        while next_url:
            resp = self._get_raw_url(next_url)
            all_dividends.extend(resp.get("results", []))
            next_url = resp.get("next_url")

        return all_dividends

    def get_dividends_for_window(
        self,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch EVERY dividend going ex in ``[start, end]``, across all tickers.

        One ``/v3/reference/dividends?ex_dividend_date.gte=<start>&
        ex_dividend_date.lte=<end>`` call (plus any ``next_url`` pages) rather
        than one call per held ticker. Returns the raw Polygon rows — each
        carries ``ticker``, ``ex_dividend_date``, ``cash_amount`` and
        ``pay_date`` — with the same semantics as :meth:`get_dividends`.

        **Why this exists** (alpha-engine-config-I10047). The per-ticker loop in
        ``executor/dividends.py`` issued 13 calls per run date (12 held names +
        SPY) against a 5-calls/min free-tier budget; a two-date postclose on
        2026-09-04 therefore issued 28 requests in ~5 minutes, logged 16
        ``Rate limited (429)`` lines and left CRUS and MU unmeasured. The book
        is intersected against this result locally, so the cost is one call
        regardless of how many positions are held. This is the exact shape
        :meth:`get_splits_for_date` took for the same reason (I9646).

        Note the bounds are a RANGE, not the exact-match filter the splits
        endpoint takes: the dividend interval the caller reconciles over is
        ``(prior_date, run_date]``, which can span a skipped session, so both
        ``.gte`` and ``.lte`` are sent and the half-open lower edge is applied
        locally by ``executor.dividends._in_interval``.

        Raises (``PolygonRateLimitError``, ``PolygonAccessError``, HTTP error)
        rather than returning an empty list it could not measure. An empty
        return from this method is a POSITIVE finding that nothing went ex in
        the window; the caller is entitled to treat it as one.
        """
        params: dict = {
            "ex_dividend_date.gte": start,
            "ex_dividend_date.lte": end,
            "limit": limit,
            "sort": "ex_dividend_date",
        }
        data = self._get("/v3/reference/dividends", params=params)
        if data.get("status") == "FORBIDDEN":
            # _get renders a 403 as an empty result set, which on this path is
            # indistinguishable from "no dividends" — refuse rather than guess.
            raise PolygonAccessError(
                f"Polygon refused /v3/reference/dividends for [{start}, {end}] "
                "(403) — the window is UNMEASURED, not empty"
            )
        all_dividends: list[dict] = list(data.get("results", []))
        next_url: str | None = data.get("next_url")
        while next_url:
            resp = self._get_raw_url(next_url)
            all_dividends.extend(resp.get("results", []))
            next_url = resp.get("next_url")
        return all_dividends

    def get_splits(
        self,
        ticker: str,
        start: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch stock-split history for a ticker (``/v3/reference/splits``).

        Returns list of dicts with keys: ``ticker``, ``execution_date``
        (the ex-date the split takes effect on the broker's book),
        ``split_from``, ``split_to`` (e.g. a 2-for-1 split is
        ``split_from=1, split_to=2`` → share count multiplies by 2; a 1-for-10
        reverse split is ``split_from=10, split_to=1`` → ×0.1).

        Mirrors :meth:`get_dividends` (same auth, rate limiter, pagination).
        Used by reconciliation to split-adjust the ledger's pre-action share
        count so a same-day corporate action that changes IB's book with no
        corresponding ledger trade does not register as a false mismatch
        (config#1682).
        """
        params: dict = {"ticker": ticker, "limit": limit, "sort": "execution_date"}
        if start:
            params["execution_date.gte"] = start
        all_splits: list[dict] = []
        next_url: str | None = None

        # First page
        data = self._get("/v3/reference/splits", params=params)
        all_splits.extend(data.get("results", []))
        next_url = data.get("next_url")

        # Paginate
        while next_url:
            resp = self._get_raw_url(next_url)
            all_splits.extend(resp.get("results", []))
            next_url = resp.get("next_url")

        return all_splits

    def get_splits_for_date(self, execution_date: str, limit: int = 1000) -> list[dict]:
        """Fetch EVERY split executing on ``execution_date``, across all tickers.

        One ``/v3/reference/splits?execution_date=<d>`` call (plus any
        ``next_url`` pages) rather than one call per held ticker. Returns the
        raw Polygon rows — ``ticker``, ``execution_date``, ``split_from``,
        ``split_to`` — with the same semantics as :meth:`get_splits`.

        **Why this exists** (alpha-engine-config-I9646). The per-ticker loop
        issued N calls against a 5-calls/min free-tier budget; on 2026-08-31 a
        twelve-name book hit ``PolygonRateLimitError`` on three consecutive EOD
        runs, leaving 1-2 names' split status unestablished each time. The book
        is intersected against this result locally, so the cost is one call
        regardless of how many positions are held.

        ``execution_date`` is an EXACT-match filter, not the ``execution_date.gte``
        cursor bound :meth:`get_splits` uses. Verified live against Polygon
        2026-08-31: ``execution_date=2026-08-31`` returned 14 rows, every one
        carrying that exact ``execution_date``, in a single page.

        Raises on failure (``PolygonRateLimitError``, HTTP error). The caller
        decides what an unestablished split status means; this method never
        reports an empty day it could not measure as a clean one.
        """
        params: dict = {"execution_date": execution_date, "limit": limit}
        data = self._get("/v3/reference/splits", params=params)
        all_splits: list[dict] = list(data.get("results", []))
        next_url: str | None = data.get("next_url")
        while next_url:
            resp = self._get_raw_url(next_url)
            all_splits.extend(resp.get("results", []))
            next_url = resp.get("next_url")
        return all_splits

    # ── Dividend adjustment ───────────────────────────────────────────────

    def get_daily_bars_dividend_adjusted(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily bars with full adjustment (splits + dividends).

        Produces prices equivalent to yfinance auto_adjust=True.
        """
        bars = self.get_daily_bars(ticker, start, end, adjusted=True)
        if bars.empty:
            return bars

        divs = self.get_dividends(ticker, start=start)
        if not divs:
            return bars  # No dividends → split-adjusted is sufficient

        return _apply_dividend_adjustment(bars, divs)

    # ── Batch helpers ─────────────────────────────────────────────────────

    def fetch_batch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        dividend_adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple tickers with rate limiting.

        Returns dict[ticker, DataFrame].
        """
        results: dict[str, pd.DataFrame] = {}
        fetch_fn = (
            self.get_daily_bars_dividend_adjusted
            if dividend_adjusted
            else self.get_daily_bars
        )
        for i, ticker in enumerate(tickers):
            try:
                df = fetch_fn(ticker, start, end)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", ticker, e)
            if (i + 1) % 50 == 0:
                logger.info("Batch progress: %d/%d tickers", i + 1, len(tickers))
        return results

    def get_single_close(self, ticker: str, date_str: str) -> float | None:
        """Get closing price for a single ticker on a single date.

        Tries grouped daily first (if we happen to have it cached),
        falls back to per-ticker bars.
        """
        bars = self.get_daily_bars(ticker, date_str, date_str, adjusted=True)
        if not bars.empty:
            return float(bars["Close"].iloc[-1])
        return None


# ── Dividend adjustment logic ─────────────────────────────────────────────

def _apply_dividend_adjustment(
    bars: pd.DataFrame,
    dividends: list[dict],
) -> pd.DataFrame:
    """Apply backward dividend adjustment to split-adjusted OHLCV bars.

    For each bar date, computes:
        factor = product(1 - div_amount / close_before_ex)
        for all dividends with ex_date > bar_date

    Then: adjusted_price = split_adjusted_price * factor
    """
    df = bars.copy()
    price_cols = ["Open", "High", "Low", "Close"]

    # Parse and sort dividends by ex-date ascending
    last_bar_date = df.index[-1]
    div_records = []
    for d in dividends:
        ex_date = d.get("ex_dividend_date")
        amount = d.get("cash_amount")
        if ex_date and amount and float(amount) > 0:
            ex_ts = pd.Timestamp(ex_date)
            # Skip future dividends not yet ex within the data range
            if ex_ts > last_bar_date:
                continue
            div_records.append({
                "ex_date": ex_ts,
                "amount": float(amount),
            })
    if not div_records:
        return df

    div_records.sort(key=lambda x: x["ex_date"])

    # For each dividend, find the close price on the trading day before ex-date
    # to compute the adjustment ratio
    adjustment_factors = []
    for div in div_records:
        ex_date = div["ex_date"]
        # Find closest trading day before ex-date
        prior_bars = df[df.index < ex_date]
        if prior_bars.empty:
            # Dividend ex-date is before our data range — skip
            continue
        close_before = prior_bars["Close"].iloc[-1]
        if close_before <= 0:
            continue
        ratio = 1.0 - div["amount"] / close_before
        if ratio <= 0 or ratio > 1:
            logger.warning(
                "Skipping suspicious dividend ratio %.4f (amount=%.2f, close=%.2f)",
                ratio, div["amount"], close_before,
            )
            continue
        adjustment_factors.append({"ex_date": ex_date, "ratio": ratio})

    if not adjustment_factors:
        return df

    # Apply cumulative backward adjustment:
    # Bars before the earliest ex-date get ALL factors applied
    # Bars between ex-dates get progressively fewer factors
    # Bars on/after the latest ex-date get no adjustment
    for col in price_cols:
        adjusted = df[col].copy()
        for af in adjustment_factors:
            mask = df.index < af["ex_date"]
            adjusted[mask] *= af["ratio"]
        df[col] = adjusted

    return df


# ── Singleton ─────────────────────────────────────────────────────────────

_singleton: PolygonClient | None = None


def polygon_client(api_key: str | None = None) -> PolygonClient:
    """Get or create a singleton PolygonClient."""
    global _singleton
    if _singleton is None:
        _singleton = PolygonClient(api_key=api_key)
    return _singleton
