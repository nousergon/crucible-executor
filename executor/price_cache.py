"""
Load OHLCV price histories, ATR, and VWAP from the ArcticDB universe
library. ArcticDB is the sole source of truth — S3 parquet staging
artifacts (price_cache_slim, daily_closes) are no longer read directly
by the executor (2026-04-17).

ArcticDB layout:
    s3://{signals_bucket}/arcticdb/ — library "universe"
    Each symbol is a ticker with a DatetimeIndex frame of
    {Open, High, Low, Close, Volume, VWAP, atr_14_pct, ...}.
"""

from __future__ import annotations

# arcticdb MUST be imported before pandas on macOS to prime its bundled
# aws-c-common allocator before pyarrow (pulled in by pandas) loads its
# own copy. The two copies otherwise collide and arcticdb's S3Storage
# constructor segfaults with `aws_fatal_assert: allocator != ((void*)0)`
# on the first get_library() call. Linux runtimes (Lambda, EC2 Amazon
# Linux) are unaffected — dynamic linker resolves differently. arcticdb
# is a hard dep of the executor as of 2026-04-16 via requirements.txt;
# no fallback path, no optional import — feedback_no_silent_fails.
import arcticdb as _arcticdb  # noqa: F401  (kept for its side effect on import ordering)

import logging
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from executor.market_hours import is_trading_day

logger = logging.getLogger(__name__)


# Max staleness (in trading days) of the ATR feature before we hard-fail.
# 1 = yesterday's close is acceptable; anything older is treated as a
# pipeline-broken state and aborts the morning planner. Aligns with the
# predictor's own DailyData dependency expectation.
_ATR_MAX_STALENESS_TRADING_DAYS = 1

# Symbols that live in the ArcticDB `macro` library rather than `universe`.
# Mirrors the canonical writer list in alpha-engine-data's
# ``builders/daily_append.py`` (macro_keys + sector_etfs). Kept in sync
# manually; any additions there need matching updates here.
_MACRO_SYMBOLS = frozenset({
    "SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
})


def _open_universe_library(signals_bucket: str):
    """Open the ArcticDB `universe` library for reads.

    Per-ticker OHLCV + features. Does NOT include SPY/VIX/sector ETFs —
    those live in the `macro` library; use ``_open_macro_library`` for
    those.

    Hard-fails on connection/library errors per feedback_no_silent_fails.
    """
    adb = _arcticdb  # already imported at module top for macOS allocator prime
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = (
        f"s3s://s3.{region}.amazonaws.com:{signals_bucket}"
        f"?path_prefix=arcticdb&aws_auth=true"
    )
    arctic = adb.Arctic(uri)
    return arctic.get_library("universe")


def _open_macro_library(signals_bucket: str):
    """Open the ArcticDB `macro` library for reads.

    Market-wide time series: SPY, VIX, VIX3M, TNX, IRX, GLD, USO, and
    the XL* sector ETFs. Written by alpha-engine-data's daily_append
    and builders.backfill. SPY in particular is what the morning
    freshness gate and EOD reconcile check.

    Hard-fails on connection/library errors per feedback_no_silent_fails.
    """
    adb = _arcticdb
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = (
        f"s3s://s3.{region}.amazonaws.com:{signals_bucket}"
        f"?path_prefix=arcticdb&aws_auth=true"
    )
    arctic = adb.Arctic(uri)
    return arctic.get_library("macro")


def load_price_histories(
    tickers: list[str],
    signals_bucket: str,
) -> dict[str, "pd.DataFrame"]:
    """Load OHLCV histories for a list of tickers from ArcticDB.

    Routes per ticker: single-stock watchlist symbols are read from the
    ``universe`` library (full OHLCV); index ETFs / macro series
    (``SPY``/``VIX``/sector ETFs/etc., see ``_MACRO_SYMBOLS``) are read
    from the ``macro`` library (Close only; Open/High/Low default to 0.0
    for those symbols — sector-relative exit veto consumes Close only).

    ArcticDB is the sole source of truth — the S3 slim-cache parquet
    fallback was removed 2026-04-17. Library/read errors hard-fail
    (infrastructure broken). Individual tickers that return an empty
    frame are omitted from the result with an INFO log — downstream
    consumers (exit_manager, sector-relative veto) already handle
    missing tickers.

    Returns:
        {ticker: pd.DataFrame[open, high, low, close]} indexed by
        DatetimeIndex (UTC-naive normalized dates), sorted ascending.

    Shape note (2026-04-27): switched from ``list[dict]`` to
    ``pd.DataFrame`` so downstream consumers (``_compute_atr``,
    ``_compute_rsi``, ``check_correlation``, ``check_atr_trailing_stop``
    post_entry filter) can vectorize their per-bar arithmetic instead
    of running Python loops over dict lookups. Backtester's simulate
    loop hot path drops by ~10-50× per call; live executor loads once
    per boot, so the conversion cost (formerly inside iterrows here)
    becomes a free win — pandas keeps the columnar layout it already
    holds in ArcticDB.
    """
    if not tickers:
        return {}

    universe = _open_universe_library(signals_bucket)
    macro = None  # lazy-open only if a macro-routed ticker appears
    histories: dict[str, "pd.DataFrame"] = {}
    read_errors: list[str] = []
    empty: list[str] = []

    for ticker in tickers:
        if ticker in _MACRO_SYMBOLS:
            if macro is None:
                macro = _open_macro_library(signals_bucket)
            lib = macro
        else:
            lib = universe
        try:
            df = lib.read(ticker).data
        except Exception as e:
            read_errors.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty:
            empty.append(ticker)
            continue
        # Normalize to a 4-column lower-case OHLCV frame indexed by
        # DatetimeIndex. Macro routes (Close-only) get zero-filled OHL
        # to preserve column shape — downstream consumers filter on
        # Close-only paths via ``_MACRO_SYMBOLS`` upstream.
        out = pd.DataFrame({
            "open":  df["Open"].astype(float)  if "Open"  in df.columns else 0.0,
            "high":  df["High"].astype(float)  if "High"  in df.columns else 0.0,
            "low":   df["Low"].astype(float)   if "Low"   in df.columns else 0.0,
            "close": df["Close"].astype(float) if "Close" in df.columns else 0.0,
        }, index=df.index)
        # Strip intraday timestamps (ArcticDB writes UTC-midnight) so
        # ``df.loc[pd.Timestamp("2024-01-15"):]`` works against bare
        # date strings the executor compares against.
        if hasattr(out.index, "normalize"):
            out.index = out.index.normalize()
        histories[ticker] = out

    if read_errors:
        raise RuntimeError(
            f"load_price_histories ArcticDB read failed for {len(read_errors)} "
            f"ticker(s): {read_errors}. ArcticDB universe+macro libraries "
            f"must be reachable."
        )

    logger.info(
        "[data_source=arcticdb] Price histories loaded for %d/%d tickers "
        "(empty=%d)",
        len(histories), len(tickers), len(empty),
    )
    if empty:
        logger.info("Empty frame for %d ticker(s): %s", len(empty), sorted(empty))
    return histories


def load_atr_14_pct(
    tickers: list[str],
    signals_bucket: str,
    max_staleness_trading_days: int = _ATR_MAX_STALENESS_TRADING_DAYS,
    reference_date: date | None = None,
) -> dict[str, float]:
    """
    Read the most recent `atr_14_pct` value per ticker from the ArcticDB
    universe library. Single source of truth for ATR across the executor —
    pullback trigger scaling, position sizing, and trailing stops all
    consume from this map to eliminate intra-executor ATR-definition drift
    (previously each call site computed its own ATR via _compute_atr from
    raw OHLC, which could subtly diverge from the predictor's feature
    store definition of atr_14_pct).

    Values are stored in ArcticDB as decimals (e.g. 0.0238 = 2.38%),
    consistent with how the pullback trigger config's pullback_pct is
    interpreted, so no unit conversion is needed downstream.

    Hard-fails per feedback_hard_fail_until_stable:
      - arcticdb import failure (missing dep) → ImportError raised
      - ArcticDB connection/library access failure → original exception
        propagated (no silent fallback)
      - Any requested ticker missing `atr_14_pct` column → RuntimeError
      - Any requested ticker whose most-recent row is older than
        `max_staleness_trading_days` → RuntimeError
      - Any ticker with a non-finite or non-positive atr_14_pct → RuntimeError

    Args:
        tickers: Tickers to look up. Must all be present in universe library.
        signals_bucket: S3 bucket hosting the ArcticDB store (same as
                        research/predictor).
        max_staleness_trading_days: Reject data older than this many trading
                                    days from reference_date.
        reference_date: Date to measure staleness against. Defaults to today
                        (UTC). Pass an explicit date in tests.

    Returns:
        {ticker: atr_14_pct} for every requested ticker. Raises if any
        fails validation.
    """
    if not tickers:
        return {}

    universe = _open_universe_library(signals_bucket)

    ref = reference_date or datetime.now(timezone.utc).date()
    staleness_cutoff = _n_trading_days_back(ref, max_staleness_trading_days)

    atr_map: dict[str, float] = {}
    missing_feature: list[str] = []
    missing_symbol: list[str] = []
    stale: list[tuple[str, str]] = []
    invalid: list[tuple[str, float]] = []

    for ticker in tickers:
        try:
            df = universe.read(ticker).data
        except Exception as e:
            missing_symbol.append(f"{ticker} ({e.__class__.__name__})")
            continue

        if "atr_14_pct" not in df.columns:
            missing_feature.append(ticker)
            continue

        if df.empty:
            missing_symbol.append(f"{ticker} (empty frame)")
            continue

        last_dt = df.index[-1]
        last_date = last_dt.date() if hasattr(last_dt, "date") else pd.Timestamp(last_dt).date()
        if last_date < staleness_cutoff:
            stale.append((ticker, str(last_date)))
            continue

        val = float(df["atr_14_pct"].iloc[-1])
        if not (val == val and val > 0):  # NaN-safe positivity check
            invalid.append((ticker, val))
            continue

        atr_map[ticker] = val

    problems = []
    if missing_symbol:
        problems.append(f"missing_symbol={missing_symbol}")
    if missing_feature:
        problems.append(f"missing_feature={missing_feature}")
    if stale:
        problems.append(
            f"stale (older than {max_staleness_trading_days} trading day"
            f"{'s' if max_staleness_trading_days != 1 else ''} before "
            f"{ref}, cutoff={staleness_cutoff})={stale}"
        )
    if invalid:
        problems.append(f"non-finite-or-non-positive={invalid}")

    if problems:
        raise RuntimeError(
            "load_atr_14_pct failed validation — executor morning planner cannot "
            "proceed without a trustworthy ATR for every signal ticker. "
            f"Requested {len(tickers)} tickers, resolved {len(atr_map)}. "
            "Problems: " + "; ".join(problems)
        )

    logger.info(
        "[data_source=arcticdb] Loaded atr_14_pct for %d/%d tickers (cutoff=%s)",
        len(atr_map), len(tickers), staleness_cutoff,
    )
    return atr_map


# Columns that are NOT considered "features" when computing coverage.
# OHLCV + VWAP are raw market data (always populated post-ingest); feature
# columns are the engineered signals (atr_14_pct, rsi_14, momentum_60d,
# dist_from_52w_high, etc.) that may be NaN for short-history tickers.
_COVERAGE_OHLCV_COLS = frozenset({
    "Open", "High", "Low", "Close", "Adj_Close", "Volume", "VWAP",
})


def load_feature_coverage(
    tickers: list[str],
    signals_bucket: str,
) -> dict[str, float]:
    """Fraction of non-NaN feature columns in the most-recent ArcticDB
    universe row for each ticker.

    Coverage is defined as::

        coverage = non_nan_feature_cols / total_feature_cols

    where "feature cols" means every column EXCEPT the OHLCV+VWAP raw
    market data. A full-history ticker (AAPL with 10y of data) returns
    ~1.0. A short-history ticker (SNDK post-2025 spinoff with ~290 bars)
    returns < 1.0 because 252-day features (``dist_from_52w_high``,
    ``return_252d``, ``momentum_252d``) stay NaN on every row until the
    252-row warmup is filled.

    Used by:
      - Position sizer — derate ``shares`` by coverage so a 70%-covered
        ticker is sized 70% of a full-coverage ticker. Aligns position
        size with information completeness (post-PR #78).
      - Admission gate — refuse buy_candidates below a hard floor
        (``min_coverage_for_admission``, e.g. 0.30). Pure pre-history
        IPOs get rejected; partially-scoreable tickers get admitted
        with a derate.

    Failure semantics (intentionally tolerant — coverage is advisory):
      - Missing ticker from universe library → 0.0 coverage logged,
        downstream admission gate will reject.
      - ArcticDB library unreachable → RuntimeError (same as
        ``load_atr_14_pct`` — infrastructure problem, not data gap).
      - Ticker frame with zero feature columns (shouldn't happen post
        PR #78 migration) → 0.0 coverage, logged as WARNING.

    Args:
        tickers: Tickers to resolve coverage for.
        signals_bucket: S3 bucket hosting the ArcticDB store.

    Returns:
        ``{ticker: coverage}`` for every requested ticker. Tickers that
        failed the per-ticker read are present with value 0.0 so callers
        never silently lose a ticker.
    """
    if not tickers:
        return {}

    universe = _open_universe_library(signals_bucket)

    coverage_map: dict[str, float] = {}
    read_errors: list[str] = []
    empty_frames: list[str] = []
    no_features: list[str] = []

    for ticker in tickers:
        try:
            df = universe.read(ticker).data
        except Exception as e:
            read_errors.append(f"{ticker} ({e.__class__.__name__})")
            coverage_map[ticker] = 0.0
            continue

        if df.empty:
            empty_frames.append(ticker)
            coverage_map[ticker] = 0.0
            continue

        feature_cols = [c for c in df.columns if c not in _COVERAGE_OHLCV_COLS]
        if not feature_cols:
            no_features.append(ticker)
            coverage_map[ticker] = 0.0
            continue

        last_row = df[feature_cols].iloc[-1]
        non_nan = int(last_row.notna().sum())
        coverage_map[ticker] = non_nan / len(feature_cols)

    # read_errors are infrastructure-level — hard-fail consistent with
    # load_atr_14_pct. Tickers absent from the universe library return 0.0
    # coverage (they fail the admission gate naturally).
    if read_errors:
        raise RuntimeError(
            "load_feature_coverage ArcticDB read failed for "
            f"{len(read_errors)} ticker(s): {read_errors}. Not a "
            "data-gap — executor cannot trust coverage values when the "
            "universe library is unreachable."
        )

    if empty_frames:
        logger.warning(
            "load_feature_coverage: %d ticker(s) have empty ArcticDB frames "
            "— coverage set to 0.0, admission gate will reject: %s",
            len(empty_frames), empty_frames,
        )
    if no_features:
        logger.warning(
            "load_feature_coverage: %d ticker(s) have no feature columns "
            "in their ArcticDB frame (OHLCV-only) — coverage set to 0.0: %s",
            len(no_features), no_features,
        )

    min_cov = min(coverage_map.values()) if coverage_map else 0.0
    max_cov = max(coverage_map.values()) if coverage_map else 0.0
    logger.info(
        "[data_source=arcticdb] Loaded feature_coverage for %d tickers "
        "(min=%.2f, max=%.2f)",
        len(coverage_map), min_cov, max_cov,
    )
    return coverage_map


def _n_trading_days_back(ref: date, n: int) -> date:
    """Walk back `n` trading days from `ref` (inclusive of today if it's
    a trading day). Weekend/holiday skipping uses the same calendar the
    rest of the executor consults."""
    current = ref
    remaining = n
    # Start on a trading day
    while not is_trading_day(current):
        current -= timedelta(days=1)
    while remaining > 0:
        current -= timedelta(days=1)
        while not is_trading_day(current):
            current -= timedelta(days=1)
        remaining -= 1
    return current


def load_daily_vwap(
    tickers: list[str],
    signals_bucket: str,
    run_date: str | None = None,
    max_lookback: int = 5,
) -> dict[str, float]:
    """Load prior-day VWAP per ticker from the ArcticDB universe library.

    For each requested ticker, walks back from run_date (skipping
    weekends/holidays) and returns the most recent VWAP value within
    `max_lookback` trading days. Hard-fails if the universe library is
    unreachable or has no VWAP column. Tickers whose entire lookback
    window has no VWAP are raised as a single failure (no silent empty
    dict) — VWAP is a daemon entry-trigger input and must be trusted.
    """
    if not tickers:
        return {}

    universe = _open_universe_library(signals_bucket)
    start = date.fromisoformat(run_date) if run_date else date.today()

    # Build the list of candidate trading dates once — all tickers scan
    # the same window. Normalize to date for filtering.
    candidates: list[date] = []
    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        if candidate.weekday() > 4:
            continue
        if not is_trading_day(candidate):
            continue
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            f"No trading-day candidates within {max_lookback} days of {start}"
        )

    # Contract:
    #   HARD FAIL on library/read errors — infrastructure problem.
    #   PARTIAL COVERAGE (INFO log) when a ticker's frame has no VWAP column
    #       or no valid VWAP in the lookback window. VWAP was added to the
    #       universe schema 2026-04-17; historical ticker frames + yfinance-
    #       sourced rows legitimately lack it. The daemon's VWAP-discount
    #       trigger explicitly skips tickers with no VWAP (entry_triggers.py:
    #       `if vwap and vwap > 0`), so a documented data gap is tolerable
    #       while other triggers (pullback, support, time expiry) carry load.
    read_errors: list[str] = []
    no_vwap_column: list[str] = []
    no_valid_vwap_in_window: list[str] = []
    vwap_map: dict[str, float] = {}

    for ticker in tickers:
        try:
            df = universe.read(ticker).data
        except Exception as e:
            read_errors.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty or "VWAP" not in df.columns:
            no_vwap_column.append(ticker)
            continue
        # Find the most recent row whose index matches one of the candidate
        # trading days (normalized). First hit wins.
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        for cand in candidates:
            match = df[idx == pd.Timestamp(cand)]
            if match.empty:
                continue
            v = match["VWAP"].iloc[-1]
            if pd.notna(v) and v > 0:
                vwap_map[ticker] = float(v)
                break
        if ticker not in vwap_map:
            no_valid_vwap_in_window.append(ticker)

    if read_errors:
        raise RuntimeError(
            f"load_daily_vwap ArcticDB read failed for {len(read_errors)} "
            f"ticker(s): {read_errors}. Daemon cannot plan triggers without "
            "a trusted universe library."
        )

    logger.info(
        "[data_source=arcticdb] VWAP resolved for %d/%d tickers "
        "(window ≤ %s, no_column=%d, no_valid=%d)",
        len(vwap_map), len(tickers), start,
        len(no_vwap_column), len(no_valid_vwap_in_window),
    )
    if no_vwap_column:
        logger.info(
            "VWAP column absent for %d ticker(s) — daemon skips VWAP "
            "trigger for these: %s",
            len(no_vwap_column), sorted(no_vwap_column),
        )
    if no_valid_vwap_in_window:
        logger.info(
            "No valid VWAP in %d-day window for %d ticker(s): %s",
            max_lookback, len(no_valid_vwap_in_window),
            sorted(no_valid_vwap_in_window),
        )
    return vwap_map
