"""
Read signals.json from S3 and return parsed, validated signal data.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.universe import filter_to_universe

logger = logging.getLogger(__name__)


def read_predictions(s3_bucket: str) -> tuple[dict[str, dict], str | None]:
    """
    Read predictor/predictions/latest.json from S3.

    Returns: ``({ticker: prediction_dict}, predictions_date)`` where
    ``predictions_date`` is the top-level ``date`` field on the JSON
    payload (the ``predictions/{date}.json`` filename date the GBM run
    produced). Returns ``({}, None)`` if not available.

    The date is surfaced separately because ``latest.json`` is a pointer
    that may resolve to a prior trading day's predictions during the
    Saturday/holiday window — readers (esp. trade logging for
    transparency lineage) need the actual filename date, not today's
    date. See ROADMAP "Phase 2 transparency-inventory" → trade execution
    decisions row.
    """
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key="predictor/predictions/latest.json")
        data = json.loads(obj["Body"].read())
        if data.get("timed_out"):
            logger.warning(
                "Predictions timed out — using partial set (%d tickers)",
                len(data.get("predictions", [])),
            )
        preds = data.get("predictions", [])
        result = {p["ticker"]: p for p in preds if "ticker" in p}
        predictions_date = data.get("date")
        logger.info(
            "Predictions loaded | n=%d | date=%s", len(result), predictions_date,
        )
        return result, predictions_date
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning("predictions/latest.json not found — running without GBM input")
            return {}, None
        raise


def read_signals(s3_bucket: str, run_date: str | None = None) -> dict:
    """
    Download signals/{date}/signals.json from S3.
    Returns parsed signals dict. Raises ClientError if not found.
    """
    d = run_date or str(date.today())
    key = f"signals/{d}/signals.json"
    s3 = boto3.client("s3")
    logger.info(f"Reading signals from s3://{s3_bucket}/{key}")
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    data = json.loads(obj["Body"].read())
    logger.info(
        f"Signals loaded | market_regime={data.get('market_regime')} "
        f"| universe={len(data.get('universe', []))} "
        f"| candidates={len(data.get('buy_candidates', []))}"
    )
    return data


def read_signals_with_fallback(s3_bucket: str, run_date: str | None = None, max_lookback: int = 14) -> dict:
    """
    Read the latest signals from S3.

    Tries signals/latest.json first (written by Research alongside the dated file).
    Falls back to date-scanning if the pointer doesn't exist.

    The default max_lookback of 14 days covers the Research pipeline's weekly
    cadence (Saturday 00:00 UTC) plus a one-week buffer for a missed Saturday
    run. Shorter windows are fragile: by Friday of any given week, the most
    recent Saturday signals file is already 6 days old, and a 5-day window
    would fail even on a normal week.

    A staleness WARNING is logged when the signals being returned are more
    than 7 calendar days old, so a quietly-missed Saturday run becomes visible
    in the executor's log stream even if the signals still load successfully.

    Returns the signals dict. Raises RuntimeError if nothing found.
    """
    s3 = boto3.client("s3")

    # Try the latest.json pointer first
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key="signals/latest.json")
        data = json.loads(obj["Body"].read())
        logger.info(
            f"Signals loaded from signals/latest.json | date={data.get('date')} "
            f"| universe={len(data.get('universe', []))}"
        )
        _warn_if_stale(data.get("date"), run_date)
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("signals/latest.json not found — falling back to date scan")
        else:
            raise

    # Fallback: scan backward by date
    start = date.fromisoformat(run_date) if run_date else date.today()
    tried: list[str] = []

    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        try:
            signals = read_signals(s3_bucket, str(candidate))
            if days_back > 0:
                log_fn = logger.warning if days_back > 7 else logger.info
                log_fn(
                    f"No signals for {start} — using {candidate} "
                    f"({days_back} calendar day(s) old). Dates tried: {tried}. "
                    f"Research pipeline may have missed a weekly run."
                    if days_back > 7
                    else f"No signals for {start} — using {candidate} "
                    f"({days_back} calendar day(s) old). Dates tried: {tried}"
                )
            return signals
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"No signals file for {candidate}, looking further back...")
                tried.append(str(candidate))
                continue
            raise

    raise RuntimeError(
        f"No signals found within {max_lookback} calendar days of {start}. "
        f"Dates tried: {tried}. Check that the research pipeline ran recently."
    )


def _warn_if_stale(signals_date: str | None, run_date: str | None) -> None:
    """Log a WARNING if the loaded signals are more than 7 days old relative to run_date.

    Staleness >7 days means the Research pipeline likely missed its Saturday
    run — the signals will still work, but something upstream needs investigation.
    """
    if not signals_date:
        return
    try:
        sig = date.fromisoformat(signals_date)
    except (ValueError, TypeError):
        return
    ref = date.fromisoformat(run_date) if run_date else date.today()
    age = (ref - sig).days
    if age > 7:
        logger.warning(
            f"Loaded signals are {age} calendar days old (signals_date={sig}, "
            f"run_date={ref}). Research pipeline may have missed a weekly run."
        )


def filter_buy_candidates_to_universe(
    signals: dict,
    signals_bucket: str,
) -> dict:
    """Drop buy_candidates whose tickers aren't in the ArcticDB universe library.

    Defense-in-depth layer. Research's ``population_selector.
    compute_exits_and_open_slots`` (alpha-engine-research#41) is the
    primary universe guardrail — it drops non-S&P incumbents at the
    exit-evaluator stage. This function is the caller-side net: if a
    ticker somehow sneaks past (manual signals.json edit, a Research
    bug, a universe-drift window where DataPhase1 hasn't repopulated
    the ArcticDB library yet), dropping here prevents sizing positions
    against data that doesn't exist.

    Scope: ``buy_candidates`` only. The ``universe`` list (EXIT/REDUCE/
    HOLD for existing holdings) is left unfiltered — if we somehow
    hold a position outside the universe we still need to exit it,
    and per-ticker ArcticDB reads for ATR/VWAP on those will surface
    as clear named errors downstream if they fail.

    If the ArcticDB read itself fails (library unreachable, IAM miss),
    the filter is skipped with a WARNING — better to let the
    executor's own ArcticDB reads surface that as their own, clearer
    error than to block on a defense-in-depth layer.

    Origin: 2026-04-20 — TSM + ASML persisted as population incumbents
    despite being absent from constituents.json; manifested as
    ``NoSuchVersionException`` deep in executor-sim replay.
    """
    buy = signals.get("buy_candidates") or []
    if not buy:
        return signals

    try:
        # Local import — avoids top-level circular (price_cache imports
        # executor.market_hours which touches signal_reader indirectly).
        from executor.price_cache import _open_universe_library
        universe_lib = _open_universe_library(signals_bucket)
        universe_symbols = frozenset(universe_lib.list_symbols())
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning(
            "Skipping buy-candidate universe filter — could not open ArcticDB "
            "universe library: %s. Executor's direct ArcticDB reads will surface "
            "any data issues as their own named errors downstream.",
            exc,
        )
        return signals

    # Membership predicate is delegated to ``alpha_engine_lib.universe`` so
    # this Layer 2 filter and research's Layer 1 ``population_selector`` filter
    # share one canonical code path (no silent divergence on universe drift —
    # see lib v0.13.0 docstring).
    allowed, dropped_entries = filter_to_universe(buy, universe_symbols)
    dropped_tickers = [
        entry["ticker"] for entry in dropped_entries
        if isinstance(entry, dict) and isinstance(entry.get("ticker"), str)
    ]

    if dropped_tickers:
        logger.warning(
            "[signal_reader] dropped %d buy_candidate(s) not in ArcticDB "
            "universe: %s. Research's population_selector (alpha-engine-"
            "research#41) should have caught these — the fact that they "
            "reached here means one of: (a) Research bug, (b) manual edit "
            "to signals.json, (c) universe-library drift window. Not hard-"
            "failing because the remaining %d buy_candidate(s) are valid.",
            len(dropped_tickers),
            dropped_tickers,
            len(allowed),
        )
        signals = dict(signals)  # shallow copy to avoid mutating caller's dict
        signals["buy_candidates"] = allowed

    return signals


def filter_buy_candidates_by_coverage(
    signals: dict,
    coverage_map: dict[str, float],
    min_coverage: float,
) -> dict:
    """Drop buy_candidates whose feature coverage is below ``min_coverage``.

    Admission gate — the hard lower bound of the graceful-degrade chain
    introduced 2026-04-21 evening + 2026-04-22 (Brian's aggressive-new-listings
    posture reconfirmation). Coverage comes from ``price_cache.load_feature_coverage``;
    tickers absent from the ArcticDB universe library appear with 0.0 and
    naturally fail this gate.

    Scope: ``buy_candidates`` only. Held positions (``universe`` list —
    EXIT/REDUCE/HOLD) are NEVER filtered here. A held ticker whose
    coverage drops below threshold still needs its exit/management path
    evaluated (stop-loss, drawdown sizing, etc.) — admission-refuse
    applies to NEW ENTRY decisions, not to unwinding existing exposure.

    Rejected tickers are logged with their named coverage + threshold +
    top missing features and emitted to the ``admission_refused``
    CloudWatch metric so low-coverage admissions are observable.
    """
    buy = signals.get("buy_candidates") or []
    if not buy:
        return signals

    allowed: list[dict] = []
    refused: list[tuple[str, float]] = []
    for entry in buy:
        ticker = entry.get("ticker") if isinstance(entry, dict) else None
        if not ticker:
            continue
        cov = coverage_map.get(ticker, 0.0)
        if cov >= min_coverage:
            allowed.append(entry)
        else:
            refused.append((ticker, cov))

    if refused:
        logger.warning(
            "[signal_reader] admission gate refused %d buy_candidate(s) "
            "below min_coverage=%.2f: %s. ``REFUSED_INSUFFICIENT_COVERAGE`` "
            "— pure pre-history IPOs or extremely short-history tickers "
            "cannot be meaningfully scored on long-window features. The %d "
            "remaining candidate(s) will be sized normally (position sizer "
            "will derate any partial-coverage tickers via "
            "``coverage_sizing_enabled``).",
            len(refused), min_coverage,
            [(t, round(c, 3)) for t, c in refused],
            len(allowed),
        )
        _emit_admission_refused_metric(len(refused))

        signals = dict(signals)  # avoid mutating caller
        signals["buy_candidates"] = allowed

    return signals


def _emit_admission_refused_metric(count: int) -> None:
    """Emit ``AlphaEngine/Executor/admission_refused_count`` gauge.

    Best-effort: CloudWatch errors WARN but don't fail the planner —
    admission gate decision is the load-bearing path, metrics are
    observability. Parallel shape to ``_emit_unscored_count_metric``.
    """
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Executor",
            MetricData=[{
                "MetricName": "admission_refused_count",
                "Value": float(count),
                "Unit": "Count",
            }],
        )
    except Exception as exc:
        logger.warning(
            "CloudWatch admission_refused_count metric failed: %s. "
            "Not blocking the planner — admission decision already made.",
            exc,
        )


class UnscoredBuyCandidatesError(RuntimeError):
    """Raised when signals.json has buy_candidates that are missing from
    predictions.json — the GBM veto gate is structurally unreachable for
    those tickers, so sizing positions would route around a risk control.

    Self-healing should happen upstream (weekday Step Function's coverage-gap
    Choice state re-invokes predictor with --tickers). This error is the
    read-time defense-in-depth backstop: if the gap reaches the executor, we
    refuse to trade rather than bypass the veto.
    """
    def __init__(self, missing: list[str], n_buy: int, n_preds: int):
        self.missing = missing
        self.n_buy = n_buy
        self.n_preds = n_preds
        super().__init__(
            f"Coverage gap: {len(missing)} of {n_buy} buy_candidate(s) "
            f"not present in predictions.json (which has {n_preds} tickers). "
            f"Missing: {', '.join(missing)}. "
            "Refusing to size positions — GBM veto gate is unreachable for "
            "these tickers. Re-run predictor with --tickers to close the gap."
        )


def _emit_unscored_count_metric(count: int) -> None:
    """Emit CloudWatch metric AlphaEngine/Predictor/unscored_buy_candidates_count.

    Best-effort; never raises. The hard-fail (UnscoredBuyCandidatesError) is
    the functional guard — this metric + CloudWatch alarm is the long-term
    guard against the self-healing mechanism silently regressing.
    """
    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Predictor",
            MetricData=[{
                "MetricName": "unscored_buy_candidates_count",
                "Value": float(count),
                "Unit": "Count",
            }],
        )
    except Exception as exc:  # noqa: BLE001 — observability, must never block trading path
        logger.warning("CloudWatch metric emission failed: %s", exc)


def assert_predictions_cover_buy_candidates(
    signals: dict,
    predictions_by_ticker: dict,
) -> None:
    """Verify every buy_candidate has a prediction row. Raise on gap.

    Always emits the `unscored_buy_candidates_count` CloudWatch metric (even
    with value 0) so alarm baselines are continuous.
    """
    buy = signals.get("buy_candidates") or []
    buy_tickers = {
        (e.get("ticker") or "").upper()
        for e in buy if isinstance(e, dict) and e.get("ticker")
    }
    pred_tickers = {
        (t or "").upper() for t in (predictions_by_ticker or {}).keys()
    }
    missing = sorted(buy_tickers - pred_tickers)
    _emit_unscored_count_metric(len(missing))
    if missing:
        raise UnscoredBuyCandidatesError(
            missing=missing,
            n_buy=len(buy_tickers),
            n_preds=len(pred_tickers),
        )


def patch_unknown_sectors_with_constituents(signals_raw: dict, s3_bucket: str) -> int:
    """Backfill sector="Unknown" or missing on ENTER signals using
    constituents.json sector_map as the authoritative GICS source.

    Defense-in-depth against an escape from research's signals.json
    preflight (alpha-engine-research#126). The 2026-05-04 EOG/NVT
    incident wrote "Unknown" into trades.db because the planner consumed
    research's first-pass file before sector_map had loaded. The research
    preflight is the primary gate; this is the executor-side catch.

    Mutates ``signals_raw["buy_candidates"]`` and ``signals_raw["universe"]``
    in place. Returns count of patches applied for caller logging.
    Lazy-loads the constituents map only when at least one ENTER signal
    needs patching, so the typical clean path pays no S3 round-trip.
    """
    needs_patch = False
    for key in ("buy_candidates", "universe"):
        for s in signals_raw.get(key) or []:
            if not isinstance(s, dict) or s.get("signal") != "ENTER":
                continue
            cur = s.get("sector")
            if not cur or cur == "Unknown":
                needs_patch = True
                break
        if needs_patch:
            break

    if not needs_patch:
        return 0

    from executor.eod_reconcile import _load_constituents_sector_map
    constituents_map = _load_constituents_sector_map(s3_bucket)
    if not constituents_map:
        return 0

    patched = 0
    for key in ("buy_candidates", "universe"):
        for s in signals_raw.get(key) or []:
            if not isinstance(s, dict):
                continue
            cur = s.get("sector")
            if cur and cur != "Unknown":
                continue
            ticker = s.get("ticker")
            if not ticker:
                continue
            mapped = constituents_map.get(ticker)
            if mapped:
                s["sector"] = mapped
                patched += 1
    return patched


def get_actionable_signals(signals: dict) -> dict:
    """
    Filter signals to actionable entries by signal type.

    Returns:
        {
            "enter":  [signal, ...],
            "exit":   [signal, ...],
            "reduce": [signal, ...],
            "hold":   [signal, ...],
            "market_regime": str,
            "sector_ratings": {sector: {"rating": str, "modifier": float, "rationale": str}},
        }
    """
    universe = signals.get("universe", [])
    candidates = signals.get("buy_candidates", [])
    # Candidates take precedence — dedupe by ticker
    seen: set[str] = set()
    all_stocks: list[dict] = []
    for s in candidates + universe:
        ticker = s.get("ticker")
        if ticker and ticker not in seen:
            seen.add(ticker)
            all_stocks.append(s)

    return {
        "enter":  [s for s in all_stocks if s.get("signal") == "ENTER"],
        "exit":   [s for s in all_stocks if s.get("signal") == "EXIT"],
        "reduce": [s for s in all_stocks if s.get("signal") == "REDUCE"],
        "hold":   [s for s in all_stocks if s.get("signal") == "HOLD"],
        "market_regime": signals.get("market_regime", "neutral"),
        "sector_ratings": signals.get("sector_ratings", {}),
    }
