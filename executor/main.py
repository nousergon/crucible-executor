"""
Alpha Engine Executor — daily morning order-book planner.

Reads signals.json from S3, applies risk rules and position sizing,
writes approved entries and urgent exits to the intraday order book.
The daemon (daemon.py) is the sole order executor — it uses technical
triggers to time entries and executes exits immediately.

No orders are placed by this module. All trade execution happens in
the daemon via IB Gateway.

Runs on boot via systemd (alpha-engine-morning.service) on the trading
instance, which is started/stopped daily by the micro instance's cron.

Usage:
    python main.py              # write order book (requires IB Gateway for NAV/positions)
    python main.py --dry-run    # print planned orders without writing order book
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ssm_secrets import load_secrets
load_secrets()

from executor.ibkr import IBKRClient, SimulatedIBKRClient
from executor.order_book import OrderBook
from executor.position_sizer import compute_position_size
from executor.risk_guard import check_order, compute_drawdown_multiplier
from executor.signal_reader import get_actionable_signals, read_signals_with_fallback
from executor.strategies.config import load_strategy_config
from executor.strategies.exit_manager import evaluate_exits, SECTOR_ETF_MAP
from executor.price_cache import (
    load_atr_14_pct,
    load_daily_vwap,
    load_feature_coverage,
    load_price_histories,
)
from executor.trade_logger import (
    backup_to_s3,
    get_entry_dates,
    init_db,
    log_risk_event,
    log_shadow_book_block,
    log_trade,
)

from alpha_engine_lib.logging import setup_logging
# Suppress benign IB Error codes that don't represent real failures:
#   10197 — "No market data during competing live session". The daemon
#     keeps receiving delayed ticks via the delayedLast fallback in
#     price_monitor.py, so flow-doctor's ERROR alert is spam.
#   10349 — "Order TIF was set to DAY based on order preset". IB echoes
#     this back every time the preset matches the submitted TIF=DAY
#     (ibkr.py sets DAY defensively after the HSY cancel cycle on
#     2026-04-13). The order is still placed and filled.
# Every executor entrypoint passes the same pattern list so all three
# fire through the shared handler.
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
_FLOW_DOCTOR_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow-doctor.yaml")
setup_logging("main", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

from executor.config_loader import get_config_path

# S3-delivered executor params (loaded once per cold-start)
_executor_params_cache: dict | None = None
_executor_params_loaded: bool = False

# Flat param name → nested config path mapping
_PARAM_MAP = {
    "atr_multiplier": ("strategy", "exit_manager", "atr_multiplier"),
    "time_decay_reduce_days": ("strategy", "exit_manager", "time_decay_reduce_days"),
    "time_decay_exit_days": ("strategy", "exit_manager", "time_decay_exit_days"),
    "min_score": ("min_score_to_enter",),
    "max_position_pct": ("max_position_pct",),
    "reduce_fraction": ("reduce_fraction",),
    "atr_sizing_target_risk": ("atr_sizing_target_risk",),
    "confidence_sizing_min": ("confidence_sizing_min",),
    "confidence_sizing_range": ("confidence_sizing_range",),
    "staleness_decay_per_day": ("staleness_decay_per_day",),
    "earnings_sizing_reduction": ("earnings_sizing_reduction",),
    "earnings_proximity_days": ("earnings_proximity_days",),
    "momentum_gate_threshold": ("momentum_gate_threshold",),
    "correlation_block_threshold": ("correlation_block_threshold",),
    "profit_take_pct": ("strategy", "exit_manager", "profit_take_pct"),
    "momentum_exit_threshold": ("strategy", "exit_manager", "momentum_exit_threshold"),
}


# (type, min, max) for each S3-delivered param — values outside range are rejected
_PARAM_VALIDATORS = {
    "atr_multiplier":              (float, 0.5, 10.0),
    "time_decay_reduce_days":      (int,   1,   30),
    "time_decay_exit_days":        (int,   1,   60),
    "min_score":                   (float, 0,   100),
    "max_position_pct":            (float, 0.01, 0.25),
    "reduce_fraction":             (float, 0.1,  1.0),
    "atr_sizing_target_risk":      (float, 0.005, 0.10),
    "confidence_sizing_min":       (float, 0.3,  1.0),
    "confidence_sizing_range":     (float, 0.1,  1.0),
    "staleness_decay_per_day":     (float, 0.0,  0.2),
    "earnings_sizing_reduction":   (float, 0.0,  1.0),
    "earnings_proximity_days":     (int,   1,    30),
    "momentum_gate_threshold":     (float, -30,  0),
    "correlation_block_threshold": (float, 0.3,  1.0),
    "profit_take_pct":             (float, 0.05, 1.0),
    "momentum_exit_threshold":     (float, -50,  0),
}

_EXECUTOR_PARAMS_CACHE_PATH = Path(__file__).resolve().parent.parent / "config" / ".executor_params_cache.json"


def _load_executor_params_from_s3(bucket: str) -> dict | None:
    """Read config/executor_params.json from S3. Cache per cold-start.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    params survive transient S3 failures.
    """
    global _executor_params_cache, _executor_params_loaded
    if _executor_params_loaded:
        return _executor_params_cache
    _executor_params_loaded = True

    try:
        import json
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/executor_params.json")
        data = json.loads(obj["Body"].read())
        # Advisory schema validation (log warnings, never block)
        _unknown_keys = [k for k in data if k not in _PARAM_MAP and k not in (
            "disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend",
            "updated_at", "best_sharpe", "best_alpha", "improvement_pct",
            "n_combos_tested", "manual_override",
        )]
        if _unknown_keys:
            logger.warning("executor_params.json contains unknown keys: %s", _unknown_keys)
        # Only keep safe-to-override params (numeric) + special non-numeric params
        safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
        # Phase 4 non-numeric params: disabled_triggers (list), p_up sizing (bool)
        for special_key in ("disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend"):
            if special_key in data:
                safe[special_key] = data[special_key]
        if safe:
            logger.info("Loaded executor params from S3: %s", safe)
            _executor_params_cache = safe
            # Persist to local cache for fault tolerance
            try:
                _EXECUTOR_PARAMS_CACHE_PATH.write_text(json.dumps(safe, indent=2))
            except Exception:
                logger.debug("Failed to write executor params cache", exc_info=True)
        return _executor_params_cache
    except Exception as e:
        logger.warning("Could not read executor params from S3: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if _EXECUTOR_PARAMS_CACHE_PATH.exists():
            import json
            data = json.loads(_EXECUTOR_PARAMS_CACHE_PATH.read_text())
            safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
            if safe:
                logger.info("Loaded executor params from local cache (last known optimal): %s", safe)
                _executor_params_cache = safe
                return _executor_params_cache
    except Exception as e2:
        logger.warning("Could not read local executor params cache: %s", e2)

    logger.warning("Both S3 and local cache failed for executor params — using hardcoded defaults")
    return None


def _merge_s3_params(config: dict, s3_params: dict) -> dict[str, Any]:
    """Merge flat S3 param names into nested config structure with validation."""
    for param, value in s3_params.items():
        # Phase 4 non-numeric params: merge directly into top-level config
        if param == "disabled_triggers" and isinstance(value, list):
            config.setdefault("intraday", {}).setdefault("entry_triggers", {})["disabled_triggers"] = value
            logger.info("S3 disabled_triggers: %s", value)
            continue
        if param == "use_p_up_sizing" and isinstance(value, bool):
            config["use_p_up_sizing"] = value
            logger.info("S3 use_p_up_sizing: %s", value)
            continue
        if param == "p_up_sizing_blend" and isinstance(value, (int, float)):
            config["p_up_sizing_blend"] = float(value)
            continue

        path = _PARAM_MAP.get(param)
        if not path:
            continue
        validator = _PARAM_VALIDATORS.get(param)
        if validator:
            expected_type, lo, hi = validator
            if not isinstance(value, (int, float)):
                logger.warning("S3 param %s: invalid type %s — skipping", param, type(value).__name__)
                continue
            value = expected_type(value)
            if not (lo <= value <= hi):
                logger.warning("S3 param %s=%s out of range [%s, %s] — skipping", param, value, lo, hi)
                continue
        target = config
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    return config


_LOAD_CONFIG_CACHE: dict | None = None


def load_config() -> dict:
    """Load and return the risk.yaml config dict.

    Cached for the process lifetime: risk.yaml is read-only at runtime
    and re-reading on every call wastes ~20 ms per executor.run()
    invocation. Live executor calls this once per boot, so caching is a
    no-op there. Backtester loops 100k+ times per predictor_param_sweep,
    so per-call cache hit drops the load_config cost from ~1 sec total
    (50-call profile) to ~1 ms.

    A deep copy is returned so that the per-call config_override merge
    in run() (which mutates nested ``config["strategy"][...]`` dicts)
    can't pollute the cache. Deepcopy of a ~30-key nested dict is sub-
    millisecond — much cheaper than re-parsing YAML.

    Tests that need to override the config path can clear the cache by
    setting ``executor.main._LOAD_CONFIG_CACHE = None`` before invoking
    ``load_config()``.
    """
    global _LOAD_CONFIG_CACHE
    import copy
    if _LOAD_CONFIG_CACHE is None:
        with open(get_config_path()) as f:
            _LOAD_CONFIG_CACHE = yaml.safe_load(f)
    return copy.deepcopy(_LOAD_CONFIG_CACHE)


def _compute_support_level(price_history, strategy_config: dict) -> float | None:
    """Compute N-day low from price history for support-bounce entry trigger.

    Accepts a pandas DataFrame indexed by date with a ``low`` column.
    """
    lookback = strategy_config.get("intraday_support_lookback_days", 20)
    if price_history is None or len(price_history) < lookback:
        return None
    lows = price_history["low"].iloc[-lookback:]
    # Drop zeros/NaNs the way the prior list-based path did via ``if bar.get("low")``
    valid = lows[(lows.notna()) & (lows > 0)]
    if valid.empty:
        return None
    return float(valid.min())


def _read_signals(
    config: dict,
    signals_bucket: str,
    run_date: str,
    simulate: bool,
    signals_override: dict | None,
    conn,
) -> tuple[dict, dict, str, dict, str | None]:
    """Read and validate signals from S3 or override.

    Returns ``(signals_raw, signals, run_date, predictions_by_ticker,
    predictions_date)``. ``predictions_date`` is the
    ``predictor/predictions/{date}.json`` filename date the GBM run
    produced (None if predictions weren't loaded — simulate mode or S3
    miss); ``signals_raw["date"]`` is the corresponding signals.json
    filename date and is read directly off ``signals_raw`` by callers
    that need it.
    """
    if signals_override is not None:
        signals_raw = signals_override
        run_date = signals_raw.get("date", run_date)
    else:
        try:
            signals_raw = read_signals_with_fallback(signals_bucket, run_date)
        except RuntimeError as e:
            logger.error(f"Cannot proceed without signals: {e}")
            if conn:
                conn.close()
            raise

    # Defense-in-depth universe filter — drop buy_candidates whose tickers
    # aren't in the ArcticDB universe library. Research's population_selector
    # (alpha-engine-research#41) is the primary guardrail; this catches
    # anything that slipped past (manual edits, Research bug, universe-drift
    # window). See filter_buy_candidates_to_universe for scope + rationale.
    #
    # Skipped in simulate mode (2026-04-27): the backtester already
    # pre-filters signals against the ArcticDB universe ONCE at the
    # simulation-loop bootstrap (``backtest.py:_run_simulation_loop``
    # line 826 calls ``get_universe_symbols`` once, then per-date
    # ``_simulate_single_date`` runs ``_filter_signals_to_universe`` against
    # that set). Re-running the filter inside ``_read_signals`` would
    # call ``universe_lib.list_symbols()`` per signal date — an
    # ArcticDB round-trip the profile measured at ~424 ms/call, which
    # blew the predictor_param_sweep budget. Live executor still pays
    # the per-call cost (runs once per trading day, where the cost is
    # negligible).
    if not simulate:
        from executor.signal_reader import filter_buy_candidates_to_universe
        signals_raw = filter_buy_candidates_to_universe(signals_raw, signals_bucket)

    # Admission gate — refuse buy_candidates below hard coverage floor
    # (default 0.30). Companion to the position sizer's coverage derate:
    # the derate handles partial-coverage tickers gracefully, this gate
    # refuses tickers whose coverage is so low that no amount of derating
    # produces a trustworthy signal (pure pre-history IPOs, OHLCV-only
    # symbols, etc.). Held positions exempt — admission applies to ENTRY
    # only, not to unwinding existing exposure. Skipped in simulate mode
    # to preserve backtester replay parity against historical signals.
    if not simulate and config.get("coverage_admission_enabled", True):
        from executor.signal_reader import filter_buy_candidates_by_coverage
        from executor.price_cache import load_feature_coverage

        buy_tickers = [
            e.get("ticker") for e in (signals_raw.get("buy_candidates") or [])
            if isinstance(e, dict) and e.get("ticker")
        ]
        if buy_tickers:
            min_cov = float(config.get("min_coverage_for_admission", 0.30))
            try:
                cov_map = load_feature_coverage(buy_tickers, signals_bucket)
                signals_raw = filter_buy_candidates_by_coverage(
                    signals_raw, cov_map, min_coverage=min_cov,
                )
            except RuntimeError as exc:
                # ArcticDB unreachable — same posture as other preflight
                # reads: hard-fail, don't silently admit everything.
                logger.error("Admission gate failed on ArcticDB read: %s", exc)
                raise

    if not simulate:
        from executor.signal_reader import patch_unknown_sectors_with_constituents
        try:
            n_patched = patch_unknown_sectors_with_constituents(signals_raw, signals_bucket)
            if n_patched:
                logger.warning(
                    "[sector_fallback] Backfilled %d sectors from constituents.json "
                    "(research signals.json escape from alpha-engine-research#126)",
                    n_patched,
                )
        except Exception as e:
            logger.warning("Sector backfill skipped: %s", e)

    signals = get_actionable_signals(signals_raw)

    # Alert if signals are stale (research didn't run recently)
    if not simulate:
        try:
            signals_date_raw = signals_raw.get("date", run_date)
            _sig_age = (date.fromisoformat(run_date) - date.fromisoformat(signals_date_raw)).days
            if _sig_age > 2:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    f"\u26a0\ufe0f *Stale signals*\n"
                    f"Using signals from {signals_date_raw} ({_sig_age} days old)\n"
                    f"Research may not have run this week."
                )
        except Exception:
            logger.debug("Stale signals Telegram notification failed", exc_info=True)

    # Load GBM predictions for rationale capture
    predictions_date: str | None = None
    if not simulate:
        try:
            from executor.signal_reader import read_predictions
            predictions_by_ticker, predictions_date = read_predictions(signals_bucket)
        except Exception as e:
            logger.warning("Failed to load GBM predictions: %s", e)
            predictions_by_ticker = {}
    else:
        predictions_by_ticker = {}

    # Coverage guard: every buy_candidate must have a prediction row, otherwise
    # the GBM veto gate is structurally unreachable for that ticker and we'd
    # be sizing positions around a risk control. Skip in simulate mode (no
    # live trading, predictions intentionally empty). The weekday Step Function
    # coverage-gap Choice state is the self-healing mechanism; this guard is
    # read-time defense-in-depth. Always emits CloudWatch metric (value 0 on
    # success) so the alarm baseline is continuous.
    if not simulate:
        from executor.signal_reader import assert_predictions_cover_buy_candidates
        assert_predictions_cover_buy_candidates(signals_raw, predictions_by_ticker)

    logger.info(
        f"Signals | regime={signals['market_regime']} "
        f"| ENTER={len(signals['enter'])} EXIT={len(signals['exit'])} "
        f"REDUCE={len(signals['reduce'])} HOLD={len(signals['hold'])}"
    )

    return signals_raw, signals, run_date, predictions_by_ticker, predictions_date


def _plan_entries(
    enter_signals: list[dict],
    signals_raw: dict,
    predictions_by_ticker: dict,
    config: dict,
    strategy_config: dict,
    market_regime: str,
    sector_ratings: dict,
    ibkr,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict,
    price_histories: dict | None,
    atr_map: dict,
    dd_multiplier: float,
    signal_age_days: int,
    earnings_by_ticker: dict,
    vwap_map: dict,
    coverage_map: dict,
    ob: OrderBook,
    run_date: str,
    dry_run: bool,
    simulate: bool,
    predictions_date: str | None = None,
) -> tuple[int, list[dict], list[dict], list[dict]]:
    """Live-shell wrapper around ``executor.deciders.decide_entries``.

    Resolves ``prices_now`` from the IB / sim client, calls the pure
    decider, and dispatches results:
      * simulate: ``ibkr.place_market_order`` per accepted entry to
        accumulate sim_client position state across dates.
      * live (not simulate, not dry_run): ``ob.add_entry`` per
        ``entries_with_meta`` to write the daemon's order book.
      * dry_run: log only, no side effects.

    Returns ``(n_entered, orders, blocked, risk_events)``. The fourth
    element is the structured veto/override log emitted by
    ``decide_entries`` (Phase 2 transparency-inventory). Caller persists
    via ``trade_logger.log_risk_event``.
    """
    from executor.deciders import decide_entries

    # Resolve prices_now from IB/sim client up-front so the decider is
    # broker-agnostic. For each enter signal we need the price; we
    # also tolerate missing prices (the decider treats them as
    # "no price available — skip").
    prices_now: dict[str, float] = {}
    for sig in enter_signals:
        t = sig.get("ticker")
        if not t:
            continue
        p = ibkr.get_current_price(t)
        if p is not None:
            prices_now[t] = p

    plan = decide_entries(
        enter_signals=enter_signals,
        signals_raw=signals_raw,
        predictions_by_ticker=predictions_by_ticker,
        config=config,
        strategy_config=strategy_config,
        market_regime=market_regime,
        sector_ratings=sector_ratings,
        portfolio_nav=portfolio_nav,
        peak_nav=peak_nav,
        current_positions=current_positions,
        prices_now=prices_now,
        price_histories=price_histories,
        atr_map=atr_map,
        vwap_map=vwap_map,
        coverage_map=coverage_map,
        dd_multiplier=dd_multiplier,
        signal_age_days=signal_age_days,
        earnings_by_ticker=earnings_by_ticker,
        run_date=run_date,
        predictions_date=predictions_date,
    )

    # Dispatch decisions to side-effecting layer.
    if simulate:
        # Accumulate sim_client position state for next-iteration
        # already-held check. See plan.orders comment in deciders.py.
        for o in plan.orders:
            if o["action"] == "ENTER":
                ibkr.place_market_order(o["ticker"], "BUY", o["shares"])
    elif not dry_run:
        # Live: persist each entry-with-meta to the daemon's order book.
        for entry in plan.entries_with_meta:
            ob.add_entry(entry)

    return plan.n_entered, plan.orders, plan.blocked, plan.risk_events


def _plan_exits_and_reduces(
    signals: dict,
    strategy_exits: list[dict],
    predictions_by_ticker: dict,
    current_positions: dict,
    ibkr,
    portfolio_nav: float,
    config: dict,
    market_regime: str,
    ob: OrderBook,
    run_date: str,
    dry_run: bool,
    simulate: bool,
    signals_date: str | None = None,
    predictions_date: str | None = None,
) -> list[dict]:
    """Live-shell wrapper around ``executor.deciders.decide_exits_and_reduces``.

    Resolves prices_now from IB / sim client, calls the pure decider,
    and dispatches results to side-effecting layer:
      * simulate: ``ibkr.place_market_order(SELL)`` per accepted exit/reduce
      * live: ``ob.add_urgent_exit`` per ``urgent_exits_with_meta``
      * dry_run: log only

    Returns the orders list (populated in simulate mode for accumulator).
    """
    from executor.deciders import decide_exits_and_reduces

    # Tickers we may need a price for (held positions referenced by
    # exit / reduce signals). Resolve via ibkr; missing prices fall
    # back to avg_cost inside the decider.
    candidate_tickers: set[str] = set()
    for sig in signals.get("exit", []) + signals.get("reduce", []):
        t = sig.get("ticker")
        if t:
            candidate_tickers.add(t)
    for sig in strategy_exits:
        t = sig.get("ticker")
        if t:
            candidate_tickers.add(t)

    prices_now: dict[str, float] = {}
    for t in candidate_tickers:
        if t not in current_positions:
            continue
        p = ibkr.get_current_price(t)
        if p is not None:
            prices_now[t] = p

    plan = decide_exits_and_reduces(
        signals=signals,
        strategy_exits=strategy_exits,
        current_positions=current_positions,
        prices_now=prices_now,
        predictions_by_ticker=predictions_by_ticker,
        config=config,
        market_regime=market_regime,
        portfolio_nav=portfolio_nav,
        run_date=run_date,
        signals_date=signals_date,
        predictions_date=predictions_date,
    )

    if simulate:
        # Apply orders to sim_client so position state carries to next sim date.
        for o in plan.orders:
            if o["action"] == "EXIT":
                ibkr.place_market_order(o["ticker"], "SELL", o["shares"])
            elif o["action"] == "REDUCE":
                ibkr.place_market_order(o["ticker"], "SELL", o["shares"])
    elif not dry_run:
        for entry in plan.urgent_exits_with_meta:
            ob.add_urgent_exit(entry)

    return plan.orders


def _write_order_book_summary(
    ob: OrderBook,
    blocked_entries: list[dict] | None,
    signals_bucket: str,
    run_date: str,
) -> None:
    """Write a public-safe order book summary to S3 for the dashboard."""
    import boto3

    summary = {
        "date": run_date,
        "entries_approved": [
            {"ticker": e["ticker"]} for e in ob.pending_entries()
        ],
        "entries_blocked": [
            {"ticker": b["ticker"], "reason": b.get("block_reason", b.get("reason", "unknown"))}
            for b in (blocked_entries or [])
        ],
        "exits": [
            {"ticker": e["ticker"], "reason": e.get("reason", "research_signal")}
            for e in ob.pending_urgent_exits()
            if e.get("signal") != "COVER"
        ],
        "covers": [
            {"ticker": e["ticker"]}
            for e in ob.pending_urgent_exits()
            if e.get("signal") == "COVER"
        ],
    }

    try:
        s3 = boto3.client("s3")
        key = f"order_books/{run_date}/summary.json"
        s3.put_object(
            Bucket=signals_bucket,
            Key=key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        logger.info("Order book summary written to s3://%s/%s", signals_bucket, key)
    except Exception as e:
        logger.warning("Failed to write order book summary (non-fatal): %s", e)


def _write_stops_and_finalize(
    ibkr,
    ob: OrderBook,
    price_histories: dict | None,
    atr_map: dict,
    strategy_config: dict,
    conn,
    run_date: str,
    blocked_entries: list[dict] | None = None,
    signals_bucket: str | None = None,
) -> None:
    """Write stop records for held positions, detect shorts, save order book, notify."""
    # ATR previously computed inline via _compute_atr(ticker_hist). Since
    # 2026-04-16 the executor reads atr_14_pct from the feature-store map
    # (load_atr_14_pct in main()) — same definition the predictor and sizing
    # path use. atr_dollar derives from entry_price × atr_pct so trailing stops
    # stay in dollar-denominated semantics (bracket_orders consumes dollars).

    # Add stop records for all current positions
    current_pos = ibkr.get_positions()
    for t, pos in current_pos.items():
        pos_shares = int(pos.get("shares", 0))
        if pos_shares <= 0:
            continue
        # Skip tickers with pending urgent exits
        urgent_exit_tickers = {u["ticker"] for u in ob.pending_urgent_exits()}
        if t in urgent_exit_tickers:
            continue
        entry_price = pos.get("avg_cost", 0)
        atr_mult = strategy_config.get("intraday_trailing_stop_atr_multiple", 2.0)
        ticker_atr_pct = atr_map.get(t)
        if not ticker_atr_pct or ticker_atr_pct <= 0 or entry_price <= 0:
            # load_atr_14_pct's hard-fail covers signal tickers + held positions
            # at the top of main(); if we still hit a missing value here it's
            # either a position that wasn't in the held-set at ATR load time
            # (race condition) or entry_price <=0 from IBKR — skip this stop.
            logger.warning(
                "No ATR or invalid entry_price for %s — skipping stop (atr_pct=%s, entry_price=%s)",
                t, ticker_atr_pct, entry_price,
            )
            continue
        atr_val = ticker_atr_pct * entry_price
        stop_price = round(entry_price - atr_val * atr_mult, 2)
        ob.add_stop({
            "ticker": t,
            "entry_price": entry_price,
            "current_stop": stop_price,
            "trail_atr": atr_val or 0,
            "atr_multiple": atr_mult,
            "high_water": entry_price,
            "entry_date": (conn and get_entry_dates(conn, [t]).get(t)) or run_date,
            "shares": pos_shares,
        })

    # Detect short positions and add urgent cover orders
    for t, pos in current_pos.items():
        pos_shares = int(pos.get("shares", 0))
        if pos_shares < 0:
            cover_shares = abs(pos_shares)
            logger.warning(
                "SHORT DETECTED: %s has %d shares — adding urgent COVER for %d shares",
                t, pos_shares, cover_shares,
            )
            ob.add_urgent_exit({
                "ticker": t,
                "signal": "COVER",
                "shares": cover_shares,
                "reason": "short_position_cover",
                "detail": f"Covering accidental short of {cover_shares} shares",
            })

    ob.save()

    # Backup full order book to S3 for audit trail
    if signals_bucket:
        ob.backup_to_s3(signals_bucket, run_date)

    # Write public-safe summary for dashboard
    if signals_bucket:
        _write_order_book_summary(ob, blocked_entries, signals_bucket, run_date)

    n_entries = len(ob.pending_entries())
    n_urgent = len(ob.pending_urgent_exits())
    n_stops = len(ob.active_stops())
    n_covers = sum(1 for u in ob.pending_urgent_exits() if u.get("signal") == "COVER")
    logger.info(
        "Order book written: %d entries, %d urgent exits (%d covers), %d stops",
        n_entries, n_urgent, n_covers, n_stops,
    )
    # Build notification with blocked entry transparency
    blocked_lines = ""
    if blocked_entries:
        blocked_lines = f"\nBlocked ({len(blocked_entries)}):\n"
        for b in blocked_entries:
            blocked_lines += f"  {b['ticker']}: {b.get('block_reason', b.get('reason', 'unknown'))}\n"

    try:
        from executor.notifier import send_daemon_status
        send_daemon_status(
            f"\u2705 *Order book written*\n"
            f"Date: {run_date}\n"
            f"Entries: {n_entries} | Urgent exits: {n_urgent} | Stops: {n_stops}"
            f"{blocked_lines}"
        )
    except Exception:
        logger.debug("Order book Telegram notification failed", exc_info=True)


def run(
    dry_run: bool = False,
    simulate: bool = False,
    ibkr_client=None,           # injected by backtester when simulate=True
    signals_override: dict = None,  # injected signals dict (skips S3 read)
    price_histories: dict = None,   # injected by backtester for exit manager
    config_override: dict = None,   # injected by backtester param sweep
    atr_map: dict | None = None,      # injected by backtester to skip per-call ArcticDB read
    vwap_map: dict | None = None,     # injected by backtester to skip per-call ArcticDB read
    coverage_map: dict | None = None, # injected by backtester to skip per-call ArcticDB read
) -> list[dict] | None:
    """
    Returns list of order dicts when simulate=True, else None.
    All other behaviour (risk guard, position sizer, trade logger) is unchanged.
    """
    orders = []
    run_date = str(date.today())
    _health_start = _time.time()
    logger.info(f"Executor starting | date={run_date} | dry_run={dry_run} | simulate={simulate}")

    config = load_config()
    if config_override:
        for key, val in config_override.items():
            if key == "strategy" and isinstance(val, dict) and "strategy" in config:
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict) and isinstance(config["strategy"].get(sub_key), dict):
                        config["strategy"][sub_key].update(sub_val)
                    else:
                        config["strategy"][sub_key] = sub_val
            elif key in _PARAM_MAP:
                # Route flat param names through the same mapping as S3 params
                # so backtester sweep keys (e.g. "min_score") land in the right
                # nested config location (e.g. "min_score_to_enter").
                path = _PARAM_MAP[key]
                target = config
                for p in path[:-1]:
                    target = target.setdefault(p, {})
                target[path[-1]] = val
            else:
                config[key] = val
    # Merge S3-delivered params (backtester recommendations) if not in simulate mode
    if not simulate and not config_override:
        s3_params = _load_executor_params_from_s3(config.get("signals_bucket", "alpha-engine-research"))
        if s3_params:
            config = _merge_s3_params(config, s3_params)

    db_path = config["db_path"]
    signals_bucket = config["signals_bucket"]
    trades_bucket = config["trades_bucket"]

    # Preflight: AWS_REGION + S3 bucket reachable. Skip in simulate mode
    # (backtester injects orders directly, no real S3 interaction).
    # Raises RuntimeError on failure → propagates to non-zero exit.
    if not simulate:
        from executor.preflight import ExecutorPreflight
        ExecutorPreflight(bucket=signals_bucket, mode="main").run()

    # ── Flow Doctor: retrieve the shared instance set up at module import ───
    from alpha_engine_lib.logging import get_flow_doctor
    fd = get_flow_doctor() if not simulate else None

    # ── 0. Check upstream health — hard-fail if anything upstream is broken ──
    # Running on stale signals or a failed predictor produces a degraded
    # portfolio that contradicts the system's design. Per the "hard-fail until
    # stable" standard, any unknown/failed/stale upstream must abort executor
    # BEFORE we read signals or touch the order book. The trading instance
    # will remain idle for the day and be stopped at 13:30 PT as usual.
    # Per-module staleness tolerance (hours):
    #   research           — 192h (runs weekly Sat; grace covers Sat→next Fri)
    #   predictor_inference — 26h (runs every weekday morning; catches a Mon
    #                               miss without false-alarming on the weekend)
    #   daily_data         — 26h (same weekday 13:05 UTC cadence as predictor;
    #                               stamp written by alpha-engine-data after
    #                               daily_closes.collect — catches ran-and-failed
    #                               states that the direct LastModified check
    #                               below would miss)
    _UPSTREAM_MAX_AGE_H = {"research": 192, "predictor_inference": 26, "daily_data": 26}

    if not simulate:
        _health_failures: list[str] = []
        try:
            from executor.health_status import check_upstream_health
            # Pass the loosest module tolerance as the library default so
            # check_upstream_health doesn't prematurely mark research stale;
            # we re-check per-module against _UPSTREAM_MAX_AGE_H below.
            upstream = check_upstream_health(
                signals_bucket,
                list(_UPSTREAM_MAX_AGE_H),
                max_age_hours=max(_UPSTREAM_MAX_AGE_H.values()),
            )
        except Exception as _ue:
            # A health-check read failure is itself a reason to hard-fail —
            # we don't know the state of upstream, so we refuse to trade.
            upstream = {}
            _health_failures.append(f"health-check error: {_ue}")

        for mod, max_hrs in _UPSTREAM_MAX_AGE_H.items():
            info = upstream.get(mod)
            if info is None:
                _health_failures.append(f"{mod}: no health data returned")
                continue
            if info["status"] == "unknown":
                _health_failures.append(f"{mod}: no health data found")
            elif info["status"] == "failed":
                _health_failures.append(f"{mod}: last run FAILED")
            elif info["age_hours"] is None or info["age_hours"] < 0:
                _health_failures.append(f"{mod}: last_success missing")
            elif info["age_hours"] > max_hrs:
                _health_failures.append(
                    f"{mod}: {info['age_hours']:.0f}h ({info['age_hours']/24:.1f}d) stale "
                    f"(max {max_hrs}h)"
                )

        if _health_failures:
            msg = (
                "Upstream health FAILED — executor aborting:\n"
                + "\n".join(f"  - {w}" for w in _health_failures)
            )
            logger.error(msg)
            try:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    "\u274c *Upstream health FAILED*\n"
                    f"Date: {run_date}\n"
                    + "\n".join(f"- {w}" for w in _health_failures)
                    + "\n\nExecutor aborted — no order book written."
                )
            except Exception:
                logger.debug("Upstream failure Telegram notification failed", exc_info=True)
            raise RuntimeError(msg)

        # Direct freshness check on the ArcticDB macro library. The
        # stamp-based check_upstream_health above covers predictor/research
        # "did it run"; this catches the "stamp green but data blob is
        # yesterday's" failure mode (partial writes, retries skipping
        # DataPhase1). SPY is the canary — written by the daily_append
        # post-close job to the macro library (NOT universe). If SPY has
        # no row for the last closed trading day, the post-close pipeline
        # did not complete and the executor must abort before any signals
        # are read.
        try:
            import pandas as _pd
            from executor.price_cache import _open_macro_library
            from alpha_engine_lib.trading_calendar import last_closed_trading_day
            _macro = _open_macro_library(signals_bucket)
            _spy_df = _macro.read("SPY").data
            _expected_min = _pd.Timestamp(last_closed_trading_day()).normalize()
            _idx = _spy_df.index.normalize() if hasattr(_spy_df.index, "normalize") else _spy_df.index
            if _spy_df.empty or (_idx >= _expected_min).sum() == 0:
                _latest = _pd.Timestamp(_spy_df.index[-1]).date() if not _spy_df.empty else "EMPTY"
                raise RuntimeError(
                    f"ArcticDB macro has no SPY row >= {_expected_min.date()} "
                    f"(latest: {_latest}). Post-close daily-data job did not "
                    f"complete for the last closed trading day."
                )
        except Exception as _freshness_err:
            msg = f"ArcticDB freshness check FAILED — executor aborting: {_freshness_err}"
            logger.error(msg)
            try:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    "\u274c *ArcticDB universe stale/missing*\n"
                    f"Date: {run_date}\n"
                    f"{_freshness_err}\n\nExecutor aborted — no order book written."
                )
            except Exception:
                logger.debug("ArcticDB freshness Telegram notification failed", exc_info=True)
            raise RuntimeError(msg) from _freshness_err

    conn = None if simulate else init_db(db_path)

    # ── 1. Read signals from S3 (or use injected override) ──────────────────
    try:
        signals_raw, signals, run_date, predictions_by_ticker, predictions_date = _read_signals(
            config, signals_bucket, run_date, simulate, signals_override, conn,
        )
    except Exception as _sig_err:
        if fd:
            fd.report(_sig_err, severity="error", context={
                "site": "signal_read", "run_date": run_date})
        if conn:
            conn.close()
        # Re-raise so systemd marks the service as 'failed' (not 'inactive (dead)').
        # Returning silently hides signal-read failures from systemctl status, which
        # is how today's (2026-04-10) incident went undetected for 3 hours.
        raise
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]

    # ── 2. Connect to IBKR (or use injected simulated client) ───────────────
    if simulate:
        ibkr = ibkr_client
    else:
        ibkr = IBKRClient(
            host=config["ibkr_host"],
            port=config["ibkr_port"],
            client_id=config["ibkr_client_id"],
            reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
        )

    try:
        portfolio_nav = ibkr.get_portfolio_nav()
        current_positions = ibkr.get_positions()
        peak_nav = ibkr.get_peak_nav(conn)

        # Enrich positions with sector data from signals
        universe_sectors = {
            s["ticker"]: s.get("sector", "")
            for s in signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])
            if s.get("ticker")
        }
        for ticker, pos in current_positions.items():
            pos["sector"] = universe_sectors.get(ticker, "")
    
        # ── 2b. Enrich positions with entry_date from trades.db ──────────────────
        # Also pulls stance + catalyst_date (stance taxonomy arc 2026-05-11):
        # exit_manager.evaluate_exits reads pos["stance"] / pos["catalyst_date"]
        # to apply stance-conditional exit rules (ATR multiplier override,
        # time-decay disable for quality/catalyst, hard exit at
        # catalyst_date+3d for catalyst). NULL for legacy positions logged
        # before this PR — falls through to baseline behavior.
        if conn and current_positions:
            from executor.trade_logger import get_entry_stance_and_catalyst
            entry_dates = get_entry_dates(conn, list(current_positions.keys()))
            stance_lookup = get_entry_stance_and_catalyst(
                conn, list(current_positions.keys()),
            )
            for ticker, pos in current_positions.items():
                pos["entry_date"] = entry_dates.get(ticker)
                stance_info = stance_lookup.get(ticker, {})
                pos["stance"] = stance_info.get("stance")
                pos["catalyst_date"] = stance_info.get("catalyst_date")
            logger.info(f"Entry dates resolved for {len(entry_dates)}/{len(current_positions)} positions")
    
        # ── 2c. Compute graduated drawdown multiplier ──────────────────────────
        # Pass an events sink so any halt/throttle event lands in
        # `risk_events` ONCE per planning cycle (the per-ticker check_order
        # call deliberately does NOT propagate events to its inner
        # compute_drawdown_multiplier — see risk_guard.py:check_order).
        dd_events: list[dict] = []
        dd_multiplier, dd_reason = compute_drawdown_multiplier(
            portfolio_nav, peak_nav, config, events=dd_events,
        )
        if dd_multiplier < 1.0:
            logger.info(f"Drawdown tier active: {dd_reason}")
        # Stamp lineage + persist immediately. Any subsequent per-ticker
        # vetoes get persisted alongside after _plan_entries returns.
        if conn and dd_events:
            signals_date_for_events = signals_raw.get("date", run_date) if signals_raw else run_date
            for ev in dd_events:
                ev.setdefault("date", run_date)
                ev.setdefault("market_regime", market_regime)
                ev.setdefault("signal_date", signals_date_for_events)
                ev.setdefault("prediction_date", predictions_date)
                try:
                    log_risk_event(conn, ev)
                except Exception as e:
                    logger.debug("risk_event log failed (drawdown): %s", e)
    
        # ── 2d. Strategy layer: evaluate exit rules on held positions ──────────
        strategy_config = load_strategy_config(config)
    
        # Build signals lookup for exit manager
        signals_by_ticker = {}
        for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and t not in signals_by_ticker:
                signals_by_ticker[t] = s
    
        # Load price histories from predictor S3 cache (unless injected by backtester)
        # Include ENTER tickers for ATR sizing, momentum gate, and correlation check
        if price_histories is None:
            enter_tickers = [s["ticker"] for s in signals.get("enter", [])]
            all_tickers = list(set(list(current_positions.keys()) + enter_tickers))
            # Also load sector ETF histories for sector-relative exit veto
            held_sectors = set(pos.get("sector", "") for pos in current_positions.values())
            etf_tickers = [SECTOR_ETF_MAP.get(s, "SPY") for s in held_sectors if s]
            etf_tickers = list(set(etf_tickers))
            all_tickers_with_etfs = list(set(all_tickers + etf_tickers))
            if all_tickers_with_etfs:
                price_histories = load_price_histories(
                    tickers=all_tickers_with_etfs,
                    signals_bucket=signals_bucket,
                )
            else:
                price_histories = {}

        # Previous-day VWAP per ENTER-signal ticker for intraday entry triggers.
        # Sourced from the ArcticDB universe library; hard-fails on any miss
        # so daemon triggers always see a trusted VWAP.
        #
        # The ``vwap_map`` kwarg lets a caller (today: the backtester) inject
        # a precomputed resolved-per-ticker VWAP map and skip the per-call
        # ArcticDB read. Live trading passes vwap_map=None and takes the
        # existing path unchanged. Contract is identical either way:
        # {ticker: vwap_value_for_run_date}. Same design as the existing
        # ``price_histories`` kwarg — injection point, not replacement.
        vwap_tickers = sorted({s["ticker"] for s in signals.get("enter", [])})
        if vwap_map is None:
            vwap_map = load_daily_vwap(vwap_tickers, signals_bucket, run_date) if vwap_tickers else {}
        # When vwap_map was injected, trust the caller's resolution —
        # the backtester precomputes per simulate date before the call.

        # Feature-coverage map per ENTER ticker. Drives the sizer's
        # coverage derate (coverage_sizing_enabled in risk.yaml):
        # short-history tickers whose long-window features are NaN get
        # sized proportional to the fraction of populated features.
        # Admission gate in _read_signals already rejected tickers below
        # the hard floor; everything reaching here should have coverage
        # ≥ min_coverage_for_admission. Scoped to enter_tickers only —
        # held positions aren't sized via this path.
        #
        # ``coverage_map`` kwarg mirrors the atr_map / vwap_map injection
        # pattern (PR #91) — backtester precomputes once per simulate
        # pipeline, skipping per-call ``universe.read(ticker)`` round-
        # trips that timed out the 2026-04-22 Saturday SF dry-run after
        # the coverage-aware-sizing PR merged. Live trading passes
        # coverage_map=None and takes the load_feature_coverage path
        # unchanged.
        enter_tickers = [s["ticker"] for s in signals.get("enter", [])]
        if coverage_map is None:
            if enter_tickers and config.get("coverage_sizing_enabled", True):
                coverage_map = load_feature_coverage(
                    tickers=enter_tickers,
                    signals_bucket=signals_bucket,
                )
            else:
                coverage_map = {}

        # Single source of truth for ATR across the executor. Replaces per-call-site
        # _compute_atr(ticker_hist) invocations (position sizing, pullback-trigger
        # scaling, trailing stops) with the predictor's feature-store atr_14_pct,
        # so executor and predictor agree on the ATR definition. Hard-fails on
        # missing ticker or stale data — feedback_hard_fail_until_stable.
        # Scope: signal tickers (ENTER) + held positions (for trailing stops).
        #
        # ETF tickers are intentionally excluded — they're used for sector-relative
        # exit veto via price_histories, not for ATR-based execution.
        #
        # ``atr_map`` kwarg mirrors the vwap_map injection pattern — backtester
        # precomputes once per simulate pipeline, skipping millions of
        # per-call universe.read(ticker) round-trips. Live trading passes
        # atr_map=None and takes the load_atr_14_pct path unchanged.
        atr_tickers = [s["ticker"] for s in signals.get("enter", [])]
        atr_tickers += list(current_positions.keys())
        atr_tickers = sorted(set(atr_tickers))
        if atr_map is None:
            if atr_tickers:
                atr_map = load_atr_14_pct(
                    tickers=atr_tickers,
                    signals_bucket=signals_bucket,
                )
            else:
                atr_map = {}

        # Separate sector ETF histories for exit manager
        sector_etf_histories = {
            t: price_histories[t] for t in SECTOR_ETF_MAP.values()
            if t in (price_histories or {})
        }
        # Also include SPY as fallback
        if "SPY" in (price_histories or {}):
            sector_etf_histories["SPY"] = price_histories["SPY"]
    
        strategy_exits = evaluate_exits(
            current_positions=current_positions,
            signals_by_ticker=signals_by_ticker,
            run_date=run_date,
            price_histories=price_histories or {},
            ibkr_client=ibkr,
            strategy_config=strategy_config,
            sector_etf_histories=sector_etf_histories or None,
        )
    
        if strategy_exits:
            logger.info(
                f"Strategy layer generated {len(strategy_exits)} exit signal(s): "
                + ", ".join(f"{s['ticker']}({s['action']}: {s['reason']})" for s in strategy_exits)
            )
    
        enter_signals = signals["enter"]

        # Initialize order book for the day (daemon reads this after main.py completes).
        # reset_pending() makes this idempotent — if main.py runs twice, the second
        # run replaces the first rather than appending duplicate orders.
        ob = OrderBook.load()
        ob.set_date(run_date)
        ob.reset_pending()

        # ── 2e. Compute signal age for staleness discount ─────────────────────
        # Signal age is fixed for the trading day (main.py runs once on boot).
        # Signals are never refreshed mid-day, so this doesn't need recomputation.
        signals_date_str = signals_raw.get("date", run_date)
        try:
            signals_date = date.fromisoformat(signals_date_str)
            signal_age_days = (date.fromisoformat(run_date) - signals_date).days
        except (ValueError, TypeError):
            signal_age_days = 0

        # ── 2f. Batch-fetch earnings dates for ENTER candidates ──────────────
        earnings_by_ticker: dict[str, int | None] = {}
        if config.get("earnings_sizing_enabled", True) and not simulate:
            for sig in enter_signals:
                t = sig["ticker"]
                try:
                    import yfinance as yf_mod
                    cal = yf_mod.Ticker(t).calendar
                    if cal is not None and not cal.empty:
                        next_date = cal.iloc[0, 0] if hasattr(cal, 'iloc') else None
                        if next_date is not None:
                            from datetime import datetime as dt_mod
                            if hasattr(next_date, 'date'):
                                next_date = next_date.date()
                            elif isinstance(next_date, str):
                                next_date = date.fromisoformat(next_date)
                            days_until = (next_date - date.fromisoformat(run_date)).days
                            if days_until >= 0:
                                earnings_by_ticker[t] = days_until
                except Exception:
                    logger.debug("Failed to load earnings data", exc_info=True)

        # ── 2g. Drawdown forced exits ─────────────────────────────────────────
        if strategy_config.get("drawdown_forced_exit_enabled", True) and dd_multiplier < 1.0:
            forced_exit_count = 0
            if dd_multiplier <= 0.25:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier3_count", 2)
            elif dd_multiplier <= 0.50:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier2_count", 1)

            if forced_exit_count > 0 and current_positions:
                existing_exit_tickers = set(
                    s["ticker"] for s in signals.get("exit", [])
                ) | set(
                    s["ticker"] for s in strategy_exits if s["action"] == "EXIT"
                )

                def _conviction_rank(ticker_pos):
                    t, pos = ticker_pos
                    sig_data = signals_by_ticker.get(t, {})
                    score = sig_data.get("score") or 50
                    mv = pos.get("market_value", 0)
                    return (score, mv)

                ranked = sorted(current_positions.items(), key=_conviction_rank)
                for t, pos in ranked[:forced_exit_count]:
                    if t not in existing_exit_tickers:
                        shares_held = int(pos.get("shares", 0))
                        if shares_held > 0:
                            forced_sig = {
                                "ticker": t,
                                "action": "EXIT",
                                "reason": "drawdown_forced_exit",
                                "detail": f"forced exit due to drawdown (dd_mult={dd_multiplier})",
                            }
                            strategy_exits.append(forced_sig)
                            logger.info(
                                f"DRAWDOWN FORCED EXIT: {t} (score={_conviction_rank((t, pos))[0]}, "
                                f"dd_multiplier={dd_multiplier})"
                            )

        # ── 3. Process ENTER signals ─────────────────────────────────────────────
        n_entered, entry_orders, blocked_entries, plan_risk_events = _plan_entries(
            enter_signals=enter_signals,
            signals_raw=signals_raw,
            predictions_by_ticker=predictions_by_ticker,
            config=config,
            strategy_config=strategy_config,
            market_regime=market_regime,
            sector_ratings=sector_ratings,
            ibkr=ibkr,
            portfolio_nav=portfolio_nav,
            peak_nav=peak_nav,
            current_positions=current_positions,
            price_histories=price_histories,
            atr_map=atr_map,
            dd_multiplier=dd_multiplier,
            signal_age_days=signal_age_days,
            earnings_by_ticker=earnings_by_ticker,
            vwap_map=vwap_map,
            coverage_map=coverage_map,
            ob=ob,
            run_date=run_date,
            dry_run=dry_run,
            simulate=simulate,
            predictions_date=predictions_date,
        )
        orders.extend(entry_orders)

        # Log blocked entries to shadow book for evaluation
        if conn and blocked_entries:
            for be in blocked_entries:
                try:
                    log_shadow_book_block(conn, be)
                except Exception as e:
                    logger.debug("Shadow book log failed for %s: %s", be.get("ticker"), e)

        # Persist structured veto/override events (Phase 2 transparency-
        # inventory — *risk decisions* row). Sibling of the shadow-book
        # log: same family, different axis. Free-text block_reason stays
        # in shadow_book; rule + value + threshold lands in risk_events.
        if conn and plan_risk_events:
            for ev in plan_risk_events:
                try:
                    log_risk_event(conn, ev)
                except Exception as e:
                    logger.debug(
                        "risk_event log failed (%s/%s): %s",
                        ev.get("event_type"),
                        ev.get("rule"),
                        e,
                    )

        # ── 4–5. Process EXIT and REDUCE signals ────────────────────────────────
        exit_orders = _plan_exits_and_reduces(
            signals=signals,
            strategy_exits=strategy_exits,
            predictions_by_ticker=predictions_by_ticker,
            current_positions=current_positions,
            ibkr=ibkr,
            portfolio_nav=portfolio_nav,
            config=config,
            market_regime=market_regime,
            ob=ob,
            run_date=run_date,
            dry_run=dry_run,
            simulate=simulate,
            signals_date=signals_raw.get("date") if signals_raw else None,
            predictions_date=predictions_date,
        )
        orders.extend(exit_orders)

        # ── 6. Write stop records and save order book for daemon ────────────────
        if not simulate and not dry_run:
            try:
                _write_stops_and_finalize(ibkr, ob, price_histories, atr_map, strategy_config, conn, run_date, blocked_entries, signals_bucket)
            except Exception as e:
                logger.warning("Failed to write order book: %s", e)

        # ── 7. Backup and disconnect ─────────────────────────────────────────
        if not dry_run and not simulate:
            backup_to_s3(db_path, run_date, trades_bucket)

        # ── 8. Write health status ────────────────────────────────────────────
        if not simulate:
            try:
                from executor.health_status import write_health
                n_exit = len(exit_orders)
                n_blocked = len(enter_signals) - n_entered
                write_health(
                    bucket=signals_bucket,
                    module_name="executor",
                    status="ok",
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    summary={
                        "n_orders": n_entered + n_exit,
                        "n_enter": n_entered,
                        "n_exit": n_exit,
                        "n_blocked": n_blocked,
                    },
                )
            except Exception as _he:
                logger.warning("Health status write failed: %s", _he)

            # ── Data manifest ──────────────────────────────────────────────────
            try:
                from executor.health_status import write_data_manifest
                write_data_manifest(
                    bucket=signals_bucket,
                    module_name="executor_morning",
                    run_date=run_date,
                    manifest={
                        "signals_date": signals_raw.get("date", run_date),
                        "signals_count": len(signals.get("enter", [])) + len(signals.get("exit", [])),
                        "predictions_available": bool(predictions_by_ticker),
                        "entries_planned": n_entered,
                        "entries_blocked": n_blocked,
                        "blocked_reasons": [
                            {"ticker": b.get("ticker"), "reason": b.get("block_reason", b.get("reason"))}
                            for b in blocked_entries[:20]
                        ] if 'blocked_entries' in dir() else [],
                        "exits_planned": n_exit,
                    },
                )
            except Exception as _me:
                logger.warning("Data manifest write failed: %s", _me)

        if fd:
            fd.log_summary(logger)
        logger.info(f"Executor complete | dry_run={dry_run} | simulate={simulate}")

        if simulate:
            return orders
    except Exception as _exc:
        logger.exception("Executor error — ensuring IBKR disconnect")
        if fd:
            fd.report(_exc, severity="critical", context={
                "site": "executor_main", "dry_run": dry_run, "run_date": run_date})
        if not simulate:
            try:
                from executor.health_status import write_health
                write_health(
                    bucket=config.get("signals_bucket", "alpha-engine-research") if 'config' in dir() else "alpha-engine-research",
                    module_name="executor",
                    status="failed",
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    error=str(sys.exc_info()[1]),
                )
            except Exception:
                logger.debug("Health status write failed on error path", exc_info=True)
        raise
    finally:
        ibkr.disconnect()
        if conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orders without placing them",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run locally with simulated IB client (no IB Gateway needed). "
             "Uses synthetic positions and real signals from S3.",
    )
    args = parser.parse_args()

    if args.simulate:
        from executor.ibkr import SimulatedIBKRClient
        # Seed prices from S3 slim cache (last close for each ticker)
        sim_prices = {}
        try:
            config = load_config()
            bucket = config.get("signals_bucket", "alpha-engine-research")
            from executor.price_cache import load_price_histories
            import json
            import subprocess
            # Read signals to know which tickers to price
            result = subprocess.run(
                ["aws", "s3", "cp", f"s3://{bucket}/signals/{date.today()}/signals.json", "-"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                sig_data = json.loads(result.stdout)
                tickers = [s["ticker"] for s in sig_data.get("universe", [])]
                histories = load_price_histories(tickers=tickers, signals_bucket=bucket)
                for t, hist in histories.items():
                    if hist is not None and len(hist) > 0:
                        sim_prices[t] = float(hist["close"].iloc[-1])
                logger.info("Seeded %d simulated prices from S3 slim cache", len(sim_prices))
        except Exception as e:
            logger.warning("Could not seed simulated prices: %s — entries will show no price", e)
        sim_client = SimulatedIBKRClient(prices=sim_prices, nav=1_000_000.0)
        logger.info("SIMULATE MODE: using SimulatedIBKRClient (no IB Gateway)")
        orders = run(simulate=True, ibkr_client=sim_client, dry_run=True)
        if orders:
            logger.info("Simulated orders: %d", len(orders))
            for o in orders:
                logger.info(
                    "  %s %s shares=%s",
                    o.get("action", "?"), o.get("ticker", "?"), o.get("shares", "?"),
                )
        else:
            logger.info("No simulated orders generated")
    else:
        run(dry_run=args.dry_run)
