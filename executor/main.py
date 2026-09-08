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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.logging import guard_entrypoint, setup_logging

from executor.ibkr import IBKRClient, SimulatedIBKRClient
from executor.order_book import OrderBook, build_stop_record
from executor.price_cache import (
    _MACRO_SYMBOLS,
    load_atr_14_pct,
    load_daily_vwap,
    load_feature_coverage,
    load_price_histories,
)
from executor.risk_guard import compute_drawdown_multiplier
from executor.signal_reader import get_actionable_signals, read_signals_with_fallback
from executor.strategies.config import load_strategy_config
from executor.strategies.exit_manager import (
    SECTOR_ETF_MAP,
    check_position_loss_floor,
    evaluate_exits,
)
from executor.trade_logger import (
    backup_to_s3,
    get_entry_dates,
    init_db,
    log_risk_event,
    log_shadow_book_block,
)

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
from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)

_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging("main", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

from executor.config_loader import get_config_path  # noqa: E402 -- must follow setup_logging above

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
    "staleness_decay_per_day": ("staleness_decay_per_day",),
    "earnings_sizing_reduction": ("earnings_sizing_reduction",),
    "earnings_proximity_days": ("earnings_proximity_days",),
    "momentum_gate_threshold": ("momentum_gate_threshold",),
    "correlation_block_threshold": ("correlation_block_threshold",),
    "profit_take_pct": ("strategy", "exit_manager", "profit_take_pct"),
    "momentum_exit_threshold": ("strategy", "exit_manager", "momentum_exit_threshold"),
    # Task B2 (dormant): barrier-win-prob sizing weights. The enable flag
    # itself is a bool, handled with the other non-numeric special keys below.
    "barrier_win_prob_sizing_min": ("barrier_win_prob_sizing_min",),
    "barrier_win_prob_sizing_range": ("barrier_win_prob_sizing_range",),
    # L4598 (config#697): stance-conditional sizing multipliers tuned by the
    # backtester's stance_sizing_optimizer (field_overlay via the assembler).
    # Consumed by position_sizer's stance_adj; gated upstream by the optimizer's
    # rank-IC promotion gate — no key reaches the live artifact until it clears.
    "stance_size_momentum": ("stance_size_momentum",),
    "stance_size_value": ("stance_size_value",),
    "stance_size_quality": ("stance_size_quality",),
    "stance_size_catalyst": ("stance_size_catalyst",),
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
    "staleness_decay_per_day":     (float, 0.0,  0.2),
    "earnings_sizing_reduction":   (float, 0.0,  1.0),
    "earnings_proximity_days":     (int,   1,    30),
    "momentum_gate_threshold":     (float, -30,  0),
    "correlation_block_threshold": (float, 0.3,  1.0),
    "profit_take_pct":             (float, 0.05, 1.0),
    "momentum_exit_threshold":     (float, -50,  0),
    "barrier_win_prob_sizing_min":   (float, 0.3, 1.0),
    "barrier_win_prob_sizing_range": (float, 0.1, 1.0),
    # Bounds mirror stance_sizing_optimizer._SIZE_FLOOR/_SIZE_CAP — a producer
    # emission widened past these is rejected loudly here (WARN + skip).
    "stance_size_momentum":          (float, 0.4, 1.1),
    "stance_size_value":             (float, 0.4, 1.1),
    "stance_size_quality":           (float, 0.4, 1.1),
    "stance_size_catalyst":          (float, 0.4, 1.1),
}

_EXECUTOR_PARAMS_CACHE_PATH = Path(__file__).resolve().parent.parent / "config" / ".executor_params_cache.json"

# Non-numeric S3-delivered params the loader passes through to the applied
# set (master switches / lists — not range-validated like _PARAM_MAP values).
# Contract: PIPELINE_CONTRACT.yaml executor_params "applied special" section +
# tests/test_executor_params_consumer_contract.py (L4520).
_EXECUTOR_PARAMS_SPECIAL_KEYS = (
    "disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend",
    "barrier_win_prob_sizing_enabled",
)

# Producer provenance/metadata keys — KNOWN (no unknown-key WARN) but never
# applied. Keeping this list complete keeps the WARN signal: an unknown key
# always means genuine producer/consumer contract drift, not noise.
_EXECUTOR_PARAMS_KNOWN_METADATA_KEYS = (
    "updated_at", "assembled_by", "fit_target",
    "best_sharpe", "best_alpha", "best_sortino",
    "improvement_pct", "n_combos_tested", "manual_override",
    "disabled_triggers_updated_at",
    "barrier_win_prob_sizing_updated_at", "barrier_win_prob_sizing_ic",
    "p_up_sizing_updated_at", "p_up_sizing_ic",
    "stance_sizing_updated_at", "stance_sizing_alpha_spread",
)

# config#2891: the Saturday Evaluator (backtester weight_optimizer.apply)
# writes config/executor_params.json weekly; a silently-failed or stalled
# write leaves this cold-start reading an arbitrarily old tuned config with
# no signal. WARN (never block trading) once the pointer is older than 2
# weekly cycles — mirrors ARTIFACT_REGISTRY.yaml's own default
# grace_period_cycles=2 for the central config_executor_params freshness-
# monitor row this assertion complements as an independent consumer-side
# signal (config#1724 doctrine), not a replacement for it.
_EXECUTOR_PARAMS_STALE_HOURS = 24 * 7 * 2


def _check_executor_params_staleness(last_modified: datetime | None) -> None:
    """Best-effort WARN when ``last_modified`` is older than
    ``_EXECUTOR_PARAMS_STALE_HOURS``. Never raises, never blocks trading."""
    if last_modified is None:
        return
    age_hours = (datetime.now(UTC) - last_modified).total_seconds() / 3600.0
    if age_hours > _EXECUTOR_PARAMS_STALE_HOURS:
        logger.error(
            "STALE config/executor_params.json: last modified %.1fh ago "
            "(> %dh / 2 weekly cycles) — the Saturday Evaluator may have "
            "silently failed or stalled; executor may be trading on stale "
            "tuned params (config#2891)",
            age_hours, _EXECUTOR_PARAMS_STALE_HOURS,
        )


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
        _check_executor_params_staleness(obj.get("LastModified"))
        data = json.loads(obj["Body"].read())
        # Advisory schema validation (log warnings, never block)
        _unknown_keys = [
            k for k in data
            if k not in _PARAM_MAP
            and k not in _EXECUTOR_PARAMS_SPECIAL_KEYS
            and k not in _EXECUTOR_PARAMS_KNOWN_METADATA_KEYS
        ]
        if _unknown_keys:
            logger.warning("executor_params.json contains unknown keys: %s", _unknown_keys)
        # Only keep safe-to-override params (numeric) + special non-numeric params
        safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
        # Phase 4 non-numeric params: disabled_triggers (list), p_up sizing (bool).
        # Task B2: barrier_win_prob_sizing_enabled (bool) — the dormant sizing
        # consumer's master switch, flipped via S3 after soak + backtester sweep.
        for special_key in _EXECUTOR_PARAMS_SPECIAL_KEYS:
            if special_key in data:
                safe[special_key] = data[special_key]
        if safe:
            logger.info("Loaded executor params from S3: %s", safe)
            _executor_params_cache = safe
            # Persist to local cache for fault tolerance
            try:
                _EXECUTOR_PARAMS_CACHE_PATH.write_text(json.dumps(safe, indent=2))
            except Exception:
                # (a) local params-cache write failed — the in-memory
                # _executor_params_cache above already has this cycle's
                # value, so nothing downstream is affected until restart.
                # (c) recorded at WARNING (visible at the INFO root level,
                # unlike the DEBUG this replaces) — the app log stream /
                # CloudWatch is the recording surface (alpha-engine-config-I10031).
                logger.warning("Failed to write executor params cache", exc_info=True)
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

    # Champion candidate-source adapter (config#2364 / config#2366;
    # thinktank_coverage arm added config-I2518 / epic I2515) — must run
    # BEFORE the universe/coverage filters below so a synthesized
    # (scanner_predictor_direct or thinktank_coverage) buy_candidates list
    # flows through the SAME gates agentic candidates already pass through.
    # No-op passthrough when the pointer resolves to agentic (including the
    # S3-404 pre-bootstrap default) — the champion pointer read + arm-feed
    # round-trip are live-trading-arm concerns, so (like the universe/
    # coverage gates below) this is skipped in simulate mode: the
    # backtester's replay path doesn't select a live champion arm, and
    # predictor_param_sweep can't afford an extra S3 round-trip per
    # simulated date.
    #
    # The constituents sector_map is only loaded when the pointer actually
    # resolves to a synthesizing arm (scanner_predictor_direct or, since
    # config-I2518/epic I2515, thinktank_coverage — neither source carries a
    # sector on its rows, so both need this to stamp synthesized entries) —
    # on the common (agentic / pre-bootstrap) path this avoids an extra S3
    # list_objects_v2 + get_object round-trip for a value
    # apply_champion_selection would otherwise discard unused on every
    # trading day. When a synthesizing arm IS active, this is the same
    # sector_map artifact patch_unknown_sectors_with_constituents loads
    # again a few lines below — one extra fetch, paid only on the arm
    # that's rarely active, not on every run.
    #
    # ``_champion_injected_predictions`` is applied to ``predictions_by_ticker``
    # AFTER the real GBM ``read_predictions`` load below (that call fully
    # replaces ``predictions_by_ticker`, so merging earlier would be
    # clobbered) but BEFORE ``assert_predictions_cover_buy_candidates`` so
    # the coverage assert sees the injected rows for synthesized tickers.
    _champion_injected_predictions: dict = {}
    if not simulate:
        from executor.champion import (
            ARMS_REQUIRING_SECTOR_MAP,
            apply_champion_selection,
            assert_producer_champion_coherence,
            load_champion_pointer,
        )

        _champion_pointer = load_champion_pointer(signals_bucket)
        # Coherence guard (config#5713): refuse to start a trading day when
        # the signals producer's buy_candidates is empty BY CONTRACT and the
        # resolved champion arm is a no-op passthrough (agentic) — that
        # pairing guarantees no new entry is ever proposed, silently. The
        # producer stamps ``producer`` at write time, the champion pointer
        # resolves here at read time, and neither side can see the other;
        # this read path is the only place both facts are in hand. Must run
        # BEFORE apply_champion_selection so the guard sees the producer's
        # own artifact, not a champion-stamped copy.
        assert_producer_champion_coherence(signals_raw, _champion_pointer, config)
        # DERIVED, not a literal (alpha-engine-config-I9299). The tuple that
        # used to be inline here was a fourth hand-maintained arm register and
        # had already gone stale: `scanner_top20_predictor` became servable on
        # 2026-08-27 and was never added, so every entry that arm synthesized
        # was stamped sector="Unknown" — silently switching the sector
        # concentration cap off for it.
        if _champion_pointer["champion"] in ARMS_REQUIRING_SECTOR_MAP:
            from executor.eod_reconcile import _load_constituents_sector_map
            sector_map = _load_constituents_sector_map(signals_bucket)
        else:
            sector_map = {}
        signals_raw, _champion_injected_predictions = apply_champion_selection(
            signals_raw,
            {},
            bucket=signals_bucket,
            run_date=run_date,
            config=config,
            sector_map=sector_map,
            pointer=_champion_pointer,
        )

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
        from executor.price_cache import load_feature_coverage
        from executor.signal_reader import filter_buy_candidates_by_coverage

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

    # Alert if signals are stale (research didn't run recently).
    # Freshness is a KNOWLEDGE-axis comparison: signals.json is keyed by
    # trading_day (last closed session), so age is measured against
    # now_dual().trading_day — NOT run_date, which is the session axis
    # (config#1610) and sits one session ahead intraday (comparing against
    # it would false-alert every Monday on perfectly fresh Friday signals).
    if not simulate:
        try:
            from nousergon_lib.dates import now_dual as _now_dual
            _knowledge_day = _now_dual().trading_day
            signals_date_raw = signals_raw.get("date", _knowledge_day)
            _sig_age = (date.fromisoformat(_knowledge_day) - date.fromisoformat(signals_date_raw)).days
            if _sig_age > 2:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    f"\u26a0\ufe0f *Stale signals*\n"
                    f"Using signals from {signals_date_raw} ({_sig_age} days old)\n"
                    f"Research may not have run this week."
                )
        except Exception:
            # (a) Telegram transport failed for the stale-signals notice.
            # (c) not recorded elsewhere — deliberate carve-out
            # (alpha-engine-config-I10031): a failed notification must
            # never abort a trading run, and the underlying stale-signals
            # condition is already visible via the caller's own logging.
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

    # Merge champion-synthesized predictions (scanner_predictor_direct or
    # thinktank_coverage — empty dict on agentic/pre-bootstrap) in AFTER the
    # real GBM load above so injected rows for synthesized tickers are
    # visible to the coverage assert below without being clobbered by (or
    # clobbering) the real predictor's own predictions.
    if _champion_injected_predictions:
        predictions_by_ticker = {**predictions_by_ticker, **_champion_injected_predictions}

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
    regime_intensity_z: float | None = None,
    adv_map: dict | None = None,
    derisk_multiplier: float = 1.0,
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
        regime_intensity_z=regime_intensity_z,
        adv_map=adv_map,
        derisk_multiplier=derisk_multiplier,
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
    champion: str | None = None,
    promotion_source: str | None = None,
    risk_flags_off: list[str] | None = None,
) -> None:
    """Write a public-safe order book summary to S3 for the dashboard.

    ``champion``/``promotion_source`` (config#2364 / config#2366) stamp
    which candidate-source arm produced this order book — additive fields,
    ``None`` when the caller doesn't have a resolved champion (e.g. an
    older call site, or a run that errored before ``_read_signals``
    resolved the pointer). Sourced from ``signals_raw["champion"]`` /
    ``signals_raw["promotion_source"]``, which
    ``executor.champion.apply_champion_selection`` stamps on the
    scanner_predictor_direct or thinktank_coverage paths; agentic runs
    leave both unset (None) here rather than hardcoding "agentic" — the
    pointer read result is the single source of truth for that label.

    ``risk_flags_off`` (alpha-engine-config-I9021) — names, from
    ``executor.risk_flag_audit.SAFETY_FLAGS_DEFAULT_OFF``, of feature-flagged
    risk-config safety gates that are OFF this run (explicitly ``false`` or
    absent from ``risk.yaml`` — behaviorally identical). Written every run,
    including an empty list, so "all gates armed" and "not observed" are
    never confused (principle 7: a component emitting nothing is
    unobserved, not healthy). ``None`` only when the caller couldn't
    compute it (defensive) — never silently omitted.
    """
    import boto3

    summary = {
        "date": run_date,
        "champion": champion,
        "promotion_source": promotion_source,
        "risk_flags_off": risk_flags_off if risk_flags_off is not None else [],
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
        _emit_writer_failure_metric("order_book_summary_write_failed", 0)
    except Exception as e:
        # SWALLOW, classified (alpha-engine-config-I10190 deliverable 4). The
        # fleet's fail-loud rule requires any deviation to name three things:
        #
        # (a) FAILURE MODE SWALLOWED — `order_books/{date}/summary.json` is not
        #     written for this session.
        # (b) WHY THE PRIMARY DELIVERABLE SURVIVES — this is a DERIVED,
        #     public-safe projection of `ob` for the dashboard. Nothing on the
        #     trading path reads it: not the daemon (which reads the stop
        #     records `_write_stops_and_finalize` produces), not the EOD
        #     reconcile, not the next session's planner. Every field in it is a
        #     re-serialization of state already persisted elsewhere in this run.
        #     Losing it costs a dashboard tile, not a risk control — which is
        #     precisely what makes it DIFFERENT from the stop-record write at
        #     §6, whose swallow I10190 was filed against and which now raises.
        # (c) RECORDING SURFACE — `AlphaEngine/Executor/
        #     order_book_summary_write_failed`, emitted in both polarities so
        #     absence of the failure is distinguishable from absence of the
        #     emitter. Deliberately NOT alarmed: a dashboard tile going missing
        #     does not warrant a page, and an alarm nobody would act on is the
        #     chronic false positive that teaches an operator to ignore the
        #     channel. It is rendered and countable, which is the surface this
        #     failure's severity earns.
        _emit_writer_failure_metric("order_book_summary_write_failed", 1)
        logger.warning("Failed to write order book summary (non-fatal): %s", e)


class StopRecordWriteError(RuntimeError):
    """The daemon's stop records were not produced for this session.

    ``_write_stops_and_finalize`` writes the stop records
    ``executor.daemon`` reads, and their ``stop_kind`` decides what the daemon
    does with every open position — ``catastrophic_gap_only`` under the
    optimizer, ``alpha`` (the full ``IntradayExitManager``) otherwise. A run
    that places or retains positions and does not produce them has left a live
    book with no per-name exit authority.
    """

    def __init__(self, run_date: str, cause: BaseException):
        self.run_date = run_date
        super().__init__(
            f"Stop records were NOT written for {run_date}: {cause!r}. The "
            f"daemon has no stop records for this session, so every open "
            f"position is running without its per-name exit authority "
            f"(trailing stop / profit-take / collapse under the alpha "
            f"stop_kind, catastrophic gap stop under the optimizer). Positions "
            f"already placed this session are LIVE. OPERATOR: re-run the "
            f"planner for this date once the underlying write failure is "
            f"resolved, or flatten manually. Failure recorded at "
            f"executor/stop_write_failures/{run_date}.json and on the "
            f"AlphaEngine/Executor gauge stop_records_write_failed."
        )


def _emit_writer_failure_metric(metric_name: str, value: float) -> None:
    """Emit one ``AlphaEngine/Executor`` writer-failure gauge, both polarities.

    Best-effort and deliberately silent on its own failure beyond a WARNING:
    this runs on the exception path of the thing it is reporting, and an
    observability error must never replace the real one.

    A DIRECT gauge rather than a CloudWatch Logs metric filter, on purpose. A
    metric filter would depend on ``/var/log/executor.log`` on the trading box
    reaching CloudWatch Logs, and this detector's whole job is to be true on the
    day something on that box is broken — a detector resting on an unverified
    shipping path is the "ran and saw nothing" class
    (``alpha-engine-config-I10184``). ``put_metric_data`` is already this
    repo's proven surface (``signal_reader._emit_admission_refused_metric``).
    """
    try:
        import boto3 as _b3

        _b3.client("cloudwatch").put_metric_data(
            Namespace="AlphaEngine/Executor",
            MetricData=[{
                "MetricName": metric_name, "Value": float(value), "Unit": "Count",
            }],
        )
    except Exception as exc:  # noqa: BLE001 — must never mask the real failure
        logger.warning(
            "CloudWatch %s metric emission failed: %s", metric_name, exc,
        )


def _record_stop_write_failure(
    bucket: str | None, run_date: str, cause: BaseException,
) -> None:
    """Durable record that this run finished WITHOUT stop records.

    The metric pages; this artifact is what an operator or a later sweep reads
    to find out which date, and why. Both are best-effort and neither may raise
    — the caller re-raises the original failure immediately after.
    """
    _emit_writer_failure_metric("stop_records_write_failed", 1)
    if not bucket:
        return
    try:
        import boto3 as _b3

        _b3.client("s3").put_object(
            Bucket=bucket,
            Key=f"executor/stop_write_failures/{run_date}.json",
            Body=json.dumps({
                "run_date": run_date,
                "stop_records_written": False,
                "error_type": type(cause).__name__,
                "error": str(cause),
                "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, indent=2).encode(),
            ContentType="application/json",
        )
    except Exception as exc:  # noqa: BLE001 — must never mask the real failure
        logger.warning("stop-write failure artifact write failed: %s", exc)


def write_stops_and_finalize_guarded(*args, **kwargs):
    """``_write_stops_and_finalize``, made LOUD. alpha-engine-config-I10190.

    Until 2026-09-08 the §6 call site was::

        try:
            _write_stops_and_finalize(...)
        except Exception as e:
            logger.warning("Failed to write order book: %s", e)

    — a bare ``except`` on the producer of the daemon's stop records, at
    WARNING, reaching no durable surface. Measured 2026-09-08:
    ``grep -rn "Failed to write order book"`` across the fleet found the two
    call sites in this file and NOTHING else — no metric filter, no alarm, no
    alert-transport entry.

    **Why RAISE, and not a page-and-continue.** The default under the fleet's
    fail-loud rule is to raise, and no exception applies here:

    * Swallowing does not save the book. If this write fails, the daemon has no
      stop records either way — continuing does not restore the protection, it
      only hides that the protection is gone.
    * Raising routes ``RunMorningPlanner`` to the preopen pipeline's
      ``HandleFailure``, which is a surface an operator already watches, within
      minutes. A WARNING in ``/var/log/executor.log`` is not one.
    * What is skipped by raising is §6b onward — the order-book rationale
      artifact, the data manifest, the shadow-optimizer log. Every one of those
      is already documented in this file as non-blocking audit output. No risk
      control is downstream of this point.
    * The counter-argument — that aborting after entries are placed leaves
      positions with no book — is real but argues for a LOUDER failure, not a
      quieter one: those positions are exactly the ones a human must be told
      about, and they are equally unstopped whether or not the process
      continues.

    **This is the hold-book safeguard's whole safety argument.**
    ``_should_hold_book`` suppresses the optimizer rebalance and retains the
    book, and the reason that is acceptable is stated in this file: stops are
    still written, and the daemon's hard-risk overrides remain active. §6 sits
    OUTSIDE the §5b hold branch, so it runs on a held day — asserted by
    ``tests/test_stop_write_is_loud.py``. On exactly the day the system has
    decided it cannot trust its own signal, the stop records are the only
    remaining protection; their failure may not be a WARNING nobody receives.

    Emits ``stop_records_write_failed`` in BOTH polarities so the alarm has a
    continuous baseline and an absent emitter is distinguishable from a healthy
    run.
    """
    bucket = kwargs.get("signals_bucket")
    run_date = kwargs.get("run_date")
    if run_date is None and len(args) > 6:
        run_date = args[6]
    if bucket is None and len(args) > 8:
        bucket = args[8]
    try:
        result = _write_stops_and_finalize(*args, **kwargs)
    except Exception as exc:
        _record_stop_write_failure(bucket, str(run_date), exc)
        raise StopRecordWriteError(str(run_date), exc) from exc
    _emit_writer_failure_metric("stop_records_write_failed", 0)
    return result


# ── Hold-book floor ownership (alpha-engine-config-I10179 / I10184) ─────────
#
# Until 2026-09-08 this module carried ``HOLD_BOOK_ALPHA_STDEV_FLOOR = 0.001``:
# a hand-set absolute floor on the SAME quantity the predictor's
# ``output_distribution_gate`` already declares a floor on (absolute 0.015,
# "derived from the measured healthy population", alpha-engine-config-I9267,
# plus a relative floor at 50% of the trailing 10-session median). Two owners,
# one invariant, 15x apart, neither aware of the other. On 2026-09-08 the
# predictor called ``alpha_stdev`` 0.005819 collapsed and this module called it
# healthy; the optimizer rebalanced on a batch the producer had declared dead
# (``executor/hold_book_flags/2026-09-08.json`` -> ``"held": false``).
#
# The fix is ownership, not a new number: the predictor owns every threshold on
# ``alpha_stdev`` and the executor CONSUMES its verdict. What the executor still
# owns is the config#1176 discrimination — WHICH gate failure justifies holding
# the book — because that is a trading decision, not a distributional one.
#
# Measured, over the 47 served sessions carrying ``metrics.alpha_stdev``
# (2026-06-30..2026-09-08, ``s3://alpha-engine-research/predictor/predictions/``):
#
#   * the predictor's ABSOLUTE floor 0.015 sits BELOW 14 of 47 served sessions
#     — 2026-06-30..07-02, 07-06..07-10, 08-24..08-28, 09-08 — and 13 of those
#     14 ran a champion nobody has ever called collapsed. Consuming the absolute
#     leg as a hold trigger would have held the book on 30% of history, so it is
#     deliberately NOT a hold trigger here. That floor was derived from a
#     TRAINING-panel population; the served population does not support it.
#   * the RELATIVE leg (today < 50% of the trailing 10-session median) fires on
#     6 of 47: 2026-08-24..08-28 and 2026-09-08. All six are champion-collapse
#     sessions — the first five served ``v3.0-meta-2026-08-21-7d3d1cce``, the
#     candidate ``training/served_slice_dispersion.py``'s behavioural veto was
#     written to refuse and which was rolled back to ``119e069b`` on 08-31; the
#     sixth is the first served batch of ``v3.0-meta-2026-09-04-cc3271ea``.
#     ZERO healthy sessions. That is the leg the hold reads.
#   * no ABSOLUTE floor on served ``alpha_stdev`` can separate the two states at
#     all: the lowest healthy session is 0.006575 (2026-07-06..10) against the
#     collapse at 0.005819 — 13% apart. Any absolute number, including the
#     0.001 removed here, is un-derivable from this population. The separating
#     statistic is relative, and it already exists upstream.
#
# So no constant on ``alpha_stdev`` is reintroduced. The only executor-side
# degeneracy test that remains is SCALE-FREE (modal fraction of the served
# alphas), used solely as the fallback for a gate artifact that carries no
# structured dispersion legs — and it mirrors the predictor's own
# ``max_alpha_modal_fraction`` default rather than inventing a second threshold.

#: Modal fraction of the served ``predicted_alpha`` at or above which the
#: tradable signal is treated as literally constant (the 2026-04-28 class).
#: Scale-free by construction, so it is NOT a second floor on a magnitude the
#: predictor already owns. Mirrors
#: ``crucible-predictor/model/output_distribution_gate.py::
#: validate_live_batch_distribution(max_alpha_modal_fraction=0.90)``.
HOLD_BOOK_ALPHA_MODAL_FRACTION = 0.90
_HOLD_BOOK_MIN_BATCH = 5

#: Gate failures that are an unambiguous breakage of the TRADABLE signal. The
#: predictor has already decided; the executor holds without re-measuring.
_TRADABLE_SIGNAL_FAILURES = frozenset({
    "alpha_nonfinite_rate",   # predicted_alpha is non-finite
    "alpha_collapse",         # predicted_alpha is essentially one value
})

#: Gate failures on ``alpha_stdev`` dispersion. Resolved against the gate's own
#: structured legs (``metrics.relative_dispersion``) rather than re-thresholded:
#: the RELATIVE leg holds, the ABSOLUTE leg alone does not (see the measurement
#: above — 13 healthy served sessions sit below 0.015).
_DISPERSION_FAILURES = frozenset({
    "alpha_stdev_relative_compression",
    "alpha_stdev_absolute_floor",
})

#: Gate failures that describe the shape of the CALIBRATED ``p_up`` (or a
#: p_up-derived quantity), which the optimizer does not trade on. config#1176:
#: these must NEVER halt the book on their own — that is the 2026-06-22 /
#: 2026-06-29 false halt, where the isotonic calibrator collapsed ``p_up`` onto
#: one staircase step on a day whose ``predicted_alpha`` was cleanly
#: differentiated and GE's 8% target was dropped on a healthy 26-name batch.
_P_UP_SHAPE_FAILURES = frozenset({
    "unique_p_up",
    "modal_fraction",
    "stdev",
    "saturation_rate",
    "direction_skew",
    "confidence_semantics",
    "alpha_sign_skew",
    "insufficient_regime_coverage",
    "input_mismatch",
})


def _served_alpha_diagnostics(predictions_by_ticker: dict) -> tuple[list[float], dict]:
    """Served ``predicted_alpha`` values plus their scale-free shape stats.

    ``alpha_stdev`` is reported for the operator surface (``order_book_rationale``
    reads it) but is deliberately NOT compared against any executor-side
    threshold — see the ownership note above.
    """
    import math
    import statistics

    alphas: list[float] = []
    for p in (predictions_by_ticker or {}).values():
        if not isinstance(p, dict):
            continue
        a = p.get("predicted_alpha")
        if a is None:
            a = p.get("canonical_predicted_alpha")
        if isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a):
            alphas.append(float(a))

    diag: dict = {"n_alpha": len(alphas)}
    if alphas:
        diag["alpha_stdev"] = round(statistics.pstdev(alphas), 6)
        rounded = [round(a, 9) for a in alphas]
        modal = max(rounded.count(v) for v in set(rounded)) / len(rounded)
        diag["alpha_modal_fraction"] = round(modal, 4)
        diag["n_unique_alpha"] = len(set(rounded))
    return alphas, diag


def _should_hold_book(
    gate: dict | None,
    predictions_by_ticker: dict,
    *,
    modal_fraction_ceiling: float = HOLD_BOOK_ALPHA_MODAL_FRACTION,
    min_batch: int = _HOLD_BOOK_MIN_BATCH,
) -> tuple[bool, dict]:
    """Decide whether the §5b hold-book safeguard should suppress the optimizer
    rebalance, returning ``(hold, diagnostics)``.

    The predictor's ``output_distribution_gate`` owns every threshold on the
    distribution of ``predicted_alpha``. This function owns only the question
    config#1176 posed: given that the gate flagged, does THIS failure justify
    holding the book the optimizer would otherwise rotate?

    * A failure of the tradable signal itself (``alpha_collapse``,
      ``alpha_nonfinite_rate``) -> HOLD. No second measurement.
    * A dispersion failure -> read the gate's own legs. The RELATIVE leg
      (``metrics.relative_dispersion.alpha_stdev.passed is False``: today below
      50% of the trailing 10-session median) is a collapse against the served
      population and HOLDS. The ABSOLUTE leg alone does NOT hold — 13 of 47
      healthy served sessions sit below its 0.015, which was derived from a
      training panel, not from what is served.
    * A ``p_up``-shape failure -> PROCEED (config#1176). The optimizer does not
      trade ``p_up``; halting on an isotonic staircase artifact is the 6/22 and
      6/29 false halt. The scale-free degeneracy test below still runs, so a
      literally-constant alpha batch is caught, but no magnitude threshold is
      applied.
    * An unrecognised failure, or a gate carrying no structured legs -> fall
      back to the scale-free degeneracy test, which needs no floor.

    Fail-safe: a missing (``None``) gate proceeds — the existing fail-open
    posture. A flagged gate with fewer than ``min_batch`` finite alphas cannot
    be judged, so the raw gate verdict is trusted and the book is held.
    """
    gate_flagged = gate is not None and gate.get("passed") is False
    failed_check = (gate or {}).get("failed_check")
    diag: dict = {"gate_flagged": gate_flagged, "failed_check": failed_check}
    if not gate_flagged:
        diag["decision"] = "proceed_gate_ok"
        return False, diag

    alphas, alpha_diag = _served_alpha_diagnostics(predictions_by_ticker)
    diag.update(alpha_diag)

    metrics = (gate or {}).get("metrics") or {}
    rel = ((metrics.get("relative_dispersion") or {}).get("alpha_stdev") or {})
    rel_passed = rel.get("passed")
    if isinstance(rel_passed, bool):
        diag["relative_dispersion_passed"] = rel_passed
        diag["relative_dispersion_ratio"] = rel.get("ratio")
        diag["relative_dispersion_history_median"] = rel.get("history_median")
    # Observe-only corroborator (alpha-engine-config-I10184 deliverable 4): a
    # rising streak alongside a dispersion failure is a dead champion, a streak
    # of 1 is noise, and it was 0 three times in early September under a healthy
    # champion — so it is recorded and NEVER drives this decision on its own.
    nhc = metrics.get("n_high_confidence") or {}
    if isinstance(nhc.get("zero_streak"), (int, float)):
        diag["n_high_confidence_zero_streak"] = int(nhc["zero_streak"])
    if isinstance(metrics.get("champion_version_id"), str):
        diag["champion_version_id"] = metrics["champion_version_id"]
    elif isinstance((metrics.get("relative_dispersion") or {}).get(
            "today_champion_version_id"), str):
        diag["champion_version_id"] = (
            metrics["relative_dispersion"]["today_champion_version_id"]
        )

    if failed_check in _TRADABLE_SIGNAL_FAILURES:
        diag["decision"] = "hold_tradable_signal_failed"
        return True, diag

    if failed_check in _DISPERSION_FAILURES:
        if rel_passed is False:
            diag["decision"] = "hold_relative_dispersion_collapse"
            return True, diag
        if rel_passed is True:
            # Only the absolute leg failed. Not a hold — see the ownership note.
            diag["decision"] = "proceed_absolute_floor_only"
            return False, diag
        # Legs absent (older artifact schema): fall through to the scale-free
        # test rather than trusting an unresolvable verdict.

    if len(alphas) < min_batch:
        diag["decision"] = "hold_signal_undeterminable"
        return True, diag

    degenerate = alpha_diag.get("alpha_modal_fraction", 0.0) >= modal_fraction_ceiling
    diag["alpha_modal_fraction_ceiling"] = modal_fraction_ceiling
    diag["signal_degenerate"] = degenerate
    if degenerate:
        diag["decision"] = "hold_signal_degenerate"
        return True, diag

    diag["decision"] = (
        "proceed_p_up_artifact_only" if failed_check in _P_UP_SHAPE_FAILURES
        else "proceed_signal_healthy"
    )
    return False, diag


def emit_distribution_gate_metrics(gate: dict | None, hold_diag: dict | None) -> None:
    """Publish the predictor gate verdict this planner run acted on.

    alpha-engine-config-I10184 deliverable 2. The verdict is already computed
    and already written to ``executor/hold_book_flags/`` — until now it reached
    no alarmable surface, so on 2026-09-08 the gate declared the batch collapsed
    and nothing said so anywhere an operator or an alarm could see.

    Three gauges into ``AlphaEngine/Executor``, emitted on EVERY planner run
    (including the healthy 0) so the alarm baseline is continuous:

    * ``predictor_dispersion_gate_failed`` — 1 when the gate's ``alpha_stdev``
      dispersion verdict failed. This is the PAGING metric: it is the leg that
      now decides the hold and it fired on 6 of 47 served sessions, every one a
      champion collapse.
    * ``predictor_output_gate_failed`` — 1 whenever the gate flagged for ANY
      reason, including the ``p_up`` staircase artifact config#1176 refuses to
      halt on. Rendered, deliberately not paged: it flagged on 2026-06-29 on a
      healthy book, so paging it would be a chronic false positive.
    * ``predictor_n_high_confidence_zero_streak`` — the consecutive-sessions
      count of ``n_high_confidence == 0`` (deliverable 4). Observe-only: it was
      0 three times in early September under a healthy champion, so the streak
      alone must never page; it is what distinguishes "one quiet day" from a
      dead champion once the dispersion metric above has fired.

    Best-effort — a CloudWatch failure WARNs and never blocks the planner.
    """
    gate = gate or {}
    diag = hold_diag or {}
    flagged = gate.get("passed") is False
    rel = (((gate.get("metrics") or {}).get("relative_dispersion") or {})
           .get("alpha_stdev") or {})
    dispersion_failed = bool(
        flagged
        and (rel.get("passed") is False
             or diag.get("decision") == "hold_relative_dispersion_collapse")
    )
    streak = ((gate.get("metrics") or {}).get("n_high_confidence") or {}).get(
        "zero_streak"
    )
    data = [
        {
            "MetricName": "predictor_dispersion_gate_failed",
            "Value": 1.0 if dispersion_failed else 0.0,
            "Unit": "Count",
        },
        {
            "MetricName": "predictor_output_gate_failed",
            "Value": 1.0 if flagged else 0.0,
            "Unit": "Count",
        },
    ]
    if isinstance(streak, (int, float)) and not isinstance(streak, bool):
        data.append({
            "MetricName": "predictor_n_high_confidence_zero_streak",
            "Value": float(streak),
            "Unit": "Count",
        })
    try:
        import boto3 as _b3

        _b3.client("cloudwatch").put_metric_data(
            Namespace="AlphaEngine/Executor", MetricData=data,
        )
    except Exception as exc:  # noqa: BLE001 — observability never blocks trading
        logger.warning(
            "CloudWatch distribution-gate metric emission failed: %s. Not "
            "blocking the planner — the hold decision is already made and the "
            "hold_book_flags artifact still records it.",
            exc,
        )


def _write_hold_book_flag(
    bucket: str, run_date: str, predictions_date: str | None, gate: dict | None,
    *, held: bool, diag: dict | None = None,
) -> None:
    """Persist a dashboard-readable record of the hold-book safeguard's verdict
    THIS run (predictor distribution gate flagged "strongly biased" AND the
    tradable signal collapsed → optimizer rebalance suppressed → current book
    held). Writes both a dated artifact and a ``latest.json`` pointer under
    ``executor/hold_book_flags/`` so the console can banner the discrepancy for
    operator review. Best-effort — the caller swallows failures so an
    audit-artifact write never blocks the planner.

    Written UNCONDITIONALLY every run (``held`` True or False) — mirrors
    ``_write_derisk_gate_artifact``'s convention. A fire-only write left
    ``latest.json`` pinned to the LAST hold event forever once one occurred,
    with no way for the dashboard to tell "held right now" from "held once, on
    2026-06-26" — the console kept bannering a 6-week-stale HOLD-BOOK SAFEGUARD
    FIRED alert through 2026-08-10 because no healthy run ever overwrote it
    with ``held: false`` (alpha-engine-config-I<ISSUE>).
    """
    import boto3 as _boto3

    gate = gate or {}
    payload = json.dumps(
        {
            "run_date": run_date,
            "predictions_date": predictions_date,
            "held": held,
            # The decision path, not just its outcome: which gate leg was read,
            # and the observe-only n_high_confidence zero-streak that
            # distinguishes one quiet session from a dead champion
            # (alpha-engine-config-I10184).
            "hold_decision": (diag or {}).get("decision"),
            "hold_diagnostics": diag or {},
            "reason": gate.get("reason"),
            "failed_check": gate.get("failed_check"),
            "gate_metrics": gate.get("metrics"),
            "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        indent=2,
    )
    _s3 = _boto3.client("s3")
    for _key in (
        f"executor/hold_book_flags/{run_date}.json",
        "executor/hold_book_flags/latest.json",
    ):
        _s3.put_object(
            Bucket=bucket, Key=_key, Body=payload.encode(),
            ContentType="application/json",
        )


def _write_derisk_gate_artifact(gate, bucket: str, run_date: str) -> None:
    """Persist the expectancy-gated de-risk stance decision for this
    planning cycle to S3 (config-I2820 deliverable 4: "Log visibility").

    Mirrors ``_write_hold_book_flag``'s dated-artifact + ``latest.json``
    pointer convention so the dashboard can read the gate state the same
    way it reads the hold-book flag. Written unconditionally (both
    active=True and active=False) whenever the gate is ``enabled`` so the
    operator can see the gate is live and clear, not just when it fires.
    Best-effort — the caller swallows failures so this audit artifact
    never blocks the planner.
    """
    import boto3 as _boto3

    payload = json.dumps(
        {
            "run_date": run_date,
            "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **gate.to_log_dict(),
        },
        indent=2,
    )
    _s3 = _boto3.client("s3")
    for _key in (
        f"executor/derisk_gate/{run_date}.json",
        "executor/derisk_gate/latest.json",
    ):
        _s3.put_object(
            Bucket=bucket, Key=_key, Body=payload.encode(),
            ContentType="application/json",
        )


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
    use_optimizer: bool = False,
    signals_raw: dict | None = None,
    config: dict | None = None,
) -> None:
    """Write stop records for held positions, detect shorts, save order book, notify.

    ``use_optimizer`` controls the ``stop_kind`` written on each stop record.
    When True (optimizer is the sole portfolio-exit authority), stops are
    marked ``catastrophic_gap_only`` — the daemon runs ONLY the per-name
    catastrophic gap stop against them and suppresses the alpha rules
    (trailing-stop, profit-take, collapse). When False, stops are marked
    ``alpha`` and the daemon runs the full legacy IntradayExitManager.
    """
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
        # ``gap_reference_price`` anchors the optimizer-mode gap check on the
        # most recent close (gap-down detection). For a position held into the
        # planning run we have its history, so use that; build_stop_record
        # falls back to entry_price if it's missing.
        gap_ref = None
        if use_optimizer:
            hist = (price_histories or {}).get(t)
            if hist is not None and len(hist):
                try:
                    gap_ref = float(hist["close"].iloc[-1])
                except (KeyError, IndexError, ValueError, TypeError):
                    gap_ref = None
        # Single chokepoint stamps stop_kind (+ gap_reference_price) from the
        # book authority — see order_book.build_stop_record.
        ob.add_stop(build_stop_record(
            ticker=t,
            entry_price=entry_price,
            current_stop=stop_price,
            trail_atr=atr_val or 0,
            atr_multiple=atr_mult,
            high_water=entry_price,
            entry_date=(conn and get_entry_dates(conn, [t]).get(t)) or run_date,
            shares=pos_shares,
            use_optimizer=use_optimizer,
            gap_reference_price=gap_ref,
        ))

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
        from executor.risk_flag_audit import list_off_safety_flags

        off_flags = list_off_safety_flags(config)
        if off_flags:
            logger.warning(
                "Risk-config safety flags OFF this run (config#I9021): %s",
                ", ".join(off_flags),
            )
        _write_order_book_summary(
            ob, blocked_entries, signals_bucket, run_date,
            champion=(signals_raw or {}).get("champion"),
            promotion_source=(signals_raw or {}).get("promotion_source"),
            risk_flags_off=off_flags,
        )

    # Consecutive zero-entries floor alarm (config#5713) — general-case
    # backstop for "no new entry ever proposed" failures that the
    # producer/champion coherence assertion in _read_signals cannot
    # enumerate. Runs after today's summary is written so the streak
    # includes the session just planned. Best-effort: never blocks the
    # planner (this call site is live-only — not simulate / not dry_run —
    # so the backtest replay path never pages).
    if signals_bucket:
        from executor.zero_entries_alarm import check_zero_entries_floor

        check_zero_entries_floor(
            signals_bucket,
            run_date,
            threshold=int(
                (config or {}).get("zero_entries_alarm_consecutive_sessions", 3)
            ),
        )

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
        # (a) Telegram transport failed for the order-book summary.
        # (c) not recorded elsewhere — deliberate carve-out
        # (alpha-engine-config-I10031): a failed notification must never
        # abort a trading run; the order book itself is already persisted.
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
    # Session axis (config#1610; supersedes the config#1016 rationale):
    # run_date keys every trade artifact this module produces —
    # order_books/{run_date}/, hold_book_flags/{run_date}.json,
    # trades/order_book/{run_date}.json, the order book's `date` field, and
    # the trades.date column via downstream log_trade. Those are all
    # session-keyed EVENT artifacts, so run_date is the session this book
    # is FOR: session_date(), the physical session in progress or next
    # upcoming — NOT now_dual().trading_day, which is the last *closed*
    # session (D-1 pre-open/intraday; #1016's comment believed it advanced
    # at the open — it does not). Must match daemon.py's run_date axis or
    # the order-book freshness check discards the morning book. Non-strict:
    # a weekend/holiday operator run legitimately builds the book for the
    # NEXT session (a Saturday book is for Monday, not the closed Friday).
    # calendar_date is retained for the audit log line (when the process ran).
    from nousergon_lib.dates import now_dual, session_date
    _dual = now_dual()
    run_date = session_date().isoformat()
    _health_start = _time.time()
    logger.info(
        "Executor starting | session=%s trading_day=%s calendar_date=%s | dry_run=%s | simulate=%s",
        run_date, _dual.trading_day, _dual.calendar_date, dry_run, simulate,
    )

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
    from nousergon_lib.logging import get_flow_doctor
    fd = get_flow_doctor() if not simulate else None

    # ── 0. Check upstream deliverables — hard-fail if inputs are missing/stale ──
    # Independent artifact-freshness gate (config#1725 Phase A): probes the
    # real S3 deliverables via nousergon_lib.artifact_freshness instead of
    # self-reported health/*.json stamps. The trading instance stays idle for
    # the day and is stopped at EOD as usual when this gate trips.
    if not simulate:
        from executor.upstream_artifact_gate import check_upstream_deliverables

        _upstream_failures = check_upstream_deliverables(signals_bucket)
        if _upstream_failures:
            msg = (
                "Upstream deliverables FAILED — executor aborting:\n"
                + "\n".join(f"  - {w}" for w in _upstream_failures)
            )
            logger.error(msg)
            try:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    "\u274c *Upstream deliverables FAILED*\n"
                    f"Date: {run_date}\n"
                    + "\n".join(f"- {w}" for w in _upstream_failures)
                    + "\n\nExecutor aborted — no order book written."
                )
            except Exception:
                # (a) Telegram transport failed for the upstream-failure
                # alert. (c) not recorded elsewhere — deliberate carve-out
                # (alpha-engine-config-I10031): the notification is
                # secondary to the RuntimeError raised immediately below,
                # which is the actual, visible failure signal.
                logger.debug("Upstream failure Telegram notification failed", exc_info=True)
            raise RuntimeError(msg)

        # Direct freshness check on the ArcticDB macro library. The
        # artifact-freshness gate above covers research/predictor/daily_closes
        # deliverables; this catches the "stamp green but data blob is
        # yesterday's" failure mode (partial writes, retries skipping
        # DataPhase1). SPY is the canary — written by the daily_append
        # post-close job to the macro library (NOT universe). If SPY has
        # no row for the last closed trading day, the post-close pipeline
        # did not complete and the executor must abort before any signals
        # are read.
        try:
            import pandas as _pd
            from nousergon_lib.trading_calendar import last_closed_trading_day

            from executor.price_cache import _open_macro_library
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
                # (a) Telegram transport failed for the ArcticDB-freshness
                # alert. (c) not recorded elsewhere — deliberate carve-out
                # (alpha-engine-config-I10031): the notification is
                # secondary to the RuntimeError raised immediately below.
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
        # signals.json::universe read: this is the executor sizing/exit path —
        # the ONE fleet-level exception to "resolve ticker lists from
        # decision_set, not universe" (alpha-engine-config#5809). Formal
        # policy-clause registration tracked separately, not yet landed:
        # alpha-engine-config#6448.
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

        # ── 2b'. Resolve regime substrate ONCE per planning cycle ──────────────
        # Used by Wire 2 (position_sizer regime multiplier) AND Wire 3
        # (regime-aware drawdown tiers). Gated by either flag so the
        # read is skipped entirely when both wires are off — avoids
        # per-cycle S3 GET when neither feature is active. Returns None
        # on any failure mode; both wires degrade to legacy behavior
        # under None.
        regime_intensity_z: float | None = None
        if (
            (config.get("regime_sizing_enabled", False)
             or config.get("regime_drawdown_enabled", False))
            and not simulate
        ):
            try:
                from executor.signal_reader import (
                    extract_intensity_z,
                    read_regime_substrate,
                )
                substrate = read_regime_substrate(signals_bucket)
                regime_intensity_z = extract_intensity_z(substrate)
                logger.info(
                    "regime wires enabled | intensity_z=%s",
                    f"{regime_intensity_z:.3f}" if regime_intensity_z is not None else "None",
                )
            except Exception as _rs_err:
                logger.warning(
                    "regime substrate read failed (%s) — falling back to legacy behavior",
                    _rs_err,
                )
                regime_intensity_z = None

        # ── 2b''. Resolve daily fast-signal forced-bear latch (F2) ─────────────
        # regime-fast-signal-260515.md Stage F2. The daily BOCPD
        # circuit-breaker (alpha-engine-predictor regime_fast_signal
        # stage) latches forced_bear on a confirmed midweek regime
        # break. When acting, the planner's EFFECTIVE market_regime
        # becomes "bear" for ALL gating reads — sizing (Wire 2), drawdown
        # tiers (Wire 3), entry-score gate, and the existing
        # ``market_regime == "bear"`` risk.yaml overrides
        # (bear_max_position_pct etc.). No separate halt knob: forced_bear
        # == bear, the system already knows what bear means. Bear caps
        # are hard ceilings so the most-protective of {continuous wires,
        # bear floor} naturally binds (guardrail 3, no double-count).
        #
        # The read is UNGATED (always in non-simulate) so the F2
        # parallel-observe window has data even with the flag off; the
        # behavior (regime override) is gated on
        # ``regime_forced_bear_enabled`` (default false). One S3 GET per
        # planning cycle — negligible (planner runs ~1×/morning).
        if not simulate:
            try:
                from executor.signal_reader import (
                    extract_forced_bear,
                    read_fast_signal,
                )
                _fb = extract_forced_bear(read_fast_signal(signals_bucket))
                if _fb and config.get("regime_forced_bear_enabled", False):
                    logger.warning(
                        "FORCED-BEAR active — overriding effective "
                        "market_regime '%s' → 'bear' for this planning "
                        "cycle (research regime preserved for audit). "
                        "Source: daily BOCPD fast signal.",
                        market_regime,
                    )
                    market_regime = "bear"
                elif _fb:
                    logger.info(
                        "regime_forced_bear OBSERVE (flag off): fast "
                        "signal latched forced_bear=True; would override "
                        "market_regime '%s' → 'bear'. No behavior change.",
                        market_regime,
                    )
            except Exception as _fb_err:
                logger.warning(
                    "fast-signal read failed (%s) — forced-bear not "
                    "applied; legacy regime behavior preserved.", _fb_err,
                )

        # ── 2b'''. Resolve drawdown-leg posture override ──────────────────────
        # regime-drawdown-hysteresis-260518.md (regime ensemble leg 3).
        #
        # Type-system separation (v0.42.0 Phase 2B —
        # caution-regime-retirement-260528.md):
        # The drawdown leg's protective state is an axis ORTHOGONAL to
        # macro market_regime. Read the severity ordinal as the
        # canonical input:
        #   severity 0 = risk_on (no escalation)
        #   severity 1 = caution (half-step protection)
        #   severity 2 = risk_off / alpha_bleed (full protection, same as macro bear)
        # Only severity 2 overrides macro market_regime to "bear" (its
        # protective tier IS macro bear). Caution-tier (severity 1)
        # logs an OBSERVE counterfactual today — the half-step posture
        # for the executor is a future L924 follow-up (position-sizer
        # caution multiplier or similar). The macro market_regime
        # 3-class invariant is preserved.
        #
        # Same UNGATED-read / GATED-behavior + parallel-observe
        # discipline as the forced-bear block above; the bear-tier
        # override gated on ``drawdown_regime_enabled`` (default false).
        if not simulate:
            try:
                from executor.signal_reader import (
                    extract_drawdown_protective_severity,
                    read_drawdown_substrate,
                )
                _dd_payload = read_drawdown_substrate(signals_bucket)
                _dd_severity = extract_drawdown_protective_severity(_dd_payload)
                _macro_rank = {"bull": 0, "neutral": 1, "bear": 2}
                _cur_rank = _macro_rank.get((market_regime or "").lower(), 1)
                if _dd_severity >= 2 and _cur_rank < _macro_rank["bear"]:
                    if config.get("drawdown_regime_enabled", False):
                        logger.warning(
                            "DRAWDOWN regime active (severity=2) — overriding "
                            "effective market_regime '%s' → 'bear' for this "
                            "planning cycle (research regime preserved for "
                            "audit). Source: daily drawdown leg "
                            "(most-protective).",
                            market_regime,
                        )
                        market_regime = "bear"
                    else:
                        logger.info(
                            "drawdown_regime OBSERVE (flag off): drawdown "
                            "severity=2; would override market_regime '%s' "
                            "→ 'bear'. No behavior change.",
                            market_regime,
                        )
                elif _dd_severity == 1:
                    # Half-step protection — no macro override (caution
                    # tier is orthogonal to 3-class macro). Logged for
                    # OBSERVE-window data + future L924 follow-on
                    # (position-sizer half-step multiplier).
                    logger.info(
                        "drawdown_regime OBSERVE: severity=1 (caution-tier) "
                        "— no macro override (3-class invariant); half-step "
                        "protective semantic lives on drawdown axis only. "
                        "market_regime='%s' unchanged.",
                        market_regime,
                    )
            except Exception as _dd_err:
                logger.warning(
                    "drawdown-leg read failed (%s) — drawdown override not "
                    "applied; legacy regime behavior preserved.", _dd_err,
                )

        # ── 2c. Compute graduated drawdown multiplier ──────────────────────────
        # Pass an events sink so any halt/throttle event lands in
        # `risk_events` ONCE per planning cycle (the per-ticker check_order
        # call deliberately does NOT propagate events to its inner
        # compute_drawdown_multiplier — see risk_guard.py:check_order).
        # ``regime_intensity_z`` resolved above is threaded in so Wire 3
        # scales the soft tier thresholds when the wire is enabled.
        dd_events: list[dict] = []
        dd_multiplier, dd_reason = compute_drawdown_multiplier(
            portfolio_nav, peak_nav, config, events=dd_events,
            regime_intensity_z=regime_intensity_z,
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
                # (alpha-engine-config-I10031) RAISE: a risk_events write
                # failure here is a contract violation on the trading
                # path — the drawdown-tier audit trail must not silently
                # go missing. Propagates to the outer except at the
                # bottom of this function, which logs at ERROR, pages
                # flow-doctor (severity=critical) and records a
                # not-produced health status before re-raising.
                log_risk_event(conn, ev)

        # ── 2c'. Expectancy-gated de-risk stance (config-I2820 / PR2071) ───────
        # Standing operator de-risk rule (halt-or-derisk-live-deployment,
        # config#978 migration): when the signal chain's expectancy metrics
        # (alpha_vs_spy, information_ratio_ci_lower, sharpe_ratio) breach
        # their red-lines, cap position sizes to `derisk_sizing_multiplier`
        # and floor the MVO risk_aversion — independent of, and NOT composed
        # into, `dd_multiplier` (drawdown is a REALIZED portfolio-loss signal
        # that also drives drawdown_forced_exit at §2g; expectancy is a
        # FORWARD-LOOKING signal-chain-quality signal with no forced-exit
        # side effect — conflating the two would spuriously force-exit held
        # positions on a pure expectancy breach). Fail-loud BY DESIGN: raises
        # DeriskGateConfigError (not caught here) when the flag is enabled
        # but config/ledger is malformed — see executor/derisk_gate.py
        # module docstring. When the flag is absent/false (default), this is
        # a zero-cost no-op (bit-identical pre-this-PR behavior).
        #
        # Skipped entirely in simulate mode (backtester): mirrors the
        # regime-substrate read above (§2b') — the backtester pre-loads all
        # state once at simulate-loop bootstrap and doesn't want a live S3
        # read per simulated date. `derisk_multiplier` stays at its 1.0
        # no-op default under simulate, same posture as `regime_intensity_z`
        # staying None.
        from executor.derisk_gate import evaluate_derisk_gate

        derisk_multiplier = 1.0
        derisk_gate = None
        if not simulate:
            derisk_gate = evaluate_derisk_gate(config, bucket=signals_bucket)
            derisk_multiplier = derisk_gate.sizing_multiplier
            if derisk_gate.active:
                logger.warning("De-risk gate ACTIVE: %s", derisk_gate.reason)
        if conn and derisk_gate is not None and derisk_gate.enabled:
            signals_date_for_events = signals_raw.get("date", run_date) if signals_raw else run_date
            derisk_event = {
                "date": run_date,
                "event_type": "throttle" if derisk_gate.active else "observation",
                "rule": "derisk_expectancy_gate",
                "reason": derisk_gate.reason,
                "value": derisk_gate.sizing_multiplier,
                "threshold": None,
                "market_regime": market_regime,
                "signal_date": signals_date_for_events,
                "prediction_date": predictions_date,
                "context": derisk_gate.to_log_dict(),
            }
            # (alpha-engine-config-I10031) RAISE: same audit-trail
            # contract as the drawdown risk_event write above — a swallow
            # here is a contract violation on the trading path.
            # Propagates to the outer except, which pages flow-doctor and
            # records a not-produced health status before re-raising.
            log_risk_event(conn, derisk_event)
        if not simulate and not dry_run and signals_bucket:
            try:
                _write_derisk_gate_artifact(derisk_gate, signals_bucket, run_date)
            except Exception as e:
                logger.warning(
                    "de-risk gate decision-artifact write failed (non-blocking): %s", e,
                )

        # ── 2d. Strategy layer: evaluate exit rules on held positions ──────────
        strategy_config = load_strategy_config(config)

        # Build signals lookup for exit manager
        # signals.json::universe read: this is the executor sizing/exit path —
        # the ONE fleet-level exception to "resolve ticker lists from
        # decision_set, not universe" (alpha-engine-config#5809). Formal
        # policy-clause registration tracked separately, not yet landed:
        # alpha-engine-config#6448.
        signals_by_ticker = {}
        for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and t not in signals_by_ticker:
                signals_by_ticker[t] = s

        # Load price histories from predictor S3 cache (unless injected by backtester)
        # Include ENTER tickers for ATR sizing, momentum gate, and correlation check
        #
        # ── The universe determines the data, never the reverse (config-I7337) ──
        #
        # `predictions_by_ticker` is included because `optimizer_shadow.
        # _build_universe` DECLARES every predicted ticker a candidate
        # (`candidates.update(predictions_by_ticker.keys())`) and then drops any
        # whose price history is absent. Loading only held + ENTER names made
        # that declaration unsatisfiable: the optimizer asked for the predictor's
        # cut and this loader had already decided it would never have it.
        #
        # Measured 2026-08-14, the defect this closes: the scanner's
        # `attractiveness_top_20` reached the predictor correctly (20 names
        # scored, `n_predictions: 23`), and `predictor/optimizer_shadow/
        # latest.json` solved over **14** tickers of which ZERO were from that
        # cut — 10 champion-injected names, 2 held positions, SPY and CASH. The
        # book was AMD 0.11 / SPY 0.86 / CASH 0.03.
        #
        # Why the cut was invisible here specifically: the entry-selection role
        # moved out of the signals producer when the champion pointer flipped to
        # `scanner_predictor_direct` (2026-07-13), so `signals.json` has emitted
        # `signal: "HOLD"` on all 903 rows with `buy_candidates: []` since
        # 2026-07-18 — correctly, by design. Nothing re-labels the predictor's
        # cut as ENTER, so a loader keyed on ENTER can never see it. The two
        # changes are individually right and jointly delete the signal.
        if price_histories is None:
            enter_tickers = [s["ticker"] for s in signals.get("enter", [])]
            all_tickers = list(set(
                list(current_positions.keys())
                + enter_tickers
                + list(predictions_by_ticker.keys())
            ))
            # Also load sector ETF histories for sector-relative exit veto
            held_sectors = {pos.get("sector", "") for pos in current_positions.values()}
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
            # What was ASKED FOR, so `_build_universe` can tell a ticker this
            # loader never requested (a plumbing contradiction — raise) from one
            # it requested and the cache did not have (a data condition —
            # record). Without this the two are indistinguishable downstream,
            # which is precisely how the defect above stayed silent.
            price_histories_requested = set(all_tickers_with_etfs)
        else:
            # Injected by the backtester: it supplies exactly what it loaded, so
            # requested == delivered and no candidate can be a plumbing bug.
            price_histories_requested = set(price_histories)

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
        # config-I7337 (second instance): `signals["enter"]` is the CHAMPION's
        # synthesized cohort, a different set from the predictor's cut that
        # `optimizer_shadow` now solves over. An optimizer-sourced entry whose
        # VWAP is missing gets `triggers.vwap = None`, which makes the VWAP
        # entry trigger inert for exactly the names the optimizer chose. Same
        # set as `price_histories` — see the loader above.
        vwap_tickers = sorted(
            {s["ticker"] for s in signals.get("enter", [])}
            | set(predictions_by_ticker)
        )
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
        # Macro-routed tickers (sector ETFs / VIX / TNX / etc., see
        # _MACRO_SYMBOLS) are intentionally excluded. They live in the
        # Close-only `macro` ArcticDB library which has no atr_14_pct
        # feature column and are used for sector-relative exit veto via
        # price_histories, not ATR-based execution.
        #
        # SPY is NOT excluded — alpha-engine-data #245 (2026-05-15) lifted
        # SPY to a full `universe` ArcticDB member (`_UNIVERSE_EXTRA =
        # frozenset({"SPY"})`, written by both builders/backfill.py and
        # builders/daily_append.py), so `universe.SPY` carries the full
        # OHLCV + atr_14_pct feature column that `load_atr_14_pct` needs.
        # SPY is removed from `_MACRO_SYMBOLS` (price_cache.py) for the
        # same reason — the executor reads SPY from `universe` like any
        # other held ticker.
        #
        # ``atr_map`` kwarg mirrors the vwap_map injection pattern — backtester
        # precomputes once per simulate pipeline, skipping millions of
        # per-call universe.read(ticker) round-trips. Live trading passes
        # atr_map=None and takes the load_atr_14_pct path unchanged.
        # config-I7337 (second instance) — the load-bearing one of the two.
        #
        # `signals["enter"]` is the CHAMPION's synthesized cohort. The optimizer
        # now enters names from the PREDICTOR's cut, which is a different set:
        # measured 2026-08-14, the champion's 10 (AVAV AXON DELL DOCS ELF ENTG
        # IT KTOS ONTO TER) and the 8 the optimizer chose (ANF CPRT FCN FFIV
        # KEX LULU OLLI TREX) were DISJOINT.
        #
        # Consequence of a missing ATR, both measured in code, both silent:
        #   * `optimizer_cutover._build_entry_record` reads `atr_map.get(t, 0.0)`
        #     -> `atr_value: 0.0` -> `daemon.py`'s
        #     `use_bracket = atr_value and atr_value > 0` is falsy, so NO
        #     BRACKET STOP is placed and the position trades unprotected until
        #     the next morning's planner writes one.
        #   * `pullback_pct` is written as 0.0 rather than omitted, and
        #     `entry_triggers.py` reads it with
        #     `triggers.get("pullback_pct", <configured default>)` — a key that
        #     is PRESENT AND ZERO beats the default, so `pullback >= 0.0` is
        #     always true and the entry fires on the first tick as a market
        #     buy with no pullback discipline.
        #
        # Same set as `price_histories` and `vwap_tickers`: held ∪ enter ∪
        # predictions. This is the sibling of the loader config-I7337 fixed —
        # a fix has to survive the class, not the instance.
        atr_tickers = [s["ticker"] for s in signals.get("enter", [])]
        atr_tickers += list(current_positions.keys())
        atr_tickers += list(predictions_by_ticker)
        atr_tickers = sorted(set(atr_tickers) - _MACRO_SYMBOLS)
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

        # Resolve the optimizer-authority flags once, early. Under
        # ``use_portfolio_optimizer`` the optimizer is the SOLE authority for
        # portfolio-level exits — it emits target-0 / scale-down SELLs in
        # step 5b. ``optimizer_owns_exits`` additionally requires live mode
        # (not simulate / not dry_run) because the optimizer cutover that
        # GENERATES those SELLs only runs live (see step 5b gating). Gating
        # the suppression on the same condition guarantees we never end up in
        # a state where legacy exits are off AND optimizer exits are not
        # generated (which would mean no exits at all). In simulate/dry_run
        # the legacy path is unchanged — preserving backtester parity.
        #
        # Legacy "alpha" strategy exits (ATR trailing-stop, fallback_stop,
        # profit_take, momentum_exit, time_decay_*, catalyst_hard_exit) fight
        # the optimizer's target book and are suppressed BY CONSTRUCTION —
        # incident 2026-05-29: the full SPY core was time-decay-exited and
        # COST/RGEN were stopped/profit-taken while the optimizer wanted to
        # keep them, dumping ~$196k to idle cash. Only hard-risk overrides
        # survive: drawdown forced-exit (§2g below) and the daemon's
        # catastrophic gap stop.
        use_optimizer = bool(config.get("use_portfolio_optimizer", False))
        optimizer_owns_exits = use_optimizer and not simulate and not dry_run
        if optimizer_owns_exits:
            strategy_exits = []
            logger.info(
                "use_portfolio_optimizer=True — suppressing legacy alpha "
                "strategy exits; optimizer owns portfolio exits (drawdown "
                "forced-exit + catastrophic gap stop remain as hard-risk "
                "overrides)"
            )
        else:
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

        # ── 2f'. Position loss-floor (MAE) hard-risk exits ────────────────────
        # Stance-agnostic maximum-adverse-excursion floor: cut any held
        # position whose loss from avg cost breaches position_loss_floor_pct.
        # Runs UNCONDITIONALLY — like drawdown forced-exit (§2g) and the
        # catastrophic gap stop, this is a hard-risk override that survives the
        # optimizer (``optimizer_owns_exits`` suppresses *alpha* strategy exits,
        # NOT risk floors). Closes the falling-knife gap that let COIN bleed
        # -19% un-cut while ranked #1 (L4549a). The non-optimizer path already
        # evaluates the floor inside evaluate_exits; the dedup below prevents a
        # double-add there. Price falls back to last mark (market_value/shares)
        # when live pricing is unavailable pre-open, so the floor still fires.
        if strategy_config.get("position_loss_floor_enabled", True) and current_positions:
            floor_existing_exits = (
                {s["ticker"] for s in signals.get("exit", [])}
                | {s["ticker"] for s in strategy_exits if s["action"] == "EXIT"}
            )
            for t, pos in current_positions.items():
                if t in floor_existing_exits:
                    continue
                if int(pos.get("shares", 0)) <= 0:
                    continue
                px = ibkr.get_current_price(t)
                if px is None:
                    shares = pos.get("shares") or 0
                    mv = pos.get("market_value")
                    px = (mv / shares) if (mv and shares) else None
                floor_exit = check_position_loss_floor(
                    ticker=t,
                    current_price=px,
                    avg_cost=pos.get("avg_cost"),
                    strategy_config=strategy_config,
                )
                if floor_exit:
                    strategy_exits.append(floor_exit)
                    logger.info(
                        f"POSITION LOSS FLOOR EXIT (hard-risk override): {t} — "
                        f"{floor_exit['detail']}"
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
                            if hasattr(next_date, 'date'):
                                next_date = next_date.date()
                            elif isinstance(next_date, str):
                                next_date = date.fromisoformat(next_date)
                            days_until = (next_date - date.fromisoformat(run_date)).days
                            if days_until >= 0:
                                earnings_by_ticker[t] = days_until
                except Exception:
                    # (a) earnings-calendar load failed for this ticker —
                    # earnings-proximity gating silently does not apply.
                    # (c) recorded at ERROR (visible at the INFO root
                    # level) — the app log stream is the recording
                    # surface. Overlaps alpha-engine-config-I7347
                    # deliverable 3; fixed here, closed there by
                    # reference (alpha-engine-config-I10031).
                    logger.error("Failed to load earnings data", exc_info=True)

        # ── 2g. Drawdown forced exits ─────────────────────────────────────────
        if strategy_config.get("drawdown_forced_exit_enabled", True) and dd_multiplier < 1.0:
            forced_exit_count = 0
            if dd_multiplier <= 0.25:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier3_count", 2)
            elif dd_multiplier <= 0.50:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier2_count", 1)

            if forced_exit_count > 0 and current_positions:
                existing_exit_tickers = {
                    s["ticker"] for s in signals.get("exit", [])
                } | {
                    s["ticker"] for s in strategy_exits if s["action"] == "EXIT"
                }

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
        # Cutover flag (PR 5 of portfolio-optimizer-260511): when true, the
        # legacy 1/n entry planner is skipped — the optimizer's MVO solution
        # in step 5b drives the order book directly. ``use_optimizer`` /
        # ``optimizer_owns_exits`` resolved once at §2d above. Under the
        # optimizer, legacy research EXIT/REDUCE and alpha strategy exits are
        # suppressed (the optimizer emits target-0/scale-down SELLs instead);
        # only drawdown forced-exit + the catastrophic gap stop remain.
        if use_optimizer and not simulate and not dry_run:
            logger.info(
                "use_portfolio_optimizer=True — skipping legacy _plan_entries; "
                "optimizer drives entries from step 5b would_be_trades"
            )
            n_entered = 0
            entry_orders: list[dict] = []
            blocked_entries: list[dict] = []
            plan_risk_events: list[dict] = []
        else:
            # Per-name ADV$ from the scanner tradeability artifact
            # (crucible-research#343) → the position_sizer ADV size cap
            # (config#1401). Fail-soft: absent artifact → empty map → no cap
            # (legacy sizing). Loaded once here and threaded through
            # _plan_entries → decide_entries → compute_position_size.
            try:
                from executor.signal_reader import (
                    extract_adv_usd,
                    read_universe_tradeability,
                )
                adv_map = extract_adv_usd(
                    read_universe_tradeability(signals_bucket, run_date)
                )
            except Exception as _adv_err:  # noqa: BLE001 — construction refinement, never a gate
                logger.warning(
                    "ADV map load failed (%s) — position sizer ADV cap disabled "
                    "this run (fail-soft).", _adv_err,
                )
                adv_map = {}
            # ``regime_intensity_z`` resolved once at §2b' above — threaded
            # into both Wire 2 (position_sizer) and Wire 3 (compute_drawdown_multiplier).
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
                derisk_multiplier=derisk_multiplier,
                signal_age_days=signal_age_days,
                earnings_by_ticker=earnings_by_ticker,
                vwap_map=vwap_map,
                coverage_map=coverage_map,
                ob=ob,
                run_date=run_date,
                dry_run=dry_run,
                simulate=simulate,
                predictions_date=predictions_date,
                regime_intensity_z=regime_intensity_z,
                adv_map=adv_map,
            )
        orders.extend(entry_orders)

        # Log blocked entries to shadow book for evaluation
        if conn and blocked_entries:
            for be in blocked_entries:
                # (alpha-engine-config-I10031) RAISE: a shadow-book write
                # failure silently drops the S-slot challenger's evidence
                # (its `n` goes quietly wrong). Propagates to the outer
                # except, which pages flow-doctor and records a
                # not-produced health status before re-raising.
                log_shadow_book_block(conn, be)

        # Persist structured veto/override events (Phase 2 transparency-
        # inventory — *risk decisions* row). Sibling of the shadow-book
        # log: same family, different axis. Free-text block_reason stays
        # in shadow_book; rule + value + threshold lands in risk_events.
        if conn and plan_risk_events:
            for ev in plan_risk_events:
                # (alpha-engine-config-I10031) RAISE: identical audit-write
                # shape to the drawdown/derisk risk_event swallows above —
                # a third occurrence of the same swallow, treated the same
                # for consistency (not individually named in the issue's
                # table of six, but the same audit surface). Propagates to
                # the outer except, which pages flow-doctor before
                # re-raising.
                log_risk_event(conn, ev)

        # ── 4–5. Process EXIT and REDUCE signals ────────────────────────────────
        # Under the optimizer, research EXIT is already handled via eligibility
        # (signal=="EXIT" → ineligible → optimizer emits a target-0 SELL), and
        # research REDUCE genuinely CONFLICTS — the optimizer may want to hold
        # or even add to a name research is trimming. Suppress both legacy
        # research-driven exit paths; only drawdown_forced_exit (carried in
        # strategy_exits) and the optimizer's own SELLs remain.
        exit_signals_in = signals
        if optimizer_owns_exits:
            exit_signals_in = {**signals, "exit": [], "reduce": []}
        exit_orders = _plan_exits_and_reduces(
            signals=exit_signals_in,
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

        # ── 5b. Portfolio optimizer — shadow (PR 2) or live (PR 5) ───────────────
        #
        # Shadow mode (``shadow_portfolio_optimizer: true``): runs the MVO
        # optimizer alongside the legacy 1/n planner, logs target weights +
        # diagnostics to S3, NEVER touches the order book.
        #
        # Cutover mode (``use_portfolio_optimizer: true``, PR 5 of
        # portfolio-optimizer-260511): the legacy ``_plan_entries`` above
        # was skipped; here we translate ``would_be_trades`` from the
        # optimizer log into ``ob.add_entry`` / ``ob.add_urgent_exit``
        # records. Exits already populated by ``_plan_exits_and_reduces``
        # take precedence (OrderBook dedup-by-ticker handles overlaps).
        #
        # On failure of the optimizer in cutover mode we leave the order
        # book empty rather than fall back to the legacy path — wrong
        # trades are strictly worse than no trades for a one-day window.
        # ``legacy_sizer_fallback`` is deferred to a later PR.
        shadow_log: dict | None = None
        # Hold-book safeguard state — bound here so it's always available to
        # the order-book rationale build below (the optimizer block that
        # assigns them only runs under ``use_optimizer``; on the legacy path
        # these stay at their no-safeguard defaults).
        _gate: dict | None = None
        _hold_book: bool = False
        _hold_diag: dict = {}
        run_optimizer_now = (
            not simulate and not dry_run
            and (
                config.get("shadow_portfolio_optimizer", False)
                or use_optimizer
            )
        )
        if run_optimizer_now:
            try:
                from executor.optimizer_shadow import run_shadow_optimizer
                shadow_log = run_shadow_optimizer(
                    signals_raw=signals_raw,
                    predictions_by_ticker=predictions_by_ticker,
                    current_positions=current_positions,
                    portfolio_nav=portfolio_nav,
                    price_histories=price_histories or {},
                    config=config,
                    signals_bucket=signals_bucket,
                    run_date=run_date,
                    legacy_orders=orders,
                    # What the loader was ASKED for, so `_build_universe` can
                    # separate a plumbing contradiction (declared a candidate,
                    # never requested → raise) from a data condition (requested,
                    # cache empty → record). config-I7337.
                    price_histories_requested=price_histories_requested,
                )
            except Exception as _shadow_err:
                logger.warning(
                    f"Shadow portfolio optimizer wrapper raised "
                    f"(non-blocking): {_shadow_err}"
                )
                shadow_log = None

        if use_optimizer and not simulate and not dry_run:
            from executor.optimizer_cutover import (
                apply_optimizer_targets_to_orderbook,
                is_log_usable,
            )
            from executor.signal_reader import read_distribution_gate
            # ── Hold-book safeguard (2026-06-01 decision; 2026-06-29 redesign) ─
            # If the predictor's output-distribution gate flagged this batch
            # AND the signal the optimizer actually trades on is itself
            # collapsed, DO NOT let the optimizer rotate the book — hold the
            # current positions and surface the discrepancy for operator review.
            # Stops are still written below; the daemon's hard-risk overrides
            # (drawdown forced-exit, catastrophic gap stop) remain active.
            #
            # Origin: the 6/1 8-model level-biased recalibration (89.7%
            # DOWN-skew) flushed the book to SPY before this safeguard existed.
            # Redesign (config#1176): the gate judges isotonic ``p_up``, which
            # collapses onto a flat staircase step on low-dispersion-but-healthy
            # days and false-halted the book (6/22, 6/29). We now gate the hold
            # on degeneracy of the LEVEL-NEUTRALIZED ``predicted_alpha`` the
            # optimizer trades on, not on the ``p_up`` calibration artifact —
            # see ``_should_hold_book``. Fail-open: a missing/None gate proceeds.
            _gate = read_distribution_gate(signals_bucket)
            _hold_book, _hold_diag = _should_hold_book(_gate, predictions_by_ticker)
            # Publish the verdict this run acted on (alpha-engine-config-I10184
            # deliverable 2). Before this, the gate verdict reached only an S3
            # artifact nobody alarms on — 2026-09-08 flagged and paged nothing.
            emit_distribution_gate_metrics(_gate, _hold_diag)
            # Written unconditionally (held True or False) every run so the
            # console banner reflects THIS cycle, not the last time the
            # safeguard ever fired — see _write_hold_book_flag docstring.
            try:
                _write_hold_book_flag(
                    signals_bucket, run_date, predictions_date, _gate,
                    held=_hold_book, diag=_hold_diag,
                )
            except Exception as _hb_err:
                logger.warning(
                    "hold-book flag artifact write failed (non-blocking): %s",
                    _hb_err,
                )
            if _hold_book:
                logger.warning(
                    "HOLD-BOOK SAFEGUARD ACTIVE: predictor gate FLAGGED (check=%s) "
                    "AND the tradable predicted_alpha signal is collapsed for "
                    "predictions %s — suppressing optimizer rebalance; current "
                    "book retained (stops written; daemon hard-risk overrides "
                    "remain active). gate_reason=%s hold_diag=%s",
                    (_gate or {}).get("failed_check"), predictions_date,
                    (_gate or {}).get("reason"), _hold_diag,
                )
            elif is_log_usable(shadow_log):
                if _hold_diag.get("gate_flagged"):
                    # Gate flagged on the isotonic p_up artifact, but the
                    # tradable predicted_alpha is well-dispersed — this is the
                    # config#1176 false-halt class. Proceed with the rebalance;
                    # log loudly so the override is auditable (flow-doctor / CW).
                    logger.warning(
                        "HOLD-BOOK gate flagged (check=%s) but the tradable "
                        "predicted_alpha signal is HEALTHY (%s) — the gate judges "
                        "isotonic p_up, a calibration artifact on low-dispersion "
                        "days (config#1176); the optimizer trades level-neutralized "
                        "predicted_alpha. NOT holding — proceeding with optimizer "
                        "rebalance. gate_reason=%s",
                        (_gate or {}).get("failed_check"), _hold_diag,
                        (_gate or {}).get("reason"),
                    )
                opt_entries, opt_exits = apply_optimizer_targets_to_orderbook(
                    log=shadow_log,
                    ob=ob,
                    ibkr=ibkr,
                    current_positions=current_positions,
                    price_histories=price_histories,
                    atr_map=atr_map,
                    strategy_config=strategy_config,
                    vwap_map=vwap_map,
                    signals_raw=signals_raw,
                    predictions_by_ticker=predictions_by_ticker,
                    market_regime=market_regime,
                    run_date=run_date,
                    predictions_date=predictions_date,
                )
                n_entered = len(opt_entries)
                if not opt_entries and not opt_exits:
                    _diag = (shadow_log or {}).get("diagnostics") or {}
                    logger.info(
                        "Optimizer solved %r with no rebalance trades "
                        "(turnover_one_way=%s below the trade threshold) — "
                        "the current portfolio already matches target. Order "
                        "book intentionally carries no optimizer entries/"
                        "exits today; existing positions are retained with "
                        "stops. This is a valid HOLD, not a fault.",
                        _diag.get("status"),
                        _diag.get("turnover_one_way"),
                    )
            else:
                logger.error(
                    "use_portfolio_optimizer=True but optimizer log is not "
                    "usable (shadow_status=%r, diag=%r) — leaving order book "
                    "empty for safety. Operator must investigate.",
                    (shadow_log or {}).get("shadow_status"),
                    ((shadow_log or {}).get("diagnostics") or {}).get("status"),
                )

        # ── 6. Write stop records and save order book for daemon ────────────────
        if not simulate and not dry_run:
            # RAISES on failure (alpha-engine-config-I10190). This produces the
            # daemon's stop records; a run that cannot write them has left a
            # live book with no per-name exit authority, and that may not be a
            # WARNING. See write_stops_and_finalize_guarded for the rationale.
            write_stops_and_finalize_guarded(ibkr, ob, price_histories, atr_map, strategy_config, conn, run_date, blocked_entries, signals_bucket, use_optimizer=use_optimizer, signals_raw=signals_raw, config=config)

        # ── 6b. Per-ticker order-book rationale artifact ────────────────────────
        # Audit-stable record answering "why is ticker X in state S
        # today" for the whole considered universe (incl. excluded /
        # vetoed). Pure join + serialize over structures already
        # materialized above — no new instrumentation. Non-blocking:
        # the planner must never fail over an audit artifact (same
        # posture as the shadow optimizer + data manifest below).
        if not simulate and not dry_run and signals_bucket:
            try:
                import boto3
                from nousergon_lib.dates import now_dual
                from nousergon_lib.eval_artifacts import new_eval_run_id

                from executor.order_book_rationale import (
                    build_order_book_rationale,
                    write_order_book_rationale,
                )

                _dual = now_dual()
                _rationale = build_order_book_rationale(
                    signals=signals,
                    predictions_by_ticker=predictions_by_ticker,
                    order_book_data=ob.data,
                    blocked_entries=blocked_entries,
                    risk_events=plan_risk_events,
                    market_regime=market_regime,
                    run_date=run_date,
                    signal_date=(
                        signals_raw.get("date", run_date)
                        if signals_raw else run_date
                    ),
                    prediction_date=predictions_date,
                    calendar_date=_dual.calendar_date,
                    trading_day=_dual.trading_day,
                    run_id=new_eval_run_id(),
                    # L121 — when the portfolio optimizer is the
                    # authoritative entry driver, legacy blocked/
                    # risk_event lists are empty; the optimizer's
                    # eligibility mask supplies the per-ticker
                    # rejection reasons instead.
                    optimizer_shadow_log=shadow_log,
                    current_positions=current_positions,
                    # Hold-book safeguard state (schema 1.3.0 book_status) —
                    # the gate verdict + the _should_hold_book dispersion
                    # diag, already in hand from §5b. Pure join; no new read.
                    distribution_gate=_gate,
                    hold_book_active=_hold_book,
                    hold_book_diag=_hold_diag,
                )
                write_order_book_rationale(
                    _rationale,
                    s3_client=boto3.client("s3"),
                    bucket=signals_bucket,
                )
            except Exception as _rat_err:
                # L171 (2026-05-22) — OBR is observability infra, not
                # load-bearing, so we don't crash the morning planner on
                # a rationale-write failure. But the failure MUST reach
                # the operator: page 16 falling back to yesterday's
                # snapshot is exactly the silent regression
                # [[feedback_no_silent_fails]] forbids. WARN-log here
                # is one recording surface; lib v0.24.0 alerts.publish
                # is the second (SNS + Telegram). Both best-effort —
                # alert-publish itself is best-effort per the same
                # secondary-observability clause as the score_aggregator
                # pillar-sanity path. dedup_key collapses repeat
                # failures within the publish window.
                logger.warning(
                    "Order-book rationale write failed (non-blocking): %s",
                    _rat_err,
                )
                try:
                    from executor.notifier import publish_ops_alert

                    publish_ops_alert(
                        message=(
                            f"[executor/main.py] Order-book rationale write "
                            f"failed for run_date={run_date}: "
                            f"{type(_rat_err).__name__}: {_rat_err}. "
                            f"Page 16 will fall back to the prior snapshot "
                            f"until next morning-planner run."
                        ),
                        severity="WARN",
                        source="alpha-engine/executor/main.py",
                        dedup_key=f"obr_write_failed_{run_date}",
                    )
                except Exception as _alert_err:  # noqa: BLE001 — secondary observability
                    logger.warning(
                        "OBR-failure alert publish itself failed: %s "
                        "(WARN log above remains the failure surface)",
                        _alert_err,
                    )

        # ── 7. Backup and disconnect ─────────────────────────────────────────
        if not dry_run and not simulate:
            backup_to_s3(db_path, run_date, trades_bucket)

        # ── 8. Write health status ────────────────────────────────────────────
        if not simulate:
            try:
                from nousergon_lib.health import Deliverable, write_health
                n_exit = len(exit_orders)
                n_blocked = len(enter_signals) - n_entered
                write_health(
                    module_name="executor",
                    deliverables=[
                        Deliverable(name="morning_plan", required=True, produced=True),
                    ],
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    summary={
                        "n_orders": n_entered + n_exit,
                        "n_enter": n_entered,
                        "n_exit": n_exit,
                        "n_blocked": n_blocked,
                    },
                    bucket=signals_bucket,
                )
            except Exception as _he:
                logger.warning("Health status write failed: %s", _he)

            # ── Data manifest ──────────────────────────────────────────────────
            try:
                from executor.data_manifest import write_data_manifest
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

        if not simulate and not dry_run:
            try:
                import shutil
                import subprocess
                bash_bin = shutil.which("bash")
                if not bash_bin:
                    raise RuntimeError("bash not found on PATH")
                subprocess.run(
                    [bash_bin, "/home/ec2-user/alpha-engine/infrastructure/emit-heartbeat.sh", "executor-morning"],
                    check=False,
                    capture_output=True,
                )
            except Exception as _e:
                logger.warning("Heartbeat emit failed: %s (non-fatal)", _e)

        if simulate:
            return orders
    except Exception as _exc:
        logger.exception("Executor error — ensuring IBKR disconnect")
        if fd:
            fd.report(_exc, severity="critical", context={
                "site": "executor_main", "dry_run": dry_run, "run_date": run_date})
        if not simulate:
            try:
                from nousergon_lib.health import Deliverable, write_health
                write_health(
                    module_name="executor",
                    deliverables=[
                        Deliverable(name="morning_plan", required=True, produced=False),
                    ],
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    error=str(sys.exc_info()[1]),
                    bucket=config.get("signals_bucket", "alpha-engine-research") if 'config' in dir() else "alpha-engine-research",
                )
            except Exception:
                # (a) the write_health call above (reporting the PRIMARY
                # failure `_exc`) itself failed. (c) recorded at ERROR
                # with a traceback (visible at the INFO root level) —
                # deliberately NOT raised: this is the last-resort error
                # handler and a `raise` here would replace/mask `_exc`
                # (the real failure) with this secondary write failure,
                # right before the `raise` two lines below re-raises
                # `_exc`. flow-doctor (`fd.report` above, severity=
                # critical) is the durable recording surface for the
                # primary failure regardless of whether this write
                # succeeds (alpha-engine-config-I10031).
                logger.error("Health status write failed on error path", exc_info=True)
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
            import json
            import shutil
            import subprocess

            from executor.price_cache import load_price_histories
            # Read signals to know which tickers to price
            aws_bin = shutil.which("aws")
            if not aws_bin:
                raise RuntimeError("aws CLI not found on PATH")
            result = subprocess.run(
                [aws_bin, "s3", "cp", f"s3://{bucket}/signals/{date.today()}/signals.json", "-"],
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
        # guard_entrypoint captures an uncaught crash (bare raise) and reports
        # it to flow-doctor before re-raising — the log handler only sees
        # logger.error/exception, not a propagating exception. No-ops when
        # flow-doctor is inactive (e.g. local dev).
        with guard_entrypoint():
            run(dry_run=args.dry_run)
