"""Pure decision functions extracted from executor/main.py.

These functions implement the Alpha Engine's decision logic — entry
gating, exit selection, drawdown response, position enrichment — as
pure functions: no I/O, no globals, no broker connections, no S3
reads, no logger handlers other than this module's logger.

Design intent (Tier 2 of the 2026-04-27 backtester perf arc):

  * Live executor (``executor.run``) wraps these functions in a thin
    shell that handles config load, IB connection, signals read,
    OrderBook persistence, S3 backup, and email. Live calls these
    functions exactly the way it always did the inline logic.

  * Backtester (``alpha-engine-backtester/backtest.py:_simulate_single_date``)
    calls these functions DIRECTLY, skipping the shell entirely. The
    backtester pre-loads all state ONCE at simulation-loop bootstrap
    (signals, positions, prices, ATR/VWAP/coverage maps) and feeds
    flat scalars + dicts to the deciders per simulate date.

  * Strict parity: live and simulate share the deciders byte-for-byte.
    ``tests/test_decider_parity.py`` pins this with frozen-input fixtures.

Forbidden inside this module (enforced by review, not the type system):
  - ``load_config``, ``load_strategy_config`` — caller passes nested dicts
  - ``OrderBook.load/save/add_entry/add_urgent_exit/add_stop``
  - Any ``ibkr.*`` write (``place_market_order`` etc.) — orders are RETURNED, not placed
  - ``ibkr.get_current_price`` — caller passes ``prices_now: dict[str, float]``
  - S3, ArcticDB, yfinance, Telegram, SES
  - ``get_flow_doctor`` reporting — log normally; live shell threads flow-doctor around the call

Permitted:
  - Project ``logger.info`` / ``warning`` / ``debug`` calls (parity with live messages — same lines as the prior inline code)
  - Delegation to ``risk_guard.check_order``, ``risk_guard.compute_drawdown_multiplier``, ``position_sizer.compute_position_size``, ``executor.strategies.exit_manager.evaluate_exits`` (and helpers)
  - Per-call dict / list construction of return values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from executor.decision_capture import (
    DecisionCaptureWriteError,
    capture_position_sizer,
    capture_risk_guard,
    is_decision_capture_enabled,
)
from executor.position_sizer import compute_position_size
from executor.risk_guard import (
    check_order,
    compute_drawdown_multiplier,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def decide_drawdown_response(
    portfolio_nav: float, peak_nav: float, config: dict,
) -> tuple[float, str]:
    """Compute drawdown-tier sizing multiplier + reason.

    Thin wrapper for naming consistency with the other deciders. The
    underlying ``compute_drawdown_multiplier`` (in ``risk_guard``) is
    already pure — this name appears in the live shell + backtester so
    both paths read symmetrically.
    """
    return compute_drawdown_multiplier(portfolio_nav, peak_nav, config)


def enrich_positions(
    current_positions: dict[str, dict],
    signals_raw: dict,
    entry_dates_lookup: dict[str, str | None] | None = None,
    universe_sectors: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return a NEW positions dict with each position's ``sector`` and
    ``entry_date`` populated from signals + entry_dates lookup.

    Does not mutate ``current_positions``.

    ``entry_dates_lookup``: ``{ticker: ISO-date-string or None}`` from
    ``trade_logger.get_entry_dates`` (live) or from the simulated
    client's per-position state (backtester). Pass ``None`` to leave
    ``entry_date`` unset on every position.

    ``universe_sectors``: optional precomputed ``{ticker: sector}`` map.
    When provided, the per-call dict comprehension over ``signals_raw``'s
    universe + buy_candidates is skipped — caller has already built the
    lookup. Live executor passes ``None`` (rebuilt per call); backtester
    passes a precomputed map shared across all 60 combos in a
    ``predictor_param_sweep`` (Tier 3 Part A amortization, 2026-04-27).
    """
    if universe_sectors is None:
        universe_sectors = {
            s["ticker"]: s.get("sector", "")
            for s in (
                signals_raw.get("universe", [])
                + signals_raw.get("buy_candidates", [])
            )
            if s.get("ticker")
        }
    out: dict[str, dict] = {}
    for ticker, pos in current_positions.items():
        new_pos = dict(pos)  # shallow copy
        new_pos["sector"] = universe_sectors.get(ticker, pos.get("sector", ""))
        if entry_dates_lookup is not None:
            new_pos["entry_date"] = entry_dates_lookup.get(ticker)
        out[ticker] = new_pos
    return out


def compute_signal_age_days(signals_raw: dict, run_date: str) -> int:
    """Calendar days between ``signals_raw['date']`` and ``run_date``.

    Returns 0 on parse failure (preserves prior behavior — staleness
    discount silently degrades to no-discount when signals lack a date).
    """
    signals_date_str = signals_raw.get("date", run_date)
    try:
        signals_date = date.fromisoformat(signals_date_str)
        return (date.fromisoformat(run_date) - signals_date).days
    except (ValueError, TypeError):
        return 0


def pick_drawdown_forced_exit_count(
    dd_multiplier: float, strategy_config: dict,
) -> int:
    """Map drawdown multiplier to count of held positions to force-exit.

    Tier 3 (dd_multiplier <= 0.25): default 2 forced exits.
    Tier 2 (dd_multiplier <= 0.50): default 1 forced exit.
    Tier 1+: 0 (drawdown shallow enough that sizing reduction suffices).
    """
    if dd_multiplier <= 0.25:
        return strategy_config.get("drawdown_forced_exit_tier3_count", 2)
    if dd_multiplier <= 0.50:
        return strategy_config.get("drawdown_forced_exit_tier2_count", 1)
    return 0


def decide_drawdown_forced_exits(
    current_positions: dict[str, dict],
    exit_signals: list[dict],
    strategy_exits: list[dict],
    signals_by_ticker: dict[str, dict],
    dd_multiplier: float,
    strategy_config: dict,
) -> list[dict]:
    """Generate forced-exit signals for the lowest-conviction held
    positions when drawdown is severe.

    Mirrors the inline block formerly at ``executor/main.py:1325-1361``.
    """
    if not strategy_config.get("drawdown_forced_exit_enabled", True):
        return []
    if dd_multiplier >= 1.0:
        return []
    forced_count = pick_drawdown_forced_exit_count(dd_multiplier, strategy_config)
    if forced_count == 0 or not current_positions:
        return []

    existing_exit_tickers: set[str] = {
        s["ticker"] for s in exit_signals
    } | {
        s["ticker"] for s in strategy_exits if s.get("action") == "EXIT"
    }

    def _conviction_rank(ticker_pos):
        t, pos = ticker_pos
        sig_data = signals_by_ticker.get(t, {})
        score = sig_data.get("score") or 50
        mv = pos.get("market_value", 0)
        return (score, mv)

    forced: list[dict] = []
    for t, pos in sorted(current_positions.items(), key=_conviction_rank)[:forced_count]:
        if t in existing_exit_tickers:
            continue
        shares_held = int(pos.get("shares", 0))
        if shares_held <= 0:
            continue
        forced.append({
            "ticker": t,
            "action": "EXIT",
            "reason": "drawdown_forced_exit",
            "detail": f"forced exit due to drawdown (dd_mult={dd_multiplier})",
        })
        logger.info(
            "DRAWDOWN FORCED EXIT: %s (score=%s, dd_multiplier=%s)",
            t, _conviction_rank((t, pos))[0], dd_multiplier,
        )
    return forced


# ── Core deciders ────────────────────────────────────────────────────────


@dataclass
class EntryPlan:
    """Result of ``decide_entries``.

    orders: list of executable order dicts. Each has shape
        ``{date, ticker, action='ENTER', shares, price_at_order,
        portfolio_nav_at_order, position_pct, research_*, sector_rating,
        market_regime, price_target_upside, thesis_summary}``.
        Used by the simulated IBKR client (``place_market_order`` per
        order) and the backtester's all_orders accumulator.

    blocked: list of dicts describing why an entry was rejected. Each
        has shape
        ``{ticker, date, sector, sector_rating, research_score,
        conviction, market_regime, portfolio_nav, block_reason,
        + optional current_price/intended_position_pct/intended_shares/
        intended_dollars/predicted_direction/prediction_confidence}``.
        Used by the live shell's ``log_shadow_book_block`` writes.

    n_entered: count of approved entries (== len(orders)).

    entries_with_meta: list of OrderBook-ready entry dicts. Each has the
        full live-shell payload — triggers (pullback_pct, vwap, support
        level), sizing_factors, predicted_*, etc. The backtester
        ignores this list; the live shell iterates and calls
        ``ob.add_entry(entry)`` per item.
    """
    orders: list[dict] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    n_entered: int = 0
    entries_with_meta: list[dict] = field(default_factory=list)
    # Phase 2 transparency-inventory: structured veto/override events.
    # Sibling of `blocked` — same family, different axis. `blocked` is the
    # per-ENTER shadow-book row; `risk_events` is the structured-rule log
    # consumed by `trade_logger.log_risk_event`. Free-text `block_reason`
    # stays in `blocked` for evaluator backtesting; rule + value + threshold
    # land here. The live shell (main.py) iterates and persists.
    risk_events: list[dict] = field(default_factory=list)


@dataclass
class ExitPlan:
    """Result of ``decide_exits_and_reduces``.

    orders: list of executable order dicts (for sim_client.place_market_order
        in simulate mode; informational copy of what the live order book
        will execute in live).

    urgent_exits_with_meta: list of OrderBook-ready urgent_exit dicts
        for the live shell's ``ob.add_urgent_exit`` writes.
    """
    orders: list[dict] = field(default_factory=list)
    urgent_exits_with_meta: list[dict] = field(default_factory=list)


def _batch_confidence_mean(predictions_by_ticker: dict) -> float | None:
    """Mean of ``prediction_confidence`` across all valid predictions in the batch.

    Returns ``None`` when the batch has no usable confidence values — caller
    interprets None as "skip the tightening gate" rather than "tightening
    triggered." Skips entries with missing or non-numeric confidence so a
    sparse predictions file (e.g. early-cutover days) doesn't drag the mean
    toward zero artificially.
    """
    if not predictions_by_ticker:
        return None
    confs: list[float] = []
    for pred in predictions_by_ticker.values():
        if not isinstance(pred, dict):
            continue
        c = pred.get("prediction_confidence")
        if isinstance(c, (int, float)) and c is not None:
            confs.append(float(c))
    if not confs:
        return None
    return sum(confs) / len(confs)


def _apply_batch_confidence_tightening(
    config: dict,
    predictions_by_ticker: dict,
    run_date: str,
) -> dict:
    """Return a derived config with ``min_score_to_enter`` tightened when the
    predictor batch's mean confidence is broadly low.

    Per the 2026-05-07 predictor audit's Phase 1: today's predictor produced
    16 of 27 tickers clamped to ``p_up=0.458`` (mean batch confidence ~0.60),
    5 saturated at ``conf=0.99``. Mean across the batch was ~0.60 — broadly
    low and a meaningful signal that the day's predictor output is degenerate
    even when individual confidences look high. Bumping ``min_score_to_enter``
    by a configurable step under that condition is a backstop tightening
    that doesn't depend on per-ticker confidence (which is already noisy in
    the degenerate case).

    Feature-flagged off by default (``batch_confidence_tightening_enabled:
    false``). Opt in via risk.yaml; validate in shadow before relying on it.

    Returns the original config object unchanged when the gate is disabled
    or the trigger doesn't fire — no copy is made in those cases. When the
    trigger fires, returns a shallow copy with the bumped ``min_score_to_enter``
    so the caller's config dict is never mutated.
    """
    if not config.get("batch_confidence_tightening_enabled", False):
        return config

    # Confidence semantics post-2026-05-12: |p_up - 0.5| * 2 ∈ [0, 1]
    # (alpha-engine-predictor PR #143). Prior 0.65 winner-prob threshold
    # rescales to 0.30 via new = (old - 0.5) * 2.
    threshold = config.get("batch_confidence_threshold", 0.30)
    bump = config.get("batch_confidence_min_score_bump", 10)
    base_min_score = config.get("min_score_to_enter", 70)

    mean_conf = _batch_confidence_mean(predictions_by_ticker)
    if mean_conf is None or mean_conf >= threshold:
        return config

    tightened_min_score = base_min_score + bump
    logger.warning(
        "Batch confidence tightening triggered: mean_confidence=%.3f < %.3f; "
        "min_score_to_enter %d → %d for run_date=%s",
        mean_conf, threshold, base_min_score, tightened_min_score, run_date,
    )
    derived = dict(config)
    derived["min_score_to_enter"] = tightened_min_score
    derived["_batch_confidence_tightening_applied"] = {
        "mean_confidence": mean_conf,
        "threshold": threshold,
        "base_min_score": base_min_score,
        "tightened_min_score": tightened_min_score,
    }
    return derived


def _entry_priority_key(sig: dict, predictions_by_ticker: dict) -> tuple[float, float]:
    """Sort key controlling the order in which ENTER candidates are processed.

    Primary: research composite ``score`` (descending — higher score first).
    Secondary: predictor ``predicted_alpha`` (descending — higher predicted
    alpha breaks ties when research scores match).

    Why this matters: ``decide_entries`` evaluates candidates one at a time,
    and the risk_guard's ``max_total_equity`` / ``max_sector`` caps bind on
    the running tally of approved entries. Without an explicit priority
    sort, the order from ``signals.json`` is whatever order research wrote
    them in — non-deterministic for the executor's purposes. Sorting here
    makes cap-binding cases drop the *lower-priority* candidates rather
    than the *later-listed* ones.

    Returns a tuple of negated values so Python's natural ascending sort
    yields highest-priority first. ``None`` and missing fields default to
    0.0 so signals from older research runs that predate the score field
    still get processed deterministically (just at the bottom).
    """
    score = sig.get("score") or 0.0
    ticker = sig.get("ticker", "")
    pred = predictions_by_ticker.get(ticker, {}) if predictions_by_ticker else {}
    predicted_alpha = pred.get("predicted_alpha") or 0.0
    return (-float(score), -float(predicted_alpha))


def _compute_support_level(price_history, strategy_config: dict) -> float | None:
    """N-day low from price history for support-bounce entry trigger.

    Lifted from ``executor/main.py`` so deciders is fully self-contained
    (no dependency on the live shell). Accepts a pandas DataFrame
    indexed by date with a ``low`` column (post-PR-#108 contract).
    """
    lookback = strategy_config.get("intraday_support_lookback_days", 20)
    if price_history is None or len(price_history) < lookback:
        return None
    lows = price_history["low"].iloc[-lookback:]
    valid = lows[(lows.notna()) & (lows > 0)]
    if valid.empty:
        return None
    return float(valid.min())


def decide_entries(
    *,
    enter_signals: list[dict],
    signals_raw: dict,
    predictions_by_ticker: dict,
    config: dict,
    strategy_config: dict,
    market_regime: str,
    sector_ratings: dict,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict,
    prices_now: dict[str, float],
    price_histories: dict | None,
    atr_map: dict,
    vwap_map: dict,
    coverage_map: dict,
    dd_multiplier: float,
    signal_age_days: int,
    earnings_by_ticker: dict,
    run_date: str,
    predictions_date: str | None = None,
    regime_intensity_z: float | None = None,
) -> EntryPlan:
    """Pure decision pipeline for ENTER signals.

    For each ``sig`` in ``enter_signals`` runs (in this order):
      1. Already-held check (skip if ticker in current_positions)
      2. Momentum confirmation gate (config-controlled)
      3. Earnings proximity warning (logs only, no behavior change)
      4. Current-price availability via ``prices_now[ticker]``
      5. Position sizing (`compute_position_size`)
      6. Shares-round-to-zero check
      7. GBM veto from `predictions_by_ticker[ticker]['gbm_veto']`
      8. Risk guard (`check_order` — score, sector, equity, drawdown,
         correlation gates)

    Returns an ``EntryPlan`` with ``orders`` (executable), ``blocked``
    (shadow book), ``n_entered`` (count), and ``entries_with_meta``
    (live OrderBook payload). Caller decides which to consume.

    All inputs are pure data; no network calls, no file I/O, no broker
    state mutation. The caller is responsible for translating
    ``orders`` into ``sim_client.place_market_order`` calls (simulate
    mode) or for translating ``entries_with_meta`` into
    ``OrderBook.add_entry`` calls (live mode).
    """
    plan = EntryPlan()

    # signal_date is the signals.json filename date the orders sourced from.
    # Distinct from signal_trading_day (NYSE attribution day inside the
    # payload). Both stamped on every structured risk event for artifact
    # lineage parity with PR #138's trades.signal_date column.
    signals_date = signals_raw.get("date", run_date) if signals_raw else run_date

    # Tighten min_score_to_enter when batch-mean prediction_confidence is
    # broadly low (degenerate-predictor backstop). Returns the original
    # config when the feature flag is off or the trigger doesn't fire.
    config = _apply_batch_confidence_tightening(config, predictions_by_ticker, run_date)

    # Process candidates in priority order: research composite score first,
    # predicted_alpha as the tie-break. The risk_guard's max_total_equity
    # and max_sector caps bind based on the order entries are approved, so
    # processing higher-priority candidates first means cap-binding cases
    # surrender lower-priority candidates rather than higher-priority ones.
    # Per the 2026-05-07 predictor audit, this is the smallest-blast-radius
    # use of `predicted_alpha` — it informs ordering without affecting
    # sizing. Stable on missing fields (None → 0.0) so signals from older
    # research runs without scores still get processed deterministically.
    enter_signals = sorted(
        enter_signals,
        key=lambda s: _entry_priority_key(s, predictions_by_ticker),
    )

    for sig in enter_signals:
        ticker = sig["ticker"]
        sector = sig.get("sector", "Technology")
        sector_info = sector_ratings.get(sector, {})
        sector_rating_str = sector_info.get("rating", "market_weight")

        _shadow_base = {
            "ticker": ticker,
            "date": run_date,
            "sector": sector,
            "sector_rating": sector_rating_str,
            "research_score": sig.get("score"),
            "conviction": sig.get("conviction"),
            "market_regime": market_regime,
            "portfolio_nav": portfolio_nav,
        }

        # Structured-event base: every risk event stamps the same lineage
        # fields (date, ticker, sector, market_regime, signal_date,
        # prediction_date) — rule-specific fields (event_type, rule, value,
        # threshold, reason) are merged in at each emit site.
        _event_base = {
            "date": run_date,
            "ticker": ticker,
            "sector": sector,
            "market_regime": market_regime,
            "signal_date": signals_date,
            "prediction_date": predictions_date,
        }

        if ticker in current_positions:
            logger.info(f"SKIP ENTER {ticker} — already in portfolio")
            plan.blocked.append({**_shadow_base, "block_reason": "already in portfolio"})
            continue

        # Momentum confirmation gate (stance-conditional).
        # As of 2026-05-11 (alpha-engine-predictor#136), the predictor
        # emits ``momentum_veto`` + ``momentum_20d`` on every prediction —
        # rule computation lives with the predictor. Stance taxonomy arc
        # (predictor#137 / lib#38/#39) adds per-pick stance routing:
        #
        #   momentum stance → existing momentum_veto applies as-is
        #   value stance    → INVERT — require drawdown to qualify (a
        #                     value pick must actually be oversold;
        #                     skip the standard veto)
        #   quality stance  → relax momentum threshold (-15% vs -5%) —
        #                     defensive names can absorb deeper
        #                     drawdowns before we abandon them
        #   catalyst stance → skip momentum check entirely; require
        #                     catalyst_date (executor's exit-boundary
        #                     signal — without it the position has no
        #                     event-driven thesis to anchor on)
        #
        # Stance read from ``pred_data["stance"]`` (predictor's argmax
        # over its softmax loadings). Continuous loadings live at
        # ``pred_data["stance_loadings"]`` for future weighted-gate
        # consumption — v1 routes by the discrete label only.
        #
        # Backward-compat: stance=None means legacy artifact (predictor
        # pre-#137) → falls through to the original momentum-stance
        # behavior (existing momentum_veto). When predictor_momentum_veto
        # is also absent → inline executor-side fallback.
        pred_data_for_veto = predictions_by_ticker.get(ticker, {})
        if config.get("momentum_gate_enabled", True):
            mom_threshold_pct = config.get("momentum_gate_threshold", -5.0)
            mom_threshold_decimal = float(mom_threshold_pct) / 100.0
            predictor_momentum_veto = pred_data_for_veto.get("momentum_veto")
            predictor_momentum_20d = pred_data_for_veto.get("momentum_20d")
            stance = pred_data_for_veto.get("stance")  # may be None (legacy)
            value_drawdown_min = config.get("value_stance_drawdown_min", -0.05)
            quality_threshold_pct = config.get(
                "quality_stance_momentum_threshold", -15.0
            )
            quality_threshold_decimal = float(quality_threshold_pct) / 100.0

            momentum_20d_decimal: float | None = None
            if isinstance(predictor_momentum_20d, (int, float)):
                momentum_20d_decimal = float(predictor_momentum_20d)

            # ── Value stance: require drawdown to qualify ───────────────
            # A value pick must actually be oversold — that's the entry
            # premise. If the ticker isn't down, the stance label is
            # mis-applied (or the underlying feature has noise) and the
            # pick doesn't fit its declared strategy. Block.
            if stance == "value":
                if (
                    momentum_20d_decimal is not None
                    and momentum_20d_decimal > value_drawdown_min
                ):
                    reason = (
                        f"value stance gate: 20d="
                        f"{momentum_20d_decimal * 100:.1f}% > drawdown min "
                        f"{value_drawdown_min * 100:.1f}% "
                        "(value picks require actual drawdown)"
                    )
                    logger.info(f"SKIP ENTER {ticker} — {reason}")
                    plan.blocked.append({**_shadow_base, "block_reason": reason})
                    plan.risk_events.append({**_event_base,
                        "event_type": "veto",
                        "rule": "stance_gate",
                        "stance": "value",
                        "reason": reason,
                        "value": momentum_20d_decimal,
                        "threshold": value_drawdown_min,
                    })
                    continue
                # Else: drawdown qualifies. Skip the standard momentum
                # gate (the predictor's veto is for trend-following,
                # inappropriate for value picks).
                logger.debug(
                    f"PASS value gate {ticker} — 20d="
                    f"{momentum_20d_decimal * 100 if momentum_20d_decimal is not None else '?':.1f}%"
                )

            # ── Quality stance: relaxed threshold ───────────────────────
            # Defensive names (low vol, low debt) can absorb deeper
            # drawdowns before we abandon them. Default -15% vs the
            # standard -5%. Backtester-tunable.
            elif stance == "quality":
                if (
                    momentum_20d_decimal is not None
                    and momentum_20d_decimal < quality_threshold_decimal
                ):
                    reason = (
                        f"quality stance gate (relaxed): 20d="
                        f"{momentum_20d_decimal * 100:.1f}% < "
                        f"{quality_threshold_pct}%"
                    )
                    logger.info(f"SKIP ENTER {ticker} — {reason}")
                    plan.blocked.append({**_shadow_base, "block_reason": reason})
                    plan.risk_events.append({**_event_base,
                        "event_type": "veto",
                        "rule": "stance_gate",
                        "stance": "quality",
                        "reason": reason,
                        "value": momentum_20d_decimal,
                        "threshold": quality_threshold_decimal,
                    })
                    continue

            # ── Catalyst stance: skip momentum, require catalyst_date ──
            # Event-driven thesis trumps momentum considerations.
            # Without catalyst_date the position has no exit boundary;
            # the executor's catalyst gate (future PR) hard-exits at
            # catalyst_date + 3 trading days — that contract requires
            # the date.
            elif stance == "catalyst":
                catalyst_date = pred_data_for_veto.get("catalyst_date")
                if not catalyst_date:
                    reason = (
                        "catalyst stance gate: catalyst_date missing — "
                        "event-driven thesis requires an exit boundary"
                    )
                    logger.info(f"SKIP ENTER {ticker} — {reason}")
                    plan.blocked.append({**_shadow_base, "block_reason": reason})
                    plan.risk_events.append({**_event_base,
                        "event_type": "veto",
                        "rule": "stance_gate",
                        "stance": "catalyst",
                        "reason": reason,
                        "value": None,
                        "threshold": None,
                    })
                    continue
                # Catalyst stance with valid date → no momentum check.
                logger.debug(
                    f"PASS catalyst gate {ticker} — catalyst_date={catalyst_date}"
                )

            # ── Default (momentum / None): existing momentum_veto ────────
            elif predictor_momentum_veto is not None:
                # Predictor-side veto is authoritative. ``momentum_20d``
                # in the prediction is a decimal (matches feature
                # engineering's ``close/close.shift(20) - 1``).
                veto_source = "predictor"
                if isinstance(predictor_momentum_20d, (int, float)):
                    momentum_20d_decimal = float(predictor_momentum_20d)
                if predictor_momentum_veto:
                    momentum_20d_pct = (
                        f"{momentum_20d_decimal * 100:.1f}%"
                        if momentum_20d_decimal is not None else "?"
                    )
                    reason = (
                        f"momentum gate (predictor): 20d={momentum_20d_pct} "
                        f"< {mom_threshold_pct}%"
                    )
                    logger.info(f"SKIP ENTER {ticker} — {reason}")
                    plan.blocked.append({**_shadow_base, "block_reason": reason})
                    plan.risk_events.append({**_event_base,
                        "event_type": "veto",
                        "rule": "momentum_gate",
                        "reason": reason,
                        "value": momentum_20d_decimal,
                        "threshold": mom_threshold_decimal,
                        "veto_source": "predictor",
                    })
                    continue
            elif price_histories:
                # Backward-compat: predictor didn't emit momentum_veto
                # for this ticker. Fall through to the inline calc.
                ticker_history = price_histories.get(ticker)
                if ticker_history is not None and len(ticker_history) >= 21:
                    veto_source = "executor_fallback"
                    close = ticker_history["close"]
                    momentum_20d_decimal = (
                        float(close.iloc[-1]) / float(close.iloc[-21]) - 1
                    )
                    if momentum_20d_decimal < mom_threshold_decimal:
                        reason = (
                            f"momentum gate (executor fallback): "
                            f"20d={momentum_20d_decimal * 100:.1f}% < "
                            f"{mom_threshold_pct}%"
                        )
                        logger.info(f"SKIP ENTER {ticker} — {reason}")
                        plan.blocked.append({**_shadow_base, "block_reason": reason})
                        plan.risk_events.append({**_event_base,
                            "event_type": "veto",
                            "rule": "momentum_gate",
                            "reason": reason,
                            "value": momentum_20d_decimal,
                            "threshold": mom_threshold_decimal,
                            "veto_source": "executor_fallback",
                        })
                        continue

        # Earnings proximity warning (logs only)
        earnings_warning_days = config.get("earnings_proximity_warning_days", 2)
        pred_data = predictions_by_ticker.get(ticker, {})
        next_earnings_days = (
            earnings_by_ticker.get(ticker)
            or pred_data.get("next_earnings_days")
            or sig.get("next_earnings_days")
        )
        if next_earnings_days is not None and next_earnings_days <= earnings_warning_days:
            logger.warning(
                f"EARNINGS WARNING: {ticker} reports in {next_earnings_days} day(s) — "
                f"entering before earnings carries elevated event risk"
            )

        current_price = prices_now.get(ticker)
        if not current_price:
            logger.warning(f"SKIP ENTER {ticker} — no price available")
            plan.blocked.append({**_shadow_base, "block_reason": "no price available"})
            continue

        atr_pct = atr_map.get(ticker) if config.get("atr_sizing_enabled", True) else None
        pred_confidence = pred_data.get("prediction_confidence")

        sizing = compute_position_size(
            ticker=ticker,
            portfolio_nav=portfolio_nav,
            enter_signals=enter_signals,
            signal=sig,
            sector_rating=sector_rating_str,
            current_price=current_price,
            config=config,
            drawdown_multiplier=dd_multiplier,
            atr_pct=atr_pct,
            prediction_confidence=pred_confidence,
            p_up=pred_data.get("p_up"),
            signal_age_days=signal_age_days,
            days_to_earnings=earnings_by_ticker.get(ticker),
            feature_coverage=coverage_map.get(ticker),
            stance=pred_data.get("stance"),
            regime_intensity_z=regime_intensity_z,
        )

        # Emit executor:position_sizer DecisionArtifact (L2308 PR 2).
        # Captured BEFORE any downstream filtering (shares==0, GBM veto,
        # risk_guard veto) so grading analytics can measure "sizing
        # decisions that got refused" (precision of refusal) alongside
        # "sized → ordered" decisions. Risk-guard + GBM veto captures
        # land in PR 3 (executor:risk_guard).
        # Best-effort: capture failure must never kill planning flow.
        if is_decision_capture_enabled():
            sized_outcome = (
                "shares_zero" if sizing.get("shares", 0) == 0 else "approved"
            )
            sized_outcome_reason = (
                f"shares round to 0 (${sizing['dollar_size']:.0f} / "
                f"${current_price:.2f})"
                if sized_outcome == "shares_zero"
                else None
            )
            try:
                capture_position_sizer(
                    run_date=run_date,
                    ticker=ticker,
                    signal=sig,
                    sector_rating=sector_rating_str,
                    current_price=current_price,
                    portfolio_nav=portfolio_nav,
                    n_enter_signals=len(enter_signals),
                    drawdown_multiplier=dd_multiplier,
                    atr_pct=atr_pct,
                    prediction_confidence=pred_confidence,
                    p_up=pred_data.get("p_up"),
                    signal_age_days=signal_age_days,
                    days_to_earnings=earnings_by_ticker.get(ticker),
                    feature_coverage=coverage_map.get(ticker),
                    stance=pred_data.get("stance"),
                    sizing_result=sizing,
                    sized_outcome=sized_outcome,
                    sized_outcome_reason=sized_outcome_reason,
                )
            except DecisionCaptureWriteError as _cap_exc:
                logger.warning(
                    "decision_capture S3 write failed for SIZE %s — "
                    "continuing planning (capture is observability, not "
                    "load-bearing): %s",
                    ticker, _cap_exc,
                )
            except Exception:  # noqa: BLE001 — capture must never kill planning
                logger.exception(
                    "decision_capture raised unexpected exception for "
                    "SIZE %s — continuing planning", ticker,
                )

        if sizing["shares"] == 0:
            logger.info(
                f"SKIP ENTER {ticker} — shares round to 0 "
                f"(weight={sizing['position_pct']:.3f}, dollar=${sizing['dollar_size']:.0f}, "
                f"price=${current_price:.2f})"
            )
            plan.blocked.append({
                **_shadow_base,
                "block_reason": f"shares round to 0 (${sizing['dollar_size']:.0f} / ${current_price:.2f})",
                "current_price": current_price,
                "intended_position_pct": sizing["position_pct"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
            continue

        # GBM veto — predictor overriding research's ENTER signal.
        # event_type="override" (predictor overrides research, distinct
        # from a risk-rule veto).
        if pred_data.get("gbm_veto"):
            predicted_alpha = pred_data.get("predicted_alpha", 0)
            reason = (
                f"GBM veto: α={predicted_alpha:.2%}, "
                f"rank={pred_data.get('combined_rank')}"
            )
            logger.info(f"VETO {ticker} — {reason}")
            plan.blocked.append({
                **_shadow_base,
                "block_reason": reason,
                "current_price": current_price,
                "intended_position_pct": sizing["position_pct"],
                "intended_shares": sizing["shares"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
            plan.risk_events.append({**_event_base,
                "event_type": "override",
                "rule": "predictor_gbm_veto",
                "reason": reason,
                "value": float(predicted_alpha) if predicted_alpha is not None else None,
                "context": {
                    "predicted_direction": pred_data.get("predicted_direction"),
                    "prediction_confidence": pred_data.get("prediction_confidence"),
                    "combined_rank": pred_data.get("combined_rank"),
                },
            })
            continue

        sig_with_sector = {**sig, "sector_rating": sector_rating_str}

        # Per-ticker events sink — risk_guard appends one event per veto
        # rule (min_score, max_position, bear_underweight, max_sector,
        # max_equity, correlation). Drawdown halt/throttle is emitted
        # once at the portfolio level by the caller; risk_guard does not
        # propagate `events` to its inner compute_drawdown_multiplier
        # call.
        check_events: list[dict] = []
        approved, reason = check_order(
            ticker=ticker,
            action="ENTER",
            dollar_size=sizing["dollar_size"],
            portfolio_nav=portfolio_nav,
            peak_nav=peak_nav,
            current_positions=current_positions,
            sector=sector,
            market_regime=market_regime,
            signal=sig_with_sector,
            config=config,
            price_histories=price_histories,
            events=check_events,
        )
        # Merge per-ticker risk-guard events into the plan-level log,
        # stamping the lineage fields that risk_guard doesn't know about
        # (signal_date, prediction_date).
        for ev in check_events:
            ev.setdefault("date", run_date)
            ev.setdefault("signal_date", signals_date)
            ev.setdefault("prediction_date", predictions_date)
            plan.risk_events.append(ev)

        # Emit executor:risk_guard DecisionArtifact (L2308 PR 3).
        # Captures BOTH the vetoed and approved paths (counterfactual
        # coverage) so backtester grading can measure precision-of-
        # refusal: vetoed entries that would have won → false-positive
        # vetoes; approved entries that became drawdowns → false-
        # negative approvals. Without both directions, only one half of
        # the precision/recall surface is gradable.
        # Best-effort: capture failure must never kill planning flow.
        if is_decision_capture_enabled():
            try:
                capture_risk_guard(
                    run_date=run_date,
                    ticker=ticker,
                    action="ENTER",
                    dollar_size=sizing["dollar_size"],
                    portfolio_nav=portfolio_nav,
                    peak_nav=peak_nav,
                    current_positions=current_positions,
                    sector=sector,
                    market_regime=market_regime,
                    signal=sig_with_sector,
                    config=config,
                    approved=approved,
                    reason=reason,
                    events=check_events,
                )
            except DecisionCaptureWriteError as _cap_exc:
                logger.warning(
                    "decision_capture S3 write failed for RISK_GUARD %s — "
                    "continuing planning (capture is observability, not "
                    "load-bearing): %s",
                    ticker, _cap_exc,
                )
            except Exception:  # noqa: BLE001 — capture must never kill planning
                logger.exception(
                    "decision_capture raised unexpected exception for "
                    "RISK_GUARD %s — continuing planning", ticker,
                )

        if not approved:
            logger.info(f"BLOCKED {ticker} — {reason}")
            plan.blocked.append({
                **_shadow_base,
                "block_reason": reason,
                "current_price": current_price,
                "intended_position_pct": sizing["position_pct"],
                "intended_shares": sizing["shares"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
            continue

        logger.info(
            f"ORDER ENTER {ticker} {sizing['shares']} shares @ ~${current_price:.2f} "
            f"(${sizing['dollar_size']:.0f}, {sizing['position_pct']*100:.1f}% NAV)"
        )

        plan.n_entered += 1

        # Executable order dict — for sim_client.place_market_order or
        # for the live shell's downstream consumers.
        plan.orders.append({
            "date": run_date,
            "ticker": ticker,
            "action": "ENTER",
            "shares": sizing["shares"],
            "price_at_order": current_price,
            "portfolio_nav_at_order": portfolio_nav,
            "position_pct": sizing["position_pct"],
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector": sig.get("sector"),
            "sector_rating": sector_rating_str,
            "market_regime": market_regime,
            "price_target_upside": sig.get("price_target_upside"),
            "thesis_summary": sig.get("thesis_summary"),
        })

        # OrderBook-ready entry dict (live shell consumes via ob.add_entry).
        # ATR sourced from feature-store atr_map (single source of truth).
        # atr_dollar = atr_pct × current_price so trailing-stop downstream
        # keeps dollar semantics. Pullback threshold scales by atr_pct.
        ticker_atr_pct = atr_map.get(ticker)
        if ticker_atr_pct is None:
            raise RuntimeError(
                f"atr_map missing {ticker} at decide_entries — "
                "load_atr_14_pct contract violated. Abort rather than ship "
                "an entry with a bogus zero ATR."
            )
        atr_dollar = ticker_atr_pct * current_price
        pullback_atr_mult = strategy_config.get("intraday_pullback_atr_multiple", 1.0)
        scaled_pullback_pct = ticker_atr_pct * pullback_atr_mult
        ticker_hist = (price_histories or {}).get(ticker)
        pred = predictions_by_ticker.get(ticker, {})

        # signal_date = signals.json filename date the order sourced from
        # (signals_raw["date"]); falls back to run_date when the payload
        # omits the field. This is the artifact-lineage column requested
        # by the Phase 2 transparency-inventory ROADMAP item — distinct
        # from signal_trading_day (NYSE attribution day inside the payload).
        # prediction_date = predictions/{date}.json filename date the GBM
        # veto gate consulted; None when predictions weren't loaded.
        signals_date = signals_raw.get("date", run_date) if signals_raw else run_date
        plan.entries_with_meta.append({
            "ticker": ticker,
            "signal": "ENTER",
            "signal_date": signals_date,
            "prediction_date": predictions_date,
            "shares": sizing["shares"],
            "current_price": current_price,
            "dollar_size": sizing["dollar_size"],
            "position_pct": sizing["position_pct"],
            "atr_value": atr_dollar,
            "atr_pct": ticker_atr_pct,
            "triggers": {
                "pullback_pct": scaled_pullback_pct,
                "pullback_atr_multiple": pullback_atr_mult,
                "atr_pct": ticker_atr_pct,
                "vwap_discount": strategy_config.get("intraday_vwap_discount_pct", 0.005),
                "vwap": vwap_map.get(ticker),
                "support_level": _compute_support_level(ticker_hist, strategy_config),
            },
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector": sig.get("sector"),
            "sector_rating": sector_rating_str,
            "market_regime": market_regime,
            "price_target_upside": sig.get("price_target_upside"),
            "thesis_summary": sig.get("thesis_summary"),
            "predicted_direction": pred.get("predicted_direction"),
            "prediction_confidence": pred.get("prediction_confidence"),
            "predicted_alpha": pred.get("predicted_alpha"),
            # Stance taxonomy arc (2026-05-11) — denormalize predictor's
            # stance + catalyst_date onto the OrderBook entry so the daemon
            # propagates them onto the trade row at ENTER fill time. Exit
            # logic reads them via trade_logger.get_entry_stance_and_catalyst.
            "stance": pred.get("stance"),
            "catalyst_date": pred.get("catalyst_date"),
            "sizing_factors": {
                "sector_adj": sizing.get("sector_adj"),
                "conviction_adj": sizing.get("conviction_adj"),
                "upside_adj": sizing.get("upside_adj"),
                "dd_multiplier": sizing.get("dd_multiplier"),
                "atr_adj": sizing.get("atr_adj"),
                "confidence_adj": sizing.get("confidence_adj"),
                "staleness_adj": sizing.get("staleness_adj"),
                "earnings_adj": sizing.get("earnings_adj"),
                "stance_adj": sizing.get("stance_adj"),
            },
        })

    if len(enter_signals) > 0 and plan.n_entered == 0:
        logger.warning("All %d ENTER signals blocked by risk guard", len(enter_signals))

    return plan


def decide_exits_and_reduces(
    *,
    signals: dict,
    strategy_exits: list[dict],
    current_positions: dict,
    prices_now: dict[str, float],
    predictions_by_ticker: dict,
    config: dict,
    market_regime: str,
    portfolio_nav: float,
    run_date: str,
    signals_date: str | None = None,
    predictions_date: str | None = None,
) -> ExitPlan:
    """Pure decision pipeline for EXIT and REDUCE signals.

    Merges Research signals (``signals['exit']``, ``signals['reduce']``)
    with strategy-generated exits (``strategy_exits`` from
    ``evaluate_strategy_exits``), dedupes by ticker (EXIT takes priority
    over REDUCE for the same ticker), and produces:

      - ``orders``: executable order dicts (action='EXIT' or 'REDUCE')
        ready for sim_client.place_market_order or live shell pass-through.
      - ``urgent_exits_with_meta``: OrderBook-ready urgent_exit dicts
        for ``ob.add_urgent_exit`` writes (live only).

    REDUCE fraction comes from ``config['reduce_fraction']`` (default 0.50).
    """
    plan = ExitPlan()

    # ── EXITs ────────────────────────────────────────────────────────
    all_exit_tickers: set[str] = set()
    all_exits: list[dict] = []
    for sig in signals.get("exit", []):
        t = sig["ticker"]
        if t not in all_exit_tickers:
            all_exit_tickers.add(t)
            all_exits.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig.get("action") == "EXIT" and strat_sig["ticker"] not in all_exit_tickers:
            all_exit_tickers.add(strat_sig["ticker"])
            all_exits.append(strat_sig)

    for sig in all_exits:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            logger.info(f"SKIP EXIT {ticker} — not in portfolio")
            continue

        shares_held = int(current_positions[ticker]["shares"])
        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(f"ORDER EXIT {ticker} {shares_held} shares{reason_tag}")

        # Resolve current price; fall back to avg_cost if missing.
        current_price = prices_now.get(ticker)
        if current_price is None:
            current_price = current_positions[ticker].get("avg_cost", 0)

        plan.orders.append({
            "date": run_date,
            "ticker": ticker,
            "action": "EXIT",
            "shares": shares_held,
            "price_at_order": current_price,
            "portfolio_nav_at_order": portfolio_nav,
            "position_pct": 0.0,
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector_rating": current_positions[ticker].get("sector", ""),
            "market_regime": market_regime,
            "exit_reason": sig.get("reason"),
        })

        pred = predictions_by_ticker.get(ticker, {})
        plan.urgent_exits_with_meta.append({
            "ticker": ticker,
            "signal": "EXIT",
            "signal_date": signals_date,
            "prediction_date": predictions_date,
            "shares": shares_held,
            "reason": sig.get("reason", "research_signal"),
            "detail": sig.get("detail", ""),
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector_rating": current_positions[ticker].get("sector", ""),
            "market_regime": market_regime,
            "predicted_direction": pred.get("predicted_direction"),
            "prediction_confidence": pred.get("prediction_confidence"),
            "predicted_alpha": pred.get("predicted_alpha"),
        })

    # ── REDUCEs ──────────────────────────────────────────────────────
    all_reduce_tickers: set[str] = set()
    all_reduces: list[dict] = []
    for sig in signals.get("reduce", []):
        t = sig["ticker"]
        if t not in all_reduce_tickers:
            all_reduce_tickers.add(t)
            all_reduces.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig.get("action") == "REDUCE" and strat_sig["ticker"] not in all_reduce_tickers:
            if strat_sig["ticker"] not in all_exit_tickers:
                all_reduce_tickers.add(strat_sig["ticker"])
                all_reduces.append(strat_sig)

    reduce_frac = config.get("reduce_fraction", 0.50)

    for sig in all_reduces:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            continue

        shares_held = int(current_positions[ticker]["shares"])
        shares_to_sell = int(shares_held * reduce_frac)
        if shares_to_sell == 0:
            logger.info(f"SKIP REDUCE {ticker} — position too small to reduce")
            continue

        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(
            f"ORDER REDUCE {ticker} {shares_to_sell} shares "
            f"({reduce_frac:.0%} reduction){reason_tag}"
        )

        current_price = prices_now.get(ticker)
        if current_price is None:
            current_price = current_positions[ticker].get("avg_cost", 0)

        remaining_value = (shares_held - shares_to_sell) * (current_price or 0)
        plan.orders.append({
            "date": run_date,
            "ticker": ticker,
            "action": "REDUCE",
            "shares": shares_to_sell,
            "price_at_order": current_price,
            "portfolio_nav_at_order": portfolio_nav,
            "position_pct": remaining_value / portfolio_nav if portfolio_nav else 0,
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector_rating": current_positions[ticker].get("sector", ""),
            "market_regime": market_regime,
            "exit_reason": sig.get("reason"),
        })

        pred = predictions_by_ticker.get(ticker, {})
        plan.urgent_exits_with_meta.append({
            "ticker": ticker,
            "signal": "REDUCE",
            "signal_date": signals_date,
            "prediction_date": predictions_date,
            "shares": shares_to_sell,
            "reason": sig.get("reason", "research_signal"),
            "detail": sig.get("detail", ""),
            "research_score": sig.get("score"),
            "research_conviction": sig.get("conviction"),
            "research_rating": sig.get("rating"),
            "sector_rating": current_positions[ticker].get("sector", ""),
            "market_regime": market_regime,
            "predicted_direction": pred.get("predicted_direction"),
            "prediction_confidence": pred.get("prediction_confidence"),
            "predicted_alpha": pred.get("predicted_alpha"),
        })

    return plan
