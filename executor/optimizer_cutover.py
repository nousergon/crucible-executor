"""
Portfolio-optimizer cutover — PR 5 of the portfolio-optimizer-260511 arc.

When ``use_portfolio_optimizer: true`` in config/risk.yaml, the legacy
1/n ``_plan_entries`` path in main.py is replaced by the optimizer's
``would_be_trades`` list: target weights from the MVO kernel drive the
order book directly. ``_plan_exits_and_reduces`` is preserved unchanged
so research EXIT/REDUCE signals + strategy_exits (ATR / time-decay) +
drawdown forced exits continue to fire as protective overrides.

The kernel + shadow wrapper are in:
    executor/portfolio_optimizer.py   (PR 1 — pure kernel)
    executor/optimizer_shadow.py      (PR 2 — read-only logger)

Plan doc: ~/Development/alpha-engine-docs/private/portfolio-optimizer-260511.md
Pre-flip checklist (alpha-engine-config flag flip is a separate PR):
  - PR 4 ad-hoc backtester cutover-gate verdict: pass (ROADMAP L156-157)
  - ≥ 2-3 trading days of plausible shadow logs (today = N=1)
  - Operator go-ahead
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import pandas as pd

from executor.order_book import OrderBook

logger = logging.getLogger(__name__)


_OK_DIAG_STATUSES = ("optimal", "optimal_inaccurate")

# Price-resolution policy for optimizer targets (L4500, 2026-06-04).
# The paper IBKR feed frequently returns an empty first snapshot for a
# ticker; a single miss must NOT silently drop the optimizer's target
# (the 2026-06-04 incident: AMD — the optimizer's LARGEST pick at 10% /
# $101.5k — was dropped on a transient no-price, leaving the order book
# 9-of-10 and surfacing as "no action — unknown (investigate)" on the
# console). We retry the live snapshot with a short backoff, then fall
# back to the last close from price_histories (the SAME panel the
# optimizer sized its weights against, so dimensionally identical to the
# target it produced). Only if every source fails do we drop — and then
# LOUDLY (WARN + CW metric + alarm), never silently. Per
# [[feedback_no_silent_fails]] + [[feedback_sota_institutional_default_no_shortcuts]].
_PRICE_MAX_ATTEMPTS = 3
_PRICE_RETRY_SLEEP_S = 1.0


def is_log_usable(log: dict | None) -> bool:
    """Return True iff the shadow log is safe to drive the order book.

    Conservative gate: refuses anything but a clean ``optimal`` /
    ``optimal_inaccurate`` solve. Callers fall back to an empty order
    book on False (safer than wrong trades).

    An ``optimal``/``ok`` solve with an EMPTY ``would_be_trades`` is
    *usable* — it is the optimizer's legitimate verdict that the current
    portfolio already matches the target within the turnover threshold
    ("hold" day). It is NOT a failure. Conflating it with a genuine
    failure (None / non-ok / non-optimal diag) was a real bug: the
    2026-05-19 weekday rerun solved ``optimal`` with turnover 0.17%
    (would_be_trades=[]) yet the planner logged a false
    ``optimizer log is not usable … Operator must investigate`` ERROR
    and framed a correct hold as a safety fallback. Emptiness is now the
    caller's concern (apply → a safe no-op; log INFO, not ERROR), not a
    usability signal.
    """
    if not log:
        return False
    if log.get("shadow_status") != "ok":
        return False
    diag = log.get("diagnostics") or {}
    if diag.get("status") not in _OK_DIAG_STATUSES:
        return False
    return True


def apply_optimizer_targets_to_orderbook(
    log: dict,
    ob: OrderBook,
    ibkr: Any,
    current_positions: dict[str, dict],
    price_histories: dict[str, pd.DataFrame] | None,
    atr_map: dict[str, float],
    strategy_config: dict,
    vwap_map: dict[str, float | None],
    signals_raw: dict | None,
    predictions_by_ticker: dict[str, dict],
    market_regime: str,
    run_date: str,
    predictions_date: str | None,
) -> tuple[list[dict], list[dict]]:
    """Translate optimizer ``would_be_trades`` into order-book records.

    Returns ``(entries_added, exits_added)`` for the caller's summary /
    health reporting. Dedup is handled by ``OrderBook.add_entry`` /
    ``add_urgent_exit`` (by ticker / by ticker+signal respectively) so
    callers may invoke this after the legacy exit path has already
    populated urgent_exits — overlapping SELLs are skipped.
    """
    trades = log.get("would_be_trades") or []
    signals_date = (
        signals_raw.get("date", run_date) if signals_raw else run_date
    )
    signals_by_ticker = (signals_raw or {}).get("signals", {}) or {}

    entries: list[dict] = []
    exits: list[dict] = []
    dropped: list[dict] = []

    for trade in trades:
        ticker = trade.get("ticker")
        if not ticker:
            continue
        action = trade.get("action")
        target_weight = float(trade.get("target_weight", 0.0))
        delta_dollars = float(trade.get("delta_dollars", 0.0))

        current_price, price_source = _resolve_price(ibkr, ticker, price_histories)
        if current_price is None:
            # Every price source failed (no live snapshot after retries AND
            # no usable price history). The optimizer's target cannot be
            # sized, so it is dropped for this run — and this is an ERROR,
            # not a benign skip: the optimizer's allocation (esp. its
            # largest target) is LOST. Logged at ERROR + CW alarm + the
            # OBR surfaces the ticker as no_action_optimizer_dropped (an
            # error state). Per [[feedback_no_silent_fails]] (L4500/L4501).
            logger.error(
                "optimizer_cutover: DROPPING %s %s target_weight=%.4f $%.2f — "
                "no price after %d ibkr attempts and no usable price_histories "
                "close. Optimizer allocation LOST for this run; emitting "
                "AlphaEngine/Executor/optimizer_target_dropped alarm.",
                ticker, action, target_weight, delta_dollars, _PRICE_MAX_ATTEMPTS,
            )
            _emit_target_dropped_metric(ticker, action, reason="no_price_all_sources")
            dropped.append({
                "ticker": ticker,
                "action": action,
                "reason": "no_price_all_sources",
                "target_weight": target_weight,
                "delta_dollars": delta_dollars,
            })
            continue
        if price_source != "ibkr":
            # Sized off the fallback close rather than a live snapshot. The
            # target survives (the whole point of L4500) but the operator
            # should know the live feed missed this name today.
            logger.warning(
                "optimizer_cutover: %s priced via FALLBACK %s=$%.2f — live "
                "ibkr snapshot unavailable after %d attempts; sizing on last "
                "close. Target preserved.",
                ticker, price_source, current_price, _PRICE_MAX_ATTEMPTS,
            )
            _emit_target_dropped_metric(ticker, action, reason="priced_via_fallback")

        if action == "BUY":
            record = _build_entry_record(
                ticker=ticker,
                delta_dollars=delta_dollars,
                target_weight=target_weight,
                current_price=current_price,
                price_source=price_source,
                atr_map=atr_map,
                price_histories=price_histories,
                strategy_config=strategy_config,
                vwap_map=vwap_map,
                signals_by_ticker=signals_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                market_regime=market_regime,
                signals_date=signals_date,
                predictions_date=predictions_date,
            )
            if record is None:
                continue
            ob.add_entry(record)
            entries.append(record)
        elif action == "SELL":
            record = _build_urgent_exit_record(
                ticker=ticker,
                delta_dollars=delta_dollars,
                target_weight=target_weight,
                current_price=current_price,
                price_source=price_source,
                current_positions=current_positions,
                signals_by_ticker=signals_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                market_regime=market_regime,
                signals_date=signals_date,
                predictions_date=predictions_date,
            )
            if record is None:
                continue
            ob.add_urgent_exit(record)
            exits.append(record)
        else:
            logger.warning(
                "optimizer_cutover: unknown action %r for %s", action, ticker,
            )

    logger.info(
        "optimizer_cutover: applied %d entries + %d urgent_exits "
        "(%d dropped — no price) from %d would_be_trades",
        len(entries), len(exits), len(dropped), len(trades),
    )
    return entries, exits


def _get_current_price(ibkr: Any, ticker: str) -> float | None:
    try:
        return ibkr.get_current_price(ticker)
    except Exception as e:
        logger.warning("optimizer_cutover: get_current_price(%s) raised: %s", ticker, e)
        return None


def _resolve_price(
    ibkr: Any,
    ticker: str,
    price_histories: dict[str, pd.DataFrame] | None,
) -> tuple[float | None, str | None]:
    """Resolve a sizing price for an optimizer target — never silently drop.

    Order of sources (L4500):
      1. Live IBKR snapshot, retried up to ``_PRICE_MAX_ATTEMPTS`` with a
         ``_PRICE_RETRY_SLEEP_S`` backoff between tries. ``get_current_price``
         *returns None* (it does not raise) on a transient empty snapshot,
         so the retry re-requests market data — the common paper-feed miss
         clears on a later attempt.
      2. Last close from ``price_histories[ticker]`` — the same panel the
         optimizer sized its weights against, so a dimensionally-consistent
         sizing price when the live feed is down.

    Returns ``(price, source)`` where source ∈ {"ibkr", "price_history_close"},
    or ``(None, None)`` when every source fails (caller drops the target
    loudly).
    """
    for attempt in range(1, _PRICE_MAX_ATTEMPTS + 1):
        price = _get_current_price(ibkr, ticker)
        if price is not None and price > 0:
            return float(price), "ibkr"
        if attempt < _PRICE_MAX_ATTEMPTS:
            time.sleep(_PRICE_RETRY_SLEEP_S)

    last_close = _last_close(price_histories, ticker)
    if last_close is not None:
        return last_close, "price_history_close"

    return None, None


def _last_close(
    price_histories: dict[str, pd.DataFrame] | None,
    ticker: str,
) -> float | None:
    """Most-recent positive close from ``price_histories`` (lowercase
    ``close`` column, matching the optimizer_shadow panel schema), or None."""
    if not price_histories:
        return None
    df = price_histories.get(ticker)
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    try:
        val = float(df["close"].iloc[-1])
    except (IndexError, ValueError, TypeError):
        return None
    return val if math.isfinite(val) and val > 0 else None


def _emit_target_dropped_metric(ticker: str, action: str, *, reason: str) -> None:
    """Emit ``AlphaEngine/Executor/optimizer_target_dropped`` (Count) with a
    ``reason`` dimension so a dropped / fallback-priced optimizer target is
    visible on the console + alarmable.

    Best-effort: CloudWatch errors WARN but never fail the planner — the
    order-book decision is the load-bearing path; this metric is the
    recording surface that keeps the drop from being silent (mirrors
    ``signal_reader._emit_admission_refused_metric``)."""
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Executor",
            MetricData=[{
                "MetricName": "optimizer_target_dropped",
                "Value": 1.0,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "reason", "Value": reason},
                ],
            }],
        )
    except Exception as exc:
        logger.warning(
            "CloudWatch optimizer_target_dropped metric failed (%s %s, %s): %s. "
            "Not blocking the planner.",
            ticker, action, reason, exc,
        )


def _build_entry_record(
    ticker: str,
    delta_dollars: float,
    target_weight: float,
    current_price: float,
    price_source: str | None,
    atr_map: dict[str, float],
    price_histories: dict[str, pd.DataFrame] | None,
    strategy_config: dict,
    vwap_map: dict[str, float | None],
    signals_by_ticker: dict[str, dict],
    predictions_by_ticker: dict[str, dict],
    market_regime: str,
    signals_date: str,
    predictions_date: str | None,
) -> dict | None:
    if delta_dollars <= 0:
        return None
    shares = int(math.floor(delta_dollars / current_price))
    if shares <= 0:
        logger.info(
            "optimizer_cutover: skip %s BUY — delta_dollars $%.2f < price $%.2f",
            ticker, delta_dollars, current_price,
        )
        return None

    ticker_atr_pct = float(atr_map.get(ticker, 0.0) or 0.0)
    atr_dollar = ticker_atr_pct * current_price
    pullback_atr_mult = strategy_config.get("intraday_pullback_atr_multiple", 1.0)
    scaled_pullback_pct = ticker_atr_pct * pullback_atr_mult

    sig = signals_by_ticker.get(ticker, {}) or {}
    pred = predictions_by_ticker.get(ticker, {}) or {}

    return {
        "ticker": ticker,
        "signal": "ENTER",
        "signal_date": signals_date,
        "prediction_date": predictions_date,
        "shares": shares,
        "current_price": current_price,
        "dollar_size": round(delta_dollars, 2),
        "position_pct": round(target_weight, 6),
        "atr_value": atr_dollar,
        "atr_pct": ticker_atr_pct,
        "triggers": {
            "pullback_pct": scaled_pullback_pct,
            "pullback_atr_multiple": pullback_atr_mult,
            "atr_pct": ticker_atr_pct,
            "vwap_discount": strategy_config.get("intraday_vwap_discount_pct", 0.005),
            "vwap": vwap_map.get(ticker) if vwap_map else None,
            "support_level": None,
        },
        "research_score": sig.get("score"),
        "research_conviction": sig.get("conviction"),
        "research_rating": sig.get("rating"),
        "sector": sig.get("sector"),
        "sector_rating": sig.get("sector_rating"),
        "market_regime": market_regime,
        "price_target_upside": sig.get("price_target_upside"),
        "thesis_summary": sig.get("thesis_summary"),
        "predicted_direction": pred.get("predicted_direction"),
        "prediction_confidence": pred.get("prediction_confidence"),
        "predicted_alpha": pred.get("predicted_alpha"),
        # B.4c (optimizer-sota-upgrades-260526.md §B.4c): surface the
        # BayesianRidge posterior std at the order-record layer so
        # downstream dashboards / emails / EOD reconciliation can show
        # operators the predictor's confidence on each sized name.
        # None when the predictor still emits via legacy Ridge (1-week
        # soak after B.1 #199); downstream consumers handle None per the
        # additive-field S3 contract.
        "predicted_alpha_std": pred.get("predicted_alpha_std"),
        "stance": pred.get("stance"),
        "catalyst_date": pred.get("catalyst_date"),
        "sizing_source": "portfolio_optimizer",
        # config#1436: which price the optimizer sized this name on — a live
        # IBKR snapshot ("ibkr") or the last-close fallback from
        # price_histories ("price_history_close") when the live feed missed.
        # Invisible-before, so an operator could not tell a position was sized
        # off a day-old price. Surfaced into the OBR decision_chain.
        "pricing_source": price_source,
        "sizing_factors": {
            "optimizer_target_weight": target_weight,
            "optimizer_delta_dollars": round(delta_dollars, 2),
            "alpha_uncertainty": pred.get("predicted_alpha_std"),
            "pricing_source": price_source,
        },
    }


def _build_urgent_exit_record(
    ticker: str,
    delta_dollars: float,
    target_weight: float,
    current_price: float,
    price_source: str | None,
    current_positions: dict[str, dict],
    signals_by_ticker: dict[str, dict],
    predictions_by_ticker: dict[str, dict],
    market_regime: str,
    signals_date: str,
    predictions_date: str | None,
) -> dict | None:
    pos = current_positions.get(ticker)
    if not pos:
        logger.info(
            "optimizer_cutover: skip %s SELL — no current position",
            ticker,
        )
        return None
    shares_held = int(pos.get("shares", 0))
    if shares_held <= 0:
        return None

    if target_weight <= 1e-9:
        shares_to_sell = shares_held
        signal_kind = "EXIT"
        reason = "optimizer_target_zero"
    else:
        # Scale-down: sell |delta_dollars| / price, clipped to held.
        shares_to_sell = int(math.floor(abs(delta_dollars) / current_price))
        shares_to_sell = max(0, min(shares_to_sell, shares_held))
        if shares_to_sell == 0:
            return None
        signal_kind = "REDUCE" if shares_to_sell < shares_held else "EXIT"
        reason = (
            "optimizer_scale_down" if signal_kind == "REDUCE"
            else "optimizer_target_zero"
        )

    sig = signals_by_ticker.get(ticker, {}) or {}
    pred = predictions_by_ticker.get(ticker, {}) or {}

    return {
        "ticker": ticker,
        "signal": signal_kind,
        "signal_date": signals_date,
        "prediction_date": predictions_date,
        "shares": shares_to_sell,
        "reason": reason,
        "detail": f"optimizer target_weight={target_weight:.4f}",
        "research_score": sig.get("score"),
        "research_conviction": sig.get("conviction"),
        "research_rating": sig.get("rating"),
        "sector_rating": pos.get("sector", ""),
        "market_regime": market_regime,
        "predicted_direction": pred.get("predicted_direction"),
        "prediction_confidence": pred.get("prediction_confidence"),
        "predicted_alpha": pred.get("predicted_alpha"),
        # B.4c — same surfacing rationale as _build_entry_record above.
        "predicted_alpha_std": pred.get("predicted_alpha_std"),
        "sizing_source": "portfolio_optimizer",
        # config#1436 — live snapshot vs last-close fallback (see entry builder).
        "pricing_source": price_source,
    }
