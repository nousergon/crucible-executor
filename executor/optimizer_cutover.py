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
from typing import Any

import pandas as pd

from executor.order_book import OrderBook

logger = logging.getLogger(__name__)


_OK_DIAG_STATUSES = ("optimal", "optimal_inaccurate")


def is_log_usable(log: dict | None) -> bool:
    """Return True iff the shadow log is safe to drive the order book.

    Conservative gate: refuses anything but a clean ``optimal`` /
    ``optimal_inaccurate`` solve. Callers fall back to an empty order
    book on False (safer than wrong trades).
    """
    if not log:
        return False
    if log.get("shadow_status") != "ok":
        return False
    diag = log.get("diagnostics") or {}
    if diag.get("status") not in _OK_DIAG_STATUSES:
        return False
    if not log.get("would_be_trades"):
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

    for trade in trades:
        ticker = trade.get("ticker")
        if not ticker:
            continue
        action = trade.get("action")
        target_weight = float(trade.get("target_weight", 0.0))
        delta_dollars = float(trade.get("delta_dollars", 0.0))

        current_price = _get_current_price(ibkr, ticker)
        if current_price is None or current_price <= 0:
            logger.warning(
                "optimizer_cutover: skip %s — no current_price from ibkr", ticker,
            )
            continue

        if action == "BUY":
            record = _build_entry_record(
                ticker=ticker,
                delta_dollars=delta_dollars,
                target_weight=target_weight,
                current_price=current_price,
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
        "optimizer_cutover: applied %d entries + %d urgent_exits from "
        "%d would_be_trades",
        len(entries), len(exits), len(trades),
    )
    return entries, exits


def _get_current_price(ibkr: Any, ticker: str) -> float | None:
    try:
        return ibkr.get_current_price(ticker)
    except Exception as e:
        logger.warning("optimizer_cutover: get_current_price(%s) raised: %s", ticker, e)
        return None


def _build_entry_record(
    ticker: str,
    delta_dollars: float,
    target_weight: float,
    current_price: float,
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
        "stance": pred.get("stance"),
        "catalyst_date": pred.get("catalyst_date"),
        "sizing_source": "portfolio_optimizer",
        "sizing_factors": {
            "optimizer_target_weight": target_weight,
            "optimizer_delta_dollars": round(delta_dollars, 2),
        },
    }


def _build_urgent_exit_record(
    ticker: str,
    delta_dollars: float,
    target_weight: float,
    current_price: float,
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
        "sizing_source": "portfolio_optimizer",
    }
