"""
Intraday exit rule engine — evaluates exit conditions on each price update.

Four independent rules:
  1. ATR trailing stop: price below high_water - (ATR × multiple)
  2. Profit-taking: price up > threshold % from entry → REDUCE 50%
  3. Intraday collapse: price drops > threshold % within the day → full EXIT
  4. Time-based tightening: after N days held, tighten trail multiplier

These are software-side rules (daemon-enforced). Broker-side trailing stops
from bracket_orders.py provide a safety net if the daemon is down.
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


class IntradayExitManager:
    """Evaluate intraday exit rules against live price state."""

    def __init__(self, strategy_config: dict):
        self._config = strategy_config

    def evaluate(self, stop: dict, price_state: dict) -> dict | None:
        """
        Check all exit rules for a position.

        Args:
            stop: active stop record from order book:
                {ticker, entry_price, current_stop, trail_atr, atr_multiple,
                 high_water, entry_date, shares}
            price_state: from PriceMonitor:
                {last, high, low, close, volume, updated_at}

        Returns:
            Exit signal dict if triggered, else None.
            {ticker, action, shares, reason, detail}
        """
        ticker = stop["ticker"]
        current_price = price_state.get("last")
        if not current_price or current_price <= 0:
            return None

        # 1. ATR trailing stop
        result = self._check_trailing_stop(stop, current_price)
        if result:
            return result

        # 2. Profit-taking
        result = self._check_profit_take(stop, current_price)
        if result:
            return result

        # 3. Intraday collapse
        result = self._check_collapse(stop, price_state)
        if result:
            return result

        # Update high-water mark (no exit triggered)
        return None

    def should_update_trail(self, stop: dict, current_price: float) -> tuple[float, float] | None:
        """
        Check if the trailing stop should be tightened upward.

        Returns (new_high_water, new_stop_price) if update needed, else None.
        """
        high_water = stop.get("high_water", 0)
        if current_price <= high_water:
            return None

        trail_atr = stop.get("trail_atr", 0)
        if not trail_atr or trail_atr <= 0:
            return None  # no ATR data — cannot compute meaningful trail distance

        atr_multiple = stop.get("atr_multiple", 2.0)

        # Time-based tightening: after N days, reduce multiplier
        entry_date = stop.get("entry_date")
        if entry_date:
            try:
                days_held = (date.today() - date.fromisoformat(entry_date)).days
            except (ValueError, TypeError):
                days_held = 0
            tighten_after = self._config.get("intraday_tighten_after_days", 3)
            if days_held >= tighten_after:
                atr_multiple = min(atr_multiple, self._config.get("intraday_tighten_atr_multiple", 1.5))

        trail_distance = trail_atr * atr_multiple
        new_stop = round(current_price - trail_distance, 2)
        current_stop = stop.get("current_stop", 0)

        # Only ratchet up, never down
        if new_stop > current_stop:
            return current_price, new_stop
        return None

    def check_catastrophic_gap(self, stop: dict, price_state: dict) -> dict | None:
        """Hard-risk per-name catastrophic gap stop (full EXIT).

        Fires when price falls at least ``catastrophic_gap_stop_pct`` below the
        record's ``gap_reference_price`` (most recent close at planning time,
        falling back to entry_price). This is the ONLY intraday exit allowed
        when the optimizer owns the book — a true risk control, distinct from
        the retired alpha rules (trailing-stop / profit-take / collapse). The
        optimizer otherwise reconciles the book; this catches the acute
        single-name crater the optimizer can't react to until next morning.
        """
        if not self._config.get("catastrophic_gap_stop_enabled", True):
            return None
        current_price = price_state.get("last")
        if not current_price or current_price <= 0:
            return None
        ref = stop.get("gap_reference_price") or stop.get("entry_price")
        if not ref or ref <= 0:
            return None
        threshold = self._config.get("catastrophic_gap_stop_pct", 0.15)
        drop = (ref - current_price) / ref
        if drop >= threshold:
            return {
                "ticker": stop["ticker"],
                "action": "EXIT",
                "shares": stop.get("shares", 0),
                "reason": "catastrophic_gap_stop",
                "detail": (
                    f"price ${current_price:.2f} <= {(1 - threshold):.0%} of "
                    f"reference ${ref:.2f} (drop {drop:.1%} >= {threshold:.1%})"
                ),
            }
        return None

    # ── Private rule checks ──────────────────────────────────────────────────

    def _check_trailing_stop(self, stop: dict, current_price: float) -> dict | None:
        """Exit if price falls below trailing stop level."""
        current_stop = stop.get("current_stop")
        if not current_stop or current_stop <= 0:
            return None

        if current_price <= current_stop:
            trail_atr = stop.get("trail_atr", 0)
            atr_multiple = stop.get("atr_multiple", 0)
            return {
                "ticker": stop["ticker"],
                "action": "EXIT",
                "shares": stop.get("shares", 0),
                "reason": "intraday_trailing_stop",
                "detail": (
                    f"price ${current_price:.2f} <= stop ${current_stop:.2f} "
                    f"(ATR ${trail_atr:.2f} × {atr_multiple})"
                ),
            }
        return None

    def _check_profit_take(self, stop: dict, current_price: float) -> dict | None:
        """Reduce 50% when profit exceeds threshold. Fires at most once per position."""
        if stop.get("profit_take_executed"):
            return None
        threshold = self._config.get("intraday_profit_take_pct", 0.08)
        entry_price = stop.get("entry_price")
        if not entry_price or entry_price <= 0:
            return None

        gain_pct = (current_price - entry_price) / entry_price
        if gain_pct >= threshold:
            shares = stop.get("shares", 0)
            reduce_shares = max(1, shares // 2)
            return {
                "ticker": stop["ticker"],
                "action": "REDUCE",
                "shares": reduce_shares,
                "reason": "intraday_profit_take",
                "detail": f"gain {gain_pct:.1%} >= {threshold:.1%} threshold",
            }
        return None

    def _check_collapse(self, stop: dict, price_state: dict) -> dict | None:
        """Full exit on severe intraday price drop."""
        threshold = self._config.get("intraday_collapse_pct", 0.05)
        current_price = price_state.get("last", 0)
        day_high = price_state.get("high", 0)

        if not day_high or day_high <= 0 or not current_price:
            return None

        intraday_drop = (day_high - current_price) / day_high
        if intraday_drop >= threshold:
            return {
                "ticker": stop["ticker"],
                "action": "EXIT",
                "shares": stop.get("shares", 0),
                "reason": "intraday_collapse",
                "detail": f"intraday drop {intraday_drop:.1%} >= {threshold:.1%} (high=${day_high:.2f})",
            }
        return None
