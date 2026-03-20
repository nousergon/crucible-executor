"""
Intraday entry trigger engine — determines when to execute approved entries.

Four trigger types (OR logic — any one fires execution):
  1. Pullback: price drops >= X% from intraday high
  2. VWAP discount: price < VWAP by >= Y%
  3. Support bounce: price within Z% of N-day support level
  4. Time expiry: execute at market if no trigger fires by cutoff time

The morning batch writes approved entries to the order book.
The daemon calls this engine on each price update.
"""

from __future__ import annotations

import logging
from datetime import datetime, time

import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")


class EntryTriggerEngine:
    """Evaluate intraday entry triggers against live price data."""

    def __init__(self, strategy_config: dict):
        self._config = strategy_config
        expiry_str = strategy_config.get("intraday_expiry_time", "15:30")
        h, m = expiry_str.split(":")
        self._expiry_time = time(int(h), int(m))

    def should_enter(self, entry: dict, price_state: dict) -> tuple[bool, str]:
        """
        Evaluate all entry triggers for an approved entry.

        Args:
            entry: from order book:
                {ticker, signal, shares, triggers, expiry, status}
                triggers: {pullback_pct, vwap_discount, support_level}
            price_state: from PriceMonitor:
                {last, high, low, close, volume, updated_at}

        Returns:
            (should_execute, trigger_reason)
        """
        current_price = price_state.get("last")
        if not current_price or current_price <= 0:
            return False, ""

        triggers = entry.get("triggers", {})

        # 1. Pullback entry
        day_high = price_state.get("high", 0)
        if day_high and day_high > 0:
            pullback_threshold = triggers.get(
                "pullback_pct",
                self._config.get("intraday_pullback_pct", 0.02),
            )
            pullback = (day_high - current_price) / day_high
            if pullback >= pullback_threshold:
                return True, f"pullback {pullback:.1%} from high ${day_high:.2f}"

        # 2. VWAP discount (if VWAP available in triggers)
        vwap = triggers.get("vwap")
        if vwap and vwap > 0:
            vwap_threshold = triggers.get(
                "vwap_discount",
                self._config.get("intraday_vwap_discount_pct", 0.005),
            )
            discount = (vwap - current_price) / vwap
            if discount >= vwap_threshold:
                return True, f"VWAP discount {discount:.1%} (VWAP=${vwap:.2f})"

        # 3. Support bounce
        support_level = triggers.get("support_level")
        if support_level and support_level > 0:
            support_threshold = triggers.get(
                "support_pct",
                self._config.get("intraday_support_pct", 0.01),
            )
            dist = (current_price - support_level) / support_level
            if 0 <= dist <= support_threshold:
                return True, f"near support ${support_level:.2f} (dist {dist:.1%})"

        # 4. Time expiry — execute at market if no trigger fired
        now_et = datetime.now(_ET).time()
        if now_et >= self._expiry_time:
            return True, "time_expiry"

        return False, ""
