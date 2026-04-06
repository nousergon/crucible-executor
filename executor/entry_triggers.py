"""
Intraday entry trigger engine — determines when to execute approved entries.

Five trigger types (OR logic — any one fires execution):
  1. Pullback: price drops >= X% from intraday high
  2. VWAP discount: price < previous day's VWAP by >= Y%
  3. Support bounce: price within Z% of N-day support level
  4. Graduated entry: after 2 PM ET, accept if price <= morning price + 1%
  5. Time expiry: unconditional market order at 3:55 PM ET

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

        expiry_str = strategy_config.get("intraday_expiry_time", "15:55")
        h, m = expiry_str.split(":")
        self._expiry_time = time(int(h), int(m))

        grad_str = strategy_config.get("intraday_graduated_start_time", "14:00")
        h, m = grad_str.split(":")
        self._graduated_start_time = time(int(h), int(m))

        self._graduated_max_premium = strategy_config.get("intraday_graduated_max_premium_pct", 0.01)
        self._disabled_triggers = set(strategy_config.get("disabled_triggers", []))

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
        if "pullback" not in self._disabled_triggers:
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
        if "vwap_discount" not in self._disabled_triggers:
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
        if "support_bounce" not in self._disabled_triggers:
            support_level = triggers.get("support_level")
            if support_level and support_level > 0:
                day_low = price_state.get("low", float("inf"))
                if day_low < support_level:
                    pass  # support broken — skip this trigger
                else:
                    support_threshold = triggers.get(
                        "support_pct",
                        self._config.get("intraday_support_pct", 0.01),
                    )
                    dist = (current_price - support_level) / support_level
                    if 0 <= dist <= support_threshold:
                        return True, f"near support ${support_level:.2f} (dist {dist:.1%})"

        # 4. Time-based entries (graduated → expiry)
        now_et = datetime.now(_ET).time()

        # 4a. True expiry — unconditional market order near close
        if now_et >= self._expiry_time:
            return True, "time_expiry"

        # 4b. Graduated window — accept entry if price is near or below morning price
        if "graduated_entry" not in self._disabled_triggers:
            if now_et >= self._graduated_start_time:
                morning_price = entry.get("current_price")
                if morning_price and morning_price > 0:
                    premium = (current_price - morning_price) / morning_price
                    if premium <= self._graduated_max_premium:
                        return True, (
                            f"graduated_entry ({premium:+.1%} vs morning "
                            f"${morning_price:.2f}, limit {self._graduated_max_premium:.1%})"
                        )

        return False, ""
