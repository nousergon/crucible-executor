"""
Market hours validation — prevents order placement outside regular trading hours.

NYSE regular session: 9:30 AM – 4:00 PM Eastern, weekdays only.
Pre-market and after-hours orders get poor fills; this gate blocks them.
"""

from __future__ import annotations

import logging
from datetime import datetime, time

import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
# NYSE closes at 4:00 PM ET, but IB Gateway free data is 15-min delayed,
# so the last real-time trades don't arrive until ~4:15 PM ET.
_MARKET_CLOSE = time(16, 15)


def is_market_hours(now: datetime | None = None) -> bool:
    """
    Return True if the current time is during NYSE regular trading hours.

    Args:
        now: Optional datetime override (for testing). If None, uses current time.

    Returns:
        True if weekday AND between 9:30 AM – 4:00 PM Eastern.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = _ET.localize(now)
    else:
        now = now.astimezone(_ET)

    # Weekday check (Monday=0, Friday=4)
    if now.weekday() > 4:
        logger.info("Market closed: weekend (day=%d)", now.weekday())
        return False

    current_time = now.time()
    if current_time < _MARKET_OPEN or current_time >= _MARKET_CLOSE:
        logger.info(
            "Market closed: current time %s ET is outside 9:30-16:00",
            current_time.strftime("%H:%M"),
        )
        return False

    return True
