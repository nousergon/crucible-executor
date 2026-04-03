"""
Market hours validation — prevents order placement outside regular trading hours.

NYSE regular session: 9:30 AM – 4:00 PM Eastern, weekdays only.
Includes NYSE holiday calendar through 2030 to prevent orders on closed days.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time

import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(
    int(os.environ.get("MARKET_CLOSE_HOUR", "16")),
    int(os.environ.get("MARKET_CLOSE_MINUTE", "15")),
)

# NYSE observed holidays through 2030.
# Source: https://www.nyse.com/markets/hours-calendars
NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
    # 2028
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29),
    date(2028, 6, 19), date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
    date(2028, 12, 25),
    # 2029
    date(2029, 1, 1), date(2029, 1, 15), date(2029, 2, 19), date(2029, 3, 30),
    date(2029, 5, 28), date(2029, 6, 19), date(2029, 7, 4), date(2029, 9, 3),
    date(2029, 11, 22), date(2029, 12, 25),
    # 2030
    date(2030, 1, 1), date(2030, 1, 21), date(2030, 2, 18), date(2030, 4, 19),
    date(2030, 5, 27), date(2030, 6, 19), date(2030, 7, 4), date(2030, 9, 2),
    date(2030, 11, 28), date(2030, 12, 25),
}


def is_trading_day(d: date | None = None) -> bool:
    """Return True if the given date is an NYSE trading day (not weekend or holiday)."""
    if d is None:
        d = date.today()
    if d.weekday() > 4:
        return False
    return d not in NYSE_HOLIDAYS


def is_market_hours(now: datetime | None = None) -> bool:
    """
    Return True if the current time is during NYSE regular trading hours.

    Checks: weekday AND not a holiday AND between 9:30 AM – 4:00 PM Eastern.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = _ET.localize(now)
    else:
        now = now.astimezone(_ET)

    if now.weekday() > 4:
        logger.info("Market closed: weekend (day=%d)", now.weekday())
        return False

    if now.date() in NYSE_HOLIDAYS:
        logger.info("Market closed: NYSE holiday (%s)", now.date())
        return False

    current_time = now.time()
    if current_time < _MARKET_OPEN or current_time >= _MARKET_CLOSE:
        logger.info(
            "Market closed: current time %s ET is outside 9:30-16:00",
            current_time.strftime("%H:%M"),
        )
        return False

    return True
