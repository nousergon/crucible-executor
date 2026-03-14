"""
Position sizing algorithm per design doc section B.3.

Inputs: portfolio NAV, signal data, sector rating, current price, config.
Output: shares, dollar_size, position_pct.

Graduated drawdown multiplier (added 2026-03-14):
  When portfolio is in a drawdown tier, all position sizes are scaled
  down by the tier's multiplier. This is applied after all other
  adjustments but before the max position cap.
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)

_SECTOR_ADJ = {
    "overweight": 1.10,
    "market_weight": 1.00,
    "underweight": 0.75,
}


def compute_position_size(
    ticker: str,
    portfolio_nav: float,
    enter_signals: list[dict],
    signal: dict,
    sector_rating: str,
    current_price: float,
    config: dict,
    drawdown_multiplier: float = 1.0,
) -> dict:
    """
    Compute position size for a new ENTER order.

    Algorithm (design doc B.3 + graduated drawdown):
      1. base_weight = 1 / n_enter_signals  (equal weight across all entries today)
      2. sector_adj: overweight→1.10, market_weight→1.00, underweight→0.75
      3. conviction_adj: rising/stable→1.00, declining→0.50
      4. upside_adj: price_target_upside < min_price_target_upside → 0.50
      5. drawdown_adj: multiplier from graduated drawdown tiers (1.0/0.50/0.25)
      6. position_weight = min(base * sector * conviction * upside * dd, max_position_pct)
      7. dollar_size = portfolio_nav * position_weight
      8. shares = floor(dollar_size / current_price)

    Returns:
        {"shares": int, "dollar_size": float, "position_pct": float}
    """
    n = max(len(enter_signals), 1)
    base_weight = 1.0 / n

    sector_adj = _SECTOR_ADJ.get(sector_rating, 1.00)

    conviction = signal.get("conviction", "stable")
    conviction_adj = 0.50 if conviction == "declining" else 1.00

    upside = signal.get("price_target_upside")
    min_upside = config.get("min_price_target_upside", 0.05)
    upside_adj = 0.50 if (upside is not None and upside < min_upside) else 1.00

    max_pct = config.get("max_position_pct", 0.05)
    raw_weight = base_weight * sector_adj * conviction_adj * upside_adj * drawdown_multiplier
    position_weight = min(raw_weight, max_pct)

    dollar_size = portfolio_nav * position_weight
    shares = math.floor(dollar_size / current_price) if current_price > 0 else 0

    logger.debug(
        f"{ticker} sizing: n={n} base={base_weight:.3f} sector_adj={sector_adj} "
        f"conviction_adj={conviction_adj} upside_adj={upside_adj} "
        f"dd_mult={drawdown_multiplier} "
        f"→ {position_weight:.3f} NAV = ${dollar_size:.0f} = {shares} shares"
    )

    return {
        "shares": shares,
        "dollar_size": round(dollar_size, 2),
        "position_pct": round(position_weight, 4),
    }
