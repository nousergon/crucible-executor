"""Reference bars on the basis the broker actually held (alpha-engine-config-I14285).

ArcticDB bars are back-adjusted for a split once it happens, so for a run date
BEFORE a journaled split the reference Close/Low/High sit on the post-split price
basis while the broker's mark, and the share count it was struck on, are still
pre-split (SNDK 1:8, journaled 2026-11-20: 2026-11-17's $799.56 mark on 262 shares
against an adjusted range of $93.67 to $102.68). Every gate that compares a mark to
the reference range, or rebuilds NAV from the reference close, then breaches on a
correct mark.

``reference_bars_on_trade_basis`` restates those bars by the split ratio, and only
when the series itself shows it was back-adjusted. It changes the reference data and
nothing else: no bound, tolerance or refusal path lives here.
"""

# pyright: basic, reportAttributeAccessIssue=false, reportIndexIssue=false
# pandas ships no type stubs in this repo's environment, so strict mode reports every
# DataFrame operation as unknown, and the set it reports differs by pandas release.
# That is why the existing pandas-heavy modules sit in `.basedpyright/baseline.json`; a
# new file cannot join the baseline, so this one is checked at basic level. Its own
# annotations are complete (ruff ANN runs on it in CI).
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import pandas as pd

from executor.v2_fills import ForwardSplit

logger = logging.getLogger(__name__)

#: The caller's frame type, passed through untouched so its own type inference is unchanged.
BarsT = TypeVar("BarsT")

#: The price columns a split restates (Open and Volume are never read by the gates).
RESTATED_COLUMNS = ("Close", "Low", "High")


def _series_is_back_adjusted(bars: Any, split: ForwardSplit) -> bool:
    """True when no ratio-sized gap separates the last bar before the split from the split day.

    ``|log(close[s-1] / close[s])| < log(ratio) / 2``. Missing either bar, or a
    non-positive close, is NOT evidence of adjustment, so the bars pass through.
    """
    ratio = split.ratio
    if not math.isfinite(ratio) or ratio <= 0 or ratio == 1.0:
        return False
    idx = pd.DatetimeIndex(bars.index).normalize()
    effective = pd.Timestamp(split.effective).normalize()
    before = idx < effective
    on_or_after = idx >= effective
    if not before.any() or not on_or_after.any():
        return False
    closes = pd.to_numeric(bars["Close"], errors="coerce")
    prev_close = float(closes[before].iloc[-1])
    next_close = float(closes[on_or_after].iloc[0])
    if not (math.isfinite(prev_close) and math.isfinite(next_close)) or prev_close <= 0 or next_close <= 0:
        return False
    return abs(math.log(prev_close / next_close)) < abs(math.log(ratio)) / 2


def reference_bars_on_trade_basis(
    ticker: str,
    run_date: str,
    bars: BarsT,
    splits: Sequence[ForwardSplit],
) -> tuple[BarsT, dict[str, object] | None]:
    """``(bars, reference_restated)`` with Close/Low/High restated by each later split's ratio.

    For every split journaled for ``ticker`` with an effective date AFTER
    ``run_date``, bars dated before that day are multiplied by its ratio, but only
    when the series is back-adjusted across that split (``_series_is_back_adjusted``).
    Bars of an unadjusted series, a ticker with no later split, and a split not after
    ``run_date`` come back as given (the same object, untouched). Never mutates
    ``bars``.

    ``reference_restated`` is None when nothing was restated, else
    ``{ratio, split_effective, journal_ref}`` for the audit row (with a ``splits`` list
    of each when more than one split applied, ``ratio`` then their product).
    """
    run_day = pd.Timestamp(run_date).normalize()
    applicable = sorted(
        (s for s in splits if s.ticker == ticker and pd.Timestamp(s.effective).normalize() > run_day),
        key=lambda s: s.effective,
    )
    applied: list[ForwardSplit] = [s for s in applicable if _series_is_back_adjusted(bars, s)]
    if not applied:
        return bars, None
    out = bars.copy()
    idx = pd.DatetimeIndex(out.index).normalize()
    for split in applied:
        mask = idx < pd.Timestamp(split.effective).normalize()
        for col in RESTATED_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype("float64")
                out.loc[mask, col] = out.loc[mask, col] * split.ratio
    detail = [
        {"ratio": s.ratio, "split_effective": s.effective.isoformat(), "journal_ref": s.journal_ref} for s in applied
    ]
    info: dict[str, object] = dict(detail[0])
    if len(detail) > 1:
        info["ratio"] = math.prod(s.ratio for s in applied)
        info["splits"] = detail
    logger.info(
        "[split_reference] %s on %s: reference Close/Low/High restated x%g for the later split(s) %s",
        ticker, run_date, info["ratio"], [d["journal_ref"] for d in detail],
    )
    return out, info
