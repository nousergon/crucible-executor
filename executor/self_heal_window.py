"""Where the post-close self-heal audit starts (alpha-engine-config-I14373).

The trailing self-heal in ``eod_reconcile.run`` audits a fixed window, so a day
that broke and aged out of it was never repaired, and its integrity breach
stood in the 63-session cumulative-residual gate until it aged out of that too.
The start now moves back to the OLDEST day inside the gate's own 63-session
window that still carries an integrity breach, but never so far that the window
spans more than ``SELF_HEAL_MAX_WINDOW_DAYS`` trading days.

Only the START date moves. No tolerance, bound, gate or refusal is read or
changed here; the audit loop still runs oldest to newest, so a repaired day
cascades into the next one through ``prior_corrected``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date

from krepis.trading_calendar import subtract_trading_days

logger = logging.getLogger(__name__)

TRAILING_DAYS = 5  # mirrors reconcile_audit.DEFAULT_TRAILING_DAYS (asserted by a test)
SELF_HEAL_BREACH_LOOKBACK_SESSIONS = 63  # pnl_integrity.RESIDUAL_CUMULATIVE_WINDOW_SESSIONS
SELF_HEAL_MAX_WINDOW_DAYS = 21
WINDOW_REASON_TRAILING = "trailing"
WINDOW_REASON_OLDEST_OPEN_BREACH = "oldest_open_breach"


@dataclass(frozen=True)
class SelfHealWindow:
    """The first date audited, and why.

    ``audit_start`` is ``None`` for the plain trailing window (``audit_window``
    is then called exactly as before) and an ISO date when an open breach
    pulled the start back. ``left_out`` lists breached dates older than the cap,
    which this run did not reach.
    """

    start: str
    reason: str
    audit_start: str | None
    left_out: tuple[str, ...] = ()


def _trailing_start(end: date, trading_days: int) -> str:
    """First of the ``trading_days`` NYSE trading days ending at ``end``."""
    return subtract_trading_days(end, trading_days - 1).isoformat()


def _has_open_breach(raw: object) -> bool:
    """True when an ``integrity_breach_json`` value is non-empty."""
    if raw is None:
        return False
    return str(raw).strip() not in ("", "[]", "{}", "null")


def resolve_self_heal_window(
    *,
    db_path: str,
    end: str,
    exclude_dates: Collection[str] = frozenset(),
    trailing_days: int = TRAILING_DAYS,
) -> SelfHealWindow:
    """The earlier of the trailing start and the oldest open-breach day, capped.

    A breach older than the cap is NOT silently dropped: the dates left out are
    logged at WARNING. A failure to read the table falls back to the trailing
    window, so the self-heal never loses its default coverage.
    """
    end_date = date.fromisoformat(end)
    trailing_start = _trailing_start(end_date, trailing_days)
    trailing = SelfHealWindow(start=trailing_start, reason=WINDOW_REASON_TRAILING, audit_start=None)
    try:
        lookback_start = _trailing_start(end_date, SELF_HEAL_BREACH_LOOKBACK_SESSIONS)
        cap_start = _trailing_start(end_date, SELF_HEAL_MAX_WINDOW_DAYS)
        # Read-only: resolving the window must never create or migrate a store.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT date, integrity_breach_json FROM eod_pnl "
                "WHERE date >= ? AND date < ? ORDER BY date ASC",
                (lookback_start, trailing_start),
            ).fetchall()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — fail soft to the trailing window, never to no window
        logger.warning(
            "[reconcile_audit] open-breach window lookup FAILED (%s: %s) — "
            "falling back to the trailing %d-day window.",
            e.__class__.__name__, e, trailing_days)
        return trailing
    breached: list[str] = [
        str(r[0]) for r in rows if str(r[0]) not in exclude_dates and _has_open_breach(r[1])
    ]
    in_cap = [d for d in breached if d >= cap_start]
    left_out = tuple(d for d in breached if d < cap_start)
    if left_out:
        logger.warning(
            "[reconcile_audit] integrity breach(es) older than the %d-trading-day self-heal "
            "cap (window floor %s) were NOT reached by this run and are left as stored: %s",
            SELF_HEAL_MAX_WINDOW_DAYS, cap_start, list(left_out))
    if not in_cap:
        return SelfHealWindow(start=trailing_start, reason=WINDOW_REASON_TRAILING,
                              audit_start=None, left_out=left_out)
    oldest = in_cap[0]
    logger.warning(
        "[reconcile_audit] self-heal window extended back to %s (oldest day still carrying an "
        "integrity breach; trailing start was %s): breached dates in reach: %s",
        oldest, trailing_start, in_cap)
    return SelfHealWindow(start=oldest, reason=WINDOW_REASON_OLDEST_OPEN_BREACH,
                          audit_start=oldest, left_out=left_out)
