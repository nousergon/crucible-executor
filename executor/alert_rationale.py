"""Per-ticker rationale lines for the order-book / daemon-start alerts.

The "Order book written" and "Daemon started" Telegram alerts used to carry
counts only (``Entries: 3 | Urgent exits: 2``), so the operator could not
see WHY a name was being bought or urgently sold without opening the order
book. This module renders one short line per pending entry and per pending
urgent exit from the fields the deciders already stamped onto the order-book
record (reason, research score / rating / conviction, GBM prediction,
optimizer sizing).

Two constraints shape it:

* **Alert text only.** It reads order-book records and returns a string; it
  never mutates them and nothing downstream of trading reads its output.
* **Our own derived fields only.** No vendor prices or fundamentals and no
  free-text thesis — only scores, labels and reasons this system computed.

It must never raise: the daemon-start alert is sent outside a try block, so
a malformed record degrades to a bare ticker instead of aborting startup.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Telegram caps a message at 4096 chars; the counts header plus flow-doctor's
# envelope take a few hundred. Bound each section so a large rebalance can
# never push the alert into truncation.
MAX_LINES_PER_SECTION = 15


def _num(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _research_parts(rec: dict) -> list[str]:
    parts: list[str] = []
    score = _num(rec.get("research_score"))
    rating = rec.get("research_rating")
    conviction = rec.get("research_conviction")
    if score is not None:
        label = f"score {score:.0f}"
        if rating:
            label += f" {rating}"
        parts.append(label)
    elif rating:
        parts.append(str(rating))
    if conviction:
        parts.append(f"conviction {conviction}")
    direction = rec.get("predicted_direction")
    alpha = _num(rec.get("predicted_alpha"))
    conf = _num(rec.get("prediction_confidence"))
    if direction or alpha is not None:
        gbm = f"GBM {direction}" if direction else "GBM"
        if alpha is not None:
            gbm += f" α {alpha * 100:+.2f}%"
        if conf is not None:
            gbm += f" ({conf * 100:.0f}% conf)"
        parts.append(gbm)
    if rec.get("stance"):
        parts.append(f"stance {rec['stance']}")
    return parts


def _humanize(reason: Any) -> str:
    return str(reason).replace("_", " ").strip()


def entry_rationale(rec: dict) -> str:
    """One line explaining why ``rec`` (an approved entry) was proposed."""
    ticker = rec.get("ticker", "?")
    try:
        parts = _research_parts(rec)
        weight = _num(rec.get("position_pct"))
        source = rec.get("sizing_source")
        sizing = "optimizer" if source == "portfolio_optimizer" else None
        if weight is not None:
            sizing = f"{sizing + ' ' if sizing else ''}target {weight * 100:.1f}%"
        if sizing:
            parts.append(sizing)
        if rec.get("champion_arm"):
            parts.append(f"arm {rec['champion_arm']}")
        return f"  {ticker}: {', '.join(parts) if parts else 'no rationale recorded'}"
    except Exception:  # noqa: BLE001 — alert text must never abort trading
        logger.warning("entry rationale failed for %s", ticker, exc_info=True)
        return f"  {ticker}"


def urgent_exit_rationale(rec: dict) -> str:
    """One line explaining why ``rec`` (an urgent exit/reduce/cover) fired."""
    ticker = rec.get("ticker", "?")
    try:
        signal = rec.get("signal") or "EXIT"
        reason = rec.get("reason")
        head = _humanize(reason) if reason else "no reason recorded"
        detail = rec.get("detail")
        if detail:
            head += f" ({detail})"
        parts = [head] + _research_parts(rec)
        return f"  {ticker} {signal}: {', '.join(parts)}"
    except Exception:  # noqa: BLE001 — alert text must never abort trading
        logger.warning("urgent-exit rationale failed for %s", ticker, exc_info=True)
        return f"  {ticker}"


def _section(title: str, records: Iterable[dict], fmt) -> str:
    records = list(records or [])
    if not records:
        return ""
    lines = [fmt(r) for r in records[:MAX_LINES_PER_SECTION]]
    extra = len(records) - MAX_LINES_PER_SECTION
    if extra > 0:
        lines.append(f"  +{extra} more")
    return f"\n{title}:\n" + "\n".join(lines)


def order_book_rationale(order_book) -> str:
    """Rationale block for every pending urgent exit and entry in the book.

    Returns ``""`` when there is nothing pending, so callers can append it
    unconditionally. Urgent exits come first: they execute at the open.
    """
    try:
        return (
            _section("Urgent exits", order_book.pending_urgent_exits(), urgent_exit_rationale)
            + _section("Entries", order_book.pending_entries(), entry_rationale)
        )
    except Exception:  # noqa: BLE001 — alert text must never abort trading
        logger.warning("order-book rationale failed", exc_info=True)
        return ""
