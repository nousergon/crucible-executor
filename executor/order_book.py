"""
JSON-based intraday order book.

The morning batch writes approved entries and active stop state here.
The intraday daemon reads and updates it throughout the trading day.
Persisted to disk so the daemon can restart mid-day without losing state.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ORDER_BOOK_DIR = Path(__file__).resolve().parent.parent / "data"
_ORDER_BOOK_PATH = _ORDER_BOOK_DIR / "order_book.json"


def _default_book(run_date: str | None = None) -> dict:
    return {
        "date": run_date or date.today().isoformat(),
        "approved_entries": [],
        "active_stops": [],
        "executed_today": [],
    }


class OrderBook:
    """Manages the intraday order book (JSON on disk)."""

    def __init__(self, data: dict, path: Path = _ORDER_BOOK_PATH):
        self._data = data
        self._path = path

    @classmethod
    def load(cls, path: Path = _ORDER_BOOK_PATH) -> "OrderBook":
        """Load order book from disk. Returns empty book if missing or stale."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Discard stale book from a previous day
                if data.get("date") != date.today().isoformat():
                    logger.info("Order book is from %s — starting fresh", data.get("date"))
                    data = _default_book()
                return cls(data, path)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt order book — starting fresh: %s", e)
        return cls(_default_book(), path)

    def save(self) -> None:
        """Write order book to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, default=str))

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def data(self) -> dict:
        return self._data

    def all_tickers(self) -> list[str]:
        """All unique tickers across entries and stops."""
        tickers = set()
        for entry in self._data.get("approved_entries", []):
            tickers.add(entry["ticker"])
        for stop in self._data.get("active_stops", []):
            tickers.add(stop["ticker"])
        return sorted(tickers)

    def pending_entries(self) -> list[dict]:
        """Return entries with status == 'pending'."""
        return [
            e for e in self._data.get("approved_entries", [])
            if e.get("status") == "pending"
        ]

    def active_stops(self) -> list[dict]:
        """Return all active stop records."""
        return self._data.get("active_stops", [])

    # ── Mutations ────────────────────────────────────────────────────────────

    def add_entry(self, entry: dict) -> None:
        """Add an approved entry to the book."""
        entry.setdefault("status", "pending")
        self._data.setdefault("approved_entries", []).append(entry)

    def add_stop(self, stop: dict) -> None:
        """Add an active stop record."""
        self._data.setdefault("active_stops", []).append(stop)

    def mark_entry_executed(self, ticker: str, trigger_reason: str) -> None:
        """Mark an entry as executed and move to executed_today."""
        entries = self._data.get("approved_entries", [])
        for entry in entries:
            if entry["ticker"] == ticker and entry.get("status") == "pending":
                entry["status"] = "executed"
                entry["trigger_reason"] = trigger_reason
                entry["executed_at"] = datetime.now().isoformat()
                self._data.setdefault("executed_today", []).append(entry)
                break
        self._data["approved_entries"] = [
            e for e in entries if not (e["ticker"] == ticker and e.get("status") == "executed")
        ]

    def remove_stop(self, ticker: str) -> None:
        """Remove a stop record (after exit execution)."""
        self._data["active_stops"] = [
            s for s in self._data.get("active_stops", [])
            if s["ticker"] != ticker
        ]

    def update_stop_high_water(self, ticker: str, new_high: float, new_stop: float) -> None:
        """Update the high-water mark and trailing stop for a position."""
        for stop in self._data.get("active_stops", []):
            if stop["ticker"] == ticker:
                stop["high_water"] = new_high
                stop["current_stop"] = new_stop
                break

    def set_date(self, run_date: str) -> None:
        """Set the book date (used by morning batch)."""
        self._data["date"] = run_date
