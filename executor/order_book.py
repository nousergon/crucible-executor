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
        "urgent_exits": [],
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
        """Load order book from disk (with file lock). Returns empty book if missing or stale."""
        import fcntl

        if path.exists():
            try:
                lock_path = path.with_suffix(".lock")
                with open(lock_path, "w") as lock_f:
                    fcntl.flock(lock_f, fcntl.LOCK_SH)
                    try:
                        data = json.loads(path.read_text())
                    finally:
                        fcntl.flock(lock_f, fcntl.LOCK_UN)
                # Discard stale book from a previous day
                if data.get("date") != date.today().isoformat():
                    logger.info("Order book is from %s — starting fresh", data.get("date"))
                    data = _default_book()
                return cls(data, path)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt order book — starting fresh: %s", e)
        return cls(_default_book(), path)

    def save(self) -> None:
        """Write order book to disk (atomic via tmp + rename, with file lock)."""
        import fcntl

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                tmp_path = self._path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(self._data, indent=2, default=str))
                tmp_path.rename(self._path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def data(self) -> dict:
        return self._data

    def all_tickers(self) -> list[str]:
        """All unique tickers across entries, urgent exits, and stops."""
        tickers = set()
        for entry in self._data.get("approved_entries", []):
            tickers.add(entry["ticker"])
        for urgent in self._data.get("urgent_exits", []):
            tickers.add(urgent["ticker"])
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

    def pending_urgent_exits(self) -> list[dict]:
        """Return urgent exits with status == 'pending'."""
        return [
            e for e in self._data.get("urgent_exits", [])
            if e.get("status") == "pending"
        ]

    def has_content(self) -> bool:
        """Return True if the book has any entries, exits, or stops."""
        return bool(
            self._data.get("approved_entries")
            or self._data.get("urgent_exits")
            or self._data.get("active_stops")
        )

    # ── Mutations ────────────────────────────────────────────────────────────

    def add_entry(self, entry: dict) -> None:
        """Add an approved entry to the book.

        Deduplicates by ticker: if a pending entry for the same ticker
        already exists, the new record is skipped.
        """
        ticker = entry.get("ticker")
        existing = self._data.get("approved_entries", [])
        for ex in existing:
            if ex.get("ticker") == ticker and ex.get("status") == "pending":
                logger.warning(
                    "Skipping duplicate entry for %s — already pending", ticker,
                )
                return
        entry.setdefault("status", "pending")
        self._data.setdefault("approved_entries", []).append(entry)

    def add_urgent_exit(self, record: dict) -> None:
        """Add an urgent exit/reduce/cover to the book (executed immediately by daemon).

        Deduplicates by ticker+signal: if a pending urgent with the same ticker
        and signal type already exists, the new record is skipped.
        """
        ticker = record.get("ticker")
        signal = record.get("signal")
        existing = self._data.get("urgent_exits", [])
        for ex in existing:
            if (ex.get("ticker") == ticker
                    and ex.get("signal") == signal
                    and ex.get("status") == "pending"):
                logger.warning(
                    "Skipping duplicate urgent %s for %s — already pending",
                    signal, ticker,
                )
                return
        record.setdefault("status", "pending")
        self._data.setdefault("urgent_exits", []).append(record)

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

    def mark_urgent_executed(self, ticker: str, action: str) -> None:
        """Mark an urgent exit as executed and move to executed_today."""
        exits = self._data.get("urgent_exits", [])
        for record in exits:
            if record["ticker"] == ticker and record.get("signal") == action and record.get("status") == "pending":
                record["status"] = "executed"
                record["executed_at"] = datetime.now().isoformat()
                self._data.setdefault("executed_today", []).append(record)
                break
        self._data["urgent_exits"] = [
            e for e in exits
            if not (e["ticker"] == ticker and e.get("signal") == action and e.get("status") == "executed")
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

    def reset_pending(self) -> None:
        """Clear all pending items, preserving executed_today and date.

        Called by main.py before rebuilding the order book. Makes main.py
        idempotent — running it twice on the same day produces the same
        order book rather than appending duplicates.
        """
        cleared = (
            len(self._data.get("approved_entries", []))
            + len(self._data.get("urgent_exits", []))
            + len(self._data.get("active_stops", []))
        )
        self._data["approved_entries"] = []
        self._data["urgent_exits"] = []
        self._data["active_stops"] = []
        if cleared:
            logger.info("Order book reset: cleared %d pending items (executed_today preserved)", cleared)

    def merge_executed(self, executed_tickers: set[str]) -> None:
        """Remove entries and urgent exits for tickers already executed today.

        Called by the daemon after reloading the order book from disk,
        in case main.py re-ran and wrote fresh 'pending' entries for
        tickers the daemon already traded.
        """
        if not executed_tickers:
            return
        before_entries = len(self._data.get("approved_entries", []))
        before_urgents = len(self._data.get("urgent_exits", []))
        self._data["approved_entries"] = [
            e for e in self._data.get("approved_entries", [])
            if e["ticker"] not in executed_tickers
        ]
        self._data["urgent_exits"] = [
            e for e in self._data.get("urgent_exits", [])
            if e["ticker"] not in executed_tickers
        ]
        removed_entries = before_entries - len(self._data["approved_entries"])
        removed_urgents = before_urgents - len(self._data["urgent_exits"])
        if removed_entries or removed_urgents:
            logger.info(
                "Merged executed state: removed %d entries, %d urgent exits for already-traded tickers",
                removed_entries, removed_urgents,
            )
