"""
JSON-based intraday order book.

The morning batch writes approved entries and active stop state here.
The intraday daemon reads and updates it throughout the trading day.
Persisted to disk so the daemon can restart mid-day without losing state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ORDER_BOOK_DIR = Path(__file__).resolve().parent.parent / "data"
_ORDER_BOOK_PATH = _ORDER_BOOK_DIR / "order_book.json"


def _current_session_date() -> str:
    """The NYSE session in progress / next upcoming, as an ISO string.

    Session axis (config#1610; supersedes the config#1016 rationale): the
    order book's ``date`` field and its load-time freshness check key on
    ``session_date`` — the session the book is FOR — so the morning batch
    (main.py) and the daemon (daemon.py), which both derive run_date from
    the same helper, agree on which book is "today's". Pre-open and
    intraday this is the physical session being traded; a book written on
    a weekend keys to the next session (a Saturday operator book is for
    Monday). The prior ``now_dual().trading_day`` axis resolved to the
    last *closed* session (D-1 intraday), which mislabeled the book and
    every downstream trade artifact one session behind.
    """
    from nousergon_lib.dates import session_date

    return session_date().isoformat()


def _default_book(run_date: str | None = None) -> dict:
    return {
        "date": run_date or _current_session_date(),
        "approved_entries": [],
        "urgent_exits": [],
        "active_stops": [],
        "executed_today": [],
    }


def _entry_id_for(entry: dict, book_date: str) -> str:
    """Deterministic per-decision identity for an order-book entry (config#2436).

    Composite of ticker + session date + sizing source — NOT a random id —
    so regenerating the SAME decision on the SAME day (e.g. a main.py rerun
    after a full order-book-loss) reproduces the identical entry_id, and a
    durable trades.db lookup keyed on it recognizes "this exact decision
    already executed." A genuinely distinct decision for the same ticker on
    the same day (legacy decider vs. portfolio optimizer vs. intraday
    redeploy each stamp a different ``sizing_source``) gets its own id, so
    it is never silently dropped as a false duplicate the way bare-ticker
    keying would drop it.
    """
    sizing_source = entry.get("sizing_source") or "legacy_decider"
    return f"{entry.get('ticker')}:{book_date}:{sizing_source}"


def build_stop_record(
    *,
    ticker: str,
    entry_price: float,
    current_stop: float,
    trail_atr: float,
    atr_multiple: float,
    high_water: float,
    entry_date: str,
    shares: int,
    use_optimizer: bool,
    gap_reference_price: float | None = None,
    **extra,
) -> dict:
    """Construct an active-stop record with book-authority semantics stamped.

    Single chokepoint for stop creation across BOTH producers — the morning
    planner (``main.py::_write_stops_and_finalize``) and the intraday daemon
    (``daemon.py::_execute_entry``). ``use_optimizer`` is a REQUIRED keyword:
    forgetting it raises ``TypeError`` at construction (fail-loud) rather than
    silently defaulting to the wrong behavior.

    When the portfolio optimizer owns the book (``use_optimizer=True``) every
    stop is ``catastrophic_gap_only`` — the daemon runs ONLY the hard-risk
    per-name catastrophic gap stop and the alpha rules (trailing-stop /
    profit-take / 5% intraday-collapse) are retired. ``gap_reference_price``
    anchors that gap check; for a same-day daemon entry there is no overnight
    gap to catch, so it falls back to ``entry_price`` (a 15% crater from where
    we actually filled). When the optimizer is off, the record is ``alpha`` and
    the daemon runs the full legacy ``IntradayExitManager``.

    Centralizing this prevents the producer-divergence bug where daemon-entered
    positions silently omitted ``stop_kind``, defaulted to the alpha rules, and
    were churned same-day by the 5%-from-intraday-high collapse rule
    (WDAY 2026-06-05: bought $146.48, force-sold $143.92 on a 5.0% drop from a
    pre-entry high of $151.50 — a peak the position never held through).
    """
    record = {
        "ticker": ticker,
        "entry_price": entry_price,
        "current_stop": current_stop,
        "trail_atr": trail_atr,
        "atr_multiple": atr_multiple,
        "high_water": high_water,
        "entry_date": entry_date,
        "shares": shares,
    }
    if use_optimizer:
        record["stop_kind"] = "catastrophic_gap_only"
        record["gap_reference_price"] = gap_reference_price or entry_price
    else:
        record["stop_kind"] = "alpha"
    record.update(extra)
    return record


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
                # Discard stale book from a previous trading session. Compare on
                # the session axis (config#1610) so the daemon doesn't treat
                # the morning batch's book as stale on a pre-open weekday read.
                if data.get("date") != _current_session_date():
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

    def backup_to_s3(self, bucket: str, run_date: str) -> None:
        """Backup full order book to S3 for audit trail and debugging."""
        try:
            import boto3
            s3 = boto3.client("s3")
            key = f"trades/order_book/{run_date}.json"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(self._data, indent=2, default=str),
                ContentType="application/json",
            )
            logger.info("Order book backed up to s3://%s/%s", bucket, key)
        except Exception as e:
            logger.warning("Order book S3 backup failed (non-fatal): %s", e)

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

        Deduplicates by ``entry_id`` (ticker + session date + sizing source,
        config#2436) rather than bare ticker: a literal re-submission of the
        SAME decision is skipped, but a genuinely distinct decision for an
        already-pending ticker (e.g. the intraday redeploy solver proposing
        a second buy alongside a still-pending morning entry) is kept
        instead of being silently dropped. ``entry_id`` is assigned here if
        the caller hasn't already stamped one.
        """
        entry.setdefault("entry_id", _entry_id_for(entry, self._data.get("date", "")))
        entry_id = entry["entry_id"]
        existing = self._data.get("approved_entries", [])
        for ex in existing:
            if ex.get("entry_id") == entry_id and ex.get("status") == "pending":
                logger.warning(
                    "Skipping duplicate entry for %s (entry_id=%s) — already pending",
                    entry.get("ticker"), entry_id,
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

    def mark_entry_executing(self, ticker: str, trigger_reason: str) -> None:
        """Stamp a write-ahead ``executing`` status on a pending entry.

        Crash-safety WAL (config#2328): the daemon calls this and ``save()``s
        BEFORE placing the IB BUY order, so an entry whose order was sent to
        the broker is durably marked in-doubt on disk. On a crash-restart the
        entry is no longer ``pending`` (``pending_entries`` excludes it), so
        the naive entry loop cannot re-place the BUY; startup reconciliation
        (``executing_entries``) resolves it against broker truth instead.

        Keeps the record in ``approved_entries`` (unlike ``mark_entry_executed``
        which moves it to ``executed_today``) — the order is not confirmed yet.
        """
        for entry in self._data.get("approved_entries", []):
            if entry["ticker"] == ticker and entry.get("status") == "pending":
                entry["status"] = "executing"
                entry["trigger_reason"] = trigger_reason
                entry["executing_at"] = datetime.now().isoformat()
                break

    def executing_entries(self) -> list[dict]:
        """Return entries left in the write-ahead ``executing`` state.

        Non-empty only after a crash between order placement and finalization
        (config#2328). The daemon reconciles each against the broker at
        startup: revert to ``pending`` if the order never landed, finalize via
        ``mark_entry_executed`` if the position/order exists, or leave as-is
        (fail-safe) if broker state is unreadable.
        """
        return [
            e for e in self._data.get("approved_entries", [])
            if e.get("status") == "executing"
        ]

    def revert_entry_to_pending(self, ticker: str) -> None:
        """Roll a write-ahead ``executing`` entry back to ``pending``.

        Used when reconciliation proves the order never reached the broker
        (Rejected order, or restart with no matching position/working order),
        so the legitimate entry can be re-driven through the guarded path.
        """
        for entry in self._data.get("approved_entries", []):
            if entry["ticker"] == ticker and entry.get("status") == "executing":
                entry["status"] = "pending"
                entry.pop("executing_at", None)
                break

    def mark_entry_executed(self, ticker: str, trigger_reason: str) -> None:
        """Mark an entry as executed and move to executed_today.

        Matches a ``pending`` entry (normal path) OR an ``executing`` one
        (the write-ahead WAL entry being finalized after its order confirmed,
        config#2328).
        """
        entries = self._data.get("approved_entries", [])
        for entry in entries:
            if entry["ticker"] == ticker and entry.get("status") in ("pending", "executing"):
                entry["status"] = "executed"
                entry["trigger_reason"] = trigger_reason
                entry["executed_at"] = datetime.now().isoformat()
                entry.pop("executing_at", None)
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

    def update_stop_shares(self, ticker: str, new_shares: int) -> None:
        """Update the share count on a stop record after a partial REDUCE."""
        for stop in self._data.get("active_stops", []):
            if stop["ticker"] == ticker:
                stop["shares"] = new_shares
                break

    def mark_profit_take_executed(self, ticker: str) -> None:
        """Mark a stop record so profit-take doesn't fire again."""
        for stop in self._data.get("active_stops", []):
            if stop["ticker"] == ticker:
                stop["profit_take_executed"] = True
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
