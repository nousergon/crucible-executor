"""Crash-restart double-BUY prevention (config#2328).

Covers the three defense layers that stop a daemon restarted between an IB
fill and the order-book save from re-placing the same ENTER:

  1. Write-ahead ``executing`` status on the order book (order_book WAL).
  2. Startup seeding of the already-traded set from durable state.
  3. Pre-BUY broker-truth duplicate guard, symmetric to the SELL cap.

plus the startup reconciliation that resolves in-doubt ``executing`` entries.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from executor.daemon import (
    _validate_buy_not_duplicate,
    _reconcile_executing_entries,
)
from executor.order_book import OrderBook, _current_session_date, _default_book
from executor.trade_logger import init_db, get_executed_entry_tickers


# ── helpers ──────────────────────────────────────────────────────────────


def _book_with_pending(tmp_path, ticker="AAPL"):
    # Key the book to the current session so OrderBook.load() doesn't discard
    # it as stale on reload (the freshness check compares on the session axis,
    # config#1610).
    data = _default_book(_current_session_date())
    data["approved_entries"].append(
        {"ticker": ticker, "status": "pending", "shares": 100}
    )
    return OrderBook(data, path=tmp_path / "order_book.json")


def _mock_ibkr(positions=None, open_buy=0, positions_raises=False, open_buy_raises=False):
    ibkr = MagicMock()
    if positions_raises:
        ibkr.get_positions.side_effect = ConnectionError("IB gateway down")
    else:
        ibkr.get_positions.return_value = positions or {}
    if open_buy_raises:
        ibkr.get_open_buy_shares.side_effect = ConnectionError("IB gateway down")
    else:
        ibkr.get_open_buy_shares.return_value = open_buy
    return ibkr


# ── Layer 1: order-book write-ahead ``executing`` state ──────────────────


class TestOrderBookExecutingWAL:
    def test_mark_executing_removes_from_pending(self, tmp_path):
        book = _book_with_pending(tmp_path)
        book.mark_entry_executing("AAPL", "gap_up")
        # No longer a candidate for the naive entry loop...
        assert book.pending_entries() == []
        # ...but durably recorded as in-doubt.
        executing = book.executing_entries()
        assert [e["ticker"] for e in executing] == ["AAPL"]
        assert executing[0]["status"] == "executing"
        assert "executing_at" in executing[0]

    def test_executing_survives_reload_from_disk(self, tmp_path):
        """The core invariant: a crash between placement and save leaves an
        'executing' entry on disk; a restarted daemon does NOT see it as
        pending, so it cannot re-place the BUY."""
        book = _book_with_pending(tmp_path)
        book.mark_entry_executing("AAPL", "gap_up")
        book.save()  # <-- pre-crash WAL write, then the process "dies"

        # Restart: fresh load off the same file (same session date).
        reloaded = OrderBook.load(tmp_path / "order_book.json")
        assert reloaded.pending_entries() == []  # NOT re-placeable
        assert [e["ticker"] for e in reloaded.executing_entries()] == ["AAPL"]

    def test_mark_executed_finalizes_an_executing_entry(self, tmp_path):
        book = _book_with_pending(tmp_path)
        book.mark_entry_executing("AAPL", "gap_up")
        book.mark_entry_executed("AAPL", "gap_up")
        assert book.pending_entries() == []
        assert book.executing_entries() == []
        assert [e["ticker"] for e in book.data["executed_today"]] == ["AAPL"]
        assert book.data["executed_today"][0]["status"] == "executed"

    def test_revert_to_pending_restores_candidacy(self, tmp_path):
        book = _book_with_pending(tmp_path)
        book.mark_entry_executing("AAPL", "gap_up")
        book.revert_entry_to_pending("AAPL")
        assert [e["ticker"] for e in book.pending_entries()] == ["AAPL"]
        assert book.executing_entries() == []


# ── Layer 2: startup seeding query ───────────────────────────────────────


class TestExecutedEntrySeeding:
    def _insert_trade(self, conn, ticker, action, date="2026-07-11"):
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"{ticker}-{action}-{date}", date, ticker, action, 100, "2026-07-11T10:00:00"),
        )
        conn.commit()

    def test_seeds_todays_enter_fills(self, tmp_path):
        conn = init_db(str(tmp_path / "trades.db"))
        self._insert_trade(conn, "AAPL", "ENTER")
        self._insert_trade(conn, "MSFT", "ENTER")
        self._insert_trade(conn, "NVDA", "EXIT")  # not an ENTER
        self._insert_trade(conn, "TSLA", "ENTER", date="2026-07-10")  # other session
        assert get_executed_entry_tickers(conn, "2026-07-11") == {"AAPL", "MSFT"}

    def test_empty_when_no_fills(self, tmp_path):
        conn = init_db(str(tmp_path / "trades.db"))
        assert get_executed_entry_tickers(conn, "2026-07-11") == set()


# ── Layer 3: pre-BUY broker-truth duplicate guard ────────────────────────


class TestValidateBuyNotDuplicate:
    def test_allows_when_flat_and_no_working_order(self):
        ibkr = _mock_ibkr(positions={}, open_buy=0)
        assert _validate_buy_not_duplicate(ibkr, "AAPL", "daemon_entry") is True

    def test_skips_when_already_holding(self):
        ibkr = _mock_ibkr(positions={"AAPL": {"shares": 100}}, open_buy=0)
        assert _validate_buy_not_duplicate(ibkr, "AAPL", "daemon_entry") is False

    def test_skips_when_working_buy_exists(self):
        ibkr = _mock_ibkr(positions={}, open_buy=100)
        assert _validate_buy_not_duplicate(ibkr, "AAPL", "daemon_entry") is False

    def test_fails_closed_on_position_read_error(self):
        ibkr = _mock_ibkr(positions_raises=True)
        assert _validate_buy_not_duplicate(ibkr, "AAPL", "daemon_entry") is False

    def test_fails_closed_on_open_order_read_error(self):
        ibkr = _mock_ibkr(positions={}, open_buy_raises=True)
        assert _validate_buy_not_duplicate(ibkr, "AAPL", "daemon_entry") is False


# ── Startup reconciliation of in-doubt ``executing`` entries ─────────────


class TestReconcileExecutingEntries:
    def _executing_book(self, tmp_path, ticker="AAPL"):
        book = _book_with_pending(tmp_path, ticker)
        book.mark_entry_executing(ticker, "gap_up")
        book.save()
        return book

    def test_noop_when_nothing_executing(self, tmp_path):
        book = _book_with_pending(tmp_path)
        ibkr = _mock_ibkr()
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        ibkr.get_positions.assert_not_called()
        assert [e["ticker"] for e in book.pending_entries()] == ["AAPL"]

    def test_finalizes_when_broker_shows_position(self, tmp_path):
        # Order LANDED at broker before the crash → must NOT be re-placed.
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions={"AAPL": {"shares": 100}}, open_buy=0)
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        assert book.executing_entries() == []
        assert book.pending_entries() == []  # not re-placeable
        assert [e["ticker"] for e in book.data["executed_today"]] == ["AAPL"]

    def test_finalizes_when_working_buy_exists(self, tmp_path):
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions={}, open_buy=100)
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        assert book.executing_entries() == []
        assert [e["ticker"] for e in book.data["executed_today"]] == ["AAPL"]

    def test_reverts_when_order_never_landed(self, tmp_path):
        # Broker flat + no working order → the order never reached IB →
        # revert to pending so the legitimate entry re-drives (guarded).
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions={}, open_buy=0)
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        assert book.executing_entries() == []
        assert [e["ticker"] for e in book.pending_entries()] == ["AAPL"]

    def test_leaves_in_doubt_when_broker_unreadable(self, tmp_path):
        # Fail-safe: cannot verify → leave 'executing' (blocked, never re-bought).
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions_raises=True)
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        assert [e["ticker"] for e in book.executing_entries()] == ["AAPL"]
        assert book.pending_entries() == []  # still not re-placeable

    def test_leaves_in_doubt_when_open_order_read_fails(self, tmp_path):
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions={}, open_buy_raises=True)
        _reconcile_executing_entries(ibkr, book, dry_run=False)
        assert [e["ticker"] for e in book.executing_entries()] == ["AAPL"]

    def test_dry_run_touches_nothing(self, tmp_path):
        book = self._executing_book(tmp_path)
        ibkr = _mock_ibkr(positions={"AAPL": {"shares": 100}})
        _reconcile_executing_entries(ibkr, book, dry_run=True)
        ibkr.get_positions.assert_not_called()
        assert [e["ticker"] for e in book.executing_entries()] == ["AAPL"]
