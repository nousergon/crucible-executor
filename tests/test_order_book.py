"""Tests for executor.order_book.OrderBook."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from executor.order_book import OrderBook, _current_trading_day, _default_book


# ── helpers ──────────────────────────────────────────────────────────────


def _today() -> str:
    return date.today().isoformat()


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _make_book(tmp_path, data: dict | None = None):
    """Write a JSON order book file and return its path."""
    path = tmp_path / "order_book.json"
    path.write_text(json.dumps(data or _default_book(), indent=2))
    return path


# ── 1. load fresh when no file exists ────────────────────────────────────


def test_load_fresh_no_file(tmp_path):
    path = tmp_path / "order_book.json"
    book = OrderBook.load(path)
    assert book.data["date"] == _today()
    assert book.data["approved_entries"] == []
    assert book.data["urgent_exits"] == []
    assert book.data["active_stops"] == []
    assert book.data["executed_today"] == []


# ── 2. load discards stale date ──────────────────────────────────────────


def test_load_discards_stale_date(tmp_path):
    stale = _default_book(run_date=_yesterday())
    stale["approved_entries"].append({"ticker": "AAPL", "status": "pending"})
    path = _make_book(tmp_path, stale)

    book = OrderBook.load(path)
    assert book.data["date"] == _today()
    assert book.data["approved_entries"] == []


# ── 3. load recovers from corrupt JSON ──────────────────────────────────


def test_load_recovers_from_corrupt_json(tmp_path):
    path = tmp_path / "order_book.json"
    path.write_text("{not valid json!!!")

    book = OrderBook.load(path)
    assert book.data["date"] == _today()
    assert book.data["approved_entries"] == []


# ── 4. save and reload round-trip ────────────────────────────────────────


def test_save_reload_round_trip(tmp_path):
    path = tmp_path / "order_book.json"
    book = OrderBook(_default_book(), path)
    book.add_entry({"ticker": "MSFT", "shares": 10})
    book.add_stop({"ticker": "MSFT", "current_stop": 400.0, "high_water": 420.0})
    book.save()

    reloaded = OrderBook.load(path)
    assert reloaded.data["approved_entries"][0]["ticker"] == "MSFT"
    assert reloaded.data["active_stops"][0]["ticker"] == "MSFT"


# ── 5. add_entry sets pending status ────────────────────────────────────


def test_add_entry_sets_pending():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "GOOG", "shares": 5})
    entry = book.data["approved_entries"][0]
    assert entry["status"] == "pending"
    assert entry["ticker"] == "GOOG"


# ── 6. add_urgent_exit deduplication ────────────────────────────────────


def test_add_urgent_exit_deduplication():
    book = OrderBook(_default_book())
    record = {"ticker": "TSLA", "signal": "EXIT"}
    book.add_urgent_exit(record.copy())
    book.add_urgent_exit(record.copy())

    assert len(book.data["urgent_exits"]) == 1


# ── 7. add_urgent_exit keeps different signals ──────────────────────────


def test_add_urgent_exit_different_signals_kept():
    book = OrderBook(_default_book())
    book.add_urgent_exit({"ticker": "TSLA", "signal": "EXIT"})
    book.add_urgent_exit({"ticker": "TSLA", "signal": "REDUCE"})

    assert len(book.data["urgent_exits"]) == 2
    signals = {e["signal"] for e in book.data["urgent_exits"]}
    assert signals == {"EXIT", "REDUCE"}


# ── 8. mark_entry_executed moves to executed_today ───────────────────────


def test_mark_entry_executed():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "NVDA", "shares": 3})
    book.mark_entry_executed("NVDA", "pullback")

    assert len(book.data["approved_entries"]) == 0
    assert len(book.data["executed_today"]) == 1
    executed = book.data["executed_today"][0]
    assert executed["ticker"] == "NVDA"
    assert executed["status"] == "executed"
    assert executed["trigger_reason"] == "pullback"
    assert "executed_at" in executed


# ── 9. mark_urgent_executed moves to executed_today ──────────────────────


def test_mark_urgent_executed():
    book = OrderBook(_default_book())
    book.add_urgent_exit({"ticker": "META", "signal": "EXIT"})
    book.mark_urgent_executed("META", "EXIT")

    assert len(book.data["urgent_exits"]) == 0
    assert len(book.data["executed_today"]) == 1
    executed = book.data["executed_today"][0]
    assert executed["ticker"] == "META"
    assert executed["status"] == "executed"
    assert "executed_at" in executed


# ── 10. remove_stop removes by ticker ───────────────────────────────────


def test_remove_stop():
    book = OrderBook(_default_book())
    book.add_stop({"ticker": "AAPL", "current_stop": 170.0})
    book.add_stop({"ticker": "MSFT", "current_stop": 400.0})
    book.remove_stop("AAPL")

    assert len(book.data["active_stops"]) == 1
    assert book.data["active_stops"][0]["ticker"] == "MSFT"


# ── 11. update_stop_high_water ───────────────────────────────────────────


def test_update_stop_high_water():
    book = OrderBook(_default_book())
    book.add_stop({"ticker": "AMZN", "current_stop": 180.0, "high_water": 190.0})
    book.update_stop_high_water("AMZN", new_high=200.0, new_stop=192.0)

    stop = book.data["active_stops"][0]
    assert stop["high_water"] == 200.0
    assert stop["current_stop"] == 192.0


# ── 12. reset_pending preserves executed_today ───────────────────────────


def test_reset_pending_preserves_executed_today():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "GOOG", "shares": 5})
    book.mark_entry_executed("GOOG", "vwap")
    book.add_entry({"ticker": "NFLX", "shares": 2})
    book.add_urgent_exit({"ticker": "TSLA", "signal": "EXIT"})
    book.add_stop({"ticker": "MSFT", "current_stop": 400.0})

    book.reset_pending()

    assert book.data["approved_entries"] == []
    assert book.data["urgent_exits"] == []
    assert book.data["active_stops"] == []
    assert len(book.data["executed_today"]) == 1
    assert book.data["executed_today"][0]["ticker"] == "GOOG"


# ── 13. merge_executed removes traded tickers ────────────────────────────


def test_merge_executed_removes_traded_tickers():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "AAPL", "shares": 10})
    book.add_entry({"ticker": "MSFT", "shares": 5})
    book.add_urgent_exit({"ticker": "AAPL", "signal": "EXIT"})

    book.merge_executed({"AAPL"})

    tickers = [e["ticker"] for e in book.data["approved_entries"]]
    assert tickers == ["MSFT"]
    assert len(book.data["urgent_exits"]) == 0


# ── 14. merge_executed noop for empty set ────────────────────────────────


def test_merge_executed_noop_empty_set():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "AAPL", "shares": 10})
    book.merge_executed(set())

    assert len(book.data["approved_entries"]) == 1


# ── 15. all_tickers deduplicates across sections ────────────────────────


def test_all_tickers_deduplicates():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "AAPL", "shares": 10})
    book.add_urgent_exit({"ticker": "AAPL", "signal": "EXIT"})
    book.add_stop({"ticker": "AAPL", "current_stop": 170.0})
    book.add_entry({"ticker": "MSFT", "shares": 5})

    tickers = book.all_tickers()
    assert tickers == ["AAPL", "MSFT"]


# ── 16. has_content true / false ─────────────────────────────────────────


def test_has_content_true_with_entries():
    book = OrderBook(_default_book())
    assert book.has_content() is False

    book.add_entry({"ticker": "GOOG", "shares": 1})
    assert book.has_content() is True


def test_has_content_true_with_stops_only():
    book = OrderBook(_default_book())
    book.add_stop({"ticker": "GOOG", "current_stop": 150.0})
    assert book.has_content() is True


# ── 17. pending_entries filters out non-pending ──────────────────────────


def test_pending_entries_filters():
    book = OrderBook(_default_book())
    book.add_entry({"ticker": "AAPL", "shares": 10})
    book.add_entry({"ticker": "MSFT", "shares": 5})
    book.mark_entry_executed("AAPL", "pullback")

    # Only MSFT should remain pending (AAPL was removed from approved_entries)
    pending = book.pending_entries()
    assert len(pending) == 1
    assert pending[0]["ticker"] == "MSFT"


# ── 18. save is atomic (no .tmp remains) ────────────────────────────────


def test_save_atomic_no_tmp(tmp_path):
    path = tmp_path / "order_book.json"
    book = OrderBook(_default_book(), path)
    book.add_entry({"ticker": "AAPL", "shares": 10})
    book.save()

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


# ── 19. trading-day axis (issue config#1016) ─────────────────────────────
#
# The order book keys its `date` field and its load-time freshness check on
# the last *closed* NYSE session (now_dual().trading_day), not the raw
# calendar date, so the morning batch (main.py) and the daemon (daemon.py) —
# which both derive run_date from now_dual().trading_day — agree on which
# book is "today's" even when read pre-open or on a non-session calendar day.


def test_current_trading_day_resolves_to_prior_session_on_weekend(monkeypatch):
    """On a Saturday, _current_trading_day() walks back to the prior Friday
    session — NOT date.today(). Mirrors the now_dual() contract in
    nousergon_lib.dates (Sat 2026-04-25 → trading_day 2026-04-24)."""
    from nousergon_lib.dates import now_dual

    saturday = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)  # 8 AM ET Sat
    # _current_trading_day imports now_dual lazily from nousergon_lib.dates,
    # so patch that source symbol.
    import nousergon_lib.dates as dates_mod
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=saturday))

    assert _current_trading_day() == "2026-04-24"
    # And it is NOT the raw calendar date.
    assert _current_trading_day() != saturday.date().isoformat()


def test_default_book_keys_on_trading_day_when_offsession(monkeypatch):
    """A fresh book built on a weekend keys `date` to the prior session."""
    import nousergon_lib.dates as dates_mod
    from nousergon_lib.dates import now_dual

    sunday = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=sunday))

    book = _default_book()
    assert book["date"] == "2026-04-24"  # Friday session, not Sunday calendar


def test_load_keeps_book_written_on_prior_calendar_day_same_session(monkeypatch, tmp_path):
    """The exact bug config#1016 guards: a book the morning batch wrote keyed
    to the trading session must NOT be discarded as 'stale' on a pre-open read
    whose calendar date has advanced but whose session has not.

    Saturday read of a book keyed to Friday's session (2026-04-24) keeps it,
    because Saturday's trading_day is still Friday."""
    import nousergon_lib.dates as dates_mod
    from nousergon_lib.dates import now_dual

    saturday = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=saturday))

    book_data = _default_book(run_date="2026-04-24")  # Friday session key
    book_data["approved_entries"].append({"ticker": "AAPL", "status": "pending"})
    path = _make_book(tmp_path, book_data)

    loaded = OrderBook.load(path)
    # Same trading session → book preserved, entries intact.
    assert loaded.data["date"] == "2026-04-24"
    assert loaded.data["approved_entries"][0]["ticker"] == "AAPL"


def test_load_discards_book_from_prior_session(monkeypatch, tmp_path):
    """A book from a genuinely earlier session is still discarded."""
    import nousergon_lib.dates as dates_mod
    from nousergon_lib.dates import now_dual

    saturday = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)  # session = Fri 4/24
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=saturday))

    stale = _default_book(run_date="2026-04-23")  # Thursday session — older
    stale["approved_entries"].append({"ticker": "AAPL", "status": "pending"})
    path = _make_book(tmp_path, stale)

    loaded = OrderBook.load(path)
    assert loaded.data["date"] == "2026-04-24"  # rebuilt on current session
    assert loaded.data["approved_entries"] == []
