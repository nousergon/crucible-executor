"""Tests for mid-session surveillance-universe re-derivation (config#897).

The daemon builds the IB surveillance universe once at startup and, until this
fix, never re-read signals.json in the poll loop — so a mid-session Research
re-run (e.g. a manual Saturday-SF re-run during a weekday session) left the IB
subscription pinned to the startup-time universe until a daemon restart.

These cover ``_refresh_surveillance_universe`` end-to-end against a MagicMock
monitor (same pattern as ``tests/test_price_monitor.py`` uses for the IB
client): new signals tickers get subscribed, dropped ones get cancelled, an
unchanged signals.json produces zero churn, and order_book / positions still
contribute to the universe.
"""
from unittest.mock import MagicMock, patch

from executor.daemon import _refresh_surveillance_universe, _signals_fingerprint


def _make_monitor(subscribed):
    """A monitor stub exposing the diff interface the daemon calls.

    ``resubscribe`` computes the add/remove delta itself (mirroring the real
    PriceMonitor contract) so the test can assert on exactly which tickers
    would hit reqMktData / cancelMktData.
    """
    monitor = MagicMock()
    state = {"tickers": set(subscribed)}

    def resubscribe(new_tickers):
        desired = set(new_tickers)
        added = desired - state["tickers"]
        removed = state["tickers"] - desired
        state["tickers"] = desired
        return added, removed

    monitor.resubscribe.side_effect = resubscribe
    monitor._state = state
    return monitor


def _make_order_book(tickers):
    ob = MagicMock()
    ob.all_tickers.return_value = list(tickers)
    return ob


def _make_ibkr(positions):
    ibkr = MagicMock()
    ibkr.get_positions.return_value = {t: {"shares": 1} for t in positions}
    return ibkr


def _signals(*tickers):
    return {"signals": {t: {"score": 1.0} for t in tickers}, "buy_candidates": []}


def _refresh(monitor, signals, *, order_book_tickers=(), positions=(),
             last_fingerprint=None, current_tickers=None, dry_run=False):
    """Drive _refresh_surveillance_universe with signals returned from the reader."""
    order_book = _make_order_book(order_book_tickers)
    ibkr = _make_ibkr(positions)
    with patch(
        "executor.signal_reader.read_signals_with_fallback",
        return_value=signals,
    ):
        return _refresh_surveillance_universe(
            monitor,
            config={"signals_bucket": "bkt"},
            run_date="2026-07-05",
            order_book=order_book,
            ibkr=ibkr,
            dry_run=dry_run,
            last_fingerprint=last_fingerprint,
            current_tickers=list(current_tickers or []),
        )


# ── (a) new tickers appear → subscribe called for exactly the new ones ───────


def test_new_signals_tickers_are_subscribed():
    monitor = _make_monitor(["AAPL", "SPY"])
    current = ["AAPL", "SPY"]

    tickers, _fp = _refresh(
        monitor,
        _signals("AAPL", "NVDA", "TSLA"),
        last_fingerprint="stale",
        current_tickers=current,
    )

    monitor.resubscribe.assert_called_once()
    # New names present in the derived universe...
    assert {"NVDA", "TSLA"}.issubset(set(tickers))
    # ...and now subscribed on the monitor.
    assert {"NVDA", "TSLA"}.issubset(monitor._state["tickers"])


def test_new_tickers_subscribe_delta_is_exactly_the_new_ones():
    monitor = _make_monitor(["AAPL", "SPY"])
    resubscribe_arg = {}
    orig = monitor.resubscribe.side_effect

    def spy(new_tickers):
        resubscribe_arg["desired"] = set(new_tickers)
        return orig(new_tickers)

    monitor.resubscribe.side_effect = spy

    _refresh(
        monitor,
        _signals("AAPL", "NVDA"),
        last_fingerprint="stale",
        current_tickers=["AAPL", "SPY"],
    )

    # Universe = {AAPL, NVDA, SPY}; previously {AAPL, SPY}. Delta added = {NVDA}.
    assert resubscribe_arg["desired"] == {"AAPL", "NVDA", "SPY"}
    assert "NVDA" in monitor._state["tickers"]


# ── (b) tickers removed → cancel called for exactly the removed ones ─────────


def test_removed_signals_tickers_are_cancelled():
    monitor = _make_monitor(["AAPL", "NVDA", "TSLA", "SPY"])

    tickers, _fp = _refresh(
        monitor,
        _signals("AAPL"),
        last_fingerprint="stale",
        current_tickers=["AAPL", "NVDA", "TSLA", "SPY"],
    )

    # NVDA + TSLA dropped from signals and not held/booked → cancelled.
    assert "NVDA" not in monitor._state["tickers"]
    assert "TSLA" not in monitor._state["tickers"]
    # AAPL + SPY survive.
    assert {"AAPL", "SPY"}.issubset(monitor._state["tickers"])
    assert "NVDA" not in tickers and "TSLA" not in tickers


# ── (c) unchanged signals.json → no subscribe/cancel churn ───────────────────


def test_unchanged_signals_is_a_no_op():
    signals = _signals("AAPL", "NVDA")
    fp = _signals_fingerprint(signals)
    monitor = _make_monitor(["AAPL", "NVDA", "SPY"])

    tickers, new_fp = _refresh(
        monitor,
        signals,
        last_fingerprint=fp,  # same fingerprint the reader will produce
        current_tickers=["AAPL", "NVDA", "SPY"],
    )

    # Fingerprint matched → the function short-circuits: no recompute, no churn.
    monitor.resubscribe.assert_not_called()
    assert new_fp == fp
    assert tickers == ["AAPL", "NVDA", "SPY"]


def test_changed_payload_same_universe_updates_fingerprint_without_churn():
    # signals.json rewritten (new fingerprint) but the derived ticker set is
    # identical (e.g. only scores changed) → adopt fingerprint, skip IB work.
    monitor = _make_monitor(["AAPL", "SPY"])

    tickers, new_fp = _refresh(
        monitor,
        _signals("AAPL"),  # universe = {AAPL, SPY}
        last_fingerprint="old-different",
        current_tickers=["AAPL", "SPY"],
    )

    monitor.resubscribe.assert_not_called()
    assert set(tickers) == {"AAPL", "SPY"}
    assert new_fp is not None and new_fp != "old-different"


# ── (d) order_book + positions still contribute to the universe ──────────────


def test_order_book_and_positions_contribute_to_universe():
    monitor = _make_monitor(["SPY"])

    tickers, _fp = _refresh(
        monitor,
        _signals("AAPL"),
        order_book_tickers=["BOOKED"],
        positions=["HELD"],
        last_fingerprint="stale",
        current_tickers=["SPY"],
    )

    # Universe = signals(AAPL) ∪ order_book(BOOKED) ∪ positions(HELD) ∪ SPY.
    assert set(tickers) == {"AAPL", "BOOKED", "HELD", "SPY"}
    assert {"AAPL", "BOOKED", "HELD"}.issubset(monitor._state["tickers"])


def test_dry_run_skips_positions_but_keeps_book_and_signals():
    monitor = _make_monitor(["SPY"])

    tickers, _fp = _refresh(
        monitor,
        _signals("AAPL"),
        order_book_tickers=["BOOKED"],
        positions=["HELD"],  # ignored under dry_run
        last_fingerprint="stale",
        current_tickers=["SPY"],
        dry_run=True,
    )

    assert set(tickers) == {"AAPL", "BOOKED", "SPY"}
    assert "HELD" not in tickers


# ── fail-soft: read failure keeps the current universe, no crash/no drop ─────


def test_read_failure_keeps_current_universe_and_does_not_touch_monitor():
    monitor = _make_monitor(["AAPL", "SPY"])
    order_book = _make_order_book([])
    ibkr = _make_ibkr([])

    with patch(
        "executor.signal_reader.read_signals_with_fallback",
        side_effect=RuntimeError("s3 down"),
    ):
        tickers, fp = _refresh_surveillance_universe(
            monitor,
            config={"signals_bucket": "bkt"},
            run_date="2026-07-05",
            order_book=order_book,
            ibkr=ibkr,
            dry_run=False,
            last_fingerprint="last-good",
            current_tickers=["AAPL", "SPY"],
        )

    # Fail-soft to last-good: unchanged tickers + fingerprint, monitor untouched.
    assert tickers == ["AAPL", "SPY"]
    assert fp == "last-good"
    monitor.resubscribe.assert_not_called()


def test_positions_read_failure_degrades_but_still_refreshes_signals():
    monitor = _make_monitor(["SPY"])
    order_book = _make_order_book([])
    ibkr = MagicMock()
    ibkr.get_positions.side_effect = RuntimeError("ib positions unavailable")

    with patch(
        "executor.signal_reader.read_signals_with_fallback",
        return_value=_signals("AAPL"),
    ):
        tickers, _fp = _refresh_surveillance_universe(
            monitor,
            config={"signals_bucket": "bkt"},
            run_date="2026-07-05",
            order_book=order_book,
            ibkr=ibkr,
            dry_run=False,
            last_fingerprint="stale",
            current_tickers=["SPY"],
        )

    # Positions degrade to empty; signals universe still applied.
    assert set(tickers) == {"AAPL", "SPY"}


# ── fingerprint helper ───────────────────────────────────────────────────────


def test_fingerprint_stable_across_key_order():
    a = {"signals": {"AAPL": {"x": 1}, "MSFT": {"y": 2}}, "buy_candidates": ["NVDA"]}
    b = {"buy_candidates": ["NVDA"], "signals": {"MSFT": {"y": 2}, "AAPL": {"x": 1}}}
    assert _signals_fingerprint(a) == _signals_fingerprint(b)


def test_fingerprint_changes_on_content_change():
    a = _signals("AAPL")
    b = _signals("AAPL", "NVDA")
    assert _signals_fingerprint(a) != _signals_fingerprint(b)


def test_fingerprint_none_for_empty_signals():
    assert _signals_fingerprint(None) is None
    assert _signals_fingerprint({}) is None
