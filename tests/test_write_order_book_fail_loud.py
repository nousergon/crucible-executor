"""Fail-loud contract for the morning order-book write (config#1234).

``_write_order_book_or_raise`` (executor/main.py) wraps
``_write_stops_and_finalize``, which builds stop records for held positions
and then calls ``ob.save()`` — the order book's local-disk persist. The
intraday daemon reads ONLY that file to place every order for the rest of
the session, so a swallowed ``ob.save()`` failure is a ghost success: the
planner reports OK while the daemon trades the whole day on a stale/empty
book.

Previously the call site in ``run()`` caught ANY exception from
``_write_stops_and_finalize`` and demoted it to a WARNING. These tests pin
the fixed contract: a genuine ``ob.save()`` failure now RE-RAISES (mirroring
crucible-research#312 / crucible-predictor#304), while a failure in one of
the best-effort sub-steps that follow it (the S3 audit-trail backup) stays
non-fatal.
"""
from __future__ import annotations

import json

import pytest

import executor.main as main_mod
from executor.ibkr import SimulatedIBKRClient
from executor.order_book import OrderBook, _default_book


def _fresh_book(tmp_path):
    path = tmp_path / "order_book.json"
    return OrderBook(_default_book(run_date="2026-07-06"), path=path)


def test_write_order_book_or_raise_reraises_on_save_failure(tmp_path, monkeypatch, caplog):
    """A genuine ob.save() failure must propagate, not be swallowed to a WARNING."""
    ob = _fresh_book(tmp_path)

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(ob, "save", _boom)

    ibkr = SimulatedIBKRClient(prices={})

    with caplog.at_level("ERROR"):
        with pytest.raises(OSError, match="disk full"):
            main_mod._write_order_book_or_raise(
                ibkr, ob,
                price_histories=None,
                atr_map={},
                strategy_config={},
                conn=None,
                run_date="2026-07-06",
                blocked_entries=None,
                signals_bucket=None,
                use_optimizer=False,
            )

    assert any("Failed to write order book" in r.message for r in caplog.records)
    # No warning-only demotion — the ERROR record above is the only trace,
    # and it must not be swallowed (the `with pytest.raises` above is the
    # real assertion; this just pins the log message contract too).


def test_write_order_book_or_raise_succeeds_and_persists(tmp_path):
    """The happy path still writes the order book to disk as before."""
    ob = _fresh_book(tmp_path)
    ibkr = SimulatedIBKRClient(prices={})

    main_mod._write_order_book_or_raise(
        ibkr, ob,
        price_histories=None,
        atr_map={},
        strategy_config={},
        conn=None,
        run_date="2026-07-06",
        blocked_entries=None,
        signals_bucket=None,
        use_optimizer=False,
    )

    persisted = json.loads(ob._path.read_text())
    assert persisted["date"] == "2026-07-06"


def test_backup_to_s3_failure_stays_non_fatal(tmp_path, monkeypatch, caplog):
    """The S3 audit-trail backup (order_book.py::backup_to_s3) is a genuinely
    secondary mirror of the registered morning_planner_orders artifact — its
    own internal try/except must keep swallowing failures (WARN, non-fatal)
    even after the ob.save() fail-loud fix, so a transient S3 hiccup can't
    block a planner run whose local order book already saved successfully."""
    ob = _fresh_book(tmp_path)
    ibkr = SimulatedIBKRClient(prices={})

    class _BoomS3:
        def put_object(self, **kwargs):
            raise RuntimeError("S3 unavailable")

    monkeypatch.setattr("boto3.client", lambda *a, **k: _BoomS3())

    with caplog.at_level("WARNING"):
        # Must NOT raise — backup_to_s3 catches internally.
        main_mod._write_order_book_or_raise(
            ibkr, ob,
            price_histories=None,
            atr_map={},
            strategy_config={},
            conn=None,
            run_date="2026-07-06",
            blocked_entries=None,
            signals_bucket="alpha-engine-research",
            use_optimizer=False,
        )

    assert any("S3 backup failed" in r.message for r in caplog.records)
    # The load-bearing local save still happened despite the S3 failure.
    persisted = json.loads(ob._path.read_text())
    assert persisted["date"] == "2026-07-06"
