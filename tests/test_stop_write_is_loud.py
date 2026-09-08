"""The stop-record write is loud, and it runs on a held day.

`alpha-engine-config-I10190`. Until 2026-09-08 the §6 call site in
`executor/main.py` was a bare `except Exception` around
`_write_stops_and_finalize` that logged a WARNING and continued. That function
produces the stop records `executor.daemon` reads, and their `stop_kind`
decides what the daemon does with every open position. Measured 2026-09-08:
`grep -rn "Failed to write order book"` across the fleet found the two call
sites in that file and nothing else — no metric filter, no alarm, no
alert-transport entry. The failure reached no durable surface at all.

Two invariants are asserted together, because the hold-book safeguard's
justification is true only while BOTH hold:

  1. §6 runs on a HELD day — it is not nested inside the §5b hold branch. This
     is what makes "suppresses the rebalance, retains the book, still writes
     stops, hard-risk overrides remain active" a defensible posture rather than
     a claim.
  2. A failure to produce those records is LOUD — it raises, it records the
     date durably, and it publishes a metric an alarm can read.

If (1) regressed, a held day would trade with no stops. If (2) regressed, the
day it happened would look identical to a healthy one.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

import executor.main as main
from executor.main import StopRecordWriteError, write_stops_and_finalize_guarded

SOURCE = Path(inspect.getfile(main)).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


class TestTheStopWriteRunsOnAHeldDay:
    """Invariant 1 — asserted against the source, because the only other way to
    show it is to run the whole planner against a broker."""

    def test_the_guarded_stop_write_is_reached_from_the_planner(self):
        assert _calls_named(TREE, "write_stops_and_finalize_guarded"), (
            "no call to write_stops_and_finalize_guarded in executor/main.py — "
            "either §6 was renamed or the guard was bypassed"
        )

    def test_the_stop_write_is_not_nested_under_the_hold_branch(self):
        """`if _hold_book:` suppresses the optimizer rebalance. It must NOT
        also suppress the stop-record write, or the safeguard's own stated
        justification is false on the one day it fires."""
        offenders = []
        for node in ast.walk(TREE):
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "_hold_book" not in names:
                continue
            for stmt in node.body:
                if _calls_named(stmt, "write_stops_and_finalize_guarded"):
                    offenders.append(f"line {node.lineno}")
        assert not offenders, (
            f"the stop-record write is nested inside an `if _hold_book:` branch "
            f"at {offenders}. On a held day the book is retained and the "
            f"optimizer is suppressed; the stops are the ONLY remaining "
            f"per-name protection, and this nesting removes them from exactly "
            f"the day they matter most (alpha-engine-config-I10190)."
        )

    def test_the_raw_writer_is_not_called_directly_from_the_planner(self):
        """A second, unguarded call site would silently reopen the swallow."""
        direct = [
            n.lineno for n in _calls_named(TREE, "_write_stops_and_finalize")
        ]
        # The only legitimate caller is the guard itself.
        guard = next(
            n for n in ast.walk(TREE)
            if isinstance(n, ast.FunctionDef)
            and n.name == "write_stops_and_finalize_guarded"
        )
        inside = {n.lineno for n in _calls_named(guard, "_write_stops_and_finalize")}
        assert set(direct) <= inside, (
            f"_write_stops_and_finalize is called outside its guard at "
            f"{sorted(set(direct) - inside)} — that call site has no failure "
            f"surface (alpha-engine-config-I10190)."
        )

    def test_the_old_swallow_is_gone(self):
        """Asserted over the AST, not the text. The guard's own docstring
        quotes the removed code verbatim, and a substring scan would match that
        explanation — the false-positive class this fleet has now hit several
        times, where a rule's own justification trips the rule."""
        offenders = [
            n.lineno for n in ast.walk(TREE)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "warning"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "Failed to write order book: %s"
        ]
        assert not offenders, (
            f"the bare-except WARNING on the stop-record write is back as "
            f"EXECUTABLE code at line(s) {offenders}"
        )


class _Boto:
    """Captures put_metric_data / put_object without touching AWS."""

    def __init__(self, *, s3_raises: bool = False):
        self.metrics: list[dict] = []
        self.objects: list[dict] = []
        self._s3_raises = s3_raises

    def client(self, name, *a, **kw):
        outer = self

        class _C:
            def put_metric_data(self, **kw):
                outer.metrics.append(kw)

            def put_object(self, **kw):
                if outer._s3_raises:
                    raise RuntimeError("s3 down too")
                outer.objects.append(kw)

        return _C()

    def gauge(self, name: str):
        for call in self.metrics:
            for m in call["MetricData"]:
                if m["MetricName"] == name:
                    return m["Value"]
        return None


@pytest.fixture()
def boto(monkeypatch):
    b = _Boto()
    monkeypatch.setitem(sys.modules, "boto3", b)
    return b


class TestAFailureToProduceStopRecordsIsLoud:
    """Invariant 2."""

    def test_a_write_failure_raises(self, monkeypatch, boto):
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ibkr timeout")),
        )
        with pytest.raises(StopRecordWriteError) as ei:
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None,
                "alpha-engine-research",
            )
        assert ei.value.__cause__ is not None
        assert "2026-09-08" in str(ei.value)

    def test_the_exception_tells_the_operator_what_is_live_and_what_to_do(
        self, monkeypatch, boto,
    ):
        """A raise that does not say the positions are unstopped is a stack
        trace, not an alert."""
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(StopRecordWriteError) as ei:
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None,
                "alpha-engine-research",
            )
        msg = str(ei.value)
        assert "LIVE" in msg
        assert "OPERATOR" in msg
        assert "stop_records_write_failed" in msg

    def test_a_write_failure_publishes_the_paging_metric(self, monkeypatch, boto):
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(StopRecordWriteError):
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None,
                "alpha-engine-research",
            )
        assert boto.gauge("stop_records_write_failed") == 1.0

    def test_a_write_failure_records_the_date_durably(self, monkeypatch, boto):
        """The metric pages; the artifact is what says WHICH date and why."""
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ibkr timeout")),
        )
        with pytest.raises(StopRecordWriteError):
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None,
                "alpha-engine-research",
            )
        assert boto.objects, "no durable record of a run that wrote no stops"
        put = boto.objects[0]
        assert put["Key"] == "executor/stop_write_failures/2026-09-08.json"
        import json

        body = json.loads(put["Body"])
        assert body["stop_records_written"] is False
        assert body["error_type"] == "RuntimeError"
        assert "ibkr timeout" in body["error"]

    def test_a_healthy_run_publishes_zero(self, monkeypatch, boto):
        """A metric that appears only on failure gives its alarm no baseline,
        and makes an absent emitter indistinguishable from a healthy run."""
        monkeypatch.setattr(main, "_write_stops_and_finalize", lambda *a, **k: None)
        write_stops_and_finalize_guarded(
            None, None, None, {}, {}, None, "2026-09-08", None,
            "alpha-engine-research",
        )
        assert boto.gauge("stop_records_write_failed") == 0.0
        assert not boto.objects

    def test_a_failure_of_the_recorder_never_masks_the_real_failure(
        self, monkeypatch,
    ):
        """The recording path runs on the exception path of the thing it
        reports. If S3 is down too, the operator must still get the stop-record
        failure — not an S3 error."""
        b = _Boto(s3_raises=True)
        monkeypatch.setitem(sys.modules, "boto3", b)
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the real one")),
        )
        with pytest.raises(StopRecordWriteError) as ei:
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None,
                "alpha-engine-research",
            )
        assert "the real one" in str(ei.value.__cause__)

    def test_no_bucket_still_raises(self, monkeypatch, boto):
        """A run with no signals_bucket cannot write the artifact; it must
        still raise and still emit."""
        monkeypatch.setattr(
            main, "_write_stops_and_finalize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(StopRecordWriteError):
            write_stops_and_finalize_guarded(
                None, None, None, {}, {}, None, "2026-09-08", None, None,
            )
        assert boto.gauge("stop_records_write_failed") == 1.0
        assert not boto.objects


class TestTheSiblingSummarySwallowIsClassifiedNotCopied:
    """Deliverable 4. `_write_order_book_summary`'s swallow is legitimate — a
    dashboard projection, nothing on the trading path reads it — but the
    fail-loud rule requires the deviation to be WRITTEN DOWN with a recording
    surface, and it had neither."""

    def test_the_summary_swallow_carries_its_three_required_fields(self):
        fn = SOURCE[SOURCE.index("def _write_order_book_summary("):]
        fn = fn[: fn.index("\nclass StopRecordWriteError")]
        assert "FAILURE MODE SWALLOWED" in fn
        assert "WHY THE PRIMARY DELIVERABLE SURVIVES" in fn
        assert "RECORDING SURFACE" in fn
        assert "alpha-engine-config-I10190" in fn

    def test_the_summary_swallow_now_has_a_countable_surface(self, monkeypatch):
        b = _Boto(s3_raises=True)
        monkeypatch.setitem(sys.modules, "boto3", b)

        class _OB:
            def pending_entries(self):
                return []

            def pending_urgent_exits(self):
                return []

        main._write_order_book_summary(_OB(), [], "alpha-engine-research", "2026-09-08")
        assert b.gauge("order_book_summary_write_failed") == 1.0

    def test_the_summary_swallow_does_not_raise(self, monkeypatch):
        """It stays non-fatal, deliberately — the classification is the fix,
        not a promotion to fatal."""
        b = _Boto(s3_raises=True)
        monkeypatch.setitem(sys.modules, "boto3", b)

        class _OB:
            def pending_entries(self):
                return []

            def pending_urgent_exits(self):
                return []

        main._write_order_book_summary(_OB(), [], "alpha-engine-research", "2026-09-08")
