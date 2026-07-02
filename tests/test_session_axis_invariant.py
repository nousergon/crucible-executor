"""Cross-component session-axis invariant (config#1610 closes-when).

For a session physically trading on day D, every component that produces or
consumes a session-keyed EVENT artifact must resolve the same key:

    daemon run_date == order-book `date` == snapshot key == reconcile
    run_date == D

The 6/23–7/2 incident: the daemon froze ``now_dual().trading_day`` (the
last CLOSED session, D-1) at startup while the post-close EOD components
resolved D, so every reconcile joined one session's trades against a
different session's snapshot. The knowledge axis (``trading_day``) is
untouched by the fix and is asserted here to remain D-1 intraday — the two
axes are exactly one session apart during a live session, and the snapshot
capturer's guard relies on them coinciding post-close.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from nousergon_lib.dates import (
    last_closed_trading_day,
    now_dual,
    previous_trading_day,
    session_date,
)

ET = ZoneInfo("America/New_York")

# A plain mid-week session: Thursday 2026-07-02 (the incident session).
SESSION = "2026-07-02"
INTRADAY = datetime(2026, 7, 2, 10, 0, tzinfo=ET)      # daemon running
PREOPEN = datetime(2026, 7, 2, 5, 45, tzinfo=ET)       # morning planner
POSTCLOSE = datetime(2026, 7, 2, 16, 15, tzinfo=ET)    # snapshot/reconcile


def _daemon_run_date(moment: datetime) -> str:
    """The axis daemon.py freezes at startup (session_date, strict)."""
    return session_date(moment, strict=True).isoformat()


def _planner_run_date(moment: datetime) -> str:
    """The axis main.py resolves (session_date, non-strict)."""
    return session_date(moment).isoformat()


def _eod_component_date(moment: datetime) -> str:
    """What snapshot_capturer/eod_reconcile resolve post-close: they call
    now_dual().trading_day, which AT/AFTER the close equals the just-closed
    session — the moment the two axes coincide."""
    return last_closed_trading_day(moment).isoformat()


class TestSessionAxisInvariant:
    def test_all_components_agree_on_session_d(self):
        """The config#1610 closes-when: planner (pre-open), daemon
        (intraday), and the EOD components (post-close) all key session D."""
        planner = _planner_run_date(PREOPEN)
        daemon = _daemon_run_date(INTRADAY)
        eod = _eod_component_date(POSTCLOSE)
        assert planner == daemon == eod == SESSION

    def test_order_book_freshness_holds_across_the_chain(self, monkeypatch, tmp_path):
        """The book main.py writes pre-open for D is accepted by a daemon
        reading it intraday — the freshness match that forces main.py and
        daemon.py onto the same axis."""
        import nousergon_lib.dates as dates_mod
        from executor.order_book import OrderBook, _default_book

        book = _default_book(run_date=_planner_run_date(PREOPEN))
        path = tmp_path / "order_book.json"
        path.write_text(__import__("json").dumps(book))

        monkeypatch.setattr(
            dates_mod, "session_date",
            lambda now=None, **kw: session_date(INTRADAY, **kw),
        )
        loaded = OrderBook.load(path)
        assert loaded.data["date"] == SESSION

    def test_eod_trigger_run_date_passes_snapshot_guard(self):
        """The daemon's frozen run_date, delivered to the EOD SF and passed
        as --date to snapshot_capturer post-close, must equal what the
        capturer's guard computes (now_dual().trading_day at capture time).
        This is the exact comparison that failed on eod-2026-07-01."""
        frozen_at_startup = _daemon_run_date(INTRADAY)
        guard_view_postclose = now_dual(now=POSTCLOSE).trading_day
        assert frozen_at_startup == guard_view_postclose

    def test_knowledge_axis_unchanged_and_one_session_behind(self):
        """trading_day (the closed-data axis) stays D-1 intraday — the fix
        must NOT move the knowledge axis; the two axes are exactly one
        session apart during a live session."""
        knowledge = now_dual(now=INTRADAY).trading_day
        session = _daemon_run_date(INTRADAY)
        assert knowledge == "2026-07-01"
        assert previous_trading_day(
            datetime.fromisoformat(session).date()
        ).isoformat() == knowledge

    def test_daemon_refuses_offsession_start(self):
        """strict=True: the daemon must not silently attribute a weekend /
        holiday / post-close start to the next session (the reverse of the
        D-1 bug). July 3 2026 is the observed Independence Day holiday."""
        holiday = datetime(2026, 7, 3, 10, 0, tzinfo=ET)
        with pytest.raises(ValueError):
            _daemon_run_date(holiday)
        with pytest.raises(ValueError):
            _daemon_run_date(POSTCLOSE)

    def test_incident_replay_old_axis_fails_guard(self):
        """Regression pin on the bug class itself: the OLD daemon axis
        (now_dual().trading_day frozen intraday) does NOT match what the
        snapshot guard computes post-close — the mismatch the guard
        correctly rejected on eod-2026-06-30 and eod-2026-07-01."""
        old_axis_frozen = now_dual(now=INTRADAY).trading_day  # 2026-07-01
        guard_view = now_dual(now=POSTCLOSE).trading_day       # 2026-07-02
        assert old_axis_frozen != guard_view

    def test_reconcile_live_guard_refuses_mismatched_date(self, monkeypatch):
        """eod_reconcile.run() with run_audit=True (live path) must refuse a
        run_date that isn't the just-closed session — symmetric with the
        snapshot guard, closing the silent-mis-join path."""
        import executor.eod_reconcile as er

        monkeypatch.setattr(
            er, "now_dual", lambda **kw: now_dual(now=POSTCLOSE)
        )
        with pytest.raises(RuntimeError, match="refusing live run"):
            er.run(run_date="2026-07-01", send_email=False, run_audit=True)

    def test_nav_series_writer_refuses_mislabeled_point(self):
        """Write-time content-vs-key guard: a NAV point ticking in session D
        must not land in a file labeled D-1 (the incident artifact:
        nav_series/2026-07-01.json full of 07-02 timestamps)."""
        from executor.intraday_snapshot import IntradayNavSeriesWriter

        class _S3Stub:
            def get_object(self, **kw):  # pragma: no cover - not reached
                raise AssertionError("write should be refused before read")

            def put_object(self, **kw):  # pragma: no cover - not reached
                raise AssertionError("mis-keyed point must not be written")

        w = IntradayNavSeriesWriter(bucket="b", s3_client=_S3Stub())
        # The writer stamps points with the real current time, so exercise
        # the guard with the label a D-1-frozen daemon would pass: yesterday
        # relative to the CURRENT session (works whenever the suite runs).
        current_session = session_date().isoformat()
        stale_label = previous_trading_day(
            datetime.fromisoformat(current_session).date()
        ).isoformat()
        assert w.write(stale_label, {"net_liquidation": 1.0}, spy_last=None) is False
