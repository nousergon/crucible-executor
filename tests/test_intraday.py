"""Tests to close coverage gaps: entry_triggers, intraday_exit_manager,
market_hours, notifier, retry.

All pure-logic tests — no IB Gateway or S3 dependencies.
"""

from datetime import date, datetime
from unittest.mock import patch

import pytest
import pytz

from executor.entry_triggers import EntryTriggerEngine
from executor.intraday_exit_manager import IntradayExitManager
from executor.market_hours import is_market_hours, is_trading_day
from executor.notifier import send_daemon_status, send_trade_alert
from executor.order_book import build_stop_record
from executor.retry import retry

_ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# EntryTriggerEngine
# ---------------------------------------------------------------------------


class TestEntryTriggerEngine:
    def _engine(self, **overrides):
        cfg = {"intraday_expiry_time": "15:55", "intraday_graduated_start_time": "14:00",
               "intraday_graduated_max_premium_pct": 0.01, "intraday_pullback_pct": 0.02,
               "intraday_vwap_discount_pct": 0.005, "intraday_support_pct": 0.01, **overrides}
        return EntryTriggerEngine(cfg)

    def test_no_price(self):
        eng = self._engine()
        ok, reason = eng.should_enter({"triggers": {}}, {"last": 0})
        assert ok is False

    def test_pullback_fires(self):
        eng = self._engine()
        ok, reason = eng.should_enter(
            {"triggers": {"pullback_pct": 0.02}},
            {"last": 145.0, "high": 150.0},
        )
        assert ok is True
        assert "pullback" in reason

    @patch("executor.entry_triggers.datetime")
    def test_pullback_no_fire(self, mock_dt):
        mock_dt.now.return_value = _ET.localize(datetime(2026, 4, 8, 10, 0))
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        eng = self._engine()
        ok, reason = eng.should_enter(
            {"triggers": {"pullback_pct": 0.02}},
            {"last": 149.0, "high": 150.0},
        )
        assert ok is False

    def test_vwap_discount_fires(self):
        eng = self._engine()
        ok, reason = eng.should_enter(
            {"triggers": {"vwap": 150.0, "vwap_discount": 0.005}},
            {"last": 148.0, "high": 150.0},
        )
        assert ok is True
        assert "VWAP" in reason

    def test_support_bounce_fires(self):
        eng = self._engine(disabled_triggers=["pullback"])
        ok, reason = eng.should_enter(
            {"triggers": {"support_level": 145.0, "support_pct": 0.01}},
            {"last": 145.5, "high": 146.0, "low": 145.2},
        )
        assert ok is True
        assert "support" in reason

    @patch("executor.entry_triggers.datetime")
    def test_support_broken_no_fire(self, mock_dt):
        mock_dt.now.return_value = _ET.localize(datetime(2026, 4, 8, 10, 0))
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        eng = self._engine(disabled_triggers=["pullback"])
        ok, reason = eng.should_enter(
            {"triggers": {"support_level": 145.0}},
            {"last": 144.0, "high": 146.0, "low": 143.0},
        )
        assert ok is False

    @patch("executor.entry_triggers.datetime")
    def test_time_expiry_fires(self, mock_dt):
        mock_dt.now.return_value = _ET.localize(datetime(2026, 4, 8, 15, 56))
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        eng = self._engine()
        ok, reason = eng.should_enter(
            {"triggers": {}},
            {"last": 150.0, "high": 150.0},
        )
        assert ok is True
        assert "time_expiry" in reason

    @patch("executor.entry_triggers.datetime")
    def test_graduated_entry_fires(self, mock_dt):
        mock_dt.now.return_value = _ET.localize(datetime(2026, 4, 8, 14, 30))
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        eng = self._engine()
        ok, reason = eng.should_enter(
            {"triggers": {}, "current_price": 150.0},
            {"last": 150.5, "high": 152.0},
        )
        assert ok is True
        assert "graduated" in reason

    @patch("executor.entry_triggers.datetime")
    def test_disabled_trigger(self, mock_dt):
        mock_dt.now.return_value = _ET.localize(datetime(2026, 4, 8, 10, 0))
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        eng = self._engine(disabled_triggers=["pullback"])
        ok, _ = eng.should_enter(
            {"triggers": {"pullback_pct": 0.02}},
            {"last": 140.0, "high": 150.0},
        )
        assert ok is False  # pullback disabled


# ---------------------------------------------------------------------------
# IntradayExitManager
# ---------------------------------------------------------------------------


class TestIntradayExitManager:
    def _mgr(self, **overrides):
        cfg = {"intraday_profit_take_pct": 0.08, "intraday_collapse_pct": 0.05,
               "intraday_tighten_after_days": 3, "intraday_tighten_atr_multiple": 1.5,
               **overrides}
        return IntradayExitManager(cfg)

    def test_trailing_stop_fires(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 145.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "high_water": 155.0, "entry_price": 140.0, "shares": 10},
            {"last": 144.0, "high": 155.0},
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert "trailing_stop" in result["reason"]

    def test_trailing_stop_no_fire(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 145.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "high_water": 155.0, "entry_price": 140.0, "shares": 10},
            {"last": 150.0, "high": 155.0},
        )
        assert result is None

    def test_profit_take_fires(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 130.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "high_water": 155.0, "entry_price": 140.0, "shares": 10},
            {"last": 152.0, "high": 155.0},
        )
        assert result is not None
        assert result["action"] == "REDUCE"
        assert result["shares"] == 5

    def test_profit_take_already_executed(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 130.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "high_water": 155.0, "entry_price": 140.0,
             "shares": 10, "profit_take_executed": True},
            {"last": 152.0, "high": 155.0},
        )
        assert result is None

    def test_collapse_fires(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 130.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "high_water": 155.0, "entry_price": 150.0, "shares": 10},
            {"last": 142.0, "high": 155.0},
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert "collapse" in result["reason"]

    def test_no_price_returns_none(self):
        mgr = self._mgr()
        result = mgr.evaluate(
            {"ticker": "AAPL", "current_stop": 145.0, "shares": 10},
            {"last": 0},
        )
        assert result is None

    def test_should_update_trail_ratchets_up(self):
        mgr = self._mgr()
        result = mgr.should_update_trail(
            {"high_water": 150.0, "current_stop": 140.0, "trail_atr": 5.0, "atr_multiple": 2.0},
            155.0,
        )
        assert result is not None
        new_hw, new_stop = result
        assert new_hw == 155.0
        assert new_stop == 145.0

    def test_should_update_trail_no_change(self):
        mgr = self._mgr()
        result = mgr.should_update_trail(
            {"high_water": 155.0, "current_stop": 145.0, "trail_atr": 5.0, "atr_multiple": 2.0},
            153.0,
        )
        assert result is None

    def test_should_update_trail_tightens_after_days(self):
        mgr = self._mgr()
        # Use a date 10 days ago
        from datetime import timedelta
        entry = (date.today() - timedelta(days=10)).isoformat()
        result = mgr.should_update_trail(
            {"high_water": 150.0, "current_stop": 140.0, "trail_atr": 5.0,
             "atr_multiple": 2.0, "entry_date": entry},
            155.0,
        )
        assert result is not None
        _, new_stop = result
        # Tightened to 1.5x ATR = 7.5, so stop = 155 - 7.5 = 147.5
        assert new_stop == 147.5

    # ── Catastrophic gap stop (optimizer-mode hard-risk override) ────────────
    def test_catastrophic_gap_fires(self):
        mgr = self._mgr()  # default pct 0.15
        result = mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 169.0},  # -15.5% from reference
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["reason"] == "catastrophic_gap_stop"
        assert result["shares"] == 10

    def test_catastrophic_gap_no_fire_above_threshold(self):
        mgr = self._mgr()
        result = mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 175.0},  # -12.5% — above the 15% threshold
        )
        assert result is None

    def test_catastrophic_gap_falls_back_to_entry_price(self):
        mgr = self._mgr()
        result = mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "entry_price": 100.0, "shares": 10},  # no gap_reference_price
            {"last": 80.0},  # -20% from entry
        )
        assert result is not None
        assert result["reason"] == "catastrophic_gap_stop"

    def test_catastrophic_gap_disabled(self):
        mgr = self._mgr(catastrophic_gap_stop_enabled=False)
        result = mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 150.0},  # -25% but disabled
        )
        assert result is None

    def test_catastrophic_gap_custom_threshold(self):
        mgr = self._mgr(catastrophic_gap_stop_pct=0.10)
        result = mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 178.0},  # -11% — fires at 10% threshold
        )
        assert result is not None

    # ── catastrophic_gap_drop — observability cohort helper (config#846) ──

    def test_gap_drop_reports_fire_and_magnitude(self):
        mgr = self._mgr()  # default pct 0.15
        gap = mgr.catastrophic_gap_drop(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 169.0},  # -15.5% from reference
        )
        assert gap is not None
        assert gap["fired"] is True
        assert gap["reference"] == 200.0
        assert gap["current"] == 169.0
        assert gap["threshold"] == 0.15
        assert abs(gap["drop"] - (200.0 - 169.0) / 200.0) < 1e-9

    def test_gap_drop_reports_non_firing_counterfactual(self):
        # The whole point of the cohort: a watched position whose drop did
        # NOT reach the threshold still yields a magnitude for offline tuning.
        mgr = self._mgr()
        gap = mgr.catastrophic_gap_drop(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 185.0},  # -7.5% — below the 15% threshold
        )
        assert gap is not None
        assert gap["fired"] is False
        assert abs(gap["drop"] - 0.075) < 1e-9

    def test_gap_drop_never_fires_when_disabled_but_still_reports_drop(self):
        # Disabled: the counterfactual drop is still meaningful (would a run
        # WITH the stop enabled have helped?), but ``fired`` must stay False so
        # check_catastrophic_gap keeps its disabled-early-return behavior.
        mgr = self._mgr(catastrophic_gap_stop_enabled=False)
        gap = mgr.catastrophic_gap_drop(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 150.0},  # -25%
        )
        assert gap is not None
        assert gap["fired"] is False
        assert abs(gap["drop"] - 0.25) < 1e-9
        # And the acting path still returns None when disabled.
        assert mgr.check_catastrophic_gap(
            {"ticker": "AAPL", "gap_reference_price": 200.0, "shares": 10},
            {"last": 150.0},
        ) is None

    def test_gap_drop_none_without_valid_reference_or_price(self):
        mgr = self._mgr()
        # No live price.
        assert mgr.catastrophic_gap_drop(
            {"ticker": "AAPL", "gap_reference_price": 200.0}, {"last": None},
        ) is None
        # No reference and no entry_price.
        assert mgr.catastrophic_gap_drop(
            {"ticker": "AAPL"}, {"last": 100.0},
        ) is None

    def test_gap_drop_falls_back_to_entry_price(self):
        mgr = self._mgr()
        gap = mgr.catastrophic_gap_drop(
            {"ticker": "AAPL", "entry_price": 100.0},  # no gap_reference_price
            {"last": 82.0},
        )
        assert gap is not None
        assert gap["reference"] == 100.0
        assert abs(gap["drop"] - 0.18) < 1e-9


# ---------------------------------------------------------------------------
# build_stop_record — book-authority chokepoint (WDAY 2026-06-05 regression)
# ---------------------------------------------------------------------------


class TestBuildStopRecord:
    """The single chokepoint that stamps stop_kind from the book authority.

    Both producers (morning planner + intraday daemon) route through this, so
    a daemon-entered position can no longer silently omit stop_kind and inherit
    the alpha exit rules (the WDAY 2026-06-05 same-day-churn bug).
    """

    def _kwargs(self, **overrides):
        base = {
            "ticker": "WDAY", "entry_price": 146.48, "current_stop": 132.0,
            "trail_atr": 4.0, "atr_multiple": 2.0, "high_water": 146.48,
            "entry_date": "2026-06-05", "shares": 306,
        }
        base.update(overrides)
        return base

    def test_optimizer_stamps_catastrophic_gap_only(self):
        rec = build_stop_record(use_optimizer=True, **self._kwargs())
        assert rec["stop_kind"] == "catastrophic_gap_only"
        # No explicit reference → anchors on entry_price (no overnight gap for a
        # same-day entry; the gap stop guards a 15% crater from the fill).
        assert rec["gap_reference_price"] == 146.48

    def test_optimizer_uses_explicit_gap_reference(self):
        rec = build_stop_record(
            use_optimizer=True, gap_reference_price=150.0, **self._kwargs()
        )
        assert rec["gap_reference_price"] == 150.0

    def test_non_optimizer_stamps_alpha(self):
        rec = build_stop_record(use_optimizer=False, **self._kwargs())
        assert rec["stop_kind"] == "alpha"
        assert "gap_reference_price" not in rec

    def test_use_optimizer_is_required(self):
        # Forgetting the authority is a TypeError at construction (fail-loud),
        # never a silent default to the wrong (alpha) behavior.
        with pytest.raises(TypeError):
            build_stop_record(**self._kwargs())

    def test_extra_fields_passthrough(self):
        rec = build_stop_record(
            use_optimizer=True, entry_trade_id="abc-123", **self._kwargs()
        )
        assert rec["entry_trade_id"] == "abc-123"

    def test_wday_scenario_no_intraday_collapse_under_optimizer(self):
        """The exact WDAY 2026-06-05 churn must not recur.

        Bought $146.48; the day's HIGH was $151.50 (set before the entry); the
        price fell to $143.92 — 5.0% below the high but only ~1.7% below entry.
        Under optimizer authority the daemon runs ONLY the catastrophic gap
        stop, which must NOT fire (a true risk control sees a 1.7% drift, not a
        crater). The retired 5% collapse rule WOULD have force-sold it.
        """
        mgr = IntradayExitManager({"catastrophic_gap_stop_pct": 0.15})
        rec = build_stop_record(use_optimizer=True, **self._kwargs())
        price_state = {"last": 143.92, "high": 151.50}

        # Production path under the optimizer: gap stop only — no exit.
        assert mgr.check_catastrophic_gap(rec, price_state) is None

        # Document the retired behavior: the alpha collapse rule WOULD fire on
        # the 5%-from-high drop. This is what the fix suppresses.
        legacy = mgr.evaluate(rec, price_state)
        assert legacy is not None and legacy["reason"] == "intraday_collapse"

        # And a genuine 15% crater from the fill still trips the gap stop.
        crater = mgr.check_catastrophic_gap(rec, {"last": 124.0, "high": 151.50})
        assert crater is not None and crater["reason"] == "catastrophic_gap_stop"


# ---------------------------------------------------------------------------
# market_hours
# ---------------------------------------------------------------------------


class TestMarketHours:
    def test_weekday_is_trading(self):
        assert is_trading_day(date(2026, 4, 8)) is True

    def test_weekend_not_trading(self):
        assert is_trading_day(date(2026, 4, 11)) is False

    def test_holiday_not_trading(self):
        assert is_trading_day(date(2026, 12, 25)) is False

    def test_market_hours_during_session(self):
        # 10:30 AM ET on a Wednesday
        now = _ET.localize(datetime(2026, 4, 8, 10, 30))
        assert is_market_hours(now) is True

    def test_market_hours_before_open(self):
        now = _ET.localize(datetime(2026, 4, 8, 8, 0))
        assert is_market_hours(now) is False

    def test_market_hours_after_close(self):
        now = _ET.localize(datetime(2026, 4, 8, 17, 0))
        assert is_market_hours(now) is False

    def test_market_hours_at_close_boundary(self):
        """Default close is 16:00 ET — aligned with session_date (config#1610)."""
        assert is_market_hours(_ET.localize(datetime(2026, 4, 8, 15, 59))) is True
        assert is_market_hours(_ET.localize(datetime(2026, 4, 8, 16, 0))) is False
        assert is_market_hours(_ET.localize(datetime(2026, 4, 8, 16, 5))) is False

    def test_market_hours_weekend(self):
        now = _ET.localize(datetime(2026, 4, 11, 10, 30))
        assert is_market_hours(now) is False

    def test_market_hours_holiday(self):
        now = _ET.localize(datetime(2026, 12, 25, 10, 30))
        assert is_market_hours(now) is False

    def test_market_hours_naive_datetime(self):
        # Naive datetime should be localized
        now = datetime(2026, 4, 8, 10, 30)
        assert is_market_hours(now) is True


# ---------------------------------------------------------------------------
# notifier
# ---------------------------------------------------------------------------


class TestNotifier:
    """Daemon-side formatter tests. Primitive send/escape behavior is locked
    upstream in nousergon_lib.telegram's own test suite. These tests cover
    message-shape contracts and flow-doctor vs legacy routing.
    """

    def test_send_trade_alert_no_config(self, monkeypatch):
        # Deliberately exercises real send_message (see conftest autouse guard).
        from nousergon_lib.telegram import send_message as real_send_message
        monkeypatch.setattr("executor.notifier.send_message", real_send_message)
        with patch.dict("os.environ", {"ALPHA_ENGINE_SECRETS_SOURCE": "env"}, clear=True):
            with patch("executor.notifier.get_flow_doctor", return_value=None):
                assert send_trade_alert("BUY", "AAPL", 10, 150.0) is False

    @patch("executor.notifier.get_flow_doctor", return_value=None)
    @patch("executor.notifier.send_message", return_value=True)
    def test_send_trade_alert_success(self, mock_send, _mock_fd):
        assert send_trade_alert("BUY", "AAPL", 10, 150.0, "pullback", "daemon") is True
        mock_send.assert_called_once()

    @patch("executor.notifier.get_flow_doctor", return_value=None)
    @patch("executor.notifier.send_message", return_value=True)
    def test_send_trade_alert_message_format(self, mock_send, _mock_fd):
        send_trade_alert("BUY", "AAPL", 10, 150.0, "pullback", "daemon")
        msg = mock_send.call_args.args[0]
        assert "*BUY AAPL*" in msg
        assert "Shares: 10 @ $150.00" in msg
        assert "Trigger: pullback" in msg
        assert "Source: daemon" in msg

    @patch("executor.notifier.get_flow_doctor", return_value=None)
    @patch("executor.notifier.send_message", return_value=True)
    def test_send_trade_alert_unknown_action_uses_fallback_emoji(self, mock_send, _mock_fd):
        send_trade_alert("WEIRD", "AAPL", 10, 150.0)
        msg = mock_send.call_args.args[0]
        assert "*WEIRD AAPL*" in msg

    def test_send_daemon_status_no_config(self, monkeypatch):
        from nousergon_lib.telegram import send_message as real_send_message
        monkeypatch.setattr("executor.notifier.send_message", real_send_message)
        with patch.dict("os.environ", {"ALPHA_ENGINE_SECRETS_SOURCE": "env"}, clear=True):
            with patch("executor.notifier.get_flow_doctor", return_value=None):
                assert send_daemon_status("test") is False

    @patch("executor.notifier.get_flow_doctor", return_value=None)
    @patch("executor.notifier.send_message", return_value=True)
    def test_send_daemon_status_success(self, mock_send, _mock_fd):
        assert send_daemon_status("daemon started") is True
        mock_send.assert_called_once_with("daemon started")


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


class TestRetry:
    @patch("executor.retry.time.sleep")
    def test_succeeds_first_try(self, mock_sleep):
        @retry(max_attempts=3)
        def good():
            return "ok"
        assert good() == "ok"
        mock_sleep.assert_not_called()

    @patch("executor.retry.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep):
        count = {"n": 0}

        @retry(max_attempts=3, backoff_base=1)
        def flaky():
            count["n"] += 1
            if count["n"] < 3:
                raise ValueError("transient")
            return "ok"

        assert flaky() == "ok"
        assert count["n"] == 3

    @patch("executor.retry.time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        @retry(max_attempts=2, backoff_base=1)
        def always_fail():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError):
            always_fail()
