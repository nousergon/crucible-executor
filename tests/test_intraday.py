"""Tests to close coverage gaps: entry_triggers, intraday_exit_manager,
market_hours, notifier, retry.

All pure-logic tests — no IB Gateway or S3 dependencies.
"""

from datetime import date, datetime, time
from unittest.mock import MagicMock, patch

import pytz
import pytest

from executor.entry_triggers import EntryTriggerEngine
from executor.intraday_exit_manager import IntradayExitManager
from executor.market_hours import is_market_hours, is_trading_day
from executor.notifier import _escape_markdown, _send_telegram, send_daemon_status, send_trade_alert
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

    def test_pullback_no_fire(self):
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

    def test_support_broken_no_fire(self):
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

    def test_disabled_trigger(self):
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
        old_date = (date.today().isoformat()[:8] + "01").replace(
            date.today().isoformat()[:8], (date.today().replace(day=1) if date.today().day > 5 else date.today()).isoformat()[:8]
        )
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
    def test_escape_markdown(self):
        assert _escape_markdown("hello_world") == "hello-world"
        assert _escape_markdown("code`block`") == "code'block'"
        assert _escape_markdown("[link]") == "(link)"

    @patch("executor.notifier.requests.post")
    def test_send_telegram_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        assert _send_telegram("token", "123", "hello") is True

    @patch("executor.notifier.requests.post")
    def test_send_telegram_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=400, text="Bad Request")
        assert _send_telegram("token", "123", "hello") is False

    @patch("executor.notifier.requests.post", side_effect=Exception("timeout"))
    def test_send_telegram_exception(self, mock_post):
        assert _send_telegram("token", "123", "hello") is False

    def test_send_trade_alert_no_config(self):
        with patch.dict("os.environ", {}, clear=True):
            assert send_trade_alert("BUY", "AAPL", 10, 150.0) is False

    @patch("executor.notifier._send_telegram", return_value=True)
    def test_send_trade_alert_success(self, mock_send):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}):
            assert send_trade_alert("BUY", "AAPL", 10, 150.0, "pullback", "daemon") is True
            mock_send.assert_called_once()

    def test_send_daemon_status_no_config(self):
        with patch.dict("os.environ", {}, clear=True):
            assert send_daemon_status("test") is False

    @patch("executor.notifier._send_telegram", return_value=True)
    def test_send_daemon_status_success(self, mock_send):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}):
            assert send_daemon_status("daemon started") is True


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
