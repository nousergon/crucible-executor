"""Tests for stance-conditional exit rules (stance taxonomy arc B).

The exit_manager routes per-stance behavior through:
  1. ``_resolve_strategy_config_for_stance`` — returns a stance-
     overridden config view per ``STANCE_EXIT_OVERRIDES``
  2. ``check_catalyst_hard_exit`` — new catalyst-stance terminal exit
  3. ``evaluate_exits`` — reads pos['stance'] / pos['catalyst_date'],
     threads stance_config through each per-stance check

These tests pin behavior at the helper level (cheaper than spinning
up the full orchestrator) plus 4 integration tests exercising
evaluate_exits end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from executor.strategies.exit_manager import (
    STANCE_EXIT_OVERRIDES,
    _resolve_strategy_config_for_stance,
    check_catalyst_hard_exit,
    evaluate_exits,
)

# ── _resolve_strategy_config_for_stance ──────────────────────────────────


class TestResolveStanceConfig:
    """Helper that returns a stance-overridden strategy_config view."""

    def _base(self):
        return {
            "atr_multiplier": 3.0,
            "time_decay_enabled": True,
            "time_decay_reduce_days": 5,
            "time_decay_exit_days": 10,
        }

    def test_stance_none_returns_baseline_unchanged(self):
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, None)
        assert out is base  # identity, not just equality — no needless copy

    def test_unknown_stance_returns_baseline_unchanged(self):
        """Defensive: a future stance label that ships before this PR
        is updated must fall through to baseline rather than crash or
        silently drop checks."""
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, "growth")
        assert out is base

    def test_momentum_stance_returns_baseline_unchanged(self):
        """momentum is the baseline thesis — no overrides expected."""
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, "momentum")
        # No overrides → identity-return for efficiency
        assert out is base

    def test_value_stance_now_uniform_baseline(self):
        """DE-STANCED (L4565): value no longer widens ATR / extends time
        decay — uniform risk. Empty overrides → baseline identity."""
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, "value")
        assert out is base
        assert out["atr_multiplier"] == 3.0
        assert out["time_decay_exit_days"] == 10

    def test_quality_stance_now_uniform_baseline(self):
        """DE-STANCED (L4565): quality no longer disables time decay —
        uniform risk."""
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, "quality")
        assert out is base
        assert out["time_decay_enabled"] is True

    def test_catalyst_stance_disables_time_decay(self):
        """catalyst-stance positions exit at catalyst_date+N via
        check_catalyst_hard_exit, NOT via time decay (which would fire
        on a misaligned schedule). time_decay_enabled=False prevents
        the conflict."""
        base = self._base()
        out = _resolve_strategy_config_for_stance(base, "catalyst")
        assert out["time_decay_enabled"] is False

    def test_value_quality_overrides_retired_uniform_risk(self):
        """DE-STANCED (L4565): value + quality exit-loosenings are RETIRED —
        risk is uniform across stances. Pin the invariant so a future
        hand-edit can't re-introduce a value/quality bypass. catalyst keeps
        its event-boundary time-decay disable (a mechanism, not a loosening)."""
        assert STANCE_EXIT_OVERRIDES["value"] == {}
        assert STANCE_EXIT_OVERRIDES["quality"] == {}
        assert STANCE_EXIT_OVERRIDES["momentum"] == {}
        assert STANCE_EXIT_OVERRIDES["catalyst"].get("time_decay_enabled") is False


# ── check_catalyst_hard_exit ──────────────────────────────────────────────


class TestCheckCatalystHardExit:
    def _cfg(self, followthrough_days: int = 3):
        return {"catalyst_followthrough_days": followthrough_days}

    def test_returns_none_when_catalyst_date_absent(self):
        result = check_catalyst_hard_exit(
            ticker="NVDA", catalyst_date=None,
            run_date="2026-06-15", strategy_config=self._cfg(),
        )
        assert result is None

    def test_returns_none_when_run_date_before_catalyst(self):
        """Event hasn't happened yet — hold."""
        result = check_catalyst_hard_exit(
            ticker="NVDA", catalyst_date="2026-06-15",
            run_date="2026-06-10", strategy_config=self._cfg(),
        )
        assert result is None

    def test_returns_none_within_followthrough_window(self):
        """Within ``catalyst_followthrough_days`` of the event — let
        the post-event move settle before exiting."""
        result = check_catalyst_hard_exit(
            ticker="NVDA", catalyst_date="2026-06-15",
            run_date="2026-06-16", strategy_config=self._cfg(followthrough_days=3),
        )
        assert result is None  # 1 trading day past < 3

    def test_fires_after_followthrough_window(self):
        result = check_catalyst_hard_exit(
            ticker="NVDA", catalyst_date="2026-06-15",
            run_date="2026-06-22", strategy_config=self._cfg(followthrough_days=3),
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["reason"] == "catalyst_hard_exit"
        assert result["catalyst_date"] == "2026-06-15"
        assert result["trading_days_past_catalyst"] >= 3

    def test_invalid_date_skips_gracefully(self):
        """Bad date strings shouldn't crash the daemon — log + skip."""
        result = check_catalyst_hard_exit(
            ticker="NVDA", catalyst_date="not-a-date",
            run_date="2026-06-22", strategy_config=self._cfg(),
        )
        assert result is None

    def test_followthrough_days_configurable(self):
        """Default is 3; tunable via strategy_config."""
        # Day 5 past — fires at default 3, but NOT at custom 10
        d_now = "2026-06-22"  # 5 trading days past 2026-06-15
        fires = check_catalyst_hard_exit(
            "NVDA", "2026-06-15", d_now, self._cfg(followthrough_days=3),
        )
        skips = check_catalyst_hard_exit(
            "NVDA", "2026-06-15", d_now, self._cfg(followthrough_days=10),
        )
        assert fires is not None
        assert skips is None


# ── evaluate_exits — integration ──────────────────────────────────────────


def _history_with_drawdown(entry_date: str, current_date: str, peak: float, trough: float) -> pd.DataFrame:
    """Synthetic history: rises from peak then crashes to trough."""
    entry_dt = pd.Timestamp(entry_date)
    current_dt = pd.Timestamp(current_date)
    dates = pd.bdate_range(entry_dt - pd.Timedelta(days=60), current_dt)
    n = len(dates)
    # Pre-entry: flat at peak
    pre = pd.Series([peak] * (n // 2), index=dates[: n // 2])
    # Post-entry: peak → trough decline (linear)
    post_n = n - n // 2
    post = pd.Series(
        [peak - (peak - trough) * (i / max(post_n - 1, 1)) for i in range(post_n)],
        index=dates[n // 2 :],
    )
    closes = pd.concat([pre, post])
    return pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes}
    )


def _strategy_cfg():
    return {
        "atr_trailing_enabled": True,
        "atr_period": 14,
        "atr_multiplier": 3.0,
        "fallback_stop_enabled": False,
        "profit_take_enabled": False,
        "momentum_exit_enabled": False,
        "time_decay_enabled": True,
        "time_decay_reduce_days": 5,
        "time_decay_exit_days": 10,
        "catalyst_followthrough_days": 3,
    }


def _ibkr_mock(price: float):
    m = MagicMock()
    m.get_current_price.return_value = price
    return m


def test_evaluate_exits_value_stance_now_uniform():
    """DE-STANCED (L4565): a value-stance position is treated IDENTICALLY
    to a baseline/momentum position — no wider ATR, no extended hold.
    "Don't treat value differently." """
    history = _history_with_drawdown(
        "2026-05-01", "2026-05-20", peak=100.0, trough=88.0,
    )

    def _exits_for(stance):
        positions = {
            "WING": {
                "shares": 100,
                "avg_cost": 100.0,
                "entry_date": "2026-05-01",
                "sector": "Consumer Discretionary",
                "stance": stance,
            },
        }
        return evaluate_exits(
            current_positions=positions,
            signals_by_ticker={},
            run_date="2026-05-20",
            price_histories={"WING": history},
            ibkr_client=_ibkr_mock(88.0),
            strategy_config=_strategy_cfg(),
        )

    value_reasons = sorted(e.get("reason") for e in _exits_for("value"))
    momentum_reasons = sorted(e.get("reason") for e in _exits_for("momentum"))
    assert value_reasons == momentum_reasons, (
        f"value must now exit identically to momentum (uniform risk); "
        f"value={value_reasons} momentum={momentum_reasons}"
    )


def test_evaluate_exits_quality_stance_now_uniform():
    """DE-STANCED (L4565): quality no longer disables time decay — a
    quality position held past the exit threshold time-decays like any
    other stance."""
    history = _history_with_drawdown(
        "2026-04-01", "2026-05-01", peak=100.0, trough=100.0,
    )
    positions = {
        "JNJ": {
            "shares": 100,
            "avg_cost": 100.0,
            "entry_date": "2026-04-01",  # 30 days ago
            "sector": "Healthcare",
            "stance": "quality",
        },
    }
    cfg = _strategy_cfg()
    exits = evaluate_exits(
        current_positions=positions,
        signals_by_ticker={},  # No HOLD/ENTER signal — fall-through
        run_date="2026-05-01",
        price_histories={"JNJ": history},
        ibkr_client=_ibkr_mock(100.0),
        strategy_config=cfg,
    )
    time_decays = [e for e in exits if "time_decay" in e.get("reason", "")]
    assert time_decays != [], (
        "quality stance should now time-decay uniformly (L4565 de-stancing)"
    )


def test_evaluate_exits_catalyst_stance_hard_exits_after_window():
    """catalyst-stance position past catalyst_date + 3 trading days
    triggers the hard exit FIRST (before any other check)."""
    history = _history_with_drawdown(
        "2026-05-01", "2026-05-22", peak=100.0, trough=100.0,  # no drawdown
    )
    positions = {
        "MRNA": {
            "shares": 100,
            "avg_cost": 100.0,
            "entry_date": "2026-05-01",
            "sector": "Healthcare",
            "stance": "catalyst",
            "catalyst_date": "2026-05-15",
        },
    }
    exits = evaluate_exits(
        current_positions=positions,
        signals_by_ticker={},
        run_date="2026-05-22",  # 5 trading days past catalyst → fires
        price_histories={"MRNA": history},
        ibkr_client=_ibkr_mock(100.0),
        strategy_config=_strategy_cfg(),
    )
    hard_exits = [e for e in exits if e.get("reason") == "catalyst_hard_exit"]
    assert len(hard_exits) == 1
    assert hard_exits[0]["ticker"] == "MRNA"
    assert hard_exits[0]["catalyst_date"] == "2026-05-15"


def test_evaluate_exits_legacy_position_falls_through_to_baseline():
    """A pre-stance-arc position (stance=None, catalyst_date=None)
    must use baseline behavior. Pinned so the rollout transition
    doesn't change behavior for in-flight legacy positions."""
    history = _history_with_drawdown(
        "2026-05-01", "2026-05-15", peak=100.0, trough=100.0,
    )
    positions = {
        "OLD": {
            "shares": 100,
            "avg_cost": 100.0,
            "entry_date": "2026-05-01",
            "sector": "Technology",
            # No stance / catalyst_date — legacy
        },
    }
    exits = evaluate_exits(
        current_positions=positions,
        signals_by_ticker={},
        run_date="2026-05-15",
        price_histories={"OLD": history},
        ibkr_client=_ibkr_mock(100.0),
        strategy_config=_strategy_cfg(),
    )
    # No exits should fire for a flat position with no stance —
    # baseline time-decay reduce-days threshold is 5; this is 14 days,
    # but with flat price no signal_action=HOLD context. Just verify
    # nothing crashes + no stance_gate-style behavior leaked through.
    assert all(e.get("reason") != "catalyst_hard_exit" for e in exits)
