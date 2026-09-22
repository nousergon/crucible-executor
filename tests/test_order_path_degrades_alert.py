"""Order-path degrades that alter the book must reach a human.

`alpha-engine-config-I11369`'s sweep deliverable. PR570 fixed the two
optimizer sites; this covers the residue the sweep found — handlers that log
at WARNING/ERROR and publish nothing, which the
`.debug-swallow-allowlist.yaml` guard does not catch because it only sees
DEBUG-level-or-`pass` swallows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import executor.notifier as notifier
from executor.daemon import _resolve_pending_sell_shares
from executor.main import _alert_regime_leg_blind


@pytest.fixture
def published(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        notifier, "publish_ops_alert",
        lambda message, **kw: calls.append((message, kw)),
    )
    return calls


class TestShortSellGuardBlind:
    """The guard subtracts in-flight sells from held, so an unknown value
    falling back to 0 OVERSTATES what may be sold — it makes the check pass.
    That is the direction of the 2026-04-22 PFE incident the guard exists to
    prevent, and until the sweep it happened on a WARNING log alone.
    """

    def test_the_happy_path_publishes_nothing(self, published):
        ibkr = MagicMock()
        ibkr.get_open_sell_shares.return_value = 77
        assert _resolve_pending_sell_shares(ibkr, "PFE", phase="urgent-exit") == 77
        assert published == [], "a working guard is not an event"

    def test_a_blind_guard_pages_and_still_returns_zero(self, published):
        ibkr = MagicMock()
        ibkr.get_open_sell_shares.side_effect = ConnectionError("IB gone")

        # The fallback is deliberately UNCHANGED — refusing the sell would
        # block an urgent risk exit on any transient IBKR error.
        assert _resolve_pending_sell_shares(ibkr, "PFE", phase="urgent-exit") == 0

        assert len(published) == 1
        message, kw = published[0]
        assert kw["severity"] == "error"
        assert "PFE" in kw["dedup_key"] and "urgent-exit" in kw["dedup_key"]
        assert "BLIND" in message
        assert "ConnectionError" in message

    def test_the_two_phases_dedup_separately(self, published):
        ibkr = MagicMock()
        ibkr.get_open_sell_shares.side_effect = ConnectionError("IB gone")
        _resolve_pending_sell_shares(ibkr, "PFE", phase="urgent-exit")
        _resolve_pending_sell_shares(ibkr, "PFE", phase="intraday")
        keys = {kw["dedup_key"] for _, kw in published}
        assert len(keys) == 2, (
            "the urgent-exit and intraday guards are different call sites; "
            "collapsing them would hide the second"
        )

    def test_an_alert_failure_never_breaks_the_guard(self, monkeypatch):
        # The order path is load-bearing; the alert is not.
        monkeypatch.setattr(
            notifier, "publish_ops_alert",
            MagicMock(side_effect=RuntimeError("SNS down")),
        )
        ibkr = MagicMock()
        ibkr.get_open_sell_shares.side_effect = ConnectionError("IB gone")
        assert _resolve_pending_sell_shares(ibkr, "PFE", phase="intraday") == 0


class TestRegimeLegBlind:
    """Conditioned on the leg's enable flag, not emitted unconditionally.

    With the flag off the leg is observe-only and the read failure changes
    nothing; paging would manufacture a daily false page.
    """

    def test_it_pages_naming_the_leg_and_the_run_date(self, published):
        _alert_regime_leg_blind("forced_bear", ValueError("bad json"), "2026-09-22")
        assert len(published) == 1
        message, kw = published[0]
        assert kw["severity"] == "error"
        assert kw["dedup_key"] == "regime-leg-blind-forced_bear-2026-09-22"
        assert "forced_bear" in message and "ValueError" in message

    def test_each_leg_dedups_separately_within_a_run_date(self, published):
        _alert_regime_leg_blind("forced_bear", ValueError("x"), "2026-09-22")
        _alert_regime_leg_blind("drawdown", ValueError("x"), "2026-09-22")
        assert len({kw["dedup_key"] for _, kw in published}) == 2

    def test_an_alert_failure_is_non_blocking(self, monkeypatch):
        monkeypatch.setattr(
            notifier, "publish_ops_alert",
            MagicMock(side_effect=RuntimeError("SNS down")),
        )
        _alert_regime_leg_blind("drawdown", ValueError("x"), "2026-09-22")


def test_the_enable_flag_gates_the_regime_alert():
    """The call sites are guarded by the leg's own flag.

    Source-level assertion rather than a planner run: the planner needs S3,
    IBKR and a full config to reach these lines, and what matters is the
    guard existing at all — an unconditional call would page daily on an
    observe-only leg.
    """
    import inspect

    import executor.main as m

    src = inspect.getsource(m)
    for flag, leg in (
        ("regime_forced_bear_enabled", "forced_bear"),
        ("drawdown_regime_enabled", "drawdown"),
    ):
        needle = (
            f'if config.get("{flag}", False):\n'
            f'                    _alert_regime_leg_blind("{leg}"'
        )
        assert needle in src, f"{leg} alert is not gated on {flag}"
