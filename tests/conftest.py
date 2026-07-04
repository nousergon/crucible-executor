"""Test fixtures + sys.path setup.

Pins ``ALPHA_ENGINE_SECRETS_SOURCE=env`` for the test process so
``nousergon_lib.secrets.get_secret()`` (post 2026-05-12 .env→SSM
migration, PR 6 of the arc) reads from monkeypatched env vars only —
never the real SSM Parameter Store.
"""
import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")


@pytest.fixture(autouse=True)
def _isolate_secrets_from_ssm(monkeypatch):
    """Re-pin ``ALPHA_ENGINE_SECRETS_SOURCE=env`` per test + clear the
    per-process secret cache. See
    ``alpha-engine-docs/private/env-to-ssm-260512.md`` § Risks.
    """
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    try:
        from nousergon_lib.secrets import clear_cache
    except ImportError:
        yield
        return
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _block_real_alert_publish(monkeypatch):
    """Stub ``nousergon_lib.alerts.publish`` so NO test fans out a real
    SNS / Telegram operator alert.

    Without this, any test exercising a code path that calls
    ``alerts.publish(..., sns=True, telegram=True)`` pages the operator for
    real. Concretely: ``test_optimizer_shadow.py``'s baseline fixture (a
    near-all-cash book) trips the turnover-governor large-move flag added in
    #237, which fired a live WARN to Telegram + SNS on every suite run
    (``run_date=2026-05-11``, observed 2026-06-07). Tests assert on the call
    inputs, not on real delivery. ``optimizer_shadow`` imports the symbol
    lazily as ``from nousergon_lib import alerts as _alerts`` and calls
    ``_alerts.publish`` at runtime, so patching the module attribute here
    intercepts it. See ROADMAP L4566; mirrored by a cross-repo guard in
    ``nousergon_lib.alerts.publish`` (PYTEST_CURRENT_TEST).
    """
    try:
        from nousergon_lib import alerts
    except ImportError:
        yield
        return
    monkeypatch.setattr(alerts, "publish", MagicMock(name="alerts.publish"))
    yield


@pytest.fixture(autouse=True)
def _block_real_telegram_send(monkeypatch):
    """Stub ``executor.notifier.send_message`` so NO test pages a real
    Telegram chat.

    Sibling guard to ``_block_real_alert_publish`` above: that fixture blocks
    ``nousergon_lib.alerts.publish``, but ``executor/notifier.py`` also calls
    ``nousergon_lib.telegram.send_message`` directly (imported by name, so the
    patch target is ``executor.notifier.send_message``) from
    ``send_daemon_status``/``send_trade_alert``. Confirmed live 2026-07-03:
    ``test_perf_simulate_mode.py``'s ``simulate=False`` coverage exercised the
    real (unmocked) path and delivered 6 live "Order book written" + 1 live
    "Stale signals" alert to the production Telegram channel (config#1692).
    """
    try:
        from executor import notifier
    except ImportError:
        yield
        return
    monkeypatch.setattr(
        notifier, "send_message", MagicMock(name="notifier.send_message", return_value=True)
    )
    yield
