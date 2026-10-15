"""Under the optimizer, the daemon's entry bracket is a hard-risk backstop set
beyond the catastrophic gap stop, never the retired 2x ATR alpha trail."""
from unittest.mock import MagicMock, patch

from executor import daemon


def _run_entry(use_optimizer, strategy_config):
    captured = {}

    def fake_place(ibkr, ticker, side, shares, label, use_bracket=False, bracket_kwargs=None):
        captured["use_bracket"] = use_bracket
        captured["bracket_kwargs"] = bracket_kwargs
        return {"status": "Rejected", "ib_order_id": None}

    entry = {"ticker": "SMCI", "shares": 10, "atr_value": 2.37, "current_price": 34.5}
    ob = MagicMock()
    with patch.object(daemon, "_place_order_with_retry", side_effect=fake_place), \
         patch.object(daemon, "_validate_buy_not_duplicate", return_value=True):
        daemon._execute_entry(
            MagicMock(), MagicMock(), ob, entry, {"last": 34.5}, "test",
            "2026-10-16", strategy_config, False, monitor=None,
            use_optimizer=use_optimizer,
        )
    return captured


def test_optimizer_book_uses_wide_stop_backstop():
    got = _run_entry(True, {"catastrophic_gap_stop_pct": 0.15})
    assert got["use_bracket"] is True
    assert got["bracket_kwargs"]["stop_pct"] == 0.20
    assert "atr_multiple" not in got["bracket_kwargs"]


def test_legacy_book_keeps_atr_trail():
    got = _run_entry(False, {"bracket_trail_atr_multiple": 2.0})
    assert got["bracket_kwargs"] == {"atr_value": 2.37, "atr_multiple": 2.0}


def test_bracket_disabled_places_no_stop_under_optimizer():
    got = _run_entry(True, {"bracket_stop_enabled": False})
    assert not got["use_bracket"]
    assert got["bracket_kwargs"] is None
