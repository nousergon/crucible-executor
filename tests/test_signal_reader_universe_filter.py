"""
Tests for filter_buy_candidates_to_universe — the executor-side
defense-in-depth guardrail that drops buy_candidates whose tickers
aren't in the ArcticDB universe library.

Origin: 2026-04-20 — Research emitted buy/enter signals for TSM +
ASML despite those tickers being absent from every constituents.json
and from the ArcticDB universe library. Research's Layer-1 fix
(alpha-engine-research#41) stops the leak at source; this is the
caller-side net that catches any future leak or manual signals.json
edit that slips past.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from executor.signal_reader import filter_buy_candidates_to_universe


def _fake_universe(symbols: list[str]) -> MagicMock:
    """Build a MagicMock that mimics an ArcticDB library with list_symbols."""
    lib = MagicMock()
    lib.list_symbols.return_value = list(symbols)
    return lib


class TestFilterBuyCandidatesToUniverse:
    def test_drops_ticker_not_in_universe(self):
        signals = {
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"ticker": "TSM", "signal": "ENTER"},
                {"ticker": "MSFT", "signal": "ENTER"},
            ],
            "universe": [{"ticker": "AAPL", "signal": "HOLD"}],
        }
        fake_lib = _fake_universe(["AAPL", "MSFT", "GOOG"])
        with patch(
            "executor.price_cache._open_universe_library",
            return_value=fake_lib,
        ):
            result = filter_buy_candidates_to_universe(signals, "alpha-engine-research")

        tickers = [b["ticker"] for b in result["buy_candidates"]]
        assert "TSM" not in tickers
        assert tickers == ["AAPL", "MSFT"]

    def test_universe_list_unchanged(self):
        """EXIT/REDUCE/HOLD for existing holdings are not filtered."""
        signals = {
            "buy_candidates": [{"ticker": "TSM", "signal": "ENTER"}],
            "universe": [{"ticker": "TSM", "signal": "HOLD"}],  # existing holding
        }
        fake_lib = _fake_universe(["AAPL"])  # TSM absent
        with patch(
            "executor.price_cache._open_universe_library",
            return_value=fake_lib,
        ):
            result = filter_buy_candidates_to_universe(signals, "alpha-engine-research")

        assert result["buy_candidates"] == []  # dropped
        # universe list untouched — executor can still process EXIT/HOLD
        # for held positions, and per-ticker ArcticDB failures downstream
        # will surface as their own named errors.
        assert result["universe"] == [{"ticker": "TSM", "signal": "HOLD"}]

    def test_empty_buy_candidates_skips_check(self):
        """No buy_candidates → no ArcticDB call, identity return."""
        signals = {"buy_candidates": [], "universe": [{"ticker": "AAPL"}]}
        with patch(
            "executor.price_cache._open_universe_library"
        ) as open_lib:
            result = filter_buy_candidates_to_universe(signals, "x")
        open_lib.assert_not_called()
        assert result is signals  # identity

    def test_missing_buy_candidates_key_skips_check(self):
        """signals.json without a buy_candidates key → identity return."""
        signals = {"universe": [{"ticker": "AAPL"}]}
        with patch(
            "executor.price_cache._open_universe_library"
        ) as open_lib:
            result = filter_buy_candidates_to_universe(signals, "x")
        open_lib.assert_not_called()
        assert result is signals

    def test_all_tickers_in_universe_returns_input_unchanged(self):
        signals = {
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"ticker": "MSFT", "signal": "ENTER"},
            ],
        }
        fake_lib = _fake_universe(["AAPL", "MSFT", "GOOG"])
        with patch(
            "executor.price_cache._open_universe_library",
            return_value=fake_lib,
        ):
            result = filter_buy_candidates_to_universe(signals, "x")
        # No drops, no mutation
        assert [b["ticker"] for b in result["buy_candidates"]] == ["AAPL", "MSFT"]

    def test_arcticdb_error_skips_filter_logs_warning(self, caplog):
        """If ArcticDB is unreachable, skip the filter — don't block trading."""
        signals = {"buy_candidates": [{"ticker": "TSM"}, {"ticker": "AAPL"}]}
        with patch(
            "executor.price_cache._open_universe_library",
            side_effect=RuntimeError("S3 unreachable"),
        ):
            import logging
            with caplog.at_level(logging.WARNING):
                result = filter_buy_candidates_to_universe(signals, "x")
        # Filter skipped — all entries pass through
        assert [b["ticker"] for b in result["buy_candidates"]] == ["TSM", "AAPL"]
        assert any(
            "Skipping buy-candidate universe filter" in r.message
            for r in caplog.records
        )

    def test_entry_without_ticker_key_ignored_gracefully(self):
        """Malformed entry with no ticker field → dropped silently."""
        signals = {
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"signal": "ENTER"},  # no ticker
                {"ticker": "TSM", "signal": "ENTER"},
            ]
        }
        fake_lib = _fake_universe(["AAPL"])
        with patch(
            "executor.price_cache._open_universe_library",
            return_value=fake_lib,
        ):
            result = filter_buy_candidates_to_universe(signals, "x")
        # AAPL kept, TSM dropped, malformed entry dropped
        assert [b["ticker"] for b in result["buy_candidates"]] == ["AAPL"]

    def test_caller_dict_not_mutated(self):
        """filter returns shallow copy when changes made — doesn't mutate input."""
        signals = {
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"ticker": "TSM", "signal": "ENTER"},
            ],
        }
        original_candidates = list(signals["buy_candidates"])
        fake_lib = _fake_universe(["AAPL"])
        with patch(
            "executor.price_cache._open_universe_library",
            return_value=fake_lib,
        ):
            _ = filter_buy_candidates_to_universe(signals, "x")
        # Caller's dict unchanged
        assert signals["buy_candidates"] == original_candidates
