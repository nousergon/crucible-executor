"""Regression tests for the 2026-04-27 simulate-mode per-call cost fix.

Two changes pinned here:

1. ``load_config()`` caches its result for the process lifetime. The
   cache is cleared between tests via the autouse fixture so each test
   sees a clean parse, but within a single executor.run() loop the
   cache prevents repeated YAML parses (~20 ms/call savings).

2. ``_read_signals`` skips ``filter_buy_candidates_to_universe`` when
   ``simulate=True``. The backtester pre-filters at simulation-loop
   bootstrap; re-running the filter inside ``_read_signals`` would
   call ``universe_lib.list_symbols()`` per signal date — an ArcticDB
   round-trip the profile measured at ~424 ms/call.

If a future PR re-introduces either round trip in the simulate path,
these tests catch it before backtester budgets blow up again.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import executor.main as main_mod


@pytest.fixture(autouse=True)
def clear_load_config_cache():
    """Reset the module-level cache between tests so each test sees a
    clean parse. Production code never resets — the cache lives for the
    process lifetime."""
    main_mod._LOAD_CONFIG_CACHE = None
    yield
    main_mod._LOAD_CONFIG_CACHE = None


@pytest.fixture
def fake_risk_yaml(tmp_path, monkeypatch):
    """Write a minimal risk.yaml to a temp path and point
    ``get_config_path`` at it. Lets the cache-behavior tests run on a
    clean CI runner that has no real risk.yaml on disk.
    """
    p = tmp_path / "risk.yaml"
    p.write_text(
        "min_score_to_enter: 70\n"
        "max_position_pct: 0.05\n"
        "strategy:\n"
        "  exit_manager:\n"
        "    atr_period: 14\n"
    )
    monkeypatch.setattr(
        "executor.main.get_config_path", lambda: str(p),
    )
    return p


class TestLoadConfigCache:
    def test_load_config_caches_result(self, fake_risk_yaml):
        """First call reads the file; second call hits the cache."""
        cfg1 = main_mod.load_config()
        assert main_mod._LOAD_CONFIG_CACHE is not None
        # Returned dict is a deep copy, not the cached object
        assert cfg1 is not main_mod._LOAD_CONFIG_CACHE

        cfg2 = main_mod.load_config()
        assert cfg2 is not cfg1  # each call returns a fresh deepcopy
        assert cfg2 == cfg1

    def test_load_config_returns_independent_copies(self, fake_risk_yaml):
        """Mutating one returned dict must not pollute the cache or
        a subsequent caller."""
        cfg1 = main_mod.load_config()
        cfg1["min_score_to_enter"] = 999
        cfg1["strategy"]["exit_manager"]["atr_period"] = 99

        cfg2 = main_mod.load_config()
        assert cfg2["min_score_to_enter"] != 999, (
            "Cache pollution: mutating cfg1 changed the cached config"
        )
        assert cfg2["strategy"]["exit_manager"]["atr_period"] != 99, (
            "Cache pollution: nested-dict mutation leaked into cache"
        )

    def test_file_only_read_once_under_repeated_calls(self, fake_risk_yaml):
        """The YAML file should only be opened on first call."""
        # First call populates the cache
        main_mod.load_config()
        # Patch builtins.open: subsequent calls must NOT hit it.
        with patch("builtins.open") as mock_open:
            for _ in range(20):
                main_mod.load_config()
            assert mock_open.call_count == 0, (
                f"load_config() opened the file {mock_open.call_count}x "
                "under 20 repeated calls — cache is not effective"
            )


class TestSimulateModeSkipsArcticDBFilter:
    """In simulate mode, _read_signals must NOT call
    filter_buy_candidates_to_universe — the backtester pre-filters at
    simulation-loop bootstrap and re-running per call costs ~424 ms/call
    via ArcticDB list_symbols round-trip.

    Live mode (simulate=False) still calls the filter as defense-in-
    depth against universe drift.
    """

    def _minimal_signals_override(self):
        return {
            "date": "2026-04-25",
            "market_regime": "neutral",
            "sector_ratings": {},
            "enter": [],
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": [],
            "buy_candidates": [],
        }

    def test_simulate_skips_arcticdb_universe_filter(self):
        """_read_signals must NOT invoke filter_buy_candidates_to_universe
        when simulate=True."""
        with patch("executor.signal_reader.filter_buy_candidates_to_universe") as mock_filter:
            main_mod._read_signals(
                config={"signals_bucket": "test-bucket"},
                signals_bucket="test-bucket",
                run_date="2026-04-25",
                simulate=True,
                signals_override=self._minimal_signals_override(),
                conn=None,
            )
            assert mock_filter.call_count == 0, (
                "Simulate mode called filter_buy_candidates_to_universe — "
                "should be skipped (backtester pre-filters)"
            )

    def test_live_still_calls_arcticdb_universe_filter(self):
        """Live mode (simulate=False) must still run the defense-in-depth
        filter — universe drift between research-time and execution-time
        is real (TSM/ASML 2026-04-20 incident)."""
        # Use an injected signals_override so we don't need real S3.
        # Mock the filter to a passthrough so we just verify it was called.
        # Also passthrough the champion adapter (config#2366) — both the
        # pointer read (_read_signals calls load_champion_pointer directly
        # to decide whether to pay for a sector-map load) and the adapter
        # itself are real S3 round-trips unrelated to what this test pins,
        # and this suite deliberately avoids live AWS calls.
        with patch(
            "executor.signal_reader.filter_buy_candidates_to_universe",
            side_effect=lambda s, b: s,  # passthrough
        ) as mock_filter, patch(
            "executor.champion.load_champion_pointer",
            return_value={"schema_version": 1, "champion": "agentic", "promotion_source": "test"},
        ), patch(
            "executor.champion.apply_champion_selection",
            side_effect=lambda signals_raw, preds, **kw: (signals_raw, preds),
        ):
            # _read_signals does more in non-simulate (predictions read,
            # stale check, telegram) — conftest's autouse fixtures mock the
            # real S3/telegram side effects, so this runs to completion
            # without needing to swallow exceptions to mask a live send.
            main_mod._read_signals(
                config={
                    "signals_bucket": "test-bucket",
                    "coverage_admission_enabled": False,
                },
                signals_bucket="test-bucket",
                run_date="2026-04-25",
                simulate=False,
                signals_override=self._minimal_signals_override(),
                conn=None,
            )
            assert mock_filter.call_count == 1, (
                f"Live mode invoked filter {mock_filter.call_count} times "
                "— expected exactly 1"
            )
