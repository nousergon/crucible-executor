"""Tests for signal_reader.patch_unknown_sectors_with_constituents.

Defense-in-depth complement to research's signals.json sector preflight
(alpha-engine-research#126). The 2026-05-04 EOG/NVT incident: research
wrote signals.json with sector="Unknown" because constituents data hadn't
loaded yet, the morning planner consumed v1, and the daemon's intraday
fills wrote "Unknown" into trades.db. The research preflight blocks the
emission; this patch catches the escape if it ever happens.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def signals_raw_unknown_eog_nvt():
    return {
        "buy_candidates": [
            {"ticker": "EOG", "signal": "ENTER", "sector": "Unknown"},
            {"ticker": "NVT", "signal": "ENTER", "sector": "Unknown"},
            {"ticker": "CTAS", "signal": "ENTER", "sector": "Industrials"},
        ],
        "universe": [
            {"ticker": "EOG", "signal": "ENTER", "sector": "Unknown"},
            {"ticker": "NVT", "signal": "ENTER", "sector": "Unknown"},
            {"ticker": "CTAS", "signal": "ENTER", "sector": "Industrials"},
            {"ticker": "META", "signal": "HOLD", "sector": "Unknown"},
        ],
    }


def test_patches_unknown_with_constituents_map(signals_raw_unknown_eog_nvt):
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    constituents = {"EOG": "Energy", "NVT": "Industrials"}

    with patch(
        "executor.eod_reconcile._load_constituents_sector_map",
        return_value=constituents,
    ):
        n = patch_unknown_sectors_with_constituents(signals_raw_unknown_eog_nvt, "bucket")

    assert n == 4

    bc_by_t = {s["ticker"]: s for s in signals_raw_unknown_eog_nvt["buy_candidates"]}
    uni_by_t = {s["ticker"]: s for s in signals_raw_unknown_eog_nvt["universe"]}

    assert bc_by_t["EOG"]["sector"] == "Energy"
    assert bc_by_t["NVT"]["sector"] == "Industrials"
    assert bc_by_t["CTAS"]["sector"] == "Industrials"
    assert uni_by_t["EOG"]["sector"] == "Energy"
    assert uni_by_t["NVT"]["sector"] == "Industrials"
    assert uni_by_t["META"]["sector"] == "Unknown"


def test_no_patch_when_all_sectors_resolved():
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    signals_raw = {
        "buy_candidates": [{"ticker": "EOG", "signal": "ENTER", "sector": "Energy"}],
        "universe": [{"ticker": "EOG", "signal": "ENTER", "sector": "Energy"}],
    }

    with patch(
        "executor.eod_reconcile._load_constituents_sector_map",
        side_effect=AssertionError("should not load constituents on clean path"),
    ):
        n = patch_unknown_sectors_with_constituents(signals_raw, "bucket")

    assert n == 0


def test_constituents_miss_leaves_unknown_in_place():
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    signals_raw = {
        "buy_candidates": [{"ticker": "RARE", "signal": "ENTER", "sector": "Unknown"}],
        "universe": [{"ticker": "RARE", "signal": "ENTER", "sector": "Unknown"}],
    }

    with patch(
        "executor.eod_reconcile._load_constituents_sector_map",
        return_value={"OTHER": "Healthcare"},
    ):
        n = patch_unknown_sectors_with_constituents(signals_raw, "bucket")

    assert n == 0
    assert signals_raw["buy_candidates"][0]["sector"] == "Unknown"


def test_empty_constituents_map_is_safe():
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    signals_raw = {
        "buy_candidates": [{"ticker": "EOG", "signal": "ENTER", "sector": "Unknown"}],
        "universe": [{"ticker": "EOG", "signal": "ENTER", "sector": "Unknown"}],
    }

    with patch("executor.eod_reconcile._load_constituents_sector_map", return_value={}):
        n = patch_unknown_sectors_with_constituents(signals_raw, "bucket")

    assert n == 0
    assert signals_raw["buy_candidates"][0]["sector"] == "Unknown"


def test_missing_or_none_sector_is_treated_as_unknown():
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    signals_raw = {
        "buy_candidates": [
            {"ticker": "A", "signal": "ENTER"},
            {"ticker": "B", "signal": "ENTER", "sector": None},
            {"ticker": "C", "signal": "ENTER", "sector": ""},
        ],
        "universe": [],
    }

    with patch(
        "executor.eod_reconcile._load_constituents_sector_map",
        return_value={"A": "Healthcare", "B": "Energy", "C": "Industrials"},
    ):
        n = patch_unknown_sectors_with_constituents(signals_raw, "bucket")

    assert n == 3
    assert signals_raw["buy_candidates"][0]["sector"] == "Healthcare"
    assert signals_raw["buy_candidates"][1]["sector"] == "Energy"
    assert signals_raw["buy_candidates"][2]["sector"] == "Industrials"


def test_empty_signals_raw_is_safe():
    from executor.signal_reader import patch_unknown_sectors_with_constituents

    with patch(
        "executor.eod_reconcile._load_constituents_sector_map",
        side_effect=AssertionError("should not load constituents on empty path"),
    ):
        assert patch_unknown_sectors_with_constituents({}, "bucket") == 0
        assert patch_unknown_sectors_with_constituents(
            {"buy_candidates": [], "universe": []}, "bucket"
        ) == 0
