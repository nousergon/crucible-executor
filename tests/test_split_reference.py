"""Reference bars on the basis the broker held (alpha-engine-config-I14285).

SNDK split 1:8 and the ledger journals it on 2026-11-20. The account held 262
PRE-split shares on 2026-11-17..19 and IB's $799.56 / $775.68 / $755.25 are correct
pre-split marks, but the ArcticDB bars for those days have since been back-adjusted.
Nothing here changes a bound: every test either shows the restated range letting a
correct mark through, or a wrong mark still breaching.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from executor import reconcile_audit, v2_fills
from executor.eod_reconcile import _detect_ib_mark_outside_range
from executor.reconcile_audit import audit_window
from executor.split_reference import reference_bars_on_trade_basis
from executor.v2_fills import ForwardSplit, v2_forward_splits
from tests.test_eod_reconcile import _run_up_to_closing_prices
from tests.test_reconcile_audit import _cfg, _seed_eod_with_positions
from tests.test_v2_fills import DAY_1120, JOURNAL_1120, _journal, _split, _store

RATIO = 8.0
SPLIT = ForwardSplit("SNDK", dt.date(2026, 11, 20), RATIO, "trader/ledger/journal/nous-ergon-paper/2026-11-20.json#split/SNDK/8.0")
DAYS = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"])


def _adjusted_bars() -> pd.DataFrame:
    """ArcticDB today: pre-split days divided by 8 (11-17 Low/High are $93.67 / $102.68)."""
    return pd.DataFrame(
        {
            "Close": [99.00, 99.655, 97.03, 94.39, 89.00, 90.00],
            "Low": [95.0, 93.67, 94.0, 92.0, 87.0, 88.0],
            "High": [101.0, 102.68, 100.0, 96.0, 91.0, 92.0],
        },
        index=DAYS,
    )


def _unadjusted_bars() -> pd.DataFrame:
    """The series as it stood on the day: pre-split prices up to 11-19, post-split from 11-20."""
    bars = _adjusted_bars()
    pre = bars.index < pd.Timestamp("2026-11-20")
    bars.loc[pre, ["Close", "Low", "High"]] = bars.loc[pre, ["Close", "Low", "High"]] * RATIO
    return bars


def _positions(mark: float, shares: int = 262) -> dict[str, dict[str, Any]]:
    return {"SNDK": {"shares": shares, "ib_market_value": mark * shares}}


def _range_of(bars: pd.DataFrame, day: str) -> tuple[float, float]:
    row = bars.loc[pd.Timestamp(day)]
    return float(row["Low"]), float(row["High"])


def _flags(bars: pd.DataFrame, mark: float, day: str = "2026-11-17") -> list[dict[str, Any]]:
    lo, hi = _range_of(bars, day)
    return _detect_ib_mark_outside_range(positions=_positions(mark), day_low={"SNDK": lo}, day_high={"SNDK": hi})


# ── the helper ───────────────────────────────────────────────────────────────


def test_adjusted_bars_before_a_journaled_split_are_restated_and_the_broker_mark_passes() -> None:
    bars = _adjusted_bars()
    restated, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [SPLIT])
    lo, hi = _range_of(restated, "2026-11-17")
    assert lo == pytest.approx(749.36)
    assert hi == pytest.approx(821.44)
    assert float(restated.loc["2026-11-17", "Close"]) == pytest.approx(797.24)
    assert info == {"ratio": RATIO, "split_effective": "2026-11-20", "journal_ref": SPLIT.journal_ref}
    # Before: the unchanged bars flag IB's correct $799.56 by the logged $182.6k.
    before = _flags(bars, 799.56)
    assert [f["ticker"] for f in before] == ["SNDK"]
    assert before[0]["mark_error_usd"] == pytest.approx(262 * (799.56 - 102.68))
    # After: the same mark, the same detector, restated reference.
    assert _flags(restated, 799.56) == []
    assert _flags(restated, 775.68, "2026-11-18") == []
    assert _flags(restated, 755.25, "2026-11-19") == []


def test_the_split_day_and_later_bars_are_not_touched_and_the_input_is_not_mutated() -> None:
    bars = _adjusted_bars()
    snapshot = bars.copy()
    restated, _ = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [SPLIT])
    pd.testing.assert_frame_equal(bars, snapshot)
    for day in ("2026-11-20", "2026-11-23"):
        pd.testing.assert_series_equal(restated.loc[pd.Timestamp(day)], bars.loc[pd.Timestamp(day)])


def test_without_a_journaled_split_the_same_day_still_breaches() -> None:
    bars = _adjusted_bars()
    same, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [])
    assert same is bars
    assert info is None
    assert len(_flags(same, 799.56)) == 1


def test_a_split_of_another_ticker_restates_nothing() -> None:
    bars = _adjusted_bars()
    other = ForwardSplit("AMD", SPLIT.effective, RATIO, "x")
    same, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [other])
    assert same is bars
    assert info is None


def test_unadjusted_source_bars_are_not_restated() -> None:
    bars = _unadjusted_bars()
    same, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [SPLIT])
    assert same is bars
    assert info is None
    assert _flags(same, 799.56) == []  # already on the broker's basis, as before


def test_a_mark_outside_the_restated_range_still_breaches() -> None:
    restated, _ = reference_bars_on_trade_basis("SNDK", "2026-11-17", _adjusted_bars(), [SPLIT])
    _, hi = _range_of(restated, "2026-11-17")
    low_flags = _flags(restated, 700.0)
    high_flags = _flags(restated, 830.0)
    assert [f["ticker"] for f in low_flags] == ["SNDK"] and low_flags[0]["mark_error_usd"] < 0
    assert [f["ticker"] for f in high_flags] == ["SNDK"] and high_flags[0]["mark_error_usd"] == pytest.approx(262 * (830.0 - hi))
    # The pre-split mark divided by the ratio is not accepted either: the share count is pre-split.
    assert len(_flags(restated, 799.56 / RATIO)) == 1


def test_a_split_on_or_before_the_run_date_is_not_a_later_split() -> None:
    bars = _adjusted_bars()
    for run_date in ("2026-11-20", "2026-11-23"):
        same, info = reference_bars_on_trade_basis("SNDK", run_date, bars, [SPLIT])
        assert same is bars and info is None


def test_a_missing_split_day_bar_is_not_evidence_of_adjustment() -> None:
    bars = _adjusted_bars().iloc[:4]  # the series ends on 11-19: no bar on or after the split day
    same, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [SPLIT])
    assert same is bars and info is None


def test_the_gap_test_is_the_documented_half_log_ratio() -> None:
    # |log(close[s-1]/close[s])| < log(ratio)/2: a gap just inside is adjusted, just outside is not.
    inside = _adjusted_bars()
    inside.loc[pd.Timestamp("2026-11-20"), "Close"] = 94.39 * 2.8  # log 2.8 < log(8)/2 = log 2.828
    assert reference_bars_on_trade_basis("SNDK", "2026-11-17", inside, [SPLIT])[1] is not None
    outside = _adjusted_bars()
    outside.loc[pd.Timestamp("2026-11-20"), "Close"] = 94.39 * 2.9
    assert reference_bars_on_trade_basis("SNDK", "2026-11-17", outside, [SPLIT])[1] is None


def test_a_close_only_series_is_restated_on_close() -> None:
    bars = _adjusted_bars()[["Close"]]
    restated, info = reference_bars_on_trade_basis("SNDK", "2026-11-17", bars, [SPLIT])
    assert info is not None
    assert float(restated.loc["2026-11-17", "Close"]) == pytest.approx(797.24)
    assert list(restated.columns) == ["Close"]


def test_two_later_splits_compound_and_say_so() -> None:
    days = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-20", "2026-11-23", "2026-11-24"])
    bars = pd.DataFrame({"Close": [10.0, 10.0, 10.0, 10.0, 10.0]}, index=days)
    s1 = ForwardSplit("T", dt.date(2026, 11, 20), 2.0, "j1")
    s2 = ForwardSplit("T", dt.date(2026, 11, 24), 3.0, "j2")
    restated, info = reference_bars_on_trade_basis("T", "2026-11-17", bars, [s2, s1])
    assert info is not None and info["ratio"] == 6.0 and len(info["splits"]) == 2  # pyright: ignore[reportArgumentType]
    assert list(restated["Close"]) == [60.0, 60.0, 30.0, 30.0, 10.0]


# ── the lookahead reader ─────────────────────────────────────────────────────


def test_forward_splits_find_the_split_journaled_after_the_run_date() -> None:
    store = _store({DAY_1120: JOURNAL_1120})
    got = v2_forward_splits("2026-11-17", through=dt.date(2026, 11, 24), store=store)
    assert got == [ForwardSplit("SNDK", dt.date(2026, 11, 20), 8.0, f"{v2_fills.journal_key(DAY_1120)}#split/SNDK/8.0")]


def test_forward_splits_exclude_the_run_date_itself_and_days_past_through() -> None:
    store = _store({DAY_1120: JOURNAL_1120})
    assert v2_forward_splits("2026-11-20", through=dt.date(2026, 11, 24), store=store) == []
    assert v2_forward_splits("2026-11-17", through=dt.date(2026, 11, 19), store=store) == []
    assert v2_forward_splits("2026-11-24", through=dt.date(2026, 11, 24), store=store) == []


def test_forward_splits_skip_a_split_with_no_stated_ratio_and_a_bad_run_date() -> None:
    no_ratio = _journal([_split("SNDK", 1834, ratio=None)], day=DAY_1120)
    assert v2_forward_splits("2026-11-17", through=dt.date(2026, 11, 24), store=_store({DAY_1120: no_ratio})) == []
    assert v2_forward_splits("not-a-date", store=_store({})) == []


def test_forward_splits_with_an_unreadable_env_give_none(tmp_path: Any) -> None:
    assert v2_forward_splits("2026-11-17", through=dt.date(2026, 11, 24), env_file=tmp_path / "absent.env") == []


# ── reconcile_audit's stale_position_close ───────────────────────────────────


def _audit(tmp_path: Any, forward: list[ForwardSplit], bars: pd.DataFrame) -> tuple[dict[str, Any], MagicMock]:
    """``audit_window``'s summary. The mocked re-reconcile changes nothing, so a day that is
    really stale ends unconverged; its summary rides the exception."""
    db = str(tmp_path / "t.db")
    stored = {"SNDK": {"closing_price": 797.24, "shares": 262}}
    _seed_eod_with_positions(db, "2026-11-17", 600.0, 0.10, 1.40, stored)
    lib = MagicMock()
    lib.read.return_value = SimpleNamespace(data=bars)
    with patch.object(reconcile_audit, "_spy_close", lambda d, c: 600.0), \
         patch("executor.price_cache._open_universe_library", lambda bucket: lib), \
         patch("executor.v2_fills.v2_forward_splits", lambda run_date, **kw: forward), \
         patch.object(reconcile_audit, "eod_run") as run_mock, \
         patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
         patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
        try:
            return audit_window(start="2026-11-17", end="2026-11-17", config=_cfg(db)), run_mock
        except reconcile_audit.ReconciliationUnconvergedError as exc:
            return exc.summary or {}, run_mock


def test_stale_position_close_no_longer_rewrites_a_correct_pre_split_close(tmp_path: Any) -> None:
    res, run_mock = _audit(tmp_path, [SPLIT], _adjusted_bars())
    assert res["corrected"] == []
    run_mock.assert_not_called()


def test_without_the_journaled_split_the_stale_close_is_still_flagged(tmp_path: Any) -> None:
    res, run_mock = _audit(tmp_path, [], _adjusted_bars())
    run_mock.assert_called_once()
    assert res["corrected"][0]["reason"] == "stale_position_close"
    assert res["corrected"][0]["before"]["stale_tickers"]["SNDK"] > 8000  # 797.24 vs 99.655: the logged 8,750bp class


def test_a_real_divergence_on_a_restated_day_is_still_flagged_and_names_the_restatement(tmp_path: Any) -> None:
    bars = _adjusted_bars()
    bars.loc[pd.Timestamp("2026-11-17"), "Close"] = 99.655 * 1.02  # settled really moved 200bp
    res, run_mock = _audit(tmp_path, [SPLIT], bars)
    run_mock.assert_called_once()
    leg = res["corrected"][0]["legs"][0]
    assert leg["leg"] == "position_close:SNDK"
    assert leg["settled"] == pytest.approx(797.24 * 1.02)
    assert leg["reference_restated"]["ratio"] == RATIO


# ── eod_reconcile's own load ─────────────────────────────────────────────────


class _Stop(Exception):
    pass


def test_eod_reconcile_prices_and_range_checks_on_the_restated_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    run_date = "2026-11-17"
    _run_up_to_closing_prices(monkeypatch, held_ticker="SNDK", frame=_adjusted_bars(), run_date=run_date)
    monkeypatch.setattr("executor.v2_fills.v2_forward_splits", lambda rd, **kw: [SPLIT])
    seen: dict[str, Any] = {}

    def _capture(*, positions: dict[str, Any], day_low: dict[str, float], day_high: dict[str, float]) -> list[Any]:
        seen.update(positions=positions, low=day_low["SNDK"], high=day_high["SNDK"])
        raise _Stop

    monkeypatch.setattr("executor.eod_reconcile._detect_ib_mark_outside_range", _capture)
    from executor.eod_reconcile import run

    with pytest.raises(_Stop):
        run(run_date=run_date)
    assert seen["low"] == pytest.approx(749.36) and seen["high"] == pytest.approx(821.44)
    assert seen["positions"]["SNDK"]["reference_restated"]["split_effective"] == "2026-11-20"


