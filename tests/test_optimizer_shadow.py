"""
Unit tests for executor/optimizer_shadow.py — PR 2 of portfolio-optimizer arc.

The shadow wrapper assembles optimizer inputs from main.py's existing state
(signals, predictions, positions, price histories), calls the kernel, and
logs to S3. Tests use synthetic inputs + a stub S3 client to verify:
  1. Happy path — universe assembly, alpha_hat, returns_panel, w_prev,
     sectors, stance_caps, eligibility all populated correctly
  2. EXIT signals → eligibility[ticker] = False
  3. GBM veto → eligibility[ticker] = False
  4. Held positions populate w_prev from market_value / NAV
  5. Cash sleeve absorbs residual weight pre-solve
  6. Universe includes SPY and CASH appended at the end
  7. Failures don't raise — sentinel written, None returned
  8. Stance multipliers apply to caps when stance is present
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from executor.optimizer_shadow import (
    _build_alpha_hat,
    _build_eligibility,
    _build_sectors,
    _build_stance_caps,
    _build_universe,
    _build_w_prev,
    _compute_trade_deltas,
    _extract_universe_tickers,
    run_shadow_optimizer,
)


def _synthetic_price_df(n_rows: int = 260, seed: int = 0) -> pd.DataFrame:
    """Build a price history DataFrame with a 'close' column."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.012, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="D")
    return pd.DataFrame({"close": prices}, index=idx)


def _baseline_inputs():
    """Construct a minimal set of shadow-optimizer inputs."""
    tickers_with_pred = ["AAPL", "MSFT", "JNJ"]
    price_histories = {t: _synthetic_price_df(seed=i) for i, t in enumerate(tickers_with_pred)}
    price_histories["SPY"] = _synthetic_price_df(seed=99)
    signals_raw = {
        "universe": tickers_with_pred,
        "signals": {
            "AAPL": {"signal": "ENTER", "score": 72, "sector": "Technology"},
            "MSFT": {"signal": "HOLD",  "score": 65, "sector": "Technology"},
            "JNJ":  {"signal": "ENTER", "score": 60, "sector": "Healthcare"},
        },
    }
    predictions_by_ticker = {
        "AAPL": {"predicted_alpha":  0.04, "gbm_veto": False, "stance": "momentum"},
        "MSFT": {"predicted_alpha":  0.02, "gbm_veto": False, "stance": "quality"},
        "JNJ":  {"predicted_alpha": -0.01, "gbm_veto": False, "stance": "value"},
    }
    current_positions = {
        "MSFT": {"market_value": 50_000.0, "sector": "Technology"},
    }
    return {
        "signals_raw": signals_raw,
        "predictions_by_ticker": predictions_by_ticker,
        "current_positions": current_positions,
        "portfolio_nav": 1_000_000.0,
        "price_histories": price_histories,
        "config": {
            "max_position_pct": 0.08,
            "min_score_to_enter": 57,
        },
        "signals_bucket": "test-bucket",
        "run_date": "2026-05-11",
    }


def test_happy_path_assembles_inputs_and_writes_to_s3():
    inputs = _baseline_inputs()
    s3 = MagicMock()

    log = run_shadow_optimizer(s3_client=s3, **inputs)

    assert log is not None, "Shadow optimizer should succeed on the happy path"
    assert log["shadow_status"] == "ok"
    assert log["run_date"] == "2026-05-11"
    assert log["portfolio_nav"] == 1_000_000.0
    assert log["tickers"][-2:] == ["SPY", "CASH"], "SPY/CASH must be appended"
    assert log["n_tickers"] == len(log["tickers"])
    assert len(log["target_weights"]) == log["n_tickers"]
    assert log["diagnostics"]["status"] in ("optimal", "optimal_inaccurate")

    assert s3.put_object.call_count == 2, "Should write dated + latest keys"
    dated_call = s3.put_object.call_args_list[0].kwargs
    assert dated_call["Key"] == "predictor/optimizer_shadow/2026-05-11.json"
    latest_call = s3.put_object.call_args_list[1].kwargs
    assert latest_call["Key"] == "predictor/optimizer_shadow/latest.json"
    body = json.loads(dated_call["Body"])
    assert body["shadow_status"] == "ok"


def test_extract_universe_tickers_accepts_production_dict_shape():
    """Production signals.json emits `universe` as a list of per-ticker dicts.

    Regression for the 2026-05-12 first-shadow-run failure where the wrapper
    blindly called `candidates.update(universe_list)` and raised
    `TypeError: unhashable type: 'dict'` on the live payload shape.
    """
    universe = [
        {"ticker": "COST", "signal": "ENTER", "score": 55.3, "rating": "BUY"},
        {"ticker": "AAPL", "signal": "HOLD", "score": 70.1},
    ]
    assert _extract_universe_tickers(universe) == ["COST", "AAPL"]


def test_extract_universe_tickers_accepts_legacy_string_shape():
    """Legacy / minimal payloads emit a flat list of ticker strings."""
    assert _extract_universe_tickers(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]


def test_extract_universe_tickers_skips_malformed_entries():
    """Mixed / malformed shapes degrade silently; valid entries still extracted."""
    universe = [
        {"ticker": "AAPL"},
        {"no_ticker_key": "foo"},
        "MSFT",
        42,
        None,
        {"ticker": ""},
        {"ticker": None},
    ]
    assert _extract_universe_tickers(universe) == ["AAPL", "MSFT"]


def test_extract_universe_tickers_handles_non_list_input():
    assert _extract_universe_tickers(None) == []
    assert _extract_universe_tickers({"unexpected": "shape"}) == []


def test_build_universe_accepts_production_universe_dict_shape():
    """Full _build_universe call must succeed when signals_raw['universe'] is
    a list of dicts (the live signals.json shape)."""
    inputs = _baseline_inputs()
    inputs["signals_raw"]["universe"] = [
        {"ticker": "AAPL", "signal": "ENTER", "score": 72.0},
        {"ticker": "MSFT", "signal": "HOLD", "score": 65.0},
        {"ticker": "JNJ", "signal": "ENTER", "score": 60.0},
    ]
    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )
    assert set(tickers[:-2]) == {"AAPL", "MSFT", "JNJ"}
    assert tickers[-2:] == ["SPY", "CASH"]


def test_universe_assembly_filters_tickers_without_history():
    inputs = _baseline_inputs()
    inputs["price_histories"]["AAPL"] = _synthetic_price_df(n_rows=30)

    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )

    assert "AAPL" not in tickers, "Tickers with <60 rows of history must be dropped"
    assert tickers[-2:] == ["SPY", "CASH"]


def test_universe_requires_spy_history():
    inputs = _baseline_inputs()
    del inputs["price_histories"]["SPY"]

    with pytest.raises(RuntimeError, match="SPY price history"):
        _build_universe(
            inputs["signals_raw"], inputs["predictions_by_ticker"],
            inputs["current_positions"], inputs["price_histories"],
        )


def test_exit_signal_makes_ticker_ineligible():
    inputs = _baseline_inputs()
    inputs["signals_raw"]["signals"]["AAPL"]["signal"] = "EXIT"

    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")
    aapl_idx = tickers.index("AAPL")

    eligibility, _reasons = _build_eligibility(
        tickers, inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"], inputs["current_positions"],
        inputs["config"], spy_idx, cash_idx,
    )
    assert eligibility[aapl_idx] == False, "EXIT signal must zero eligibility"
    assert eligibility[spy_idx] == True
    assert eligibility[cash_idx] == True


def test_gbm_veto_makes_ticker_ineligible():
    inputs = _baseline_inputs()
    inputs["predictions_by_ticker"]["AAPL"]["gbm_veto"] = True

    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )
    aapl_idx = tickers.index("AAPL")
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")

    eligibility, _reasons = _build_eligibility(
        tickers, inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"], inputs["current_positions"],
        inputs["config"], spy_idx, cash_idx,
    )
    assert eligibility[aapl_idx] == False


def test_w_prev_reflects_current_positions():
    inputs = _baseline_inputs()
    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )
    cash_idx = tickers.index("CASH")
    msft_idx = tickers.index("MSFT")

    w_prev = _build_w_prev(
        tickers, inputs["current_positions"], inputs["portfolio_nav"],
        cash_idx, {},
    )
    assert w_prev[msft_idx] == pytest.approx(0.05, abs=1e-6), \
        "MSFT mkt_val=50k on 1M NAV → 5% weight"
    assert w_prev[cash_idx] == pytest.approx(0.95, abs=1e-6), \
        "Cash should absorb the residual pre-optimization"
    assert w_prev.sum() == pytest.approx(1.0, abs=1e-6)


def test_stance_caps_apply_multiplier_when_stance_present():
    inputs = _baseline_inputs()
    tickers = _build_universe(
        inputs["signals_raw"], inputs["predictions_by_ticker"],
        inputs["current_positions"], inputs["price_histories"],
    )
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")
    aapl_idx = tickers.index("AAPL")
    msft_idx = tickers.index("MSFT")
    jnj_idx = tickers.index("JNJ")

    caps = _build_stance_caps(
        tickers, inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"], inputs["config"], {},
        spy_idx, cash_idx,
    )
    assert caps[aapl_idx] == pytest.approx(0.08 * 1.0), "momentum mult = 1.0"
    assert caps[msft_idx] == pytest.approx(0.08 * 0.8), "quality mult = 0.8"
    assert caps[jnj_idx]  == pytest.approx(0.08 * 0.7), "value mult = 0.7"
    assert caps[spy_idx] == 1.0
    assert caps[cash_idx] == 1.0


def test_wrapper_never_raises_writes_sentinel_on_failure():
    inputs = _baseline_inputs()
    del inputs["price_histories"]["SPY"]
    s3 = MagicMock()

    result = run_shadow_optimizer(s3_client=s3, **inputs)

    assert result is None, "Failures return None — never raise"
    assert s3.put_object.call_count == 2, "Sentinel still written (dated + latest)"
    body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
    assert body["shadow_status"] == "failed"
    assert "SPY price history" in body["error"]
