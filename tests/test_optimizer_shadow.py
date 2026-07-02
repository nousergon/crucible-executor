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
    _build_alpha_uncertainty,
    _build_eligibility,
    _build_sectors,
    _build_stance_caps,
    _build_universe,
    _build_w_prev,
    _compute_trade_deltas,
    _extract_universe_tickers,
    run_shadow_optimizer,
)


@pytest.fixture(autouse=True)
def _isolate_universe_tradeability_read(monkeypatch):
    """Isolate the scanner-tradeability S3 read from AWS by default.

    ``_build_and_solve`` calls ``read_universe_tradeability`` (config#1401) to
    key the √-impact cost term on per-name ADV$. That helper opens a real boto3
    client; with no AWS creds (CI) it fails soft to ``{}``, but relying on the
    live boto path made these tests environment-dependent (they only passed on
    a box with ambient creds — see the #321 CI red). Default the read to ``{}``
    (no ADV coverage → the optimizer's flat-L1 fallback, the bit-identical
    pre-1401 behavior) so the shadow tests are deterministic and creds-free.
    Tests that WANT ADV coverage re-patch it explicitly.
    """
    monkeypatch.setattr(
        "executor.signal_reader.read_universe_tradeability",
        lambda *a, **k: {},
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


def test_adv_coverage_engages_participation_aware_tcost(monkeypatch):
    """When the scanner tradeability artifact supplies ADV$, the shadow solve
    uses the participation-aware √-impact cost term (not the flat-L1 fallback)
    and records ADV coverage in the log (config#1401)."""
    inputs = _baseline_inputs()
    # Patch the tradeability read to return ADV$ for the real names.
    monkeypatch.setattr(
        "executor.signal_reader.read_universe_tradeability",
        lambda *a, **k: {
            "AAPL": {"adv_usd": 8.0e9, "tradeability_score": 95.0},
            "MSFT": {"adv_usd": 6.0e9, "tradeability_score": 92.0},
            "JNJ":  {"adv_usd": 3.0e9, "tradeability_score": 80.0},
        },
    )
    s3 = MagicMock()
    log = run_shadow_optimizer(s3_client=s3, **inputs)

    assert log is not None
    assert log["shadow_status"] == "ok"
    diag = log["diagnostics"]
    assert diag["tcost_term_mode"] == "sqrt_impact"
    assert diag["tcost_n_names_with_adv"] == 3
    assert diag["max_pct_adv_applied"] is True
    assert log["adv_coverage"]["adv_names_covered"] == 3
    assert log["adv_coverage"]["adv_source"] == "scanner_universe_tradeability"
    # ADV$ vector is emitted (SPY/CASH → None).
    assert log["adv_usd"][-2:] == [None, None]


def test_shadow_optimizer_failsoft_when_tradeability_read_has_no_credentials(monkeypatch):
    """REGRESSION (#321 CI red): in a no-AWS-creds environment the scanner
    tradeability read raises NoCredentialsError deep in botocore. That must
    degrade to 'no ADV coverage → flat-L1 tcost fallback', NEVER crash the
    shadow optimizer. Verify both layers: (a) read_universe_tradeability itself
    swallows NoCredentialsError → {}, and (b) the end-to-end shadow solve still
    succeeds with the flat-L1 term."""
    from botocore.exceptions import NoCredentialsError

    import executor.signal_reader as sr

    # Layer (a): the reader swallows the BotoCoreError-family credential error.
    boto_client = MagicMock()
    boto_client.get_object.side_effect = NoCredentialsError()
    monkeypatch.setattr(sr.boto3, "client", lambda *a, **k: boto_client)
    assert sr.read_universe_tradeability("test-bucket", "2026-05-11") == {}

    # Layer (b): drive the real reader (still no creds) through the shadow solve.
    # Undo the autouse {} stub so the genuine read path runs; boto3.client is
    # still the no-creds mock above.
    monkeypatch.setattr(
        "executor.signal_reader.read_universe_tradeability",
        sr.read_universe_tradeability,
    )
    inputs = _baseline_inputs()
    s3 = MagicMock()
    log = run_shadow_optimizer(s3_client=s3, **inputs)

    assert log is not None, (
        "Shadow optimizer must survive a no-credentials tradeability read"
    )
    assert log["shadow_status"] == "ok"
    diag = log["diagnostics"]
    assert diag["tcost_term_mode"] == "flat_l1"
    assert diag["max_pct_adv_applied"] is False
    assert log["adv_coverage"]["adv_names_covered"] == 0


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


# ─── B.4 uncertainty wiring + ablation tests ────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.4
#
# The shadow wrapper reads B.1's predicted_alpha_std from
# predictions_by_ticker, threads it to solve_target_weights, and logs an
# ablation comparison (with/without penalty) when γ > 0.


class TestBuildAlphaUncertainty:
    """The new helper that reads predicted_alpha_std from the predictor
    output. NaN-tolerant to handle the 1-week soak case."""

    def _tickers_with_sentinels(self):
        return ["AAPL", "MSFT", "JNJ", "SPY", "CASH"]

    def test_predicted_alpha_std_present_populates_array(self):
        tickers = self._tickers_with_sentinels()
        preds = {
            "AAPL": {"predicted_alpha": 0.04, "predicted_alpha_std": 0.021},
            "MSFT": {"predicted_alpha": 0.02, "predicted_alpha_std": 0.018},
            "JNJ":  {"predicted_alpha": -0.01, "predicted_alpha_std": 0.030},
        }
        sigma = _build_alpha_uncertainty(tickers, preds, spy_idx=3, cash_idx=4)
        assert sigma[0] == pytest.approx(0.021)
        assert sigma[1] == pytest.approx(0.018)
        assert sigma[2] == pytest.approx(0.030)
        assert sigma[3] == 0.0  # SPY sentinel
        assert sigma[4] == 0.0  # CASH sentinel

    def test_missing_field_yields_nan_per_partial_rollout(self):
        """During the 1-week soak window the legacy Ridge model is still
        in production — predictions JSON has no predicted_alpha_std for
        any ticker. Result: all-NaN for non-sentinel tickers, optimizer's
        B.3 path falls through to zero penalty."""
        tickers = self._tickers_with_sentinels()
        preds = {
            "AAPL": {"predicted_alpha": 0.04},  # no std (legacy Ridge)
            "MSFT": {"predicted_alpha": 0.02},
            "JNJ":  {"predicted_alpha": -0.01},
        }
        sigma = _build_alpha_uncertainty(tickers, preds, spy_idx=3, cash_idx=4)
        assert np.isnan(sigma[0])
        assert np.isnan(sigma[1])
        assert np.isnan(sigma[2])
        # Sentinels still zero
        assert sigma[3] == 0.0 and sigma[4] == 0.0

    def test_none_field_yields_nan(self):
        """Predictor explicitly emits None when legacy Ridge model loaded
        (B.1 fallback path). Treat as missing."""
        tickers = ["AAPL", "SPY", "CASH"]
        preds = {"AAPL": {"predicted_alpha": 0.04, "predicted_alpha_std": None}}
        sigma = _build_alpha_uncertainty(tickers, preds, spy_idx=1, cash_idx=2)
        assert np.isnan(sigma[0])
        assert sigma[1] == 0.0
        assert sigma[2] == 0.0

    def test_negative_or_non_numeric_yields_nan(self):
        """Invalid σ (negative, non-numeric) → NaN. The optimizer's B.3
        coercion path will then treat them as zero penalty."""
        tickers = ["A", "B", "C", "SPY", "CASH"]
        preds = {
            "A": {"predicted_alpha": 0.01, "predicted_alpha_std": -0.05},
            "B": {"predicted_alpha": 0.02, "predicted_alpha_std": "not-a-number"},
            "C": {"predicted_alpha": 0.03, "predicted_alpha_std": float("inf")},
        }
        sigma = _build_alpha_uncertainty(tickers, preds, spy_idx=3, cash_idx=4)
        assert np.isnan(sigma[0])
        assert np.isnan(sigma[1])
        assert np.isnan(sigma[2])

    def test_partial_rollout_some_tickers_have_std_others_dont(self):
        """Mixed case: AAPL has BR std (fresh BayesianRidge inference),
        MSFT was scored by legacy Ridge inside the same predictions JSON
        (transient mid-rollout state). Must work."""
        tickers = ["AAPL", "MSFT", "SPY", "CASH"]
        preds = {
            "AAPL": {"predicted_alpha": 0.04, "predicted_alpha_std": 0.025},
            "MSFT": {"predicted_alpha": 0.02},  # legacy path
        }
        sigma = _build_alpha_uncertainty(tickers, preds, spy_idx=2, cash_idx=3)
        assert sigma[0] == pytest.approx(0.025)
        assert np.isnan(sigma[1])
        assert sigma[2] == 0.0 and sigma[3] == 0.0


class TestShadowWiringAndAblation:
    """End-to-end: shadow wrapper threads predicted_alpha_std into the
    optimizer and emits the ablation block when γ > 0."""

    def _inputs_with_std(self, gamma=0.0):
        inputs = _baseline_inputs()
        # Augment predictions with BR std field per B.1
        inputs["predictions_by_ticker"]["AAPL"]["predicted_alpha_std"] = 0.040  # diffuse
        inputs["predictions_by_ticker"]["MSFT"]["predicted_alpha_std"] = 0.005  # confident
        inputs["predictions_by_ticker"]["JNJ"]["predicted_alpha_std"] = 0.020
        if gamma > 0:
            inputs["config"] = {
                **inputs["config"],
                "portfolio_optimizer": {"alpha_uncertainty_penalty": gamma},
            }
        return inputs

    def test_shadow_log_includes_alpha_uncertainty_field(self):
        """Per-ticker σ_α̂ emitted in the shadow JSON for full diagnostic
        visibility. NaN-as-None per JSON-safe conversion."""
        inputs = self._inputs_with_std(gamma=0.0)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        assert "alpha_uncertainty" in log
        assert len(log["alpha_uncertainty"]) == log["n_tickers"]
        # Ordering: same as `tickers` list. SPY/CASH at end are 0.0 (sentinels).
        # The 3 real tickers should have populated σ_α̂ (sorted alphabetically
        # by _build_universe → AAPL, JNJ, MSFT order).
        tickers = log["tickers"]
        spy_pos = tickers.index("SPY")
        cash_pos = tickers.index("CASH")
        assert log["alpha_uncertainty"][spy_pos] == 0.0
        assert log["alpha_uncertainty"][cash_pos] == 0.0
        # AAPL appears in the universe with predicted_alpha_std=0.040
        assert log["alpha_uncertainty"][tickers.index("AAPL")] == pytest.approx(0.040)
        assert log["alpha_uncertainty"][tickers.index("MSFT")] == pytest.approx(0.005)
        assert log["alpha_uncertainty"][tickers.index("JNJ")]  == pytest.approx(0.020)

    def test_ablation_skipped_when_gamma_zero(self):
        """Default γ=0 → no ablation block (active solve already IS no-penalty)."""
        inputs = self._inputs_with_std(gamma=0.0)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        assert "uncertainty_ablation" not in log
        # Diagnostics should also report penalty_used=False
        assert log["diagnostics"]["alpha_uncertainty_penalty_used"] is False

    def test_ablation_emitted_when_gamma_positive(self):
        """γ > 0 with usable σ_α̂ signal → ablation block populated with
        side-by-side no-penalty weights + diff summary."""
        inputs = self._inputs_with_std(gamma=500.0)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        ab = log.get("uncertainty_ablation")
        assert ab is not None
        assert ab["gamma"] == 500.0
        assert len(ab["no_penalty_weights"]) == log["n_tickers"]
        assert "no_penalty_diagnostics" in ab
        assert ab["l1_delta"] >= 0
        assert ab["max_abs_delta"] >= 0
        # Diagnostics on the canonical (with-penalty) solve must report
        # penalty_used=True
        assert log["diagnostics"]["alpha_uncertainty_penalty_used"] is True

    def test_ablation_skipped_when_no_usable_std_signal(self):
        """γ > 0 but predictions JSON has no predicted_alpha_std anywhere
        (legacy Ridge inference) → ablation skipped (canonical IS
        no-penalty already since the B.3 path treats all-NaN as inactive)."""
        inputs = _baseline_inputs()
        inputs["config"] = {
            **inputs["config"],
            "portfolio_optimizer": {"alpha_uncertainty_penalty": 500.0},
        }
        # No predicted_alpha_std added to any ticker
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        assert "uncertainty_ablation" not in log
        # And the canonical solve also reports no penalty active
        assert log["diagnostics"]["alpha_uncertainty_penalty_used"] is False

    def test_per_ticker_delta_only_lists_names_that_moved(self):
        """The shadow JSON's per_ticker_delta list only includes names that
        moved ≥ 1bp — keeps the log compact while preserving observability
        on the names that actually changed under the penalty."""
        # Construct a case where most names don't move: γ very low so the
        # penalty is dominated by α̂ gain
        inputs = self._inputs_with_std(gamma=0.01)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        ab = log.get("uncertainty_ablation")
        if ab is not None:
            # All listed names actually moved by ≥ 1bp
            for entry in ab["per_ticker_delta"]:
                assert abs(entry["delta"]) >= 1e-4
            # n_names_moved must equal length of the list
            assert ab["n_names_moved"] == len(ab["per_ticker_delta"])


# ── _load_auto_tuned_optimizer_cfg (config#1057 inc 2) ───────────────────────


def _s3_returning(payload):
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
    return s3


class TestLoadAutoTunedOptimizerCfg:
    def _cfg(self, **po):
        return {"signals_bucket": "bkt", "portfolio_optimizer": po}

    def test_loads_allowlisted_and_clamps(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        s3 = _s3_returning({"risk_aversion": 4.0, "tcost_bps": 3.0,
                            "max_sector_pct": 0.99, "updated_at": "2026-06-14"})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        # only the two writable knobs survive
        assert out == {"risk_aversion": 4.0, "tcost_bps": 3.0}

    def test_out_of_band_value_is_reclamped(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg, _AUTO_TUNED_BOUNDS
        s3 = _s3_returning({"risk_aversion": 999.0, "tcost_bps": -5.0})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        assert out["risk_aversion"] == _AUTO_TUNED_BOUNDS["risk_aversion"][1]  # hi
        assert out["tcost_bps"] == _AUTO_TUNED_BOUNDS["tcost_bps"][0]          # lo

    def test_private_floor_override_admits_aggressive_lambda(self):
        # Public default floor is 3.0 → λ=2.0 clamps up to 3.0 ...
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        s3 = _s3_returning({"risk_aversion": 2.0})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        assert out["risk_aversion"] == 3.0
        # ... but the PRIVATE risk.yaml override (floor 1.0) lets λ=2.0 through,
        # so a more aggressive auto-tuned book is admitted without shipping the
        # aggressive floor in the public default.
        s3b = _s3_returning({"risk_aversion": 2.0})
        out2 = _load_auto_tuned_optimizer_cfg(
            self._cfg(tuner_risk_aversion_floor=1.0), s3_client=s3b)
        assert out2["risk_aversion"] == 2.0

    def test_kill_switch_disables_consumption(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        s3 = _s3_returning({"risk_aversion": 4.0})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(consume_auto_tuned=False), s3_client=s3)
        assert out == {}
        s3.get_object.assert_not_called()

    def test_failsafe_on_s3_error_returns_empty(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("NoSuchKey")
        assert _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3) == {}

    def test_no_bucket_returns_empty(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        assert _load_auto_tuned_optimizer_cfg({"portfolio_optimizer": {}}, s3_client=MagicMock()) == {}

    def test_non_numeric_value_ignored(self):
        from executor.optimizer_shadow import _load_auto_tuned_optimizer_cfg
        s3 = _s3_returning({"risk_aversion": "oops", "tcost_bps": 3.0})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        assert out == {"tcost_bps": 3.0}
