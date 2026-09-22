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

from executor.alpha_contract import OPTIMIZER_ALPHA_ANCHOR
from executor.optimizer_shadow import (
    _build_alpha_uncertainty,
    _build_eligibility,
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
            "MSFT": {"signal": "HOLD", "score": 65, "sector": "Technology"},
            "JNJ": {"signal": "ENTER", "score": 60, "sector": "Healthcare"},
        },
    }
    # `alpha_anchor` is REQUIRED on every numeric predicted_alpha reaching the
    # solve (alpha-engine-config-I7337) — the executor's read/inject adapters
    # stamp it, and `_build_alpha_hat` refuses a batch that mixes anchors or
    # omits one. Fixtures declare it for the same reason the live path does.
    predictions_by_ticker = {
        "AAPL": {
            "predicted_alpha": 0.04,
            "gbm_veto": False,
            "stance": "momentum",
            "alpha_anchor": OPTIMIZER_ALPHA_ANCHOR,
        },
        "MSFT": {
            "predicted_alpha": 0.02,
            "gbm_veto": False,
            "stance": "quality",
            "alpha_anchor": OPTIMIZER_ALPHA_ANCHOR,
        },
        "JNJ": {"predicted_alpha": -0.01, "gbm_veto": False, "stance": "value", "alpha_anchor": OPTIMIZER_ALPHA_ANCHOR},
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
            "JNJ": {"adv_usd": 3.0e9, "tradeability_score": 80.0},
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

    assert log is not None, "Shadow optimizer must survive a no-credentials tradeability read"
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
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )
    assert set(tickers[:-2]) == {"AAPL", "MSFT", "JNJ"}
    assert tickers[-2:] == ["SPY", "CASH"]


def test_universe_assembly_filters_tickers_without_history():
    inputs = _baseline_inputs()
    inputs["price_histories"]["AAPL"] = _synthetic_price_df(n_rows=30)

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )

    assert "AAPL" not in tickers, "Tickers with <60 rows of history must be dropped"
    assert tickers[-2:] == ["SPY", "CASH"]


def test_universe_requires_spy_history():
    inputs = _baseline_inputs()
    del inputs["price_histories"]["SPY"]

    with pytest.raises(RuntimeError, match="SPY price history"):
        _build_universe(
            inputs["signals_raw"],
            inputs["predictions_by_ticker"],
            inputs["current_positions"],
            inputs["price_histories"],
        )


def test_exit_signal_makes_ticker_ineligible():
    inputs = _baseline_inputs()
    inputs["signals_raw"]["signals"]["AAPL"]["signal"] = "EXIT"

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")
    aapl_idx = tickers.index("AAPL")

    eligibility, _reasons = _build_eligibility(
        tickers,
        inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["config"],
        spy_idx,
        cash_idx,
    )
    assert not eligibility[aapl_idx], "EXIT signal must zero eligibility"
    assert eligibility[spy_idx]
    assert eligibility[cash_idx]


def test_gbm_veto_makes_ticker_ineligible():
    inputs = _baseline_inputs()
    inputs["predictions_by_ticker"]["AAPL"]["gbm_veto"] = True

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )
    aapl_idx = tickers.index("AAPL")
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")

    eligibility, _reasons = _build_eligibility(
        tickers,
        inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["config"],
        spy_idx,
        cash_idx,
    )
    assert not eligibility[aapl_idx]


def test_w_prev_reflects_current_positions():
    inputs = _baseline_inputs()
    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )
    cash_idx = tickers.index("CASH")
    msft_idx = tickers.index("MSFT")

    w_prev = _build_w_prev(
        tickers,
        inputs["current_positions"],
        inputs["portfolio_nav"],
        cash_idx,
        {},
    )
    assert w_prev[msft_idx] == pytest.approx(0.05, abs=1e-6), "MSFT mkt_val=50k on 1M NAV → 5% weight"
    assert w_prev[cash_idx] == pytest.approx(0.95, abs=1e-6), "Cash should absorb the residual pre-optimization"
    assert w_prev.sum() == pytest.approx(1.0, abs=1e-6)


def test_stance_caps_apply_multiplier_when_stance_present():
    inputs = _baseline_inputs()
    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
    )
    spy_idx = tickers.index("SPY")
    cash_idx = tickers.index("CASH")
    aapl_idx = tickers.index("AAPL")
    msft_idx = tickers.index("MSFT")
    jnj_idx = tickers.index("JNJ")

    caps = _build_stance_caps(
        tickers,
        inputs["signals_raw"]["signals"],
        inputs["predictions_by_ticker"],
        inputs["config"],
        {},
        spy_idx,
        cash_idx,
    )
    assert caps[aapl_idx] == pytest.approx(0.08 * 1.0), "momentum mult = 1.0"
    assert caps[msft_idx] == pytest.approx(0.08 * 0.8), "quality mult = 0.8"
    assert caps[jnj_idx] == pytest.approx(0.08 * 0.7), "value mult = 0.7"
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


class TestFailureSentinelForensics:
    """The failure artifact must carry the numbers that describe the failure.

    Until alpha-engine-config-I11369 it carried ``repr(e)`` and nothing else,
    so the 2026-09-22 ``TurnoverBudgetError`` that held the whole book had to
    be diagnosed by rebuilding the inputs from scratch.
    """

    def test_sentinel_carries_structured_type_message_and_traceback(self):
        from executor.optimizer_shadow import _build_failure_sentinel

        try:
            raise ValueError("no SPY price history")
        except ValueError as exc:
            sentinel = _build_failure_sentinel(exc, "2026-09-22")

        assert sentinel["shadow_status"] == "failed"
        assert sentinel["error_type"] == "ValueError"
        assert sentinel["error_message"] == "no SPY price history"
        # The old field is kept verbatim — every existing reader uses it.
        assert sentinel["error"] == repr(ValueError("no SPY price history"))
        assert "ValueError" in sentinel["traceback"]
        assert sentinel["run_date"] == "2026-09-22"

    def test_diagnostics_attached_to_the_exception_survive_into_the_sentinel(self):
        from executor.portfolio_optimizer import TurnoverBudgetError
        from executor.optimizer_shadow import _build_failure_sentinel

        exc = TurnoverBudgetError("solved one-way turnover 0.017503 exceeds ...")
        exc.diagnostics = {
            "turnover_constraint_cap": 0.016209,
            "turnover_pre_clip_one_way": 0.017503,
            "turnover_solver_status": "optimal_inaccurate",
        }
        sentinel = _build_failure_sentinel(exc, "2026-09-22")
        assert sentinel["diagnostics_partial"]["turnover_constraint_cap"] == 0.016209
        assert (
            sentinel["diagnostics_partial"]["turnover_solver_status"]
            == "optimal_inaccurate"
        )

    def test_the_field_is_present_and_null_when_nothing_was_attached(self):
        # A key that appears only on the interesting path is
        # indistinguishable from a dead emitter.
        from executor.optimizer_shadow import _build_failure_sentinel

        sentinel = _build_failure_sentinel(RuntimeError("early"), "2026-09-22")
        assert "diagnostics_partial" in sentinel
        assert sentinel["diagnostics_partial"] is None


def test_a_shadow_failure_publishes_an_ops_alert(monkeypatch):
    """The day the optimizer refuses to trade must page (I11369).

    On 2026-09-22 the only surface that carried it was a log line on the
    trading box; the alert must say the book was HELD or it gets triaged as
    a crash.
    """
    published = []
    import executor.notifier as notifier

    monkeypatch.setattr(
        notifier, "publish_ops_alert",
        lambda message, **kw: published.append((message, kw)),
    )

    inputs = _baseline_inputs()
    del inputs["price_histories"]["SPY"]
    run_shadow_optimizer(s3_client=MagicMock(), **inputs)

    assert len(published) == 1, "exactly one page per failed session"
    message, kw = published[0]
    assert kw["severity"] == "error"
    assert kw["dedup_key"].startswith("optimizer-shadow-failed-")
    assert "HELD" in message
    assert "not a crash" in message


def test_an_alert_failure_never_breaks_the_sentinel_write(monkeypatch):
    # The S3 sentinel is the durable surface and is written first; the alert
    # is best-effort, like every other secondary-observability path here.
    import executor.notifier as notifier

    def _boom(message, **kw):
        raise RuntimeError("SNS down")

    monkeypatch.setattr(notifier, "publish_ops_alert", _boom)
    inputs = _baseline_inputs()
    del inputs["price_histories"]["SPY"]
    s3 = MagicMock()

    assert run_shadow_optimizer(s3_client=s3, **inputs) is None
    assert s3.put_object.call_count == 2


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
            "JNJ": {"predicted_alpha": -0.01, "predicted_alpha_std": 0.030},
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
            "JNJ": {"predicted_alpha": -0.01},
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

    def _inputs_with_std(self, gamma=0.0, epistemic=True):
        inputs = _baseline_inputs()
        # Augment predictions with BR std field per B.1
        inputs["predictions_by_ticker"]["AAPL"]["predicted_alpha_std"] = 0.040  # diffuse
        inputs["predictions_by_ticker"]["MSFT"]["predicted_alpha_std"] = 0.005  # confident
        inputs["predictions_by_ticker"]["JNJ"]["predicted_alpha_std"] = 0.020
        if epistemic:
            # I9452: the GUW Omega reads the DECOMPOSED estimation-error half,
            # emitted alongside the total by crucible-predictor PR596. Values
            # are the total's shape at the measured epistemic scale.
            inputs["predictions_by_ticker"]["AAPL"]["predicted_alpha_std_epistemic"] = 0.0220
            inputs["predictions_by_ticker"]["MSFT"]["predicted_alpha_std_epistemic"] = 0.0090
            inputs["predictions_by_ticker"]["JNJ"]["predicted_alpha_std_epistemic"] = 0.0150
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
        assert log["alpha_uncertainty"][tickers.index("JNJ")] == pytest.approx(0.020)

    def test_shadow_log_carries_the_epistemic_vector_as_its_own_field(self):
        """CONSUMER CONTRACT on crucible-predictor's `predicted_alpha_std_epistemic`
        (alpha-engine-config-I9452, M0 cross-repo-artifact rule).

        Two independent obligations are pinned here:

        1. The executor READS the producer's field, per ticker, aligned to the
           universe, with SPY/CASH as 0.0 sentinels.
        2. It PERSISTS it on the shadow artifact under its own key, because the
           daemon's intraday re-solve rebuilds Omega from that artifact. Drop the
           key and the afternoon re-solve silently sizes the same book on a
           different Omega than the morning solve did.
        """
        inputs = self._inputs_with_std(gamma=0.0)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        assert "alpha_uncertainty_epistemic" in log, (
            "the intraday re-solve reads this key off the artifact"
        )
        eps = log["alpha_uncertainty_epistemic"]
        assert len(eps) == log["n_tickers"]
        tickers = log["tickers"]
        assert eps[tickers.index("SPY")] == 0.0
        assert eps[tickers.index("CASH")] == 0.0
        assert eps[tickers.index("AAPL")] == pytest.approx(0.0220)
        assert eps[tickers.index("MSFT")] == pytest.approx(0.0090)
        assert eps[tickers.index("JNJ")] == pytest.approx(0.0150)
        # And it is a DIFFERENT vector from the total — a wiring mistake that
        # aliased the two would otherwise pass every other assertion here.
        assert eps != log["alpha_uncertainty"]

    def test_absent_producer_field_is_reported_not_silently_replaced(self):
        """A pre-PR596 predictions artifact carries no epistemic field.

        The penalty must go inoperative with a NAMED reason on the artifact,
        and must NOT fall back to `predicted_alpha_std` — that field is
        cross-sectionally flat (CV <= 0.008 on every measured session), so the
        fallback would silently reinstate the uniform ridge this change exists
        to remove, on a solve whose artifact claimed the penalty was applied.
        """
        inputs = self._inputs_with_std(gamma=500.0, epistemic=False)
        s3 = MagicMock()
        log = run_shadow_optimizer(s3_client=s3, **inputs)
        assert log is not None
        assert log["diagnostics"]["alpha_uncertainty_penalty_used"] is False
        assert log["diagnostics"]["alpha_uncertainty_inoperative_reason"] == (
            "epistemic_field_absent"
        )
        # The total is still present and still populated — the conviction gate
        # keeps reading it, and its own diagnostics are still computed.
        assert any(v for v in log["alpha_uncertainty"] if v)
        assert "conviction_ir_xs" in log["diagnostics"]
        # No ablation: there is no penalty to ablate against.
        assert "uncertainty_ablation" not in log

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

        s3 = _s3_returning({"risk_aversion": 4.0, "tcost_bps": 3.0, "max_sector_pct": 0.99, "updated_at": "2026-06-14"})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        # only the two writable knobs survive
        assert out == {"risk_aversion": 4.0, "tcost_bps": 3.0}

    def test_out_of_band_value_is_reclamped(self):
        from executor.optimizer_shadow import _AUTO_TUNED_BOUNDS, _load_auto_tuned_optimizer_cfg

        s3 = _s3_returning({"risk_aversion": 999.0, "tcost_bps": -5.0})
        out = _load_auto_tuned_optimizer_cfg(self._cfg(), s3_client=s3)
        assert out["risk_aversion"] == _AUTO_TUNED_BOUNDS["risk_aversion"][1]  # hi
        assert out["tcost_bps"] == _AUTO_TUNED_BOUNDS["tcost_bps"][0]  # lo

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
        out2 = _load_auto_tuned_optimizer_cfg(self._cfg(tuner_risk_aversion_floor=1.0), s3_client=s3b)
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


# ── config-I7337: a dropped candidate is NAMED, and a plumbing bug RAISES ────
#
# The defect: `_build_universe` declares every predicted ticker a candidate,
# then drops any whose price history is absent — while `executor.main` loaded
# histories only for held + ENTER names. Measured 2026-08-14, the live solve
# ran over 14 tickers containing ZERO of the day's `attractiveness_top_20`,
# and the artifact said nothing about it. `_has_usable_history` collapsed
# "never loaded" and "too short" into one False, so a data-plumbing bug and a
# legitimate exclusion were byte-identical downstream.


def test_dropped_candidates_are_named_with_a_typed_reason():
    """A dropped candidate leaves a record. `eligibility_reasons` explains an
    INCLUDED name's zero weight and is blind to names deleted before it."""
    inputs = _baseline_inputs()
    inputs["price_histories"]["AAPL"] = _synthetic_price_df(n_rows=30)
    dropped: list[dict] = []

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
        price_histories_requested=set(inputs["price_histories"]),
        dropped_out=dropped,
    )

    assert "AAPL" not in tickers
    grp = next(g for g in dropped if "AAPL" in g["tickers"])
    assert grp["reason"] == "history_too_short"
    assert grp["source"] in {"prediction", "position", "signals_universe"}
    assert grp["count"] == len(grp["tickers"])


def test_a_predicted_ticker_never_requested_raises_rather_than_vanishing():
    """THE regression guard. A name this module declared a candidate, that the
    loader was never asked for, is a contract violated inside one run — it must
    not be solved around silently."""
    inputs = _baseline_inputs()
    inputs["predictions_by_ticker"]["NVDA"] = {
        "ticker": "NVDA",
        "predicted_alpha": 0.05,
        "predicted_direction": "UP",
    }
    # NVDA is predicted but was never requested from the loader — exactly the
    # live shape: the whole attractiveness cut, absent from price_histories.
    requested = set(inputs["price_histories"])

    with pytest.raises(RuntimeError, match="never requested"):
        _build_universe(
            inputs["signals_raw"],
            inputs["predictions_by_ticker"],
            inputs["current_positions"],
            inputs["price_histories"],
            price_histories_requested=requested,
        )


def test_a_predicted_ticker_the_cache_lacked_is_recorded_not_raised():
    """The distinction that keeps this from being a trading halt: a name the
    loader ASKED for and the cache did not have is a data condition, not a
    plumbing bug. Record it; do not raise."""
    inputs = _baseline_inputs()
    inputs["predictions_by_ticker"]["NVDA"] = {
        "ticker": "NVDA",
        "predicted_alpha": 0.05,
        "predicted_direction": "UP",
    }
    requested = set(inputs["price_histories"]) | {"NVDA"}
    dropped: list[dict] = []

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
        price_histories_requested=requested,
        dropped_out=dropped,
    )

    assert "NVDA" not in tickers
    grp = next(g for g in dropped if "NVDA" in g["tickers"])
    assert grp["reason"] == "history_absent_in_cache"
    assert grp["source"] == "prediction"


def test_unknown_requested_set_degrades_rather_than_raising():
    """`price_histories_requested=None` means the caller could not say what was
    asked for — the plumbing distinction is unprovable, so it must degrade to
    the data-condition reason rather than raise on a claim it cannot support."""
    inputs = _baseline_inputs()
    inputs["predictions_by_ticker"]["NVDA"] = {
        "ticker": "NVDA",
        "predicted_alpha": 0.05,
        "predicted_direction": "UP",
    }
    dropped: list[dict] = []

    tickers = _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
        price_histories_requested=None,
        dropped_out=dropped,
    )

    assert "NVDA" not in tickers
    assert next(g for g in dropped if "NVDA" in g["tickers"])["reason"] == ("history_absent_in_cache")


def test_dropped_candidates_is_emitted_on_the_clean_path_too():
    """Emitted every run, including empty. A field that appears only when
    something is wrong is indistinguishable from a dead emitter."""
    inputs = _baseline_inputs()
    dropped: list[dict] = []

    _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
        price_histories_requested=set(inputs["price_histories"]),
        dropped_out=dropped,
    )

    assert dropped == []


def test_dropped_candidates_groups_repetition_but_keeps_every_member():
    """A count without its members is the defect config-I7324 exists for, so the
    grouping must carry the full ticker list — it removes repetition of the two
    constant fields, never a name."""
    inputs = _baseline_inputs()
    inputs["signals_raw"]["universe"] = [{"ticker": t, "signal": "HOLD"} for t in ("ZZA", "ZZB", "ZZC")]
    dropped: list[dict] = []

    _build_universe(
        inputs["signals_raw"],
        inputs["predictions_by_ticker"],
        inputs["current_positions"],
        inputs["price_histories"],
        price_histories_requested=set(inputs["price_histories"]),
        dropped_out=dropped,
    )

    grp = next(g for g in dropped if g["source"] == "signals_universe")
    assert grp["reason"] == "no_history_loaded"
    assert sorted(grp["tickers"]) == ["ZZA", "ZZB", "ZZC"]
    assert grp["count"] == 3
    # One record for three names, not three records.
    assert len([g for g in dropped if g["source"] == "signals_universe"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# band_dropped_trades — naming every trade the rebalance band removes
# (alpha-engine-config-I7346)
# ═══════════════════════════════════════════════════════════════════════════


class TestBandDroppedTrades:
    """The anti-churn rebalance band suppresses per-name DRIFT. It cannot, on
    its own, tell drift apart from an intended NEW position that something
    upstream shrank into the band — and until this record existed nothing
    named what it removed: the solve reported ``optimal``, ``entries_blocked``
    stayed empty, and a deleted entry cohort rendered exactly like a quiet
    hold day.
    """

    CFG = {"rebalance_band_pct": 0.005}
    NAV = 1_000_000.0

    def test_drift_under_the_band_is_recorded_not_silently_dropped(self):
        tickers = ["AAA", "SPY", "CASH"]
        target = np.array([0.102, 0.868, 0.03])
        current = np.array([0.100, 0.870, 0.03])
        trades, dropped = _compute_trade_deltas(
            tickers, target, current, self.NAV, self.CFG
        )
        assert trades == []
        by_ticker = {d["ticker"]: d for d in dropped}
        assert set(by_ticker) == {"AAA", "SPY"}
        assert by_ticker["AAA"]["delta"] == pytest.approx(0.002, abs=1e-9)
        assert by_ticker["AAA"]["band"] == 0.005
        assert by_ticker["AAA"]["is_new_position"] is False

    def test_a_shrunken_entry_is_marked_as_a_new_position(self):
        # The failure mode itself: an intended entry sized under the band.
        # `is_new_position` is what separates it from drift on a held name.
        tickers = ["NEW", "SPY", "CASH"]
        target = np.array([0.004, 0.966, 0.03])
        current = np.array([0.000, 0.970, 0.03])
        _trades, dropped = _compute_trade_deltas(
            tickers, target, current, self.NAV, self.CFG
        )
        new = next(d for d in dropped if d["ticker"] == "NEW")
        assert new["is_new_position"] is True
        assert new["current_weight"] == 0.0
        assert new["target_weight"] == pytest.approx(0.004)

    def test_universe_names_the_optimizer_never_picked_are_not_recorded(self):
        # A zero-target, zero-current name is not a trade the band removed —
        # it is a name the optimizer did not pick. Recording those would bury
        # the real drops under the whole universe.
        tickers = ["UNPICKED", "SPY", "CASH"]
        target = np.array([0.0, 0.97, 0.03])
        current = np.array([0.0, 0.97, 0.03])
        trades, dropped = _compute_trade_deltas(
            tickers, target, current, self.NAV, self.CFG
        )
        assert trades == []
        assert dropped == []

    def test_tradeable_deltas_are_not_recorded_as_dropped(self):
        tickers = ["AAA", "SPY", "CASH"]
        target = np.array([0.10, 0.87, 0.03])
        current = np.array([0.00, 0.97, 0.03])
        trades, dropped = _compute_trade_deltas(
            tickers, target, current, self.NAV, self.CFG
        )
        assert {t["ticker"] for t in trades} == {"AAA", "SPY"}
        assert dropped == []

    def test_cash_sentinel_is_never_a_trade_or_a_drop(self):
        tickers = ["AAA", "SPY", "CASH"]
        target = np.array([0.10, 0.87, 0.03])
        current = np.array([0.10, 0.869, 0.031])
        _trades, dropped = _compute_trade_deltas(
            tickers, target, current, self.NAV, self.CFG
        )
        assert "CASH" not in {d["ticker"] for d in dropped}


def test_band_dropped_trades_is_on_the_artifact_every_run(monkeypatch):
    """Emitted even when empty. A field that appears only on the bad path is
    indistinguishable from a dead emitter — the same rule dropped_candidates
    was given in config-I7337."""
    inputs = _baseline_inputs()
    s3 = MagicMock()
    log = run_shadow_optimizer(**inputs, s3_client=s3)
    assert log is not None
    assert log["shadow_status"] == "ok"
    assert "band_dropped_trades" in log
    assert isinstance(log["band_dropped_trades"], list)
    body = json.loads(s3.put_object.call_args.kwargs["Body"].decode())
    assert "band_dropped_trades" in body
