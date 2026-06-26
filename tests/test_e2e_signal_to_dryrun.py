"""End-to-end executor-slice test: synthetic signals → inference → dry-run plan.

Advances nousergon/alpha-engine-config#872 ("End-to-end integration test —
synthetic signals → inference → executor dry-run → verify"). #872 describes a
cross-repo harness (crucible-research → crucible-predictor → crucible-executor).
This module builds the deterministic *executor end* of that chain in-repo: it
feeds a SYNTHETIC, canned upstream artifact (research signals + a stand-in
predictor "inference" payload) through the executor's real planning path and
asserts a sane dry-run order plan is produced — without touching live brokers
(IBKR), AWS/S3, ArcticDB, or any LLM.

Two legs are covered, mirroring the three nouns in #872 ("signals →
inference → executor dry-run"):

  1. ``test_signal_to_dryrun_plan_full_run`` — the SIGNAL → EXECUTOR DRY-RUN
     leg, driven through the top-level orchestration entry point
     ``executor.main.run(simulate=True)``. This is the same function the live
     daemon/planner calls; ``simulate=True`` is the executor's own broker-less
     mode (it returns the planned order list instead of writing the order book
     or hitting IB Gateway). Synthetic signals in → planned ENTER orders out.

  2. ``test_inference_veto_blocks_entry`` — the INFERENCE leg. The top-level
     ``run(simulate=True)`` path intentionally zeroes predictions (see
     ``executor/main.py::_read_signals``: ``predictions_by_ticker = {}`` under
     simulate), so the predictor handoff is exercised one seam lower, at the
     ``decide_entries`` planner that actually consumes the inference artifact.
     A canned prediction carrying ``gbm_veto=True`` must override the research
     ENTER and block exactly that ticker, leaving the others to order — proving
     the synthetic inference payload reaches and steers the executor's plan.

Determinism: fixed synthetic OHLCV (mild upward drift so the momentum gate
passes), flat $100 prices, $1M NAV, ``max_position_pct=0.05`` → every entry
sizes to exactly 500 shares. No network, no clock dependence, no broker.

DEFERRED (the cross-repo legs #872 ultimately wants, not in this slice):
  * Real crucible-research signal generation feeding this test's input.
  * Real crucible-predictor inference producing the predictions artifact
    (here it is a hand-built dict with the predictor's documented veto fields).
  * A shared cross-repo fixture/contract pinning the signals.json and
    predictions.json schemas across the three repos.

Stubbing/fixture patterns here mirror the repo's existing deterministic
executor tests — ``tests/test_decider_parity.py`` (SimulatedIBKRClient +
injected price/atr/vwap/coverage maps) and ``tests/test_perf_simulate_mode.py``
(``fake_risk_yaml`` + ``_LOAD_CONFIG_CACHE`` reset to run config-dependent code
on a clean CI runner with no real risk.yaml).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import executor.main as main_mod
from executor.deciders import decide_entries
from executor.ibkr import SimulatedIBKRClient

# Sector ETFs the exit-veto path may load price histories for; seed them so no
# ArcticDB read is ever attempted in the full-run leg.
_SECTOR_ETFS = [
    "SPY", "XLK", "XLV", "XLF", "XLY", "XLP",
    "XLE", "XLU", "XLRE", "XLB", "XLI", "XLC",
]


def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV with mild upward drift so the momentum gate passes.

    Matches the helper in tests/test_decider_parity.py so the two suites stay
    in lock-step on what a "passing" history looks like.
    """
    return pd.DataFrame(
        {
            "open":  [base + i * 0.1 for i in range(n_bars)],
            "high":  [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low":   [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


def _enter_signals(n: int = 3) -> list[dict]:
    """Canned research ENTER signals — the synthetic 'signals' leg of #872."""
    return [
        {
            "ticker": f"TKR{i:03d}",
            "signal": "ENTER",
            "score": 80,
            "conviction": "rising",
            "sector": "Technology",
            "rating": "BUY",
            "price_target_upside": 0.15,
            "thesis_summary": "synthetic e2e fixture",
        }
        for i in range(n)
    ]


def _signals_artifact(enter: list[dict]) -> dict:
    """The synthetic signals.json artifact the executor would read from S3."""
    universe = enter + [
        {"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology", "score": 50}
    ]
    return {
        "date": "2026-04-25",
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "market_weight"}},
        "enter": enter,
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": universe,
        "buy_candidates": enter,
    }


def _risk_config_text() -> str:
    """Minimal risk.yaml sufficient for the planner; no placeholder buckets
    (config_loader refuses the .example template, so the file is real)."""
    return (
        "signals_bucket: test-bucket\n"
        "trades_bucket: test-trades-bucket\n"
        "db_path: ':memory:'\n"
        "min_score_to_enter: 70\n"
        "min_conviction_to_enter: [rising, stable]\n"
        "max_position_pct: 0.05\n"
        "bear_max_position_pct: 0.025\n"
        "max_sector_pct: 0.25\n"
        "max_equity_pct: 0.90\n"
        "drawdown_circuit_breaker: 0.08\n"
        "momentum_gate_enabled: true\n"
        "momentum_gate_threshold: -50.0\n"
        "atr_sizing_enabled: true\n"
        "correlation_block_enabled: false\n"
        "coverage_sizing_enabled: false\n"
        "coverage_admission_enabled: false\n"
        "reduce_fraction: 0.50\n"
        "use_portfolio_optimizer: false\n"
        "strategy:\n"
        "  exit_manager:\n"
        "    atr_period: 14\n"
    )


def _planner_config() -> dict:
    """In-memory config matching _risk_config_text(), for the decide_entries
    leg (which takes config as a plain dict rather than loading risk.yaml)."""
    return {
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "momentum_gate_enabled": True,
        "momentum_gate_threshold": -50.0,
        "atr_sizing_enabled": True,
        "correlation_block_enabled": False,
        "coverage_sizing_enabled": False,
        "reduce_fraction": 0.50,
        "strategy": {
            "graduated_drawdown": {
                "enabled": True,
                "tiers": [(-0.02, 1.00, "tier1"), (-0.04, 0.50, "tier2")],
            },
        },
    }


def _strategy_config() -> dict:
    return {
        "intraday_pullback_atr_multiple": 1.0,
        "intraday_vwap_discount_pct": 0.005,
        "intraday_support_lookback_days": 20,
        "drawdown_forced_exit_enabled": True,
        "drawdown_forced_exit_tier3_count": 2,
        "drawdown_forced_exit_tier2_count": 1,
    }


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """load_config() caches for the process lifetime; reset around each test so
    the fake risk.yaml is re-read cleanly (mirrors test_perf_simulate_mode.py)."""
    main_mod._LOAD_CONFIG_CACHE = None
    yield
    main_mod._LOAD_CONFIG_CACHE = None


@pytest.fixture
def fake_risk_yaml(tmp_path, monkeypatch):
    """Write a minimal risk.yaml and point get_config_path at it so the
    full-run leg works on a CI runner that has no real config repo on disk."""
    p = tmp_path / "risk.yaml"
    p.write_text(_risk_config_text())
    monkeypatch.setattr("executor.main.get_config_path", lambda: str(p))
    return p


def test_signal_to_dryrun_plan_full_run(fake_risk_yaml):
    """SIGNAL → EXECUTOR DRY-RUN leg, top-level orchestration.

    Drive ``executor.main.run(simulate=True)`` — the executor's own broker-less
    mode — with a synthetic signals artifact and a SimulatedIBKRClient. Assert
    it returns a sane planned order list: one ENTER per buy-candidate, correctly
    sized, with no live broker / S3 / ArcticDB interaction (all upstream maps
    injected).
    """
    enter = _enter_signals(3)
    signals = _signals_artifact(enter)

    enter_tickers = [s["ticker"] for s in enter]
    all_tickers = list(set(enter_tickers + _SECTOR_ETFS))
    price_histories = {t: _df_history(base=100 + i) for i, t in enumerate(all_tickers)}
    atr_map = {t: 0.02 for t in all_tickers}
    vwap_map = {t: 100.0 for t in all_tickers}
    coverage_map = {t: 1.0 for t in all_tickers}
    prices_now = {t: 100.0 for t in all_tickers}

    sim_client = SimulatedIBKRClient(prices=prices_now, nav=1_000_000.0)

    orders = main_mod.run(
        dry_run=False,
        simulate=True,            # executor's broker-less mode → returns orders
        ibkr_client=sim_client,
        signals_override=signals,  # skip S3 read; feed synthetic signals
        price_histories=price_histories,
        atr_map=atr_map,
        vwap_map=vwap_map,
        coverage_map=coverage_map,
    )

    # run(simulate=True) returns the planned order list.
    assert isinstance(orders, list), "simulate=True must return an order list"

    entries = [o for o in orders if o.get("action") == "ENTER"]
    assert len(entries) == 3, (
        f"Expected one ENTER per buy-candidate (3); got {len(entries)}: "
        f"{[o.get('ticker') for o in entries]}"
    )

    ordered_tickers = sorted(o["ticker"] for o in entries)
    assert ordered_tickers == sorted(enter_tickers), (
        f"Planned tickers {ordered_tickers} != synthetic ENTER signals "
        f"{sorted(enter_tickers)}"
    )

    # Sizing sanity: $1M NAV × 5% max_position_pct ÷ $100 = 500 shares each.
    for o in entries:
        assert o["shares"] == 500, (
            f"{o['ticker']} sized to {o['shares']} shares; expected 500 "
            "(5% of $1M NAV at $100/share)"
        )
        assert o["price_at_order"] == 100.0
        assert o["shares"] > 0 and isinstance(o["shares"], int)

    # Determinism: a second identical run produces an identical plan.
    sim_client2 = SimulatedIBKRClient(prices=prices_now, nav=1_000_000.0)
    orders2 = main_mod.run(
        dry_run=False,
        simulate=True,
        ibkr_client=sim_client2,
        signals_override=signals,
        price_histories=price_histories,
        atr_map=atr_map,
        vwap_map=vwap_map,
        coverage_map=coverage_map,
    )
    assert orders2 == orders, "Dry-run plan is not deterministic across runs"


def test_inference_veto_blocks_entry():
    """INFERENCE → EXECUTOR leg, planner seam.

    Feed a canned predictor 'inference' artifact in which one ticker carries
    ``gbm_veto=True``. The executor's entry planner (``decide_entries``, the
    same pure decider the live shell delegates to) must override that research
    ENTER and block exactly the vetoed ticker, while the un-vetoed tickers still
    produce orders. This proves the synthetic inference payload reaches and
    steers the dry-run plan — the leg the top-level simulate path zeroes out.
    """
    enter = _enter_signals(3)
    signals = _signals_artifact(enter)
    enter_tickers = [s["ticker"] for s in enter]
    vetoed = enter_tickers[1]  # "TKR001"

    all_tickers = enter_tickers + ["SPY"]
    price_histories = {t: _df_history(base=100 + i) for i, t in enumerate(all_tickers)}

    # Canned predictor inference: TKR001 is vetoed by the GBM model.
    predictions_by_ticker = {
        vetoed: {
            "gbm_veto": True,
            "predicted_alpha": -0.03,
            "combined_rank": 900,
            "predicted_direction": "down",
            "prediction_confidence": 0.80,
        },
    }

    plan = decide_entries(
        enter_signals=enter,
        signals_raw=signals,
        predictions_by_ticker=predictions_by_ticker,
        config=_planner_config(),
        strategy_config=_strategy_config(),
        market_regime="neutral",
        sector_ratings=signals["sector_ratings"],
        portfolio_nav=1_000_000.0,
        peak_nav=1_000_000.0,
        current_positions={},
        prices_now={t: 100.0 for t in all_tickers},
        price_histories=price_histories,
        atr_map={t: 0.02 for t in all_tickers},
        vwap_map={t: 100.0 for t in all_tickers},
        coverage_map={t: 1.0 for t in all_tickers},
        dd_multiplier=1.0,
        signal_age_days=0,
        earnings_by_ticker={},
        run_date="2026-04-25",
    )

    ordered_tickers = {o["ticker"] for o in plan.orders}
    blocked_tickers = {b["ticker"] for b in plan.blocked}

    # The vetoed ticker is blocked, not ordered.
    assert vetoed not in ordered_tickers, (
        f"{vetoed} carried gbm_veto=True but still produced an order"
    )
    assert vetoed in blocked_tickers, f"{vetoed} should be blocked by GBM veto"

    veto_block = next(b for b in plan.blocked if b["ticker"] == vetoed)
    assert "GBM veto" in veto_block["block_reason"], (
        f"Block reason should cite the GBM veto; got {veto_block['block_reason']!r}"
    )

    # The two un-vetoed tickers still order — the inference payload is targeted,
    # not a blanket block.
    expected_ordered = sorted(set(enter_tickers) - {vetoed})
    assert sorted(ordered_tickers) == expected_ordered, (
        f"Un-vetoed tickers {expected_ordered} should still order; "
        f"got {sorted(ordered_tickers)}"
    )
    assert plan.n_entered == 2

    # The override is recorded as a structured risk event for audit.
    veto_events = [
        e for e in plan.risk_events if e.get("rule") == "predictor_gbm_veto"
    ]
    assert len(veto_events) == 1, (
        f"Expected one predictor_gbm_veto risk event; got {len(veto_events)}"
    )
