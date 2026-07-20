"""Unit tests for executor/intraday_resolve.py — the daemon's intraday
reconcile-to-target layer (real-time drawdown overlay + event-driven re-solve).

Covers the 2026-05-29 fix: under the optimizer, cash freed intraday by
hard-risk exits is redeployed back to the sleeve same-day, and a real-time
drawdown overlay can force de-risking + suppress redeploy.
"""

from __future__ import annotations

import numpy as np
import pytest

from executor.intraday_resolve import (
    available_redeploy_cash,
    build_conviction_map,
    build_redeploy_entry,
    compute_drawdown_overlay,
    select_forced_exits,
    solve_redeploy,
    w_prev_from_live,
)
from executor.portfolio_optimizer import (
    OPTIMIZER_CONFIG_DEFAULTS,
    _estimate_covariance_daily,
)


def test_build_conviction_map_flattens_universe_and_buy_candidates():
    # config#844: the daemon builds its forced-exit conviction map from the
    # signals.json payload (universe + buy_candidates lists), keyed by ticker.
    payload = {
        "universe": [{"ticker": "AAA", "score": 80}, {"ticker": "BBB", "score": 40}],
        "buy_candidates": [{"ticker": "CCC", "score": 90}, {"ticker": "AAA", "score": 5}],
    }
    m = build_conviction_map(payload)
    assert set(m) == {"AAA", "BBB", "CCC"}
    assert m["AAA"]["score"] == 80  # first record wins on duplicate
    assert m["BBB"]["score"] == 40 and m["CCC"]["score"] == 90


def test_build_conviction_map_degrades_on_missing_or_malformed():
    assert build_conviction_map(None) == {}
    assert build_conviction_map({}) == {}
    # malformed records / missing ticker are skipped, not raised
    assert build_conviction_map({"universe": ["NOTADICT", {"score": 1}]}) == {}


def test_conviction_map_feeds_lowest_conviction_first_ranking():
    # End-to-end of the config#844 fix: a real signals payload → conviction map
    # → select_forced_exits ranks lowest-score-first (NOT smallest-position).
    payload = {"universe": [
        {"ticker": "HIGH", "score": 90},
        {"ticker": "LOW", "score": 20},
        {"ticker": "MID", "score": 55},
    ]}
    # LOW has the LARGEST position — smallest-position-first would spare it, but
    # conviction ranking must exit it first.
    positions = {
        "HIGH": {"shares": 10, "market_value": 1_000},
        "LOW": {"shares": 10, "market_value": 9_000},
        "MID": {"shares": 10, "market_value": 5_000},
    }
    m = build_conviction_map(payload)
    out = select_forced_exits(positions, m, set(), target_count=1)
    assert [o["ticker"] for o in out] == ["LOW"]


def test_empty_conviction_map_falls_back_to_smallest_position_first():
    # When signals are unreadable (empty map), ranking degrades to
    # smallest-position-first — the prior conservative behavior is preserved.
    positions = {
        "BIG": {"shares": 10, "market_value": 9_000},
        "SMALL": {"shares": 10, "market_value": 1_000},
    }
    out = select_forced_exits(positions, build_conviction_map(None), set(), target_count=1)
    assert [o["ticker"] for o in out] == ["SMALL"]


# ── Drawdown overlay ─────────────────────────────────────────────────────────

def _dd_config():
    return {
        "drawdown_circuit_breaker": 0.08,
        "strategy": {
            "graduated_drawdown": {
                "enabled": True,
                "tiers": [
                    [-0.02, 1.00, "0 to -2%"],
                    [-0.04, 0.50, "-2 to -4%"],
                    [-0.06, 0.25, "-4 to -6%"],
                ],
            },
        },
    }


def _strategy_config():
    from executor.strategies.config import load_strategy_config
    return load_strategy_config(_dd_config())


def test_overlay_no_drawdown_allows_redeploy():
    overlay = compute_drawdown_overlay(1_000_000, 1_000_000, _dd_config(), _strategy_config())
    assert overlay["multiplier"] == pytest.approx(1.0)
    assert overlay["forced_exit_count"] == 0
    assert overlay["redeploy_suppressed"] is False


def test_overlay_deep_drawdown_forces_exits_and_suppresses_redeploy():
    # -7% from peak → deepest soft tier (mult 0.25) → tier3 forced exits.
    overlay = compute_drawdown_overlay(930_000, 1_000_000, _dd_config(), _strategy_config())
    assert overlay["multiplier"] <= 0.25
    assert overlay["forced_exit_count"] == 2  # tier3 default
    assert overlay["redeploy_suppressed"] is True


# ── Forced-exit selection ────────────────────────────────────────────────────

def test_select_forced_exits_lowest_conviction_first_and_idempotent():
    positions = {
        "AAA": {"shares": 10, "market_value": 1000},
        "BBB": {"shares": 10, "market_value": 5000},
        "CCC": {"shares": 10, "market_value": 9000},
    }
    signals = {"AAA": {"score": 80}, "BBB": {"score": 40}, "CCC": {"score": 90}}
    out = select_forced_exits(positions, signals, set(), target_count=2)
    tickers = [o["ticker"] for o in out]
    assert tickers == ["BBB", "AAA"]  # lowest score first, then next-lowest
    assert all(o["reason"] == "drawdown_forced_exit" for o in out)
    # Idempotent: with one already exited, only the remaining headroom returns.
    out2 = select_forced_exits(positions, signals, {"BBB"}, target_count=2)
    assert [o["ticker"] for o in out2] == ["AAA"]
    # Target already met → nothing.
    assert select_forced_exits(positions, signals, {"BBB", "AAA"}, 2) == []


# ── Live weights / available cash ────────────────────────────────────────────

def test_w_prev_from_live_and_available_cash():
    tickers = ["AAA", "BBB", "SPY", "CASH"]
    positions = {"AAA": {"market_value": 80_000}, "SPY": {"market_value": 100_000}}
    nav = 1_000_000.0
    w = w_prev_from_live(tickers, positions, nav, cash_idx=3)
    assert w[0] == pytest.approx(0.08)
    assert w[2] == pytest.approx(0.10)
    assert w[3] == pytest.approx(1.0 - 0.18)  # cash absorbs the rest
    # Excess over a 3% sleeve, net of pending entries.
    avail = available_redeploy_cash(tickers, positions, nav, sleeve_pct=0.03, pending_entry_dollars=0.0)
    assert avail == pytest.approx((0.82 - 0.03) * nav)
    avail2 = available_redeploy_cash(tickers, positions, nav, 0.03, pending_entry_dollars=200_000)
    assert avail2 == pytest.approx((0.82 - 0.03) * nav - 200_000)


# ── solve_redeploy ───────────────────────────────────────────────────────────

def _shadow_log(n_active=2, seed=1):
    tickers = [f"T{i}" for i in range(n_active)] + ["SPY", "CASH"]
    N = len(tickers)
    cash_idx = N - 1
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(250, N))
    returns[:, cash_idx] = 0.0
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS}
    sigma_daily = _estimate_covariance_daily(returns, cfg)
    alpha_hat = [0.05, 0.04, 0.03][:n_active] + [0.0, -1e-6]  # active picks + SPY + CASH
    stance = [0.08] * n_active + [1.0, 1.0]
    sectors = ["tech"] * n_active + ["__benchmark__", "__cash__"]
    return {
        "tickers": tickers,
        "alpha_hat": alpha_hat,
        "eligibility": [True] * N,
        "stance_caps": stance,
        "sectors": sectors,
        "covariance_daily": [[float(x) for x in row] for row in sigma_daily],
        "alpha_uncertainty": None,
        "optimizer_cfg": cfg,
    }


def test_solve_redeploy_deploys_excess_cash():
    log = _shadow_log(n_active=2)
    # Lots of idle cash: only T0 held at 8%, rest cash.
    positions = {"T0": {"market_value": 80_000}}
    res = solve_redeploy(shadow_log=log, current_positions=positions, nav=1_000_000.0, stopped_out=set())
    assert res["status"] in ("optimal", "optimal_inaccurate")
    buy_tickers = {b["ticker"] for b in res["buys"]}
    # Should redeploy into T1 and SPY (T0 already at target cap).
    assert "SPY" in buy_tickers
    assert "T1" in buy_tickers
    assert "CASH" not in buy_tickers


def test_solve_redeploy_excludes_stopped_out_name():
    log = _shadow_log(n_active=2)
    positions = {"T0": {"market_value": 80_000}}
    res = solve_redeploy(
        shadow_log=log, current_positions=positions, nav=1_000_000.0,
        stopped_out={"T1"},
    )
    assert res["status"] in ("optimal", "optimal_inaccurate")
    assert "T1" not in {b["ticker"] for b in res["buys"]}, \
        "A name a gap stop just exited must not be re-bought same-day"


def test_solve_redeploy_nonoptimal_returns_no_buys():
    log = _shadow_log(n_active=2)
    # Make it infeasible: caps too small to reach 0.97 with cash pinned at 0.03.
    log["stance_caps"] = [0.01, 0.01, 0.01, 1.0]
    positions = {"T0": {"market_value": 80_000}}
    res = solve_redeploy(shadow_log=log, current_positions=positions, nav=1_000_000.0, stopped_out=set())
    assert res["status"] not in ("optimal", "optimal_inaccurate")
    assert res["buys"] == []


# ── Redeploy entry record ────────────────────────────────────────────────────

def test_build_redeploy_entry_shape():
    e = build_redeploy_entry("SPY", 100, 755.0, 0.81, "2026-05-29")
    assert e["ticker"] == "SPY"
    assert e["signal"] == "ENTER"
    assert e["shares"] == 100
    assert e["sizing_source"] == "optimizer_redeploy"
    assert e["status"] == "pending"
    assert e["triggers"]["vwap"] is None
