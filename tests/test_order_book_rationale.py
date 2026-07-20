"""Tests for executor.order_book_rationale.

Covers the pure builder's terminal-state resolution across the full
considered universe, the decision-chain construction, the summary
counts, and the canonical eval_artifacts write (dated artifact +
sidecar, round-trippable via the lib's load_latest_eval_artifact).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from executor.order_book_rationale import (
    DEFAULT_S3_PREFIX,
    SCHEMA_VERSION,
    STATE_APPROVED_ENTRY,
    STATE_HELD,
    STATE_NO_ACTION,
    STATE_NO_ACTION_OPTIMIZER_DROPPED,
    STATE_NO_ACTION_OPTIMIZER_ZERO,
    STATE_NO_ACTION_UNKNOWN,
    STATE_PREDICTOR_VETOED,
    STATE_REDUCE,
    STATE_RISK_BLOCKED,
    STATE_URGENT_EXIT,
    build_order_book_rationale,
    write_order_book_rationale,
)

# ── fixtures ─────────────────────────────────────────────────────────────


def _sig(ticker, signal, **kw):
    base = {
        "ticker": ticker,
        "signal": signal,
        "score": 72.0,
        "conviction": "rising",
        "rating": "BUY",
        "sector": "Technology",
        "sector_rating": "overweight",
        "price_target_upside": 0.18,
        "thesis_summary": f"{ticker} thesis",
    }
    base.update(kw)
    return base


def _entry_with_meta(ticker):
    return {
        "ticker": ticker,
        "signal": "ENTER",
        "shares": 100,
        "current_price": 50.0,
        "dollar_size": 5000.0,
        "position_pct": 0.041,
        "triggers": {"pullback_pct": 0.012, "vwap": 49.5, "support_level": 48.0},
        "sizing_factors": {
            "sector_adj": 1.2,
            "conviction_adj": 1.05,
            "upside_adj": 1.0,
            "dd_multiplier": 1.0,
        },
        "predicted_direction": "UP",
        "prediction_confidence": 0.62,
        "status": "pending",
    }


@pytest.fixture
def scenario():
    """A run spanning every terminal state.

    AAPL → approved entry, MSFT → urgent exit, TSLA → reduce,
    NVDA → risk-blocked (min_score), AMD → predictor-vetoed
    (stance_gate), KO → held, XOM → considered but no action.
    """
    signals = {
        "enter": [_sig("AAPL", "ENTER"), _sig("NVDA", "ENTER", score=40.0),
                  _sig("AMD", "ENTER")],
        "exit": [_sig("MSFT", "EXIT"), _sig("XOM", "EXIT")],
        "reduce": [_sig("TSLA", "REDUCE")],
        "hold": [_sig("KO", "HOLD")],
    }
    predictions = {
        "AAPL": {"predicted_direction": "UP", "prediction_confidence": 0.62,
                 "predicted_alpha": 0.03, "stance": "momentum",
                 "catalyst_date": "2026-05-20"},
        "AMD": {"predicted_direction": "DOWN", "prediction_confidence": 0.71},
    }
    order_book = {
        "date": "2026-05-15",
        "approved_entries": [_entry_with_meta("AAPL")],
        "urgent_exits": [
            {"ticker": "MSFT", "signal": "EXIT", "shares": 50,
             "reason": "research_exit"},
            {"ticker": "TSLA", "signal": "REDUCE", "shares": 20,
             "reason": "trim"},
        ],
        "active_stops": [],
        "executed_today": [],
    }
    blocked = [
        {"ticker": "NVDA", "block_reason": "score 40.0 < min 55.0"},
        {"ticker": "AMD", "block_reason": "stance gate: DOWN veto"},
    ]
    risk_events = [
        {"ticker": "NVDA", "event_type": "veto", "rule": "min_score",
         "value": 40.0, "threshold": 55.0, "reason": "below min score"},
        {"ticker": "AMD", "event_type": "veto", "rule": "stance_gate",
         "value": 0.71, "threshold": 0.6, "reason": "predictor DOWN veto"},
    ]
    return {
        "signals": signals,
        "predictions_by_ticker": predictions,
        "order_book_data": order_book,
        "blocked_entries": blocked,
        "risk_events": risk_events,
        "market_regime": "neutral",
        "run_date": "2026-05-15",
        "signal_date": "2026-05-15",
        "prediction_date": "2026-05-15",
        "calendar_date": "2026-05-15",
        "trading_day": "2026-05-15",
        "run_id": "2605151400",
        # KO is the held ticker — must come from portfolio truth, not
        # the research `hold` bucket (research HOLD on a non-held ticker
        # is a recommendation that wasn't acted on, not a held state).
        "current_positions": {"KO": {"mkt_val": 1000}},
    }


# ── builder: terminal-state resolution ───────────────────────────────────


def test_every_terminal_state_resolved(scenario):
    payload = build_order_book_rationale(**scenario)
    by_ticker = {r["ticker"]: r["terminal_state"] for r in payload["tickers"]}
    assert by_ticker["AAPL"] == STATE_APPROVED_ENTRY
    assert by_ticker["MSFT"] == STATE_URGENT_EXIT
    assert by_ticker["TSLA"] == STATE_REDUCE
    assert by_ticker["NVDA"] == STATE_RISK_BLOCKED
    assert by_ticker["AMD"] == STATE_PREDICTOR_VETOED
    assert by_ticker["KO"] == STATE_HELD
    # XOM: research EXIT signal on a non-held ticker — operationally
    # a dead signal (no possible order) so filtered out of the
    # considered universe entirely under 1.2.0+. Pre-1.2.0 emitted a
    # bare STATE_NO_ACTION row for it.
    assert "XOM" not in by_ticker


def test_considered_universe_is_union_of_all_sources(scenario):
    # XOM (research EXIT on a non-held ticker) is filtered out under
    # the 1.2.0+ rule — operationally dead signal, nothing to sell.
    payload = build_order_book_rationale(**scenario)
    tickers = {r["ticker"] for r in payload["tickers"]}
    assert tickers == {"AAPL", "MSFT", "TSLA", "NVDA", "AMD", "KO"}
    assert payload["summary"]["n_considered"] == 6


def test_excluded_ticker_carries_specific_gate(scenario):
    payload = build_order_book_rationale(**scenario)
    nvda = next(r for r in payload["tickers"] if r["ticker"] == "NVDA")
    assert nvda["exclusion"]["rule"] == "min_score"
    assert nvda["exclusion"]["value"] == 40.0
    assert nvda["exclusion"]["threshold"] == 55.0
    # The risk_guard stage in the chain mirrors the slug.
    rg = next(s for s in nvda["decision_chain"] if s["stage"] == "risk_guard")
    assert rg["result"] == "blocked"
    assert rg["rule"] == "min_score"


def test_predictor_veto_distinguished_from_hard_risk(scenario):
    payload = build_order_book_rationale(**scenario)
    amd = next(r for r in payload["tickers"] if r["ticker"] == "AMD")
    assert amd["terminal_state"] == STATE_PREDICTOR_VETOED
    assert amd["exclusion"]["rule"] == "stance_gate"


def test_approved_entry_chain_has_sizing_and_trigger(scenario):
    payload = build_order_book_rationale(**scenario)
    aapl = next(r for r in payload["tickers"] if r["ticker"] == "AAPL")
    stages = {s["stage"] for s in aapl["decision_chain"]}
    assert {"signal_read", "predictor", "risk_guard", "position_sizer",
            "entry_trigger"} <= stages
    sizer = next(s for s in aapl["decision_chain"]
                 if s["stage"] == "position_sizer")
    assert sizer["sizing_factors"]["sector_adj"] == 1.2
    assert sizer["shares"] == 100
    # config#1436: position_sizer stage always carries pricing_source
    # (None on this legacy non-optimizer entry; "ibkr"/"price_history_close"
    # on optimizer-sized names).
    assert "pricing_source" in sizer
    trig = next(s for s in aapl["decision_chain"]
                if s["stage"] == "entry_trigger")
    assert trig["triggers"]["vwap"] == 49.5


def test_research_and_predictor_blocks_projected(scenario):
    payload = build_order_book_rationale(**scenario)
    aapl = next(r for r in payload["tickers"] if r["ticker"] == "AAPL")
    assert aapl["research"]["score"] == 72.0
    assert aapl["research"]["sector_rating"] == "overweight"
    assert aapl["predictor"]["predicted_direction"] == "UP"
    assert aapl["predictor"]["catalyst_date"] == "2026-05-20"
    # KO has no prediction → empty predictor block, not a crash.
    ko = next(r for r in payload["tickers"] if r["ticker"] == "KO")
    assert ko["predictor"] == {}


def test_summary_counts_match_records(scenario):
    payload = build_order_book_rationale(**scenario)
    s = payload["summary"]
    assert s[f"n_{STATE_APPROVED_ENTRY}"] == 1
    assert s[f"n_{STATE_URGENT_EXIT}"] == 1
    assert s[f"n_{STATE_REDUCE}"] == 1
    assert s[f"n_{STATE_RISK_BLOCKED}"] == 1
    assert s[f"n_{STATE_PREDICTOR_VETOED}"] == 1
    assert s[f"n_{STATE_HELD}"] == 1
    # XOM (research EXIT on a non-held ticker) is filtered out — no
    # no-action sub-state appears in this scenario's summary at all.
    assert f"n_{STATE_NO_ACTION}" not in s
    assert f"n_{STATE_NO_ACTION_OPTIMIZER_ZERO}" not in s
    assert f"n_{STATE_NO_ACTION_UNKNOWN}" not in s


def test_records_sorted_actioned_first(scenario):
    payload = build_order_book_rationale(**scenario)
    states = [r["terminal_state"] for r in payload["tickers"]]
    assert states[0] == STATE_APPROVED_ENTRY
    # KO (STATE_HELD) is the lowest-priority remaining state after
    # XOM was filtered out — held tickers always sit last among the
    # remaining actioned/considered states.
    assert states[-1] == STATE_HELD


def test_payload_is_audit_stable_and_serializable(scenario):
    payload = build_order_book_rationale(**scenario)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_id"] == "2605151400"
    assert payload["trading_day"] == "2026-05-15"
    # Reproducible from the artifact alone → must round-trip JSON.
    assert json.loads(json.dumps(payload, default=str)) == json.loads(
        json.dumps(payload, default=str)
    )


def test_empty_run_produces_empty_but_valid_payload():
    payload = build_order_book_rationale(
        signals={"enter": [], "exit": [], "reduce": [], "hold": []},
        predictions_by_ticker={},
        order_book_data={"approved_entries": [], "urgent_exits": []},
        blocked_entries=[],
        risk_events=[],
        market_regime="bear",
        run_date="2026-05-15",
        signal_date="2026-05-15",
        prediction_date=None,
        calendar_date="2026-05-15",
        trading_day="2026-05-15",
        run_id="2605151400",
    )
    assert payload["summary"]["n_considered"] == 0
    assert payload["tickers"] == []
    assert payload["prediction_date"] is None


# ── writer: canonical eval_artifacts shape ───────────────────────────────


class _StubS3:
    """Minimal in-memory S3 supporting put_object + get_object."""

    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        body = self.store[(Bucket, Key)]

        class _Body:
            def read(self_inner):
                return body

        return {"Body": _Body()}


def test_write_creates_dated_artifact_and_sidecar(scenario):
    payload = build_order_book_rationale(**scenario)
    s3 = _StubS3()
    result = write_order_book_rationale(
        payload, s3_client=s3, bucket="alpha-engine-research"
    )
    assert result["artifact_key"] == (
        f"{DEFAULT_S3_PREFIX}/2605151400.json"
    )
    assert result["latest_key"] == f"{DEFAULT_S3_PREFIX}/latest.json"
    sidecar = json.loads(
        s3.store[("alpha-engine-research", result["latest_key"])]
    )
    # Sidecar must carry artifact_key so load_latest_eval_artifact resolves.
    assert sidecar["artifact_key"] == result["artifact_key"]
    assert sidecar["run_id"] == "2605151400"
    assert sidecar["n_considered"] == 6


def test_write_latest_false_skips_sidecar(scenario):
    payload = build_order_book_rationale(**scenario)
    s3 = _StubS3()
    result = write_order_book_rationale(
        payload, s3_client=s3, bucket="b", write_latest=False
    )
    assert "latest_key" not in result
    assert ("b", f"{DEFAULT_S3_PREFIX}/latest.json") not in s3.store


def test_round_trip_via_lib_loader(scenario):
    """The artifact resolves through the lib's canonical reader —
    proves the dashboard's load_latest_eval_artifact path works."""
    from nousergon_lib.eval_artifacts import load_latest_eval_artifact

    payload = build_order_book_rationale(**scenario)
    s3 = _StubS3()
    write_order_book_rationale(
        payload, s3_client=s3, bucket="alpha-engine-research"
    )
    loaded = load_latest_eval_artifact(
        s3, bucket="alpha-engine-research", prefix=DEFAULT_S3_PREFIX
    )
    assert loaded["run_id"] == payload["run_id"]
    assert loaded["summary"]["n_considered"] == 6


# ── optimizer-aware path (L121) ─────────────────────────────────────────


class TestOptimizerShadowLogSynthesis:
    """When use_portfolio_optimizer: true, the legacy _plan_entries path
    is bypassed and blocked_entries / risk_events arrive EMPTY. The
    rationale producer must source per-ticker rejection reasons from
    the optimizer's shadow log to answer "why didn't ticker X enter?"
    on the optimizer-driven path (ROADMAP L121)."""

    def _optimizer_scenario(self):
        """Optimizer-driven run. legacy lists are empty (legacy path skipped).
        The shadow log carries the eligibility mask + reasons.
        order_book_data carries the optimizer-translated approved + exits."""
        signals = {
            "enter": [
                _sig("AAPL", "ENTER"),
                _sig("NVDA", "ENTER", score=40.0),  # below min_score
                _sig("AMD", "ENTER"),               # gbm_veto
                _sig("F", "ENTER"),                 # universe but no score
            ],
            "exit": [_sig("MSFT", "EXIT")],
            "reduce": [],
            "hold": [_sig("KO", "HOLD")],
        }
        signals["enter"][3].pop("score", None)  # F has no score
        predictions = {
            "AAPL": {"predicted_direction": "UP", "predicted_alpha": 0.03},
            "AMD": {"predicted_direction": "DOWN", "gbm_veto": True},
        }
        # Optimizer-translated order book: AAPL is the lone approved
        # entry; MSFT comes through urgent_exits from the EXIT signal.
        order_book = {
            "date": "2026-05-15",
            "approved_entries": [_entry_with_meta("AAPL")],
            "urgent_exits": [
                {"ticker": "MSFT", "signal": "EXIT", "shares": 50,
                 "reason": "research_exit"},
            ],
            "active_stops": [],
            "executed_today": [],
        }
        # Optimizer-shadow log shape per executor.optimizer_shadow._build_and_solve.
        shadow_log = {
            "shadow_status": "ok",
            "tickers": ["AAPL", "NVDA", "AMD", "F", "MSFT", "KO", "SPY", "CASH"],
            "eligibility": [True, False, False, False, False, True, True, True],
            "eligibility_reasons": [
                None,
                "score_below_min",
                "gbm_veto",
                "no_score",
                "signal_exit",
                None,
                None,
                None,
            ],
            "optimizer_cfg": {"min_score_to_enter": 57.0},
            "diagnostics": {"status": "optimal"},
        }
        return {
            "signals": signals,
            "predictions_by_ticker": predictions,
            "order_book_data": order_book,
            "blocked_entries": [],   # legacy path skipped
            "risk_events": [],       # legacy path skipped
            "market_regime": "neutral",
            "run_date": "2026-05-15",
            "signal_date": "2026-05-15",
            "prediction_date": "2026-05-15",
            "calendar_date": "2026-05-15",
            "trading_day": "2026-05-15",
            "run_id": "2605151400",
            "optimizer_shadow_log": shadow_log,
            "current_positions": {"KO": {"mkt_val": 1000}},
        }

    def test_optimizer_rejected_tickers_surface_in_rationale(self):
        scenario = self._optimizer_scenario()
        payload = build_order_book_rationale(**scenario)
        by_ticker = {r["ticker"]: r for r in payload["tickers"]}
        # NVDA: score below min → STATE_RISK_BLOCKED with min_score rule.
        assert by_ticker["NVDA"]["terminal_state"] == STATE_RISK_BLOCKED
        assert by_ticker["NVDA"]["exclusion"]["rule"] == "min_score_to_enter"
        assert by_ticker["NVDA"]["exclusion"]["value"] == 40.0
        assert by_ticker["NVDA"]["exclusion"]["threshold"] == 57.0

    def test_optimizer_gbm_veto_surfaces_as_predictor_vetoed(self):
        scenario = self._optimizer_scenario()
        payload = build_order_book_rationale(**scenario)
        by_ticker = {r["ticker"]: r for r in payload["tickers"]}
        # AMD: gbm_veto → STATE_PREDICTOR_VETOED (event_type=override or
        # rule in _PREDICTOR_RULES).
        assert by_ticker["AMD"]["terminal_state"] == STATE_PREDICTOR_VETOED
        assert by_ticker["AMD"]["exclusion"]["rule"] == "gbm_veto"

    def test_optimizer_no_score_surfaces_as_risk_blocked(self):
        scenario = self._optimizer_scenario()
        payload = build_order_book_rationale(**scenario)
        by_ticker = {r["ticker"]: r for r in payload["tickers"]}
        # F: no score → STATE_RISK_BLOCKED with missing_score rule.
        assert by_ticker["F"]["terminal_state"] == STATE_RISK_BLOCKED
        assert by_ticker["F"]["exclusion"]["rule"] == "missing_score"

    def test_optimizer_held_ticker_does_not_get_synthetic_rejection(self):
        # KO is held (current_positions). Even if it were ineligible, we
        # should NOT fabricate a "blocked" record — exits go through
        # urgent_exits. Here KO IS eligible, so this is also a sanity
        # check that the eligible branch doesn't produce noise.
        scenario = self._optimizer_scenario()
        payload = build_order_book_rationale(**scenario)
        by_ticker = {r["ticker"]: r for r in payload["tickers"]}
        # KO is held with no rebalance trade → STATE_HELD (research said
        # HOLD; no order_book record).
        assert by_ticker["KO"]["terminal_state"] == STATE_HELD

    def test_legacy_path_with_no_shadow_log_unchanged(self, scenario):
        # Backwards compat: omitting optimizer_shadow_log → identical
        # output to pre-L121 behavior. Both paths still receive the
        # scenario's `current_positions` (the held-ticker source), so
        # the diff isolates shadow-log presence only.
        payload_legacy = build_order_book_rationale(**scenario)
        scenario_no_log = {**scenario, "optimizer_shadow_log": None}
        payload_no_log = build_order_book_rationale(**scenario_no_log)
        assert payload_legacy["summary"] == payload_no_log["summary"]
        assert payload_legacy["tickers"] == payload_no_log["tickers"]

    def test_legacy_blocked_takes_precedence_over_synth(self):
        # If both legacy and shadow_log produce rejection records for
        # the same ticker, the legacy entry wins (it ran, it's
        # authoritative). Mixed-path defensive case — production never
        # produces this, but the dedup must be deterministic.
        scenario = self._optimizer_scenario()
        scenario["blocked_entries"] = [
            {"ticker": "NVDA", "block_reason": "LEGACY reason"}
        ]
        scenario["risk_events"] = [
            {"ticker": "NVDA", "event_type": "veto", "rule": "legacy_rule",
             "value": 1.0, "threshold": 2.0, "reason": "LEGACY reason"}
        ]
        payload = build_order_book_rationale(**scenario)
        nvda = next(r for r in payload["tickers"] if r["ticker"] == "NVDA")
        # Legacy rule slug survives — synth row was filtered.
        assert nvda["exclusion"]["rule"] == "legacy_rule"

    def test_reconciliation_fields_projected_from_shadow_log(self):
        # Producer must surface portfolio_nav, would_be_trades, and the
        # rebalance band into the rationale payload so the dashboard
        # renders the target-vs-current-vs-planned reconciliation view
        # without a second S3 read of the optimizer shadow log.
        scenario = self._optimizer_scenario()
        scenario["optimizer_shadow_log"] = {
            **scenario["optimizer_shadow_log"],
            "portfolio_nav": 125_000.0,
            "would_be_trades": [
                {"ticker": "AAPL", "action": "BUY",
                 "delta_weight": 0.041, "delta_dollars": 5125.0,
                 "target_weight": 0.041, "current_weight": 0.0},
                {"ticker": "MSFT", "action": "SELL",
                 "delta_weight": -0.032, "delta_dollars": -4000.0,
                 "target_weight": 0.0, "current_weight": 0.032},
            ],
            "optimizer_cfg": {
                **scenario["optimizer_shadow_log"]["optimizer_cfg"],
                "rebalance_band_pct": 0.005,
            },
        }
        payload = build_order_book_rationale(**scenario)
        assert payload["portfolio_nav"] == 125_000.0
        assert payload["rebalance_band_pct"] == 0.005
        trades = payload["optimizer_trades"]
        assert isinstance(trades, list) and len(trades) == 2
        aapl_trade = next(t for t in trades if t["ticker"] == "AAPL")
        assert aapl_trade["action"] == "BUY"
        assert aapl_trade["delta_dollars"] == 5125.0
        assert aapl_trade["target_weight"] == 0.041

    def test_reconciliation_fields_none_when_no_shadow_log(self, scenario):
        # Legacy non-optimizer run → fields default to None so the
        # consumer can detect "no reconciliation view available" without
        # KeyErrors.
        payload = build_order_book_rationale(**scenario)
        assert payload["portfolio_nav"] is None
        assert payload["optimizer_trades"] is None
        assert payload["rebalance_band_pct"] is None

    def test_reconciliation_fields_partial_shadow_log_safe(self):
        # Defensive: a shadow log that omits portfolio_nav /
        # would_be_trades / rebalance_band_pct (e.g. an older
        # producer or a sentinel-state log) must not crash the
        # rationale build. Missing fields → None.
        scenario = self._optimizer_scenario()
        scenario["optimizer_shadow_log"] = {
            "shadow_status": "ok",
            "tickers": scenario["optimizer_shadow_log"]["tickers"],
            "eligibility": scenario["optimizer_shadow_log"]["eligibility"],
            "eligibility_reasons":
                scenario["optimizer_shadow_log"]["eligibility_reasons"],
            "optimizer_cfg": scenario["optimizer_shadow_log"]["optimizer_cfg"],
        }
        payload = build_order_book_rationale(**scenario)
        assert payload["portfolio_nav"] is None
        assert payload["optimizer_trades"] is None
        assert payload["rebalance_band_pct"] is None

    def test_optimizer_shadow_eligibility_reasons_field_in_log(self):
        # Sanity: the field name the producer reads matches what
        # optimizer_shadow.py emits. Pins the contract between the two
        # modules so a rename of either side is caught.
        import numpy as np

        from executor.optimizer_shadow import _build_eligibility
        elig, reasons = _build_eligibility(
            tickers=["AAA", "BBB", "SPY", "CASH"],
            signals_by_ticker={"AAA": {"score": 80, "signal": "ENTER"},
                               "BBB": {"score": 30, "signal": "ENTER"}},
            predictions_by_ticker={},
            current_positions={},
            config={"min_score_to_enter": 57},
            spy_idx=2,
            cash_idx=3,
        )
        assert isinstance(elig, np.ndarray)
        assert elig.tolist() == [True, False, True, True]
        assert reasons == [None, "score_below_min", None, None]


# ── held detection sourced from portfolio truth (current_positions) ──
#
# Regression coverage for the 2026-05-20 incident: AXP/LMT were held in
# the IB portfolio with the optimizer maintaining their target weights,
# but the producer was deriving STATE_HELD from `signals.json["hold"]`
# (a research recommendation, not portfolio truth) — so AXP/LMT fell
# through to STATE_NO_ACTION while SYK (research HOLD but cur_w=0)
# wrongly showed as STATE_HELD. The fix routes the held branch through
# `current_positions` and surfaces the optimizer's maintain-decision.


class TestHeldFromPortfolioTruth:
    def _empty_books(self) -> dict[str, Any]:
        return {
            "date": "2026-05-20",
            "approved_entries": [],
            "urgent_exits": [],
            "active_stops": [],
            "executed_today": [],
        }

    def test_held_ticker_with_research_enter_signal_state_held(self):
        # AXP case: held in portfolio, research signal is ENTER (wants
        # to add more), optimizer maintains target weight ≈ current
        # weight, no order_book entry. Must resolve to STATE_HELD.
        payload = build_order_book_rationale(
            signals={"enter": [_sig("AXP", "ENTER", score=75.0)],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={
                "AXP": {"predicted_direction": "UP",
                        "prediction_confidence": 0.33}},
            order_book_data=self._empty_books(),
            blocked_entries=[],
            risk_events=[],
            market_regime="neutral",
            run_date="2026-05-20",
            signal_date="2026-05-20",
            prediction_date="2026-05-20",
            calendar_date="2026-05-20",
            trading_day="2026-05-20",
            run_id="2605201400",
            current_positions={"AXP": {"mkt_val": 8000}},
        )
        axp = next(r for r in payload["tickers"] if r["ticker"] == "AXP")
        assert axp["terminal_state"] == STATE_HELD
        assert axp["held"] is True

    def test_non_held_ticker_with_research_hold_filtered_out(self):
        # SYK case: research recommends HOLD but the ticker is not in
        # the portfolio. Research HOLD on a non-held ticker is an
        # informational opinion with NO possible order-book interaction
        # (we can't "hold" something we don't own; HOLD does not
        # initiate a buy). Under 1.2.0+ such rows are filtered from
        # the considered universe rather than added as a no-action
        # row — they would only bulk the table with noise the
        # operator cannot act on.
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [],
                     "hold": [_sig("SYK", "HOLD")]},
            predictions_by_ticker={},
            order_book_data=self._empty_books(),
            blocked_entries=[],
            risk_events=[],
            market_regime="neutral",
            run_date="2026-05-20",
            signal_date="2026-05-20",
            prediction_date="2026-05-20",
            calendar_date="2026-05-20",
            trading_day="2026-05-20",
            run_id="2605201400",
            current_positions={},
        )
        tickers = {r["ticker"] for r in payload["tickers"]}
        assert "SYK" not in tickers
        assert payload["summary"]["n_considered"] == 0

    def test_held_ticker_silent_from_research_appears_in_table(self):
        # A position the IB portfolio holds but research has no opinion
        # on this week must STILL appear in the considered universe so
        # the table can answer "why is X held" for the full portfolio.
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            order_book_data=self._empty_books(),
            blocked_entries=[],
            risk_events=[],
            market_regime="neutral",
            run_date="2026-05-20",
            signal_date="2026-05-20",
            prediction_date="2026-05-20",
            calendar_date="2026-05-20",
            trading_day="2026-05-20",
            run_id="2605201400",
            current_positions={"LMT": {"mkt_val": 7000}},
        )
        tickers = {r["ticker"] for r in payload["tickers"]}
        assert "LMT" in tickers
        lmt = next(r for r in payload["tickers"] if r["ticker"] == "LMT")
        assert lmt["terminal_state"] == STATE_HELD
        assert lmt["held"] is True

    def test_optimizer_block_populated_from_shadow_log(self):
        # When the shadow log is present, each record carries the
        # per-ticker optimizer view (current_weight, target_weight,
        # alpha_hat, eligible) so the dashboard can render the
        # maintain/reduce/select decision without re-loading the log.
        shadow = {
            "tickers": ["AXP", "MSFT"],
            "current_weights": [0.08, 0.0],
            "target_weights": [0.08, 0.0],
            "alpha_hat": [0.039, -0.02],
            "eligibility": [True, True],
        }
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            order_book_data=self._empty_books(),
            blocked_entries=[],
            risk_events=[],
            market_regime="neutral",
            run_date="2026-05-20",
            signal_date="2026-05-20",
            prediction_date="2026-05-20",
            calendar_date="2026-05-20",
            trading_day="2026-05-20",
            run_id="2605201400",
            optimizer_shadow_log=shadow,
            current_positions={"AXP": {"mkt_val": 8000}},
        )
        axp = next(r for r in payload["tickers"] if r["ticker"] == "AXP")
        assert axp["optimizer"]["current_weight"] == 0.08
        assert axp["optimizer"]["target_weight"] == 0.08
        assert axp["optimizer"]["eligible"] is True
        # Optimizer maintain-decision is in the chain for held tickers.
        opt_stage = next(
            (s for s in axp["decision_chain"] if s["stage"] == "optimizer"),
            None,
        )
        assert opt_stage is not None
        assert opt_stage["result"] == "maintain"
        assert "tgt 8.00%" in opt_stage["detail"]
        assert "cur 8.00%" in opt_stage["detail"]

    def test_legacy_path_records_carry_held_false_and_null_optimizer(self):
        # No shadow_log + no current_positions → backwards-compat:
        # `held` is False for every record, `optimizer` is None.
        payload = build_order_book_rationale(
            signals={"enter": [_sig("AAPL", "ENTER")],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            order_book_data={"date": "2026-05-20",
                             "approved_entries": [_entry_with_meta("AAPL")],
                             "urgent_exits": [], "active_stops": [],
                             "executed_today": []},
            blocked_entries=[],
            risk_events=[],
            market_regime="neutral",
            run_date="2026-05-20",
            signal_date="2026-05-20",
            prediction_date="2026-05-20",
            calendar_date="2026-05-20",
            trading_day="2026-05-20",
            run_id="2605201400",
        )
        aapl = next(r for r in payload["tickers"] if r["ticker"] == "AAPL")
        assert aapl["held"] is False
        assert aapl["optimizer"] is None


# ── L171: OBR write-failure surfaces via alerts.publish ───────────────────


def test_obr_write_failure_publishes_alert_not_silent_swallow():
    """Pin the L171 fix (2026-05-22): an OBR write failure in
    `executor/main.py` must reach the operator via
    `nousergon_lib.alerts.publish` — silent warning-log only is
    exactly the [[feedback_no_silent_fails]] failure mode (page 16
    falls back to yesterday's snapshot with zero signal that today's
    write never landed).

    Source-inspection regression test (the surrounding `executor/main.py`
    requires full executor harness to exercise); pin the structural
    contract so a future refactor that drops the publish call breaks
    at CI time.
    """
    import inspect

    import executor.main as main_mod

    src = inspect.getsource(main_mod)
    # Locate the OBR-rationale write block — the WARN log + the
    # nousergon_lib.alerts.publish must BOTH live in the except
    # handler that wraps write_order_book_rationale.
    assert "Order-book rationale write failed" in src, (
        "OBR write-failure WARN log appears to have been refactored — "
        "re-audit the new except handler for the L171 alert publish "
        "before merging."
    )
    assert "obr_write_failed_" in src, (
        "OBR write-failure handler must call `publish_ops_alert` "
        "with a `dedup_key` starting `obr_write_failed_` so the failure "
        "reaches Telegram/SNS rather than silently swallowing "
        "([[feedback_no_silent_fails]])."
    )
    assert "publish_ops_alert" in src, (
        "OBR write-failure handler must route via executor.notifier.publish_ops_alert "
        "(config#1740 T3 — flow-doctor forum topics, not raw telegram=True)."
    )


# ── considered-universe filter + no-action sub-states (schema 1.2.0+) ──
#
# 1.2.0+ filters dead-signal rows (research HOLD / EXIT / REDUCE on a
# non-held ticker) out of the considered universe entirely — those
# signals have no possible order-book interaction and would only add
# noise. Post-filter, the only way to land in the no-action bucket
# is a research ENTER signal that didn't materialize as an order,
# block, or risk event. Two sub-states remain:
#   - optimizer_zero_weight (the "optimizer chose 0" case)
#   - unknown (should be 0; flags a producer bug per
#     [[feedback_no_silent_fails]])


class TestDeadSignalFiltering:
    """Research HOLD / EXIT / REDUCE on a non-held ticker is dead — no
    order is possible. 1.2.0+ filters these out of the considered
    universe entirely so the table only carries rows the operator can
    act on (or already has)."""

    def _empty_books(self) -> dict[str, Any]:
        return {
            "date": "2026-05-27",
            "approved_entries": [],
            "urgent_exits": [],
            "active_stops": [],
            "executed_today": [],
        }

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "order_book_data": self._empty_books(),
            "blocked_entries": [],
            "risk_events": [],
            "market_regime": "neutral",
            "run_date": "2026-05-27",
            "signal_date": "2026-05-27",
            "prediction_date": "2026-05-27",
            "calendar_date": "2026-05-27",
            "trading_day": "2026-05-27",
            "run_id": "2605271400",
            "current_positions": {},
        }

    def test_research_hold_on_non_position_filtered_out(self):
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [],
                     "hold": [_sig("SYK", "HOLD")]},
            predictions_by_ticker={},
            **self._base_kwargs(),
        )
        assert payload["summary"]["n_considered"] == 0

    def test_research_exit_on_non_position_filtered_out(self):
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [_sig("XOM", "EXIT")],
                     "reduce": [], "hold": []},
            predictions_by_ticker={},
            **self._base_kwargs(),
        )
        assert payload["summary"]["n_considered"] == 0

    def test_research_reduce_on_non_position_filtered_out(self):
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [_sig("BAC", "REDUCE")],
                     "hold": []},
            predictions_by_ticker={},
            **self._base_kwargs(),
        )
        assert payload["summary"]["n_considered"] == 0

    def test_research_hold_on_held_position_still_surfaces(self):
        # The filter is "non-held only" — HOLD on a position we DO own
        # must still surface (as STATE_HELD) so the table answers
        # "what positions am I carrying?" for the full portfolio.
        kwargs = self._base_kwargs()
        kwargs["current_positions"] = {"AAPL": {"mkt_val": 5000}}
        payload = build_order_book_rationale(
            signals={"enter": [], "exit": [], "reduce": [],
                     "hold": [_sig("AAPL", "HOLD")]},
            predictions_by_ticker={},
            **kwargs,
        )
        aapl = next(r for r in payload["tickers"] if r["ticker"] == "AAPL")
        assert aapl["terminal_state"] == STATE_HELD


class TestNoActionSubStates:
    def _empty_books(self) -> dict[str, Any]:
        return {
            "date": "2026-05-27",
            "approved_entries": [],
            "urgent_exits": [],
            "active_stops": [],
            "executed_today": [],
        }

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "order_book_data": self._empty_books(),
            "blocked_entries": [],
            "risk_events": [],
            "market_regime": "neutral",
            "run_date": "2026-05-27",
            "signal_date": "2026-05-27",
            "prediction_date": "2026-05-27",
            "calendar_date": "2026-05-27",
            "trading_day": "2026-05-27",
            "run_id": "2605271400",
            "current_positions": {},
        }

    def test_research_enter_optimizer_zero_target_classified_optimizer_zero(self):
        # The load-bearing operator case: research said ENTER, the
        # optimizer ran, the ticker was eligible, but the optimizer
        # assigned ~0 target weight → "we looked and chose not to."
        # Research bucket alone is what surfaces the ticker into the
        # considered universe; the optimizer view supplies the detail.
        shadow = {
            "tickers": ["F"],
            "current_weights": [0.0],
            "target_weights": [0.0],
            "alpha_hat": [0.001],
            "eligibility": [True],
            "eligibility_reasons": [None],
            "optimizer_cfg": {"min_score_to_enter": 55.0},
        }
        payload = build_order_book_rationale(
            signals={"enter": [_sig("F", "ENTER", score=65.0)],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            optimizer_shadow_log=shadow,
            **self._base_kwargs(),
        )
        f = next(r for r in payload["tickers"] if r["ticker"] == "F")
        assert f["terminal_state"] == STATE_NO_ACTION_OPTIMIZER_ZERO
        na = next(s for s in f["decision_chain"] if s["stage"] == "no_action")
        assert "optimizer" in na["detail"].lower()
        # The optimizer view is on the record so the dashboard can
        # surface the weights inline with the sub-state.
        assert f["optimizer"]["target_weight"] == 0.0
        assert f["optimizer"]["eligible"] is True

    def test_research_enter_nonzero_target_no_order_classified_dropped(self):
        # The ERROR case (L4501 / the 2026-06-04 AMD incident): research
        # ENTER, optimizer assigned a NON-ZERO target (10%), eligible —
        # but no approved-entry / block / held record exists. The
        # allocation was dropped downstream (price-resolve failure in
        # optimizer_cutover). Must classify as the distinct ERROR
        # sub-state, NOT the benign optimizer_zero or generic unknown.
        shadow = {
            "tickers": ["AMD"],
            "current_weights": [0.0],
            "target_weights": [0.10],
            "alpha_hat": [0.0316],
            "eligibility": [True],
            "eligibility_reasons": [None],
        }
        payload = build_order_book_rationale(
            signals={"enter": [_sig("AMD", "ENTER", score=75.2)],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            optimizer_shadow_log=shadow,
            **self._base_kwargs(),
        )
        amd = next(r for r in payload["tickers"] if r["ticker"] == "AMD")
        assert amd["terminal_state"] == STATE_NO_ACTION_OPTIMIZER_DROPPED
        na = next(s for s in amd["decision_chain"] if s["stage"] == "no_action")
        assert "ERROR" in na["detail"]
        assert "10.00%" in na["detail"]  # the targeted weight is named
        assert amd["optimizer"]["target_weight"] == 0.10
        # Counted distinctly in the summary so the console can banner it.
        assert payload["summary"]["n_no_action_optimizer_dropped"] == 1
        assert "n_no_action_unknown" not in payload["summary"]

    def test_research_enter_without_optimizer_view_classified_unknown(self):
        # Defensive — should be rare in production. Research said ENTER,
        # no optimizer shadow log (legacy non-optimizer run OR shadow
        # log absent for the ticker), no approved-entry record, no
        # risk-event. The legacy path would normally produce a
        # blocked-entry or approved-entry; reaching no-action here means
        # something upstream silently dropped the signal. Surfacing it
        # as STATE_NO_ACTION_UNKNOWN gives the dashboard a distinct
        # row to flag for investigation.
        payload = build_order_book_rationale(
            signals={"enter": [_sig("LOST", "ENTER")],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            **self._base_kwargs(),
        )
        lost = next(r for r in payload["tickers"] if r["ticker"] == "LOST")
        assert lost["terminal_state"] == STATE_NO_ACTION_UNKNOWN
        na = next(s for s in lost["decision_chain"] if s["stage"] == "no_action")
        assert "no" in na["detail"].lower()

    def test_chain_always_carries_no_action_stage_for_subclass_rows(self):
        # Every sub-state row must append a `no_action` chain stage so
        # the per-ticker drill-down explains the fallthrough — no
        # silent "state slug only" records. Tested against the two
        # reachable sub-states (optimizer_zero + unknown).
        shadow = {
            "tickers": ["F"],
            "current_weights": [0.0],
            "target_weights": [0.0],
            "alpha_hat": [0.001],
            "eligibility": [True],
            "eligibility_reasons": [None],
        }
        payload = build_order_book_rationale(
            signals={"enter": [_sig("F", "ENTER"), _sig("LOST", "ENTER")],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={},
            optimizer_shadow_log=shadow,
            **self._base_kwargs(),
        )
        for ticker in ("F", "LOST"):
            rec = next(r for r in payload["tickers"] if r["ticker"] == ticker)
            stages = [s["stage"] for s in rec["decision_chain"]]
            assert "no_action" in stages, (
                f"{ticker}: no_action stage missing from chain"
            )


class TestBookStatus:
    """The schema-1.3.0 ``book_status`` banner field — the single
    "why did/didn't the book move today" status the console renders
    above the per-ticker table. Asserts each of the four states fires
    on the right evidence and that dispersion is surfaced authoritatively.
    """

    def _empty_books(self) -> dict[str, Any]:
        return {
            "date": "2026-06-30",
            "approved_entries": [],
            "urgent_exits": [],
            "active_stops": [],
            "executed_today": [],
        }

    def _base_kwargs(self, **over) -> dict[str, Any]:
        base = {
            "signals": {"enter": [], "exit": [], "reduce": [], "hold": []},
            "predictions_by_ticker": {},
            "order_book_data": self._empty_books(),
            "blocked_entries": [],
            "risk_events": [],
            "market_regime": "neutral",
            "run_date": "2026-06-30",
            "signal_date": "2026-06-30",
            "prediction_date": "2026-06-30",
            "calendar_date": "2026-06-30",
            "trading_day": "2026-06-30",
            "run_id": "2606301400",
            "current_positions": {},
        }
        base.update(over)
        return base

    def test_schema_version_is_1_4_0(self):
        payload = build_order_book_rationale(**self._base_kwargs())
        assert payload["schema_version"] == "1.4.0"
        assert "book_status" in payload

    def test_no_rebalance_at_target_on_zero_turnover(self):
        # Today's 6/30 case: optimizer solved optimal, 0 entries/exits,
        # one-way turnover below band → benign HOLD.
        shadow = {
            "shadow_status": "ok",
            "diagnostics": {"status": "optimal", "turnover_one_way": 0.0041},
            "optimizer_cfg": {"rebalance_band_pct": 0.25},
        }
        payload = build_order_book_rationale(
            optimizer_shadow_log=shadow, **self._base_kwargs()
        )
        bs = payload["book_status"]
        assert bs["state"] == "no_rebalance_at_target"
        assert bs["n_entries"] == 0 and bs["n_exits"] == 0
        assert bs["turnover_one_way"] == 0.0041
        assert bs["rebalance_band_pct"] == 0.25
        assert "0.41%" in bs["headline"]
        assert bs["safeguard"]["fired"] is False

    def test_rebalanced_when_entries_or_exits_written(self):
        ob = {
            **self._empty_books(),
            "approved_entries": [_entry_with_meta("AAPL")],
            "urgent_exits": [
                {"ticker": "MSFT", "signal": "EXIT", "shares": 50, "reason": "x"}
            ],
        }
        payload = build_order_book_rationale(
            **self._base_kwargs(
                signals={"enter": [_sig("AAPL", "ENTER")],
                         "exit": [_sig("MSFT", "EXIT")], "reduce": [], "hold": []},
                order_book_data=ob,
                current_positions={"MSFT": {"mkt_val": 1000}},
            )
        )
        bs = payload["book_status"]
        assert bs["state"] == "rebalanced"
        assert bs["n_entries"] == 1 and bs["n_exits"] == 1

    def test_hold_book_safeguard_fired(self):
        gate = {
            "passed": False,
            "failed_check": "direction_skew",
            "reason": "89.7% DOWN-skew — strongly biased batch",
        }
        diag = {
            "gate_flagged": True,
            "alpha_stdev": 0.0008,
            "signal_degenerate": True,
            "decision": "hold_signal_degenerate",
        }
        payload = build_order_book_rationale(
            distribution_gate=gate,
            hold_book_active=True,
            hold_book_diag=diag,
            **self._base_kwargs(),
        )
        bs = payload["book_status"]
        assert bs["state"] == "hold_book_safeguard"
        assert bs["safeguard"]["fired"] is True
        assert bs["safeguard"]["failed_check"] == "direction_skew"
        assert "DOWN-skew" in bs["headline"]
        # Authoritative dispersion comes from the hold-book diag.
        assert bs["dispersion"]["alpha_stdev"] == 0.0008
        assert bs["dispersion"]["signal_degenerate"] is True

    def test_allocations_dropped_takes_top_precedence(self):
        # A dropped allocation is an ERROR and outranks every other state,
        # even if the safeguard also fired.
        shadow = {
            "tickers": ["AMD"],
            "current_weights": [0.0],
            "target_weights": [0.10],
            "alpha_hat": [0.05],
            "eligibility": [True],
            "eligibility_reasons": [None],
            "optimizer_cfg": {"min_score_to_enter": 55.0},
            "would_be_trades": [
                {"ticker": "AMD", "action": "BUY", "target_weight": 0.10,
                 "delta_dollars": 12500.0}
            ],
            "diagnostics": {"status": "optimal", "turnover_one_way": 0.10},
        }
        payload = build_order_book_rationale(
            signals={"enter": [_sig("AMD", "ENTER", score=80.0)],
                     "exit": [], "reduce": [], "hold": []},
            predictions_by_ticker={"AMD": {"predicted_direction": "UP",
                                           "predicted_alpha": 0.05}},
            optimizer_shadow_log=shadow,
            hold_book_active=True,  # even with safeguard set, dropped wins
            **{k: v for k, v in self._base_kwargs().items()
               if k not in ("signals", "predictions_by_ticker")},
        )
        # Sanity: the ticker resolved to the dropped terminal state.
        assert payload["summary"].get("n_no_action_optimizer_dropped", 0) >= 1
        assert payload["book_status"]["state"] == "allocations_dropped"
        assert payload["book_status"]["n_dropped"] >= 1

    def test_dispersion_computed_from_predictions_when_no_diag(self):
        # Normal day (gate ok → hold_diag carries no alpha_stdev): the
        # producer recomputes cross-sectional stdev + direction skew from
        # the batch so the dispersion sub-line still renders.
        preds = {
            "A": {"predicted_direction": "UP", "predicted_alpha": 0.02},
            "B": {"predicted_direction": "UP", "predicted_alpha": 0.01},
            "C": {"predicted_direction": "DOWN", "predicted_alpha": -0.03},
            "D": {"predicted_direction": "FLAT", "predicted_alpha": 0.0},
        }
        payload = build_order_book_rationale(
            predictions_by_ticker=preds,
            **{k: v for k, v in self._base_kwargs().items()
               if k != "predictions_by_ticker"},
        )
        disp = payload["book_status"]["dispersion"]
        assert disp["n_predictions"] == 4
        assert disp["n_up"] == 2 and disp["n_down"] == 1 and disp["n_flat"] == 1
        assert disp["alpha_stdev"] is not None and disp["alpha_stdev"] > 0
        assert disp["signal_degenerate"] is None  # not measured off-gate

    def test_book_status_present_on_legacy_path(self):
        # No optimizer/gate/hold args at all → still a valid book_status
        # (state derived from the empty book). Backward-safe default args.
        payload = build_order_book_rationale(**self._base_kwargs())
        bs = payload["book_status"]
        assert bs["state"] == "no_rebalance_at_target"
        assert bs["safeguard"]["fired"] is False
        assert bs["turnover_one_way"] is None
