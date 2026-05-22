"""Tests for executor.order_book_rationale.

Covers the pure builder's terminal-state resolution across the full
considered universe, the decision-chain construction, the summary
counts, and the canonical eval_artifacts write (dated artifact +
sidecar, round-trippable via the lib's load_latest_eval_artifact).
"""

from __future__ import annotations

import json

import pytest

from executor.order_book_rationale import (
    DEFAULT_S3_PREFIX,
    SCHEMA_VERSION,
    STATE_APPROVED_ENTRY,
    STATE_HELD,
    STATE_NO_ACTION,
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
    return dict(
        signals=signals,
        predictions_by_ticker=predictions,
        order_book_data=order_book,
        blocked_entries=blocked,
        risk_events=risk_events,
        market_regime="neutral",
        run_date="2026-05-15",
        signal_date="2026-05-15",
        prediction_date="2026-05-15",
        calendar_date="2026-05-15",
        trading_day="2026-05-15",
        run_id="2605151400",
        # KO is the held ticker — must come from portfolio truth, not
        # the research `hold` bucket (research HOLD on a non-held ticker
        # is a recommendation that wasn't acted on, not a held state).
        current_positions={"KO": {"mkt_val": 1000}},
    )


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
    assert by_ticker["XOM"] == STATE_NO_ACTION


def test_considered_universe_is_union_of_all_sources(scenario):
    payload = build_order_book_rationale(**scenario)
    tickers = {r["ticker"] for r in payload["tickers"]}
    assert tickers == {"AAPL", "MSFT", "TSLA", "NVDA", "AMD", "KO", "XOM"}
    assert payload["summary"]["n_considered"] == 7


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
    assert s[f"n_{STATE_NO_ACTION}"] == 1


def test_records_sorted_actioned_first(scenario):
    payload = build_order_book_rationale(**scenario)
    states = [r["terminal_state"] for r in payload["tickers"]]
    assert states[0] == STATE_APPROVED_ENTRY
    assert states[-1] == STATE_NO_ACTION


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
    assert sidecar["n_considered"] == 7


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
    from alpha_engine_lib.eval_artifacts import load_latest_eval_artifact

    payload = build_order_book_rationale(**scenario)
    s3 = _StubS3()
    write_order_book_rationale(
        payload, s3_client=s3, bucket="alpha-engine-research"
    )
    loaded = load_latest_eval_artifact(
        s3, bucket="alpha-engine-research", prefix=DEFAULT_S3_PREFIX
    )
    assert loaded["run_id"] == payload["run_id"]
    assert loaded["summary"]["n_considered"] == 7


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
        return dict(
            signals=signals,
            predictions_by_ticker=predictions,
            order_book_data=order_book,
            blocked_entries=[],   # legacy path skipped
            risk_events=[],       # legacy path skipped
            market_regime="neutral",
            run_date="2026-05-15",
            signal_date="2026-05-15",
            prediction_date="2026-05-15",
            calendar_date="2026-05-15",
            trading_day="2026-05-15",
            run_id="2605151400",
            optimizer_shadow_log=shadow_log,
            current_positions={"KO": {"mkt_val": 1000}},
        )

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

    def test_optimizer_shadow_eligibility_reasons_field_in_log(self):
        # Sanity: the field name the producer reads matches what
        # optimizer_shadow.py emits. Pins the contract between the two
        # modules so a rename of either side is caught.
        from executor.optimizer_shadow import _build_eligibility
        import numpy as np
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

    def test_non_held_ticker_with_research_hold_is_no_action(self):
        # SYK case: research recommends HOLD but the ticker is not in
        # the portfolio (cur_w=0). Research HOLD on a non-held ticker
        # is an informational opinion, NOT a held state — resolve to
        # STATE_NO_ACTION so the table doesn't claim we hold it.
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
        syk = next(r for r in payload["tickers"] if r["ticker"] == "SYK")
        assert syk["terminal_state"] == STATE_NO_ACTION
        assert syk["held"] is False

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
    `alpha_engine_lib.alerts.publish` — silent warning-log only is
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
    # alpha_engine_lib.alerts.publish must BOTH live in the except
    # handler that wraps write_order_book_rationale.
    assert "Order-book rationale write failed" in src, (
        "OBR write-failure WARN log appears to have been refactored — "
        "re-audit the new except handler for the L171 alert publish "
        "before merging."
    )
    assert "obr_write_failed_" in src, (
        "OBR write-failure handler must call `alpha_engine_lib.alerts.publish` "
        "with a `dedup_key` starting `obr_write_failed_` so the failure "
        "reaches Telegram/SNS rather than silently swallowing "
        "([[feedback_no_silent_fails]])."
    )
    assert "from alpha_engine_lib import alerts" in src, (
        "OBR write-failure handler must import alpha_engine_lib.alerts "
        "lazily (inside the except) so a missing lib at boot doesn't "
        "break the planner cold-start."
    )
