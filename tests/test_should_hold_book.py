"""Tests for ``_should_hold_book`` — the §5b hold-book decision.

Two eras are asserted together, because either alone is a defect.

**config#1176 (2026-06-29).** The predictor's ``output_distribution_gate``
judges, among other things, the CALIBRATED isotonic ``p_up``, which collapses
onto a flat staircase step on low-dispersion-but-healthy days. Halting the book
on that artifact false-halted a healthy 26-name batch on 2026-06-22 and
2026-06-29 (GE's 8% target dropped). A ``p_up``-shape failure alone must never
hold the book, and the replay of that exact shape is asserted below.

**alpha-engine-config-I10179 / I10184 (2026-09-08).** The fix for the false halt
was a hand-set ``HOLD_BOOK_ALPHA_STDEV_FLOOR = 0.001`` in this repo — a SECOND
floor on ``alpha_stdev``, a quantity the predictor already declares floors on
(absolute 0.015, relative 50% of the trailing 10-session median). Two owners,
one invariant, 15x apart: on 2026-09-08 the predictor called ``alpha_stdev``
0.005819 collapsed and the executor called it healthy, and the optimizer
rebalanced a batch its producer had declared dead. The floor is gone. The
executor now reads the predictor's own dispersion legs and owns only WHICH
failure justifies a hold.

The measurement that shaped it, over the 47 served sessions carrying
``metrics.alpha_stdev`` (2026-06-30..2026-09-08): the ABSOLUTE 0.015 leg sits
below 14 of 47, 13 of them healthy — so it does not hold; the RELATIVE leg fires
on 6 of 47, all six champion-collapse sessions (2026-08-24..28 under the
rolled-back ``7d3d1cce``, and 2026-09-08) — so it does.
"""
from __future__ import annotations

import executor.main as main
from executor.main import (
    HOLD_BOOK_ALPHA_MODAL_FRACTION,
    _should_hold_book,
    emit_distribution_gate_metrics,
)


def _gate(
    passed: bool,
    check: str | None = "unique_p_up",
    *,
    relative_passed: bool | None = None,
    absolute_passed: bool | None = None,
    zero_streak: int | None = None,
    champion: str | None = None,
) -> dict:
    """A gate artifact in the shape ``read_distribution_gate`` returns."""
    metrics: dict = {}
    if relative_passed is not None or absolute_passed is not None:
        rd: dict = {}
        if relative_passed is not None:
            rd["alpha_stdev"] = {
                "passed": relative_passed,
                "ratio": 0.41 if not relative_passed else 0.98,
                "history_median": 0.014249,
                "history_n": 10,
            }
        if absolute_passed is not None:
            rd["absolute_floor_alpha_stdev"] = {
                "passed": absolute_passed, "floor": 0.015,
            }
        if champion:
            rd["today_champion_version_id"] = champion
        metrics["relative_dispersion"] = rd
    if zero_streak is not None:
        metrics["n_high_confidence"] = {"zero_streak": zero_streak, "applied": 0}
    return {
        "passed": passed,
        "failed_check": check,
        "reason": "test",
        "metrics": metrics,
    }


def _preds(alphas: list[float], *, key: str = "predicted_alpha") -> dict:
    return {f"T{i}": {"ticker": f"T{i}", key: a} for i, a in enumerate(alphas)}


#: The 2026-06-29 batch shape: 26 names, spread ~-0.022..+0.018, every alpha
#: distinct. ``alpha_stdev`` ~0.0122 — BELOW the predictor's absolute 0.015.
_JUNE_29_ALPHAS = [round(-0.022 + i * 0.0016, 5) for i in range(26)]


class TestNoExecutorSideFloorOnAlphaStdev:
    """The defect I10179 names: a second, hand-set floor on a quantity the
    predictor already owns."""

    def test_the_hand_set_floor_constant_is_gone(self):
        assert not hasattr(main, "HOLD_BOOK_ALPHA_STDEV_FLOOR"), (
            "HOLD_BOOK_ALPHA_STDEV_FLOOR is back. A magnitude floor on "
            "alpha_stdev belongs to the predictor's output_distribution_gate "
            "and nowhere else — two owners 15x apart is what let 2026-09-08's "
            "collapsed batch rebalance the book (alpha-engine-config-I10179)."
        )

    def test_the_only_remaining_threshold_is_scale_free(self):
        """A modal FRACTION is invariant to the scale of the alphas, so it
        cannot disagree with the predictor about a magnitude."""
        assert 0.0 < HOLD_BOOK_ALPHA_MODAL_FRACTION <= 1.0

    def test_scaling_every_alpha_does_not_change_the_verdict(self):
        """The behavioural statement of the line above. A batch rescaled by
        1000x is the same cross-section; an absolute floor would flip on it."""
        gate = _gate(False, "unique_p_up")
        small = _should_hold_book(gate, _preds([a / 1000 for a in _JUNE_29_ALPHAS]))
        large = _should_hold_book(gate, _preds([a * 1000 for a in _JUNE_29_ALPHAS]))
        assert small[0] is False and large[0] is False
        assert small[1]["decision"] == large[1]["decision"]


class TestConfig1176ProtectionSurvives:
    """The 2026-06-29 false halt must stay not-held. This is the regression
    the whole reconciliation is measured against."""

    def test_the_2026_06_29_replay_does_not_hold_the_book(self):
        """Gate flagged on the isotonic ``p_up`` staircase (``unique_p_up``),
        no dispersion legs in the artifact of that era, alphas cleanly
        differentiated → PROCEED."""
        gate = _gate(False, "unique_p_up")
        hold, diag = _should_hold_book(gate, _preds(_JUNE_29_ALPHAS))
        assert hold is False, (
            "config#1176 regression: the 2026-06-29 shape halted the book again"
        )
        assert diag["decision"] == "proceed_p_up_artifact_only"
        assert diag["n_alpha"] == 26
        # And it sits BELOW the predictor's absolute floor, which is exactly why
        # that leg may not be a hold trigger.
        assert diag["alpha_stdev"] < 0.015

    def test_every_p_up_shape_failure_proceeds_on_a_healthy_cross_section(self):
        for check in (
            "unique_p_up", "modal_fraction", "stdev", "saturation_rate",
            "direction_skew", "confidence_semantics", "alpha_sign_skew",
        ):
            hold, diag = _should_hold_book(
                _gate(False, check), _preds(_JUNE_29_ALPHAS)
            )
            assert hold is False, f"{check} halted a healthy cross-section"
            assert diag["decision"] == "proceed_p_up_artifact_only"

    def test_a_p_up_failure_still_holds_a_literally_constant_batch(self):
        """The narrowing may not go so far that a genuinely dead batch trades.
        Scale-free: one value over every ticker."""
        hold, diag = _should_hold_book(_gate(False, "unique_p_up"), _preds([0.0149] * 26))
        assert hold is True
        assert diag["decision"] == "hold_signal_degenerate"
        assert diag["alpha_modal_fraction"] == 1.0


class TestDispersionVerdictIsConsumedNotRethresholded:
    def test_the_2026_09_08_batch_holds(self):
        """The session this was filed for: relative leg failed, 29 distinct
        alphas, stdev 0.005819. The executor must now hold."""
        alphas = [round(-0.008 + i * 0.00057, 6) for i in range(29)]
        gate = _gate(
            False, "alpha_stdev_relative_compression",
            relative_passed=False, absolute_passed=False, zero_streak=1,
            champion="v3.0-meta-2026-09-04-cc3271ea",
        )
        hold, diag = _should_hold_book(gate, _preds(alphas))
        assert hold is True
        assert diag["decision"] == "hold_relative_dispersion_collapse"
        assert diag["relative_dispersion_passed"] is False
        assert diag["n_high_confidence_zero_streak"] == 1
        assert diag["champion_version_id"] == "v3.0-meta-2026-09-04-cc3271ea"

    def test_absolute_floor_alone_does_not_hold(self):
        """13 of 47 served sessions sit below the predictor's absolute 0.015
        under a champion nobody called collapsed. Consuming that leg as a hold
        would have held the book on 30% of history."""
        gate = _gate(
            False, "alpha_stdev_absolute_floor",
            relative_passed=True, absolute_passed=False,
        )
        hold, diag = _should_hold_book(gate, _preds(_JUNE_29_ALPHAS))
        assert hold is False
        assert diag["decision"] == "proceed_absolute_floor_only"

    def test_a_dispersion_failure_with_no_legs_falls_back_scale_free(self):
        """An artifact from before the legs existed must not be trusted as an
        unresolvable verdict, and must not invent a magnitude either."""
        gate = _gate(False, "alpha_stdev_relative_compression")
        hold, diag = _should_hold_book(gate, _preds(_JUNE_29_ALPHAS))
        assert hold is False
        assert diag["decision"] == "proceed_signal_healthy"

    def test_relative_leg_holds_even_when_the_named_check_is_the_absolute_one(self):
        """The verdict is read from the legs, not from the failed_check label."""
        gate = _gate(
            False, "alpha_stdev_absolute_floor",
            relative_passed=False, absolute_passed=True,
        )
        hold, _ = _should_hold_book(gate, _preds(_JUNE_29_ALPHAS))
        assert hold is True


class TestTradableSignalFailuresHoldWithoutReMeasuring:
    def test_alpha_collapse_holds(self):
        hold, diag = _should_hold_book(
            _gate(False, "alpha_collapse"), _preds(_JUNE_29_ALPHAS)
        )
        assert hold is True
        assert diag["decision"] == "hold_tradable_signal_failed"

    def test_alpha_nonfinite_rate_holds(self):
        hold, diag = _should_hold_book(
            _gate(False, "alpha_nonfinite_rate"), _preds(_JUNE_29_ALPHAS)
        )
        assert hold is True
        assert diag["decision"] == "hold_tradable_signal_failed"


class TestFailOpenAndEdges:
    def test_gate_ok_proceeds(self):
        hold, diag = _should_hold_book(_gate(True, None), _preds([0.0] * 26))
        assert hold is False
        assert diag["decision"] == "proceed_gate_ok"

    def test_missing_gate_proceeds(self):
        hold, diag = _should_hold_book(None, _preds([0.01, -0.01] * 13))
        assert hold is False
        assert diag["gate_flagged"] is False

    def test_undeterminable_falls_back_to_gate(self):
        hold, diag = _should_hold_book(_gate(False), _preds([0.01, 0.02]))
        assert hold is True
        assert diag["decision"] == "hold_signal_undeterminable"

    def test_canonical_alpha_fallback(self):
        hold, diag = _should_hold_book(
            _gate(False), _preds(_JUNE_29_ALPHAS, key="canonical_predicted_alpha")
        )
        assert hold is False
        assert diag["n_alpha"] == 26

    def test_nan_and_none_alphas_ignored(self):
        preds = _preds([0.01 * i for i in range(-13, 13)])
        preds["BAD1"] = {"ticker": "BAD1", "predicted_alpha": float("nan")}
        preds["BAD2"] = {"ticker": "BAD2", "predicted_alpha": None}
        preds["BAD3"] = {"ticker": "BAD3"}
        hold, diag = _should_hold_book(_gate(False), preds)
        assert hold is False
        assert diag["n_alpha"] == 26

    def test_bool_not_treated_as_numeric(self):
        preds = {"T0": {"predicted_alpha": True}, "T1": {"predicted_alpha": False}}
        hold, diag = _should_hold_book(_gate(False), preds)
        assert hold is True
        assert diag["decision"] == "hold_signal_undeterminable"

    def test_an_unrecognised_failed_check_is_judged_scale_free(self):
        hold, diag = _should_hold_book(
            _gate(False, "some_check_invented_next_quarter"), _preds(_JUNE_29_ALPHAS)
        )
        assert hold is False
        assert diag["decision"] == "proceed_signal_healthy"

    def test_alpha_stdev_is_still_reported_for_the_operator_surface(self):
        """``order_book_rationale`` reads ``hold_book_diag['alpha_stdev']``."""
        _, diag = _should_hold_book(_gate(False), _preds(_JUNE_29_ALPHAS))
        assert isinstance(diag["alpha_stdev"], float)


class TestGateVerdictReachesAnAlarmableSurface:
    """I10184 deliverable 2. Before this the verdict reached only S3."""

    def _capture(self, monkeypatch) -> list[dict]:
        sent: list[dict] = []

        class _CW:
            def put_metric_data(self, **kw):
                sent.append(kw)

        class _B3:
            @staticmethod
            def client(name, *a, **kw):
                assert name == "cloudwatch"
                return _CW()

        import sys
        monkeypatch.setitem(sys.modules, "boto3", _B3)
        return sent

    def test_a_dispersion_failure_publishes_one(self, monkeypatch):
        sent = self._capture(monkeypatch)
        gate = _gate(
            False, "alpha_stdev_relative_compression",
            relative_passed=False, absolute_passed=False, zero_streak=3,
        )
        emit_distribution_gate_metrics(
            gate, {"decision": "hold_relative_dispersion_collapse"}
        )
        by_name = {m["MetricName"]: m["Value"] for m in sent[0]["MetricData"]}
        assert sent[0]["Namespace"] == "AlphaEngine/Executor"
        assert by_name["predictor_dispersion_gate_failed"] == 1.0
        assert by_name["predictor_output_gate_failed"] == 1.0
        assert by_name["predictor_n_high_confidence_zero_streak"] == 3.0

    def test_a_p_up_only_failure_does_not_raise_the_paging_metric(self, monkeypatch):
        """config#1176 again, on the alarm surface: the artifact flags, the
        paging metric stays 0, and the un-paged one records that it flagged."""
        sent = self._capture(monkeypatch)
        emit_distribution_gate_metrics(
            _gate(False, "unique_p_up"), {"decision": "proceed_p_up_artifact_only"}
        )
        by_name = {m["MetricName"]: m["Value"] for m in sent[0]["MetricData"]}
        assert by_name["predictor_dispersion_gate_failed"] == 0.0
        assert by_name["predictor_output_gate_failed"] == 1.0

    def test_a_healthy_run_still_publishes_zero(self, monkeypatch):
        """A metric that only appears on failure gives the alarm no baseline
        and makes absence indistinguishable from health."""
        sent = self._capture(monkeypatch)
        emit_distribution_gate_metrics(_gate(True, None), {"decision": "proceed_gate_ok"})
        by_name = {m["MetricName"]: m["Value"] for m in sent[0]["MetricData"]}
        assert by_name["predictor_dispersion_gate_failed"] == 0.0
        assert by_name["predictor_output_gate_failed"] == 0.0

    def test_a_cloudwatch_failure_never_blocks_the_planner(self, monkeypatch):
        class _B3:
            @staticmethod
            def client(*a, **kw):
                raise RuntimeError("no credentials")

        import sys
        monkeypatch.setitem(sys.modules, "boto3", _B3)
        emit_distribution_gate_metrics(_gate(False, "alpha_collapse"), {})
