"""Tests for ``_should_hold_book`` — the §5b hold-book decision (2026-06-29
redesign, config#1176).

The predictor's output_distribution_gate judges the CALIBRATED isotonic
``p_up``, which collapses onto a flat staircase step on low-dispersion-but-
healthy days and false-halts the book. The redesigned decision holds ONLY when
the gate flagged AND the tradable, level-neutralized ``predicted_alpha`` the
optimizer trades on is itself collapsed (near-zero cross-sectional stdev).

Invariant under test: the change strictly *reduces* false holds — it never adds
a hold the old ``gate.passed is False`` path wouldn't already have taken, and it
releases the book on the 2026-06-29 class (gate-flagged but alpha differentiated).
"""
from __future__ import annotations

from executor.main import _should_hold_book, HOLD_BOOK_ALPHA_STDEV_FLOOR


def _gate(passed: bool, check: str = "unique_p_up") -> dict:
    return {"passed": passed, "failed_check": check, "reason": "test"}


def _preds(alphas: list[float], *, key: str = "predicted_alpha") -> dict:
    return {f"T{i}": {"ticker": f"T{i}", key: a} for i, a in enumerate(alphas)}


class TestShouldHoldBook:
    def test_gate_ok_proceeds(self):
        """Gate passed → never hold, regardless of dispersion."""
        hold, diag = _should_hold_book(_gate(True), _preds([0.0] * 26))
        assert hold is False
        assert diag["decision"] == "proceed_gate_ok"

    def test_missing_gate_proceeds(self):
        """Fail-open: a None gate never halts."""
        hold, diag = _should_hold_book(None, _preds([0.01, -0.01] * 13))
        assert hold is False
        assert diag["gate_flagged"] is False

    def test_gate_flagged_but_alpha_healthy_proceeds(self):
        """The 2026-06-29 case: gate failed on isotonic p_up, but
        predicted_alpha is cleanly differentiated → DO NOT hold."""
        # Mirrors 6/29: spread ~ -0.022..+0.018, stdev ~0.011.
        alphas = [round(-0.022 + i * 0.0016, 5) for i in range(26)]
        hold, diag = _should_hold_book(_gate(False), _preds(alphas))
        assert hold is False
        assert diag["decision"] == "proceed_signal_healthy"
        assert diag["alpha_stdev"] >= HOLD_BOOK_ALPHA_STDEV_FLOOR

    def test_gate_flagged_and_alpha_collapsed_holds(self):
        """Genuine collapse: gate failed AND predicted_alpha is near-constant
        (stdev below floor) → hold (the legitimate safeguard)."""
        hold, diag = _should_hold_book(_gate(False), _preds([0.0149] * 26))
        assert hold is True
        assert diag["decision"] == "hold_signal_degenerate"
        assert diag["alpha_stdev"] < HOLD_BOOK_ALPHA_STDEV_FLOOR

    def test_gate_flagged_alpha_tiny_dispersion_below_floor_holds(self):
        """Dispersion present but below the floor still counts as collapsed."""
        alphas = [0.0149 + (i % 2) * 0.0002 for i in range(26)]  # stdev ~1e-4
        hold, diag = _should_hold_book(_gate(False), _preds(alphas))
        assert hold is True
        assert diag["alpha_stdev"] < HOLD_BOOK_ALPHA_STDEV_FLOOR

    def test_undeterminable_falls_back_to_gate(self):
        """Gate flagged but too few finite alphas to judge → trust the gate
        verdict and hold (conservative)."""
        hold, diag = _should_hold_book(_gate(False), _preds([0.01, 0.02]))
        assert hold is True
        assert diag["decision"] == "hold_signal_undeterminable"

    def test_canonical_alpha_fallback(self):
        """canonical_predicted_alpha is used when predicted_alpha is absent."""
        alphas = [round(-0.02 + i * 0.0016, 5) for i in range(26)]
        hold, diag = _should_hold_book(
            _gate(False), _preds(alphas, key="canonical_predicted_alpha")
        )
        assert hold is False
        assert diag["n_alpha"] == 26

    def test_nan_and_none_alphas_ignored(self):
        """Non-finite / missing alphas don't count toward the batch and don't
        crash the stdev computation."""
        preds = _preds([0.01 * i for i in range(-13, 13)])  # 26 healthy
        preds["BAD1"] = {"ticker": "BAD1", "predicted_alpha": float("nan")}
        preds["BAD2"] = {"ticker": "BAD2", "predicted_alpha": None}
        preds["BAD3"] = {"ticker": "BAD3"}  # no alpha key at all
        hold, diag = _should_hold_book(_gate(False), preds)
        assert hold is False
        assert diag["n_alpha"] == 26  # only the finite ones counted

    def test_bool_not_treated_as_numeric(self):
        """A stray bool in the alpha field must not be counted as a number."""
        preds = {"T0": {"predicted_alpha": True}, "T1": {"predicted_alpha": False}}
        hold, diag = _should_hold_book(_gate(False), preds)
        # 0 finite numeric alphas → undeterminable → fall back to gate (hold)
        assert hold is True
        assert diag["decision"] == "hold_signal_undeterminable"
