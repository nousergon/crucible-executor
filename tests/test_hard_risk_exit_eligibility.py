"""A name the planner already queued for a hard-risk full EXIT (loss floor,
drawdown forced exit) is pinned to zero weight, so the optimizer cannot plan a
BUY into a name the daemon sells at the open (SMCI 2026-10-15)."""
from executor.optimizer_shadow import _build_eligibility

TICKERS = ["SMCI", "MU", "NEW", "SPY", "CASH"]
SPY, CASH = 3, 4
SIGNALS = {t: {"signal": "ENTER", "score": 94} for t in TICKERS[:3]}
PREDS = {t: {"predicted_alpha": 0.048} for t in TICKERS[:3]}
HELD = {"SMCI": {}, "MU": {}}
CFG = {"min_score_to_enter": 57}


def test_no_hard_exits_leaves_held_buy_candidate_eligible():
    elig, reasons = _build_eligibility(TICKERS, SIGNALS, PREDS, HELD, CFG, SPY, CASH)
    assert elig.all()
    assert reasons == [None] * len(TICKERS)


def test_hard_exit_pins_held_name_out_despite_enter_signal():
    elig, reasons = _build_eligibility(
        TICKERS, SIGNALS, PREDS, HELD, CFG, SPY, CASH,
        hard_exit_tickers={"SMCI"},
    )
    assert not elig[0] and reasons[0] == "hard_risk_exit"
    assert elig[1] and elig[2]
    assert elig[SPY] and elig[CASH]


def test_hard_exit_takes_precedence_over_other_exit_gates():
    signals = {**SIGNALS, "SMCI": {"signal": "EXIT", "score": 94}}
    _elig, reasons = _build_eligibility(
        TICKERS, signals, PREDS, HELD, CFG, SPY, CASH,
        hard_exit_tickers={"SMCI"},
    )
    assert reasons[0] == "hard_risk_exit"
