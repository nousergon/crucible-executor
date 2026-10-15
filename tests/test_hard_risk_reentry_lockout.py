"""Re-entry lockout after a hard-risk EXIT: the risk layer outranks the
optimizer, so a name cut by the loss floor / gap stop / drawdown forced exit
cannot be re-bought for ``hard_risk_reentry_lockout_days`` sessions (SMCI was
loss-floor exited 2026-10-15 while the optimizer still wanted 10%)."""
from executor.main import _reentry_lockouts
from executor.optimizer_shadow import _build_eligibility
from executor.order_book_rationale import _synthesize_optimizer_rejections
from executor.trade_logger import get_hard_risk_exits_since, init_db


def _conn(rows):
    conn = init_db(":memory:")
    for i, (ticker, d, action, reason, status) in enumerate(rows):
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, exit_reason, status, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (f"t{i}", d, ticker, action, reason, status, f"{d}T13:30:00"),
        )
    return conn


ROWS = [
    ("SMCI", "2026-10-15", "EXIT", "position_loss_floor", "Filled"),
    ("HL", "2026-10-15", "EXIT", "position_loss_floor", "Filled"),
    ("COIN", "2026-10-08", "EXIT", "catastrophic_gap_stop", "Filled"),
    ("AMD", "2026-10-14", "EXIT", "optimizer_target_zero", "Filled"),
    ("MU", "2026-10-14", "EXIT", "drawdown_forced_exit", "Rejected"),
    ("ANF", "2026-10-14", "ENTER", None, "Filled"),
]


def test_only_filled_hard_risk_exits_count():
    out = get_hard_risk_exits_since(_conn(ROWS), "2026-10-01")
    assert set(out) == {"SMCI", "HL", "COIN"}
    assert out["SMCI"] == {"date": "2026-10-15", "exit_reason": "position_loss_floor"}


def test_lockout_covers_the_next_five_sessions_then_releases():
    conn = _conn(ROWS)
    cfg = {"hard_risk_reentry_lockout_days": 5}
    # 2026-10-15 (Thu) exit → locked 10-16, 10-19, 10-20, 10-21, 10-22.
    assert "SMCI" in _reentry_lockouts(conn, "2026-10-16", cfg)
    assert "SMCI" in _reentry_lockouts(conn, "2026-10-22", cfg)
    assert "SMCI" not in _reentry_lockouts(conn, "2026-10-23", cfg)
    # COIN 10-08 exit has aged out by 10-16 (5 sessions: 10-09..10-15).
    assert "COIN" not in _reentry_lockouts(conn, "2026-10-16", cfg)
    assert "COIN" in _reentry_lockouts(conn, "2026-10-15", cfg)


def test_weekend_run_date_counts_from_the_last_session():
    cfg = {"hard_risk_reentry_lockout_days": 5}
    assert "SMCI" in _reentry_lockouts(_conn(ROWS), "2026-10-18", cfg)


def test_lockout_disabled_or_no_db():
    conn = _conn(ROWS)
    assert _reentry_lockouts(conn, "2026-10-16", {"hard_risk_reentry_lockout_days": 0}) == {}
    assert _reentry_lockouts(conn, "2026-10-16", {"hard_risk_reentry_lockout_days": None}) == {}
    assert _reentry_lockouts(None, "2026-10-16", {}) == {}
    # Default is on (5 sessions) with no config key.
    assert "SMCI" in _reentry_lockouts(conn, "2026-10-16", {})


TICKERS = ["SMCI", "MU", "SPY", "CASH"]
SPY, CASH = 2, 3
SIGNALS = {"SMCI": {"signal": "ENTER", "score": 94}, "MU": {"signal": "ENTER", "score": 90}}
PREDS = {"SMCI": {"predicted_alpha": 0.048}, "MU": {"predicted_alpha": 0.03}}


def test_locked_out_name_is_ineligible_and_surfaced_as_a_block():
    elig, reasons = _build_eligibility(
        TICKERS, SIGNALS, PREDS, {}, {"min_score_to_enter": 57}, SPY, CASH,
        reentry_lockout_tickers={"SMCI"},
    )
    assert not elig[0] and reasons[0] == "hard_risk_lockout"
    assert elig[1]
    blocked, events = _synthesize_optimizer_rejections(
        shadow_log={
            "tickers": TICKERS,
            "eligibility": [bool(x) for x in elig],
            "eligibility_reasons": reasons,
        },
        signals_by_ticker=SIGNALS,
        current_positions=set(),
    )
    assert [b["ticker"] for b in blocked] == ["SMCI"]
    assert events[0]["rule"] == "hard_risk_lockout"
