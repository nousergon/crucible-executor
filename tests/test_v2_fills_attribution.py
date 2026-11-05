"""v2 trader fills in the EOD attribution -- alpha-engine-config-I13227.

On 2026-11-04 the v2 trader's first fills sold the v1 book and bought into a
held name and a new one. ``eod_reconcile`` read fills from v1's ``trades``
table only, so rotation read $0, the add was valued from the prior close, and
the day's trading P&L (+$70,995 against a correct NAV) failed the residual
bounds gate as "unattributed".

The run() tests below reproduce that day's SHAPE on a synthetic book (no real
account, prices or names): three v1 names sold away from the prior close, an
add to a held name below the prior close, a new name bought, a commission on
every fill, and no row in v1's ``trades`` table. Without the journal the gate
fails exactly as it did; with it the residual is the commission alone.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from executor.config_loader import NAV_BASIS_IB_NETLIQ
from executor.eod_report import compute_rotation_realized
from executor.trade_logger import init_db, log_eod
from executor.v2_fills import (
    V2_FIRST_SESSION,
    V2FillsError,
    fills_from_journal,
    journal_key,
    load_v2_fills,
    merge_for_attribution,
)

RUN_DATE = "2026-11-04"
PRIOR_DATE = "2026-11-03"
COMMISSION_CENTS = 100  # $1.00 per fill
ACCRUED = 100.0
CASH_PRIOR = 200_000.0

# ticker -> (prior shares, prior close, today close)
SOLD = {"AAA": (1000, 100.00, 101.00), "BBB": (1000, 50.00, 48.50), "CCC": (1000, 200.00, 228.00)}
# ticker -> list of (signed shares, fill price)
FILLS = {
    "AAA": [(-600, 110.00), (-400, 110.00)],
    "BBB": [(-1000, 49.00)],
    "CCC": [(-1000, 230.00)],
    "HHH": [(800, 170.93), (800, 170.93)],
    "NNN": [(240, 1042.11)],
}
HELD = {"HHH": (400, 175.40, 171.70)}
NEW = {"NNN": (1037.40, 1042.11)}  # today close, avg_cost

N_FILLS = sum(len(v) for v in FILLS.values())
ROTATION = 1000 * (110.00 - 100.00) + 1000 * (49.00 - 50.00) + 1000 * (230.00 - 200.00)  # +39,000
COMMISSIONS = N_FILLS * COMMISSION_CENTS / 100.0


def _cents(usd: float) -> int:
    return round(usd * 100)


def _journal(*, extra_fills: list[dict] | None = None, day: str = RUN_DATE) -> dict:
    """A ``ledger_journal.v1`` document shaped like trader.reconcile's."""
    entries: list[dict] = []
    for ticker, legs in FILLS.items():
        for i, (shares, price) in enumerate(legs):
            ref = f"0000{ticker}.{i}"
            consideration = -_cents(shares * price)  # cash: a buy is negative
            entries.append({
                "kind": "fill", "event_ref": ref, "trading_day": day,
                "evidence": {"commission_separated": True, "source": "ib_execution"},
                "postings": [
                    {"account": "assets:cash:USD", "amount_cents": consideration, "quantity": None},
                    {"account": f"assets:securities:{ticker}", "amount_cents": -consideration,
                     "quantity": shares},
                ],
            })
            entries.append({
                "kind": "commission", "event_ref": ref, "trading_day": day,
                "evidence": {"source": "ib_commission_report"},
                "postings": [
                    {"account": "expenses:commissions", "amount_cents": COMMISSION_CENTS, "quantity": None},
                    {"account": "assets:cash:USD", "amount_cents": -COMMISSION_CENTS, "quantity": None},
                ],
            })
    entries.extend(extra_fills or [])
    entries.append({
        "kind": "interest_accrual", "event_ref": "interest", "trading_day": day,
        "evidence": {}, "postings": [],
    })
    return {"schema_version": "ledger_journal.v1", "tenant": "nous-ergon-paper",
            "trading_day": day, "entries": entries}


class _FakeStore:
    def __init__(self, documents: dict[str, dict]):
        self.documents = documents
        self.keys: list[str] = []

    def read_json(self, relative: str) -> dict | None:
        self.keys.append(relative)
        return self.documents.get(relative)


# ── The journal reader ─────────────────────────────────────────────────────


class TestFillsFromJournal:
    def test_every_fill_becomes_one_row_with_its_gross_price_and_commission(self):
        rows = fills_from_journal(_journal(), RUN_DATE)
        assert len(rows) == N_FILLS
        aaa = [r for r in rows if r["ticker"] == "AAA"]
        assert [r["action"] for r in aaa] == ["SELL", "SELL"]
        assert [r["shares"] for r in aaa] == [600, 400]
        assert all(r["fill_price"] == pytest.approx(110.00) for r in aaa)
        nnn = next(r for r in rows if r["ticker"] == "NNN")
        assert nnn["action"] == "BUY"
        assert nnn["fill_price"] == pytest.approx(1042.11)
        assert all(r["commission_usd"] == pytest.approx(1.00) for r in rows)
        assert all(r["status"] == "Filled" and r["price_at_order"] is None for r in rows)

    def test_a_v1_fill_exported_into_the_journal_is_not_counted_again(self):
        v1 = {
            "kind": "fill", "event_ref": "v1:abc", "trading_day": RUN_DATE, "evidence": {},
            "postings": [
                {"account": "assets:cash:USD", "amount_cents": 1000, "quantity": None},
                {"account": "assets:securities:ZZZ", "amount_cents": -1000, "quantity": -1},
            ],
        }
        rows = fills_from_journal(_journal(extra_fills=[v1]), RUN_DATE)
        assert "ZZZ" not in {r["ticker"] for r in rows}

    def test_commission_not_separated_is_absent_not_zero(self):
        bare = {
            "kind": "fill", "event_ref": "x1", "trading_day": RUN_DATE,
            "evidence": {"commission_separated": False},
            "postings": [
                {"account": "assets:cash:USD", "amount_cents": -5000, "quantity": None},
                {"account": "assets:securities:YYY", "amount_cents": 5000, "quantity": 10},
            ],
        }
        rows = fills_from_journal(_journal(extra_fills=[bare]), RUN_DATE)
        assert next(r for r in rows if r["ticker"] == "YYY")["commission_usd"] is None

    @pytest.mark.parametrize(
        "doc",
        [
            {"schema_version": "ledger_journal.v2", "trading_day": RUN_DATE, "entries": []},
            {"schema_version": "ledger_journal.v1", "trading_day": PRIOR_DATE, "entries": []},
            {"schema_version": "ledger_journal.v1", "trading_day": RUN_DATE, "entries": [{
                "kind": "fill", "event_ref": "bad", "evidence": {},
                "postings": [{"account": "assets:securities:QQQ", "amount_cents": 500,
                              "quantity": -5}],
            }]},
        ],
    )
    def test_a_journal_it_cannot_read_as_fills_is_refused(self, doc):
        with pytest.raises(V2FillsError):
            fills_from_journal(doc, RUN_DATE)


class TestLoadV2Fills:
    def test_dates_before_the_v2_cutover_never_read_the_store(self):
        def _boom():
            raise AssertionError("store opened for a pre-v2 date")

        assert load_v2_fills("2026-11-02", store_factory=_boom) == ([], None)
        assert V2_FIRST_SESSION == "2026-11-03"

    def test_reads_the_days_journal_under_the_paper_tenant(self):
        store = _FakeStore({journal_key(RUN_DATE): _journal()})
        rows, warning = load_v2_fills(RUN_DATE, store_factory=lambda: store)
        assert warning is None
        assert len(rows) == N_FILLS
        assert store.keys == ["trader/ledger/journal/nous-ergon-paper/2026-11-04.json"]

    def test_a_missing_journal_is_a_named_warning(self):
        rows, warning = load_v2_fills(RUN_DATE, store_factory=lambda: _FakeStore({}))
        assert rows == []
        assert warning is not None
        assert "no ledger journal" in warning and "unattributed_true_usd" in warning

    def test_an_unreadable_store_is_a_named_warning_never_a_raise(self):
        def _denied():
            raise PermissionError("AccessDenied")

        rows, warning = load_v2_fills(RUN_DATE, store_factory=_denied)
        assert rows == []
        assert warning is not None and "AccessDenied" in warning

    def test_no_trader_env_on_this_host_is_a_named_warning(self, tmp_path):
        rows, warning = load_v2_fills(RUN_DATE, env_path=tmp_path / "absent.env")
        assert rows == []
        assert warning is not None and "absent" in warning


class TestMerge:
    def test_v1_rows_are_kept_and_v2_appended(self):
        v1 = [{"ticker": "AAA", "action": "EXIT", "shares": 10}]
        v2 = [{"ticker": "BBB", "action": "SELL", "shares": 5}]
        merged, kept, overlap = merge_for_attribution(v1, v2)
        assert merged == v1 + v2
        assert kept == v2
        assert overlap == []

    def test_a_ticker_v1_also_traded_today_keeps_v1s_rows_only(self):
        v1 = [{"ticker": "AAA", "action": "EXIT", "shares": 10}]
        v2 = [{"ticker": "AAA", "action": "SELL", "shares": 10},
              {"ticker": "BBB", "action": "SELL", "shares": 5}]
        merged, kept, overlap = merge_for_attribution(v1, v2)
        assert [r["ticker"] for r in merged] == ["AAA", "BBB"]
        assert overlap == ["AAA"]

    def test_rotation_sleeve_prices_the_v2_sells(self):
        prior = {t: {"shares": s, "closing_price": pc} for t, (s, pc, _) in SOLD.items()}
        rows = fills_from_journal(_journal(), RUN_DATE)
        assert compute_rotation_realized({}, prior, []) == 0.0
        assert compute_rotation_realized({}, prior, rows) == pytest.approx(ROTATION)


# ── run(): the 2026-11-04 shape, end to end ────────────────────────────────


def _frame(prior_close: float, close: float) -> pd.DataFrame:
    lo, hi = min(prior_close, close) * 0.9, max(prior_close, close) * 1.1
    return pd.DataFrame(
        {"Close": [prior_close, close], "Low": [lo, lo], "High": [hi, hi]},
        index=pd.to_datetime([PRIOR_DATE, RUN_DATE]),
    )


def _universe_library() -> MagicMock:
    frames = {t: _frame(pc, c) for t, (_, pc, c) in {**SOLD, **HELD}.items()}
    frames.update({t: _frame(c, c) for t, (c, _) in NEW.items()})
    frames["SPY"] = _frame(500.0, 505.0)
    lib = MagicMock()

    def _read(sym):
        if sym not in frames:
            raise KeyError(sym)
        return SimpleNamespace(data=frames[sym])

    lib.read.side_effect = _read
    return lib


def _cash_today() -> float:
    flow = sum(-shares * price for legs in FILLS.values() for shares, price in legs)
    return CASH_PRIOR + flow - COMMISSIONS


def _positions_today() -> dict:
    out = {}
    for t, (s, _, c) in HELD.items():
        out[t] = {"shares": s + sum(q for q, _ in FILLS[t]), "market_value": 0.0, "sector": "Tech"}
        out[t]["market_value"] = out[t]["shares"] * c
    for t, (c, avg) in NEW.items():
        shares = sum(q for q, _ in FILLS[t])
        out[t] = {"shares": shares, "market_value": shares * c, "avg_cost": avg, "sector": "Tech"}
    return out


def _nav_prior() -> float:
    mv = sum(s * pc for s, pc, _ in {**SOLD, **HELD}.values())
    return CASH_PRIOR + ACCRUED + mv


def _nav_today() -> float:
    return _cash_today() + ACCRUED + sum(p["market_value"] for p in _positions_today().values())


def _seed_prior_row(conn: sqlite3.Connection) -> None:
    prior_positions = {
        t: {"shares": s, "market_value": s * pc, "ib_market_value": s * pc,
            "closing_price": pc, "sector": "Tech"}
        for t, (s, pc, _) in {**SOLD, **HELD}.items()
    }
    log_eod(conn, {
        "date": PRIOR_DATE, "portfolio_nav": _nav_prior(), "daily_return_pct": 0.0,
        "spy_return_pct": 0.0, "daily_alpha_pct": 0.0,
        "positions_snapshot": prior_positions, "spy_close": 500.0,
        "total_cash": CASH_PRIOR, "accrued_interest": ACCRUED,
        "nav_basis": NAV_BASIS_IB_NETLIQ, "nav_ib_usd": _nav_prior(),
        "nav_settled_usd": _nav_prior(), "nav_basis_diff_usd": 0.0, "nav_basis_diff_bps": 0.0,
    })


def _snapshot() -> dict:
    positions = _positions_today()
    gross = sum(p["market_value"] for p in positions.values())
    return {
        "account": {
            "net_liquidation": _nav_today(), "total_cash": _cash_today(),
            "accrued_interest": ACCRUED, "unrealized_pnl": 0.0, "realized_pnl": 0.0,
            "gross_position_value": gross,
        },
        "positions": positions,
        "accrued_dividends": {},
        "captured_at": f"{RUN_DATE}T21:20:00Z",
    }


def _vendor_unavailable(_rows):
    raise RuntimeError("vendor unavailable (test stub)")


def _run_eod(store: _FakeStore) -> SimpleNamespace:
    from executor import eod_reconcile

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    conn = init_db(db_path)
    _seed_prior_row(conn)
    captured: dict = {}

    def _load(run_date):
        return load_v2_fills(run_date, store_factory=lambda: store)

    error: Exception | None = None
    try:
        with patch.object(eod_reconcile, "now_dual") as m_now, \
             patch.object(eod_reconcile, "load_config") as m_cfg, \
             patch("executor.preflight.ExecutorPreflight") as m_pre, \
             patch.object(eod_reconcile, "init_db", return_value=conn), \
             patch("executor.snapshot_capturer.load_snapshot", return_value=_snapshot()), \
             patch("executor.price_cache._open_universe_library", return_value=_universe_library()), \
             patch.object(eod_reconcile, "_spy_close", return_value=505.0), \
             patch.object(eod_reconcile, "fetch_ex_dividends", return_value=({}, {}, True, None)), \
             patch.object(eod_reconcile, "fetch_live_anchor_window", side_effect=_vendor_unavailable), \
             patch.object(eod_reconcile, "_load_signals_from_s3", return_value=({}, None)), \
             patch.object(eod_reconcile, "_load_predictions_from_s3", return_value=({}, None)), \
             patch.object(eod_reconcile, "load_v2_fills", side_effect=_load), \
             patch.object(eod_reconcile, "write_eod_report",
                          side_effect=lambda report, **kw: captured.__setitem__("report", report)), \
             patch.object(eod_reconcile, "send_eod_email"), \
             patch.object(eod_reconcile, "backup_to_s3"), \
             patch("boto3.client"), \
             patch("nousergon_lib.logging.get_flow_doctor", return_value=MagicMock()):
            m_now.return_value = SimpleNamespace(trading_day=RUN_DATE, calendar_date=RUN_DATE)
            m_cfg.return_value = {
                "db_path": db_path, "trades_bucket": "alpha-engine-research",
                "signals_bucket": "alpha-engine-research", "aws_region": "us-east-1",
                "email_sender": "x@x.com", "email_recipients": "y@y.com",
                "nav_basis": NAV_BASIS_IB_NETLIQ,
            }
            m_pre.return_value.run.return_value = None
            try:
                eod_reconcile.run(run_date=RUN_DATE, run_audit=False)
            except RuntimeError as exc:
                error = exc
        read = sqlite3.connect(db_path)
        cols = [r[1] for r in read.execute("PRAGMA table_info(eod_pnl)")]
        raw = read.execute("SELECT * FROM eod_pnl WHERE date=?", (RUN_DATE,)).fetchone()
        read.close()
        return SimpleNamespace(row=dict(zip(cols, raw, strict=True)), error=error,
                               report=captured.get("report", {}))
    finally:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass
        os.unlink(db_path)


@pytest.fixture(scope="module")
def without_journal() -> SimpleNamespace:
    return _run_eod(_FakeStore({}))


@pytest.fixture(scope="module")
def with_journal() -> SimpleNamespace:
    return _run_eod(_FakeStore({journal_key(RUN_DATE): _journal()}))


class TestTheNovemberFourthShape:
    def test_fixture_nav_change_is_rotation_plus_marks_minus_commission(self):
        held = sum(s * (c - pc) for s, pc, c in HELD.values())
        add = sum(q * (HELD[t][2] - px) for t in HELD for q, px in FILLS[t])
        new = sum(q * (NEW[t][0] - px) for t in NEW for q, px in FILLS[t])
        assert _nav_today() - _nav_prior() == pytest.approx(ROTATION + held + add + new - COMMISSIONS)

    def test_without_the_journal_the_gate_fails_as_it_did(self, without_journal):
        """The defect, reproduced: same NAV, every fill unseen."""
        row = without_journal.row
        assert row["rotation_realized_usd"] == pytest.approx(0.0)
        assert row["traded_notional_usd"] == pytest.approx(0.0)
        assert row["unattributed_true_usd"] > 40_000
        assert without_journal.error is not None
        assert "P&L residual bound breached" in str(without_journal.error)

    def test_with_the_journal_the_residual_is_the_commission(self, with_journal):
        row = with_journal.row
        assert row["rotation_realized_usd"] == pytest.approx(ROTATION)
        assert row["unattributed_true_usd"] == pytest.approx(-COMMISSIONS, abs=0.01)
        assert row["commission_usd"] == pytest.approx(COMMISSIONS)
        notional = sum(abs(q) * px for legs in FILLS.values() for q, px in legs)
        assert row["traded_notional_usd"] == pytest.approx(notional)
        assert with_journal.error is None

    def test_nav_is_identical_either_way(self, without_journal, with_journal):
        assert with_journal.row["portfolio_nav"] == pytest.approx(without_journal.row["portfolio_nav"])
        assert with_journal.row["portfolio_nav"] == pytest.approx(_nav_today())

    def test_the_report_prices_the_same_fills(self, with_journal):
        attribution = with_journal.report["alpha_attribution"]
        rotation = [c for c in attribution["components"] if c["kind"] == "rotation"]
        assert sum(c["contrib_usd"] for c in rotation) == pytest.approx(ROTATION)
        sources = {t.get("source") for t in with_journal.report["trades_today"]}
        assert sources == {"v2_ledger_journal"}

    def test_integrity_breach_column_is_clean(self, with_journal):
        breaches = json.loads(with_journal.row["integrity_breach_json"] or "[]")
        assert not [b for b in breaches if "residual" in str(b.get("kind", ""))]
