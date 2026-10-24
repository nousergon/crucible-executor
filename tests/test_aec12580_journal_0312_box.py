"""scripts/aec12580_journal_0312_box.py (alpha-engine-config-I12580).

Every case runs on a temporary trades.db. The real database, S3 and the trading box are never
touched.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from executor import money_ledger
from executor.trade_logger import init_db

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aec12580_journal_0312_box.py"
_spec = importlib.util.spec_from_file_location("aec12580_journal_0312_box", SCRIPT)
box = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(box)

SAT_NIGHT = dt.datetime(2026, 10, 24, 23, 0, tzinfo=dt.UTC)  # Sat 19:00 ET: allowed
TUE_NOON = dt.datetime(2026, 10, 27, 16, 0, tzinfo=dt.UTC)  # Tue 12:00 ET: refused
ACCRUED = 113.21


def _db(tmp_path: Path, **row) -> str:
    path = str(tmp_path / "trades.db")
    conn = init_db(path)
    base = {
        "date": box.RUN_DATE,
        "created_at": "2026-10-24T03:30:00Z",
        "portfolio_nav": box.RULED_NAV,
        "total_cash": round(box.RULED_NAV - ACCRUED, 2),
        "accrued_interest": ACCRUED,
        "positions_snapshot": "{}",
        "nav_basis": "settled_close",
        "nav_ib_raw_usd": box.RULED_NAV,
    }
    base.update(row)
    conn.execute(
        "INSERT INTO eod_pnl (date, created_at, portfolio_nav, total_cash, accrued_interest, "
        "positions_snapshot, nav_basis, nav_ib_raw_usd) VALUES (:date, :created_at, :portfolio_nav, "
        ":total_cash, :accrued_interest, :positions_snapshot, :nav_basis, :nav_ib_raw_usd)",
        base,
    )
    conn.commit()
    conn.close()
    return path


def _entries(path: str) -> int:
    c = sqlite3.connect(path)
    money_ledger.ensure_schema(c)
    n = len(money_ledger.entries_for_date(c, box.RUN_DATE))
    c.close()
    return n


def test_the_flat_backfilled_row_journals_nothing_on_the_normal_path(tmp_path):
    path = _db(tmp_path)
    out = box.run("check", apply=False, db_path=path, trades_bucket=None, now=SAT_NIGHT)
    assert out["status"] == "no_entry"
    assert out["existing_entries"] == []
    out = box.run("apply", apply=True, db_path=path, trades_bucket=None, now=SAT_NIGHT)
    assert out["status"] == "no_entry" and out["written"] is False
    assert "backup_path" not in out
    assert _entries(path) == 0


def test_check_never_writes_even_when_an_entry_is_due(tmp_path):
    snap = {"AAA": {"market_value": 1000.0, "ib_market_value": 1012.5, "close_source": "polygon"}}
    path = _db(tmp_path, positions_snapshot=json.dumps(snap), nav_ib_raw_usd=box.RULED_NAV + 12.5)
    before = Path(path).read_bytes()
    out = box.run("check", apply=False, db_path=path, trades_bucket=None, now=TUE_NOON)
    assert out["status"] == "entry"
    assert Path(path).read_bytes() == before
    assert _entries(path) == 0


def test_apply_posts_once_through_the_live_helper_and_balances(tmp_path, monkeypatch):
    monkeypatch.setattr(box, "BACKUP_DIR", str(tmp_path / "bk"))
    snap = {"AAA": {"market_value": 1000.0, "ib_market_value": 1012.5, "close_source": "polygon"}}
    path = _db(tmp_path, positions_snapshot=json.dumps(snap), nav_ib_raw_usd=box.RULED_NAV + 12.5)
    out = box.run("apply", apply=True, db_path=path, trades_bucket=None, now=SAT_NIGHT)
    assert out["status"] == "posted", out
    assert out["problems"] == []
    assert Path(out["backup_path"]).exists() and len(out["backup_sha256"]) == 64
    c = sqlite3.connect(path)
    assert all(v == 0 for v in money_ledger.trial_balance(c).values())
    c.close()
    assert _entries(path) == 1
    again = box.run("apply", apply=True, db_path=path, trades_bucket=None, now=SAT_NIGHT)
    assert again["newly_written"] is False  # idempotent: content-hashed entry_id
    assert _entries(path) == 1


def test_verify_ties_the_asset_postings_to_the_rows_correction(tmp_path):
    row = {"portfolio_nav": 100.0, "nav_ib_raw_usd": 90.0}
    good = {
        "entry_id": "e",
        "postings": [
            {"account": "assets:cash:USD", "amount": "10.00"},
            {"account": money_ledger.ACCOUNT_VALUATION_CORRECTION, "amount": "-10.00"},
        ],
    }
    c = sqlite3.connect(":memory:")
    money_ledger.ensure_schema(c)
    assert box.verify(c, good, row) == []
    off = {
        "entry_id": "e",
        "postings": [
            {"account": "assets:cash:USD", "amount": "9.00"},
            {"account": money_ledger.ACCOUNT_VALUATION_CORRECTION, "amount": "-9.00"},
        ],
    }
    assert any("correction" in p for p in box.verify(c, off, row))
    assert Decimal("10.00") == money_ledger.to_cents(100.0) - money_ledger.to_cents(90.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"portfolio_nav": box.RULED_NAV + 5}, "is not the ruled"),
        ({"nav_basis": "ib_netliq"}, "nav_basis"),
        ({"nav_ib_raw_usd": None}, "nav_ib_raw_usd"),
    ],
)
def test_preconditions_refuse(tmp_path, change, match):
    path = _db(tmp_path, **change)
    with pytest.raises(box.Refuse, match=match):
        box.run("check", apply=False, db_path=path, trades_bucket=None, now=SAT_NIGHT)


def test_a_missing_row_refuses(tmp_path):
    path = str(tmp_path / "trades.db")
    init_db(path).close()
    with pytest.raises(box.Refuse, match="no eod_pnl row"):
        box.run("check", apply=False, db_path=path, trades_bucket=None, now=SAT_NIGHT)


def test_apply_needs_the_flag_and_refuses_the_market_window(tmp_path):
    path = _db(tmp_path)
    with pytest.raises(box.Refuse, match="--apply"):
        box.run("apply", apply=False, db_path=path, trades_bucket=None, now=SAT_NIGHT)
    with pytest.raises(box.Refuse, match="07:00-16:30 ET"):
        box.run("apply", apply=True, db_path=path, trades_bucket=None, now=TUE_NOON)


def test_main_prints_one_result_line(tmp_path, capsys):
    path = _db(tmp_path)
    assert box.main(["check", "--db-path", path]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("RESULT_JSON=") and json.loads(line.split("=", 1)[1])["status"] == "no_entry"
