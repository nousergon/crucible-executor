"""Tests for executor/ledger_completeness.py (alpha-engine-config-I12411).

Thirteen historical gaps are declared exceptions under one ruling id: seven
(Brian 2026-10-21 16:34Z) and six more (Brian 2026-10-22 15:37Z). Any other
divergence must still fail the check -- including CRUS +136, which is restated
by its own script and deliberately NOT declared.
"""

from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from executor.ledger_completeness import (
    DECLARED_EXCEPTIONS_PATH,
    check_ledger_completeness,
    load_declared_exceptions,
    main,
    run_on_db,
)

RULING = "nousergon/alpha-engine-config#12411"
SEVEN = {
    "COKE": ("2026-04-01", 315), "UNH": ("2026-04-07", 156), "NVT": ("2026-04-10", 62),
    "ABBV": ("2026-04-13", 218), "PFE": ("2026-04-22", 231), "CME": ("2026-05-26", 5),
    "BRO": ("2026-05-26", -1089),
}
SIX = {
    "AKAM": ("2026-03-20", 382), "CIEN": ("2026-03-20", 118), "COST": ("2026-03-25", 60),
    "PLTR": ("2026-03-27", 130), "TER": ("2026-04-02", -86), "VRTX": ("2026-04-06", -312),
}
DECLARED = {**SEVEN, **SIX}
RULED_BY = {**dict.fromkeys(SEVEN, "Brian 2026-10-21T16:34Z"),
            **dict.fromkeys(SIX, "Brian 2026-10-22T15:37Z")}
# CRUS +136 (the 2026-10-12 EXIT) is restated by its own script, not declared.
CRUS_DELTA = 136
AS_OF = "2026-10-20"


def _trade(ticker, action, shares, date="2026-03-02", filled=None):
    return {"trade_id": f"{ticker}-{date}-{action}", "date": date, "ticker": ticker,
            "action": action, "shares": shares, "filled_shares": filled}


def _ledger_with(gaps):
    """Ledger rows reproducing each divergence in ``gaps`` against an empty snapshot."""
    rows = []
    for t, (d, delta) in gaps.items():
        if delta > 0:
            rows += [_trade(t, "ENTER", delta + 100, "2026-03-02"), _trade(t, "EXIT", 100, d)]
        else:  # BRO/TER/VRTX: an unlogged BUY leaves the logged EXIT larger than the logged ENTER
            rows += [_trade(t, "ENTER", 170, "2026-03-02"), _trade(t, "EXIT", 170 - delta, d)]
    return rows


def _ledger_declared():
    """Ledger rows reproducing every declared divergence."""
    return _ledger_with(DECLARED)


class TestDeclaredFile:
    def test_file_declares_exactly_the_thirteen_with_ruling(self):
        entries = load_declared_exceptions()
        assert {e["ticker"]: (e["date"], e["delta_shares"]) for e in entries} == DECLARED
        for e in entries:
            assert e["ruling"] == RULING
            assert e["ruled_by"] == RULED_BY[e["ticker"]]
            assert e["reason"].strip()

    def test_crus_is_not_declared(self):
        assert "CRUS" not in {e["ticker"] for e in load_declared_exceptions()}

    def test_missing_file_declares_nothing(self, tmp_path):
        assert load_declared_exceptions(tmp_path / "absent.json") == []


class TestDeclaredGapsPass:
    def test_declared_pass_with_the_shipped_declarations(self):
        res = check_ledger_completeness(
            _ledger_declared(), {}, as_of=AS_OF, declared_exceptions=load_declared_exceptions())
        assert res["status"] == "ok", res["message"]
        assert res["breaches"] == []
        assert {d["ticker"] for d in res["declared"]} == set(DECLARED)
        assert all(d["ruling"] == RULING for d in res["declared"])
        assert RULING in res["message"]

    def test_declared_fail_without_declarations(self):
        res = check_ledger_completeness(_ledger_declared(), {}, as_of=AS_OF, declared_exceptions=[])
        assert res["status"] == "breach"
        assert {b["ticker"] for b in res["breaches"]} == set(DECLARED)
        assert {b["kind"] for b in res["breaches"]} == {"undeclared_gap"}


class TestUndeclaredGapFails:
    def test_synthetic_gap_fails_while_the_declared_still_pass(self):
        rows = _ledger_declared() + [_trade("ZZZZ", "ENTER", 40, "2026-09-01")]
        res = check_ledger_completeness(
            rows, {}, as_of=AS_OF, declared_exceptions=load_declared_exceptions())
        assert res["status"] == "breach"
        assert [(b["kind"], b["ticker"], b["delta_shares"]) for b in res["breaches"]] == [
            ("undeclared_gap", "ZZZZ", 40)]
        assert {d["ticker"] for d in res["declared"]} == set(DECLARED)

    def test_crus_still_shows_as_undeclared(self):
        """The live 2026-10-20 shape: all thirteen declared gaps plus CRUS +136.
        Only CRUS breaches until its restatement script runs."""
        rows = _ledger_with(DECLARED) + _ledger_with({"CRUS": ("2026-10-12", CRUS_DELTA)})
        res = check_ledger_completeness(
            rows, {}, as_of=AS_OF, declared_exceptions=load_declared_exceptions())
        assert res["status"] == "breach"
        assert [(b["kind"], b["ticker"], b["delta_shares"]) for b in res["breaches"]] == [
            ("undeclared_gap", "CRUS", CRUS_DELTA)]
        assert {d["ticker"] for d in res["declared"]} == set(DECLARED)

    def test_snapshot_position_with_no_ledger_backing_fails(self):
        res = check_ledger_completeness(
            _ledger_declared(), {"ZZZZ": {"shares": 10}}, as_of=AS_OF,
            declared_exceptions=load_declared_exceptions())
        assert [(b["kind"], b["ticker"], b["delta_shares"]) for b in res["breaches"]] == [
            ("undeclared_gap", "ZZZZ", -10)]

    def test_second_gap_on_a_declared_ticker_fails(self):
        """The exception is the exact delta, not the ticker: a further unlogged
        fill on COKE changes the delta and is a NEW gap."""
        rows = _ledger_declared() + [_trade("COKE", "ENTER", 7, "2026-06-01")]
        res = check_ledger_completeness(
            rows, {}, as_of=AS_OF, declared_exceptions=load_declared_exceptions())
        kinds = {(b["kind"], b["ticker"]) for b in res["breaches"]}
        assert ("undeclared_gap", "COKE") in kinds
        assert ("declared_exception_not_a_gap", "COKE") in kinds


class TestLedgerSemantics:
    def test_clean_ledger_and_snapshot_pass(self):
        rows = [_trade("AAA", "ENTER", 10), _trade("BBB", "ENTER", 5), _trade("BBB", "EXIT", 5)]
        res = check_ledger_completeness(rows, {"AAA": {"shares": 10}}, as_of=AS_OF)
        assert res["status"] == "ok" and res["n_findings"] == 0

    def test_partial_fill_counts_filled_shares(self):
        rows = [_trade("AXP", "ENTER", 261), _trade("AXP", "EXIT", 50, "2026-05-26", filled=261)]
        assert check_ledger_completeness(rows, {}, as_of=AS_OF)["status"] == "ok"

    def test_rows_after_as_of_are_ignored(self):
        rows = [_trade("AAA", "ENTER", 10), _trade("NEW", "ENTER", 3, "2026-10-21")]
        assert check_ledger_completeness(rows, {"AAA": {"shares": 10}}, as_of=AS_OF)["status"] == "ok"

    def test_share_count_difference_on_a_held_ticker_is_out_of_scope(self):
        rows = [_trade("AAA", "ENTER", 10)]
        assert check_ledger_completeness(rows, {"AAA": {"shares": 9}}, as_of=AS_OF)["status"] == "ok"

    def test_unknown_action_fails_closed(self):
        rows = [_trade("AAA", "TELEPORT", 10)]
        res = check_ledger_completeness(rows, {}, as_of=AS_OF)
        assert [b["kind"] for b in res["breaches"]] == ["unknown_action"]


class TestExceptionValidation:
    def _decl(self):
        return copy.deepcopy(load_declared_exceptions())

    def _run(self, decl):
        return check_ledger_completeness(_ledger_declared(), {}, as_of=AS_OF, declared_exceptions=decl)

    @pytest.mark.parametrize("bad", [None, "", "per Brian", "alpha-engine-config#12411"])
    def test_entry_without_a_ruling_id_excuses_nothing(self, bad):
        decl = self._decl()
        decl[0]["ruling"] = bad
        res = self._run(decl)
        kinds = {(b["kind"], b["ticker"]) for b in res["breaches"]}
        assert ("declared_exception_without_ruling", decl[0]["ticker"]) in kinds
        assert ("undeclared_gap", decl[0]["ticker"]) in kinds

    def test_entry_without_a_reason_excuses_nothing(self):
        decl = self._decl()
        decl[1]["reason"] = " "
        assert self._run(decl)["status"] == "breach"

    def test_malformed_and_duplicate_entries_excuse_nothing(self):
        decl = self._decl() + ["COKE", {"ticker": "NVT", "delta_shares": "62", "date": "2026-04-10"}]
        res = self._run(decl)
        assert res["status"] == "breach"
        assert "declared_exception_malformed" in {b["kind"] for b in res["breaches"]}

    def test_duplicated_ticker_is_malformed(self):
        decl = self._decl()
        decl.append(copy.deepcopy(decl[0]))
        res = self._run(decl)
        assert {b["kind"] for b in res["breaches"]} >= {"declared_exception_malformed", "undeclared_gap"}

    def test_declaration_for_a_gap_that_does_not_exist_fails(self):
        decl = self._decl() + [{**decl_entry(), "ticker": "GHOST", "delta_shares": 9}]
        res = self._run(decl)
        assert [(b["kind"], b["ticker"]) for b in res["breaches"]] == [
            ("declared_exception_not_a_gap", "GHOST")]

    def test_wrong_delta_does_not_excuse_the_gap(self):
        decl = self._decl()
        decl[0]["delta_shares"] += 1
        kinds = {(b["kind"], b["ticker"]) for b in self._run(decl)["breaches"]}
        assert ("undeclared_gap", decl[0]["ticker"]) in kinds


def decl_entry():
    return {"date": "2026-04-01", "ruling": RULING, "reason": "synthetic"}


def _make_db(path, trades, snapshot, snap_date=AS_OF):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, action TEXT, shares INTEGER, "
              "filled_shares INTEGER, created_at TEXT)")
    c.execute("CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, positions_snapshot TEXT)")
    for i, t in enumerate(trades):
        c.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?)",
                  (t["trade_id"], t["date"], t["ticker"], t["action"], t["shares"], t["filled_shares"], f"{i:06d}"))
    c.execute("INSERT INTO eod_pnl VALUES (?,?)", (snap_date, json.dumps(snapshot)))
    c.commit()
    c.close()


class TestDbRunnerAndCli:
    def test_db_run_passes_with_the_declared_and_fails_with_another(self, tmp_path, capsys):
        ok_db, bad_db = tmp_path / "ok.db", tmp_path / "bad.db"
        _make_db(ok_db, _ledger_declared(), {})
        _make_db(bad_db, _ledger_declared() + [_trade("ZZZZ", "ENTER", 40, "2026-09-01")], {})
        assert run_on_db(ok_db)["status"] == "ok"
        assert main(["--db", str(ok_db)]) == 0
        assert main(["--db", str(bad_db)]) == 1
        assert '"ZZZZ"' in capsys.readouterr().out

    def test_db_is_opened_read_only(self, tmp_path):
        db = tmp_path / "ro.db"
        _make_db(db, _ledger_declared(), {})
        before = db.read_bytes()
        run_on_db(db)
        assert db.read_bytes() == before

    def test_shipped_declarations_path_exists(self):
        assert DECLARED_EXCEPTIONS_PATH.is_file()
