"""Tests for executor/money_ledger.py — every correction is a balanced entry.

Brian, 2026-10-22 14:20Z: "we should also make sure we are using a ledger which
must always remain balanced. the daily correction should represent a journal
entry." Schema shape: alpha-engine-config-I12510.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from executor import money_ledger as ml

D = Decimal


def _entry(**overrides):
    kwargs = {
        "effective_date": "2026-10-20",
        "source": "daily_auto_correct",
        "kind": ml.KIND_SETTLED_RESTRIKE,
        "reason": "test",
        "postings": (
            ml.Posting("assets:securities:UAL", D("2404.64")),
            ml.Posting(ml.ACCOUNT_VALUATION_CORRECTION, D("-2404.64")),
        ),
    }
    kwargs.update(overrides)
    return ml.JournalEntry(**kwargs)


class TestBalancedByConstruction:
    def test_a_balanced_entry_constructs(self):
        assert _entry().entry_id.startswith("je_")

    def test_an_unbalanced_entry_cannot_be_constructed(self):
        with pytest.raises(ml.UnbalancedEntryError, match="does not balance"):
            _entry(postings=(
                ml.Posting("assets:securities:UAL", D("2404.64")),
                ml.Posting(ml.ACCOUNT_VALUATION_CORRECTION, D("-2404.63")),
            ))

    def test_balance_is_per_currency(self):
        with pytest.raises(ml.UnbalancedEntryError):
            _entry(postings=(
                ml.Posting("assets:cash:USD", D("1.00"), "USD"),
                ml.Posting("assets:cash:EUR", D("-1.00"), "EUR"),
            ))

    def test_one_posting_is_not_an_entry(self):
        with pytest.raises(ml.UnbalancedEntryError, match="at least two"):
            _entry(postings=(ml.Posting("assets:cash:USD", D("1.00")),))

    def test_zero_and_sub_cent_postings_are_refused(self):
        with pytest.raises(ValueError, match="zero posting"):
            ml.Posting("assets:cash:USD", D("0.00"))
        with pytest.raises(ValueError, match="exact cents"):
            ml.Posting("assets:cash:USD", D("0.001"))
        with pytest.raises(ValueError, match="exact cents"):
            ml.Posting("assets:cash:USD", 1.5)  # a float is never an amount

    def test_source_kind_reason_and_account_are_closed(self):
        with pytest.raises(ValueError, match="source"):
            _entry(source="manual")
        with pytest.raises(ValueError, match="kind"):
            _entry(kind="whatever")
        with pytest.raises(ValueError, match="reason"):
            _entry(reason="  ")
        with pytest.raises(ValueError, match="chart-of-accounts"):
            ml.Posting("securities UAL", D("1.00"))

    def test_entry_id_is_content_addressed(self):
        assert _entry().entry_id == _entry(reason="different words").entry_id
        other = _entry(effective_date="2026-10-21")
        assert other.entry_id != _entry().entry_id


class TestCents:
    def test_float_noise_does_not_leak_into_cents(self):
        assert ml.to_cents(0.1 + 0.2) == D("0.30")
        assert ml.to_cents(1045825.66) == D("1045825.66")

    def test_non_finite_refused(self):
        with pytest.raises(ValueError):
            ml.to_cents(float("nan"))


class TestViewsAndCorrections:
    def test_view_sums_to_nav_exactly(self):
        view = ml.balance_view(nav=1000.00, cash=100.00, accrued=1.25,
                               securities={"A": 450.10, "B": 448.10})
        assert sum(view.values()) == D("1000.00")
        assert view[ml.ACCOUNT_BROKER_MARK] == D("0.55")

    def test_ib_row_carries_securities_at_the_broker_mark(self):
        snap = {"UAL": {"market_value": 9777.0, "ib_market_value": 9544.0}}
        ib = ml.view_from_eod_row(portfolio_nav=19544.0, total_cash=10000.0,
                                  accrued_interest=0, positions_snapshot=json.dumps(snap),
                                  nav_basis=None)
        settled = ml.view_from_eod_row(portfolio_nav=19777.0, total_cash=10000.0,
                                       accrued_interest=0, positions_snapshot=snap,
                                       nav_basis="settled_close")
        assert ib["assets:securities:UAL"] == D("9544.00")
        assert settled["assets:securities:UAL"] == D("9777.00")
        assert ib[ml.ACCOUNT_BROKER_MARK] == settled[ml.ACCOUNT_BROKER_MARK] == D("0.00")

    def test_row_without_cash_has_no_view(self):
        with pytest.raises(ValueError):
            ml.view_from_eod_row(portfolio_nav=1.0, total_cash=None, accrued_interest=0,
                                 positions_snapshot="{}", nav_basis="settled_close")

    def test_the_ual_after_hours_mark_restrike(self):
        """2026-10-20: IB marked UAL on an after-hours print (95.44) while the
        settled close was 97.77. Re-striking the row onto the settled basis is
        one entry whose asset lines sum to exactly the NAV change."""
        snap = {
            "UAL": {"market_value": 1140 * 97.77, "ib_market_value": 1140 * 95.44 + 251.6},
            "ANF": {"market_value": 41241.15, "ib_market_value": 41248.80},
        }
        before = ml.view_from_eod_row(
            portfolio_nav=1043421.02, total_cash=26438.92, accrued_interest=87.52,
            positions_snapshot=snap, nav_basis="ib_netliq")
        after = ml.view_from_eod_row(
            portfolio_nav=1045825.66, total_cash=26438.92, accrued_interest=87.52,
            positions_snapshot=snap, nav_basis="settled_close")
        entry = ml.correction_entry(effective_date="2026-10-20",
                                    kind=ml.KIND_SETTLED_RESTRIKE, reason="restrike",
                                    before=before, after=after)
        assert entry.total("assets:") == D("2404.64")
        assert entry.total(ml.ACCOUNT_VALUATION_CORRECTION) == D("-2404.64")
        assert entry.evidence["nav_change_usd"] == "2404.64"
        assert {p.account for p in entry.postings} >= {
            "assets:securities:UAL", "assets:securities:ANF"}
        assert "UAL" in ml.describe(entry)

    def test_no_change_is_no_entry(self):
        view = ml.balance_view(nav=10, cash=10, accrued=0, securities={})
        assert ml.correction_entry(effective_date="2026-10-20",
                                   kind=ml.KIND_SETTLED_RESTRIKE, reason="r",
                                   before=view, after=dict(view)) is None

    def test_reclass_with_no_nav_change_needs_no_offset(self):
        before = ml.balance_view(nav=100, cash=50, accrued=0, securities={"A": 50})
        after = ml.balance_view(nav=100, cash=50, accrued=0, securities={"A": 49})
        entry = ml.correction_entry(effective_date="2026-10-20",
                                    kind=ml.KIND_SETTLED_RESTRIKE, reason="r",
                                    before=before, after=after)
        assert {p.account for p in entry.postings} == {
            "assets:securities:A", ml.ACCOUNT_BROKER_MARK}


class TestStore:
    @pytest.fixture
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "t.db")
        yield c
        c.close()

    def test_record_is_idempotent(self, conn):
        doc1, new1 = ml.record_entry(conn, _entry())
        doc2, new2 = ml.record_entry(conn, _entry())
        assert new1 and not new2
        assert doc1["entry_id"] == doc2["entry_id"]
        assert conn.execute("SELECT COUNT(*) FROM ledger_postings").fetchone()[0] == 2

    def test_append_only(self, conn):
        ml.record_entry(conn, _entry())
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE ledger_postings SET amount='1.00'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM ledger_journal_entries")

    def test_trial_balance_is_zero_and_detects_a_planted_break(self, conn):
        ml.record_entry(conn, _entry())
        ml.record_entry(conn, _entry(effective_date="2026-10-21"))
        assert ml.trial_balance(conn) == {"USD": D("0.00")}
        conn.execute("INSERT INTO ledger_postings VALUES ('x', 0, 'assets:cash:USD', "
                     "'0.01', 'USD')")
        with pytest.raises(ml.LedgerIntegrityError, match="not zero"):
            ml.trial_balance(conn)

    def test_a_broken_ledger_refuses_new_entries_and_rolls_them_back(self, conn):
        ml.ensure_schema(conn)
        conn.execute("INSERT INTO ledger_postings VALUES ('x', 0, 'assets:cash:USD', "
                     "'0.01', 'USD')")
        conn.commit()
        with pytest.raises(ml.LedgerIntegrityError):
            ml.record_entry(conn, _entry())
        assert conn.execute(
            "SELECT COUNT(*) FROM ledger_journal_entries").fetchone()[0] == 0

    def test_post_publishes_an_immutable_s3_copy(self, conn):
        s3 = MagicMock()
        doc = ml.post(conn, _entry(), s3_client=s3, bucket="b")
        kwargs = s3.put_object.call_args.kwargs
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["Key"] == f"trades/ledger/journal/2026-10-20/{doc['entry_id']}.json"
        body = json.loads(kwargs["Body"])
        assert body["schema_version"] == ml.LEDGER_SCHEMA_VERSION
        assert sum(D(p["amount"]) for p in body["postings"]) == 0

    def test_existing_s3_object_is_the_same_entry(self, conn):
        err = Exception("exists")
        err.response = {"Error": {"Code": "PreconditionFailed"}}
        s3 = MagicMock()
        s3.put_object.side_effect = err
        doc = ml.post(conn, _entry(), s3_client=s3, bucket="b")
        assert "s3_error" not in doc and doc["s3_key"].endswith(".json")

    def test_s3_failure_is_recorded_not_swallowed(self, conn):
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("down")
        doc = ml.post(conn, _entry(), s3_client=s3, bucket="b")
        assert doc["s3_error"].startswith("RuntimeError")
        assert ml.entries_for_date(conn, "2026-10-20")  # the db copy stands

    def test_none_entry_posts_nothing(self, conn):
        assert ml.post(conn, None) is None


class TestStoreEnforcedBalance:
    """Policy Rule 2 (alpha-engine-config-I12558): the STORE refuses an
    unbalanced or partial entry, whatever path writes it. Every write here is
    a raw ``sqlite3`` statement that never goes through ``record_entry``."""

    @pytest.fixture
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "t.db")
        ml.ensure_schema(c)
        yield c
        c.close()

    @staticmethod
    def _lines(conn, entry_id, *lines):
        conn.executemany(
            "INSERT INTO ledger_postings VALUES (?, ?, ?, ?, ?)",
            [(entry_id, i, acct, amt, cur) for i, (acct, amt, cur) in enumerate(lines)],
        )

    @staticmethod
    def _header(conn, entry_id):
        conn.execute(
            "INSERT INTO ledger_journal_entries VALUES (?, '2026-10-20', 'now', "
            "'daily_auto_correct', 'settled_restrike', 'bypass', NULL, '{}')",
            (entry_id,),
        )

    def test_the_guard_triggers_are_installed(self, conn):
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert set(ml.STORE_GUARD_TRIGGERS) <= names

    def test_a_header_with_no_postings_cannot_be_posted(self, conn):
        with pytest.raises(sqlite3.IntegrityError, match="does not balance"):
            self._header(conn, "je_partial")

    def test_a_single_posting_cannot_be_posted(self, conn):
        self._lines(conn, "je_one", ("assets:cash:USD", "1.00", "USD"))
        with pytest.raises(sqlite3.IntegrityError, match="does not balance"):
            self._header(conn, "je_one")

    def test_an_unbalanced_entry_cannot_be_posted_by_a_raw_write(self, conn):
        self._lines(conn, "je_bad",
                    ("assets:securities:UAL", "2404.64", "USD"),
                    (ml.ACCOUNT_VALUATION_CORRECTION, "-2404.63", "USD"))
        with pytest.raises(sqlite3.IntegrityError, match="does not balance"):
            self._header(conn, "je_bad")
        assert conn.execute("SELECT COUNT(*) FROM ledger_journal_entries").fetchone()[0] == 0

    def test_balance_is_enforced_per_currency_by_the_store(self, conn):
        self._lines(conn, "je_fx",
                    ("assets:cash:USD", "1.00", "USD"),
                    ("assets:cash:EUR", "-1.00", "EUR"))
        with pytest.raises(sqlite3.IntegrityError, match="does not balance"):
            self._header(conn, "je_fx")

    def test_the_store_balance_is_exact_cents_at_large_magnitudes(self, conn):
        self._lines(conn, "je_big",
                    ("assets:cash:USD", "98765432109.87", "USD"),
                    ("assets:securities:A", "-98765432109.86", "USD"))
        with pytest.raises(sqlite3.IntegrityError, match="does not balance"):
            self._header(conn, "je_big")
        conn.rollback()  # the refused header leaves its lines orphaned until rolled back
        self._lines(conn, "je_big_ok",
                    ("assets:cash:USD", "98765432109.87", "USD"),
                    ("assets:securities:A", "-98765432109.80", "USD"),
                    ("assets:securities:B", "-0.07", "USD"))
        self._header(conn, "je_big_ok")
        assert ml.verify_entries(conn) == 1

    def test_a_posted_entry_is_sealed(self, conn):
        doc, _ = ml.record_entry(conn, _entry())
        with pytest.raises(sqlite3.IntegrityError, match="sealed"):
            self._lines(conn, doc["entry_id"], ("assets:cash:USD", "5.00", "USD"))
        assert conn.execute(
            "SELECT COUNT(*) FROM ledger_postings WHERE entry_id = ?", (doc["entry_id"],)
        ).fetchone()[0] == 2

    def test_a_failed_append_leaves_neither_header_nor_postings(self, conn, monkeypatch):
        def broken(_conn):
            raise ml.LedgerIntegrityError("planted")

        monkeypatch.setattr(ml, "verify_entries", broken)
        with pytest.raises(ml.LedgerIntegrityError, match="planted"):
            ml.record_entry(conn, _entry())
        assert conn.execute("SELECT COUNT(*) FROM ledger_postings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ledger_journal_entries").fetchone()[0] == 0


class TestVerifyEntries:
    """A ledger written before the store guards existed (the trades.db on the
    box was created by executor-PR616's schema) is checked entry by entry."""

    LEGACY_SCHEMA = """
    CREATE TABLE ledger_journal_entries (
        entry_id TEXT PRIMARY KEY, effective_date TEXT NOT NULL, posted_at TEXT NOT NULL,
        source TEXT NOT NULL, kind TEXT NOT NULL, reason TEXT NOT NULL,
        reverses_entry_id TEXT, evidence_json TEXT NOT NULL);
    CREATE TABLE ledger_postings (
        entry_id TEXT NOT NULL, line_no INTEGER NOT NULL, account TEXT NOT NULL,
        amount TEXT NOT NULL, currency TEXT NOT NULL, PRIMARY KEY (entry_id, line_no));
    """

    @pytest.fixture
    def legacy(self, tmp_path):
        c = sqlite3.connect(tmp_path / "legacy.db")
        c.executescript(self.LEGACY_SCHEMA)
        yield c
        c.close()

    @staticmethod
    def _write(conn, entry_id, *amounts, header=True):
        if header:
            conn.execute(
                "INSERT INTO ledger_journal_entries VALUES (?, '2026-10-20', 'now', "
                "'daily_auto_correct', 'settled_restrike', 'legacy', NULL, '{}')",
                (entry_id,),
            )
        conn.executemany(
            "INSERT INTO ledger_postings VALUES (?, ?, 'assets:cash:USD', ?, 'USD')",
            [(entry_id, i, a) for i, a in enumerate(amounts)],
        )

    def test_a_clean_legacy_ledger_verifies_and_gains_the_guards(self, legacy):
        self._write(legacy, "je_a", "2404.64", "-2404.64")
        ml.ensure_schema(legacy)
        assert ml.verify_entries(legacy) == 1
        names = {r[0] for r in legacy.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert set(ml.STORE_GUARD_TRIGGERS) <= names
        ml.record_entry(legacy, _entry())  # appends still work over the old rows
        assert ml.verify_entries(legacy) == 2

    def test_offsetting_broken_entries_pass_the_trial_balance_but_not_verify(self, legacy):
        self._write(legacy, "je_up", "1.00", "-0.99")
        self._write(legacy, "je_down", "-1.00", "0.99", "0.00")
        assert ml.trial_balance(legacy) == {"USD": D("0.00")}
        with pytest.raises(ml.LedgerIntegrityError, match="je_down.*je_up|je_up"):
            ml.verify_entries(legacy)

    def test_a_partial_entry_and_an_orphan_line_are_breaks(self, legacy):
        self._write(legacy, "je_partial", "1.00")
        self._write(legacy, "je_orphan", "-1.00", header=False)
        with pytest.raises(ml.LedgerIntegrityError) as exc:
            ml.verify_entries(legacy)
        assert "je_partial: 1 posting(s)" in str(exc.value)
        assert "je_orphan: 1 posting(s) with no entry header" in str(exc.value)

    def test_a_broken_legacy_ledger_refuses_new_entries(self, legacy):
        self._write(legacy, "je_up", "1.00", "-0.99")
        self._write(legacy, "je_down", "-1.00", "0.99")
        with pytest.raises(ml.LedgerIntegrityError, match="broken entr"):
            ml.record_entry(legacy, _entry())
        assert legacy.execute(
            "SELECT COUNT(*) FROM ledger_journal_entries").fetchone()[0] == 2

    def test_an_empty_store_verifies_to_zero_entries(self, tmp_path):
        c = sqlite3.connect(tmp_path / "empty.db")
        assert ml.verify_entries(c) == 0
        c.close()
