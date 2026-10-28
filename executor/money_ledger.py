"""Double-entry journal for every correction to a published NAV.

Brian, 2026-10-22 14:20Z: "in terms of money tracking, we should also make sure
we are using a ledger which must always remain balanced. the daily correction
should represent a journal entry." Ground-truth ruling the same morning
(alpha-engine-config-I9638 / I10052): **IB is the source for holdings (shares
and cash); Polygon settled data is the source for prices.**

This module writes the journal in the shape the Postgres schema v1 declares
(alpha-engine-config-I12510): ``journal_entries`` (id, effective_date,
posted_at, source, reason, reverses_entry_id) and ``postings`` (entry_id,
account, amount as an exact decimal, currency). It lands in Postgres unchanged
when dual-write (alpha-engine-config-I12511) starts. Until then the store is
``trades.db`` plus one immutable S3 object per entry.

What the journal holds
----------------------
A published ``eod_pnl`` row is a statement of position. Its BALANCE VIEW is::

    assets:cash:USD                  total_cash
    assets:accrued_interest:USD      accrued_interest
    assets:securities:<TICKER>       the market value the headline NAV used
    assets:broker_mark_adjustment    headline NAV − (cash + accrued + Σ securities)

The four lines sum to the headline NAV exactly, by construction. Every change
to a published figure is the DELTA between two views, posted as one entry whose
offset is a named income or equity account:

* ``daily_auto_correct`` / ``ib_mark_to_settled_close`` — the live run moving
  the broker's marks (IB, including after-hours prints) onto Polygon's settled
  closes. Offset: ``income:unrealized_gain:valuation_correction``.
* ``daily_auto_correct`` / ``settled_restrike`` — the T+1 audit re-striking a
  published row on corrected settled data, or onto the settled basis. Offset:
  the same account. ``reverses_entry_id`` is never set here: the correction is
  a new entry, and the original row's view is carried in ``evidence``.

One entry is not a correction: the OPENING BALANCE (``opening_balance`` /
``statement_anchor``, alpha-engine-config-I12557 deliverable 1) anchors the
ledger on one broker statement. Shares and cash come from IB (the statement);
each position is valued at its Polygon settled close. Its postings are cash,
accrued interest and one line per security, offset to
``equity:opening_balance``. A ledger holds at most ONE opening balance; the
store refuses a second (invariant 7), so the anchor cannot be silently moved.
It is built by :func:`opening_balance_entry` and posted only by an explicit
caller; no scheduled run posts it.

Invariants, enforced here and NOT by the caller
------------------------------------------------
1. **Balanced.** Every entry's postings sum to exactly zero per currency, in
   cents, as :class:`decimal.Decimal`. An entry that does not balance cannot be
   constructed, let alone written.
2. **Append-only.** ``trades.db`` refuses UPDATE and DELETE on both tables by
   trigger. A correction is a new entry.
3. **Idempotent.** ``entry_id`` is a hash of the entry's content, so the same
   correction re-run (a repeated reconcile) is one entry, not two.
4. **Trial balance is zero.** :func:`trial_balance` sums every posting ever
   written; a non-zero total is a broken ledger and raises.
5. **Balanced at commit, by the store** (policy Rule 2; alpha-engine-config-
   I12558). An entry's postings are written first and its header LAST, and
   the header is what makes an entry posted. A ``BEFORE INSERT`` trigger on
   ``ledger_journal_entries`` refuses a header unless the entry already has
   at least two postings summing to exactly zero per currency (in integer
   cents). A ``BEFORE INSERT`` trigger on ``ledger_postings`` refuses a line
   for an entry whose header exists, so a posted entry is sealed. Both hold
   for a raw ``sqlite3`` write that never calls :func:`record_entry`: an
   unbalanced or partial entry cannot become posted through ANY write path.
6. **Every entry verified, not only the total.** :func:`verify_entries`
   checks each stored entry in exact :class:`~decimal.Decimal` (balanced per
   currency, two or more lines) and that no posting lacks a header and no
   header lacks postings. It runs inside every append, next to the trial
   balance; a zero total over two offsetting broken entries is still a break.
7. **One anchor.** A ``BEFORE INSERT`` trigger refuses an
   ``opening_balance`` header when the ledger already holds one. Re-posting the
   SAME anchor is a no-op (its ``entry_id`` names the account and statement
   date, invariant 3); a different one is an error, never a second anchor.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, cast

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = "money_ledger.v1"
CURRENCY = "USD"
CENT = Decimal("0.01")

ACCOUNT_CASH = "assets:cash:USD"
ACCOUNT_ACCRUED = "assets:accrued_interest:USD"
ACCOUNT_SECURITIES_PREFIX = "assets:securities:"
ACCOUNT_BROKER_MARK = "assets:broker_mark_adjustment"
ACCOUNT_VALUATION_CORRECTION = "income:unrealized_gain:valuation_correction"
ACCOUNT_OPENING_BALANCE = "equity:opening_balance"

SOURCE_DAILY_AUTO_CORRECT = "daily_auto_correct"
SOURCE_OPENING_BALANCE = "opening_balance"

#: The closed set of correction kinds a ``daily_auto_correct`` entry carries.
KIND_IB_MARK_TO_SETTLED = "ib_mark_to_settled_close"
KIND_SETTLED_RESTRIKE = "settled_restrike"
#: The one kind an ``opening_balance`` entry carries.
KIND_STATEMENT_ANCHOR = "statement_anchor"

#: Every source and the kinds it may carry. A closed vocabulary on purpose: a
#: new event class (alpha-engine-config-I12557's catalogue) is a code change
#: that adds its row here.
KINDS_BY_SOURCE: Mapping[str, tuple[str, ...]] = {
    SOURCE_DAILY_AUTO_CORRECT: (KIND_IB_MARK_TO_SETTLED, KIND_SETTLED_RESTRIKE),
    SOURCE_OPENING_BALANCE: (KIND_STATEMENT_ANCHOR,),
}

#: The closed set of journal sources. A new source is a code change on purpose.
SOURCES: tuple[str, ...] = tuple(KINDS_BY_SOURCE)
KINDS: tuple[str, ...] = tuple(k for kinds in KINDS_BY_SOURCE.values() for k in kinds)

#: Broker statements the opening balance may anchor on, and the one broker read
#: that files them (crucible-trader ``broker_statement``). A book rebuilt from
#: fills or from this ledger is not a statement and cannot be an anchor.
ANCHOR_STATEMENT_SCHEMAS: tuple[str, ...] = ("broker_statement.v1", "broker_statement.v2")
ANCHOR_STATEMENT_SOURCES: tuple[str, ...] = ("ib_paper",)

_ACCOUNT_RE = re.compile(
    r"^(assets|liabilities|equity|income|expenses)(:[A-Za-z0-9_.\-/ ]+)+$"
)

#: S3 key for the immutable copy of one entry.
S3_KEY_TEMPLATE = "trades/ledger/journal/{effective_date}/{entry_id}.json"


class UnbalancedEntryError(ValueError):
    """A journal entry whose postings do not sum to zero."""


class LedgerIntegrityError(RuntimeError):
    """The stored ledger is not balanced, or an append-only rule was broken."""


def to_cents(value: Any) -> Decimal:
    """An exact cents amount. Floats go through ``repr`` so 0.1 stays 0.1."""
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, float):
        d = Decimal(repr(value))
    else:
        d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError(f"amount {value!r} is not finite")
    return d.quantize(CENT, rounding=ROUND_HALF_EVEN)


@dataclasses.dataclass(frozen=True)
class Posting:
    account: str
    amount: Decimal
    currency: str = CURRENCY

    def __post_init__(self) -> None:
        if not _ACCOUNT_RE.match(self.account):
            raise ValueError(f"account {self.account!r} is not a chart-of-accounts path")
        if not isinstance(self.amount, Decimal) or self.amount != self.amount.quantize(CENT):
            raise ValueError(f"posting amount {self.amount!r} is not an exact cents Decimal")
        if self.amount == 0:
            raise ValueError(f"zero posting on {self.account}: carry no empty lines")

    def to_document(self) -> dict[str, str]:
        return {"account": self.account, "amount": str(self.amount), "currency": self.currency}


@dataclasses.dataclass(frozen=True)
class JournalEntry:
    effective_date: str
    source: str
    kind: str
    reason: str
    postings: tuple[Posting, ...]
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reverses_entry_id: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"journal source {self.source!r} not in {SOURCES}")
        if self.kind not in KINDS_BY_SOURCE[self.source]:
            raise ValueError(
                f"journal kind {self.kind!r} not in {KINDS_BY_SOURCE[self.source]} "
                f"for source {self.source!r}"
            )
        if not self.reason or not self.reason.strip():
            raise ValueError("a journal entry must carry a reason")
        datetime.strptime(self.effective_date, "%Y-%m-%d")
        if len(self.postings) < 2:
            raise UnbalancedEntryError(
                f"entry for {self.effective_date} has {len(self.postings)} posting(s); "
                "a balanced entry needs at least two"
            )
        totals: dict[str, Decimal] = {}
        for p in self.postings:
            totals[p.currency] = totals.get(p.currency, Decimal("0")) + p.amount
        broken = {c: t for c, t in totals.items() if t != 0}
        if broken:
            raise UnbalancedEntryError(
                f"entry for {self.effective_date} ({self.kind}) does not balance: "
                + ", ".join(f"{c} {t:+}" for c, t in sorted(broken.items()))
            )

    @property
    def entry_id(self) -> str:
        """Content hash: the same correction is the same entry.

        An opening balance also hashes its anchor (account and statement
        date). The key is added for that source only, so every existing
        correction keeps the id it was stored under.
        """
        content: dict[str, object] = {
            "effective_date": self.effective_date,
            "source": self.source,
            "kind": self.kind,
            "reverses_entry_id": self.reverses_entry_id,
            "postings": sorted(
                (p.account, str(p.amount), p.currency) for p in self.postings
            ),
            # The before/after NAVs make two identical-delta corrections on
            # different bases two different entries.
            "basis": {
                k: self.evidence.get(k)
                for k in ("nav_before_usd", "nav_after_usd")
            },
        }
        if self.source == SOURCE_OPENING_BALANCE:
            content["anchor"] = {
                k: self.evidence.get(k) for k in ("account", "statement_date")
            }
        canonical = json.dumps(content, sort_keys=True)
        return "je_" + hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def total(self, account_prefix: str) -> Decimal:
        return sum(
            (p.amount for p in self.postings if p.account.startswith(account_prefix)),
            Decimal("0"),
        )

    def to_document(self, *, posted_at: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "entry_id": self.entry_id,
            "effective_date": self.effective_date,
            "posted_at": posted_at or datetime.now(UTC).isoformat(),
            "source": self.source,
            "kind": self.kind,
            "reason": self.reason,
            "reverses_entry_id": self.reverses_entry_id,
            "postings": [p.to_document() for p in self.postings],
            "evidence": dict(self.evidence),
        }


# ── Balance views ─────────────────────────────────────────────────────────────


def balance_view(
    *,
    nav: Any,
    cash: Any,
    accrued: Any,
    securities: Mapping[str, Any],
) -> dict[str, Decimal]:
    """The four-line statement of one published NAV, in exact cents.

    ``assets:broker_mark_adjustment`` is the plug that makes the lines sum to
    ``nav`` exactly. On the settled basis it is zero up to rounding; on the IB
    basis it is whatever IB NetLiquidation carried that is not cash, accrued
    interest or a position at the stated mark.
    """
    view: dict[str, Decimal] = {
        ACCOUNT_CASH: to_cents(cash),
        ACCOUNT_ACCRUED: to_cents(accrued or 0),
    }
    for ticker, mv in securities.items():
        if mv is None:
            raise ValueError(f"security {ticker!r} has no market value in the view")
        view[ACCOUNT_SECURITIES_PREFIX + ticker] = to_cents(mv)
    view[ACCOUNT_BROKER_MARK] = to_cents(nav) - sum(view.values(), Decimal("0"))
    return view


def view_from_eod_row(
    *,
    portfolio_nav: Any,
    total_cash: Any,
    accrued_interest: Any,
    positions_snapshot: Any,
    nav_basis: str | None,
) -> dict[str, Decimal]:
    """The balance view of a stored ``eod_pnl`` row.

    Securities are carried at the mark the headline NAV was struck on: the
    broker's (``ib_market_value``) on ``ib_netliq`` rows, the settled close
    (``market_value``) on ``settled_close`` rows. A NULL ``nav_basis`` is a
    pre-flag row, which means IB.
    """
    if portfolio_nav is None or total_cash is None:
        raise ValueError("an eod_pnl row without portfolio_nav or total_cash has no view")
    positions = (
        json.loads(positions_snapshot)
        if isinstance(positions_snapshot, str)
        else (positions_snapshot or {})
    )
    use_ib = (nav_basis or "ib_netliq") == "ib_netliq"
    securities: dict[str, Any] = {}
    for ticker, pos in positions.items():
        mv = pos.get("ib_market_value") if use_ib else None
        if mv is None:
            mv = pos.get("market_value")
        if mv is None:
            continue
        securities[ticker] = mv
    return balance_view(
        nav=portfolio_nav, cash=total_cash, accrued=accrued_interest, securities=securities
    )


def correction_entry(
    *,
    effective_date: str,
    kind: str,
    reason: str,
    before: Mapping[str, Decimal],
    after: Mapping[str, Decimal],
    evidence: Mapping[str, Any] | None = None,
    offset_account: str = ACCOUNT_VALUATION_CORRECTION,
) -> JournalEntry | None:
    """One balanced entry moving the books from ``before`` to ``after``.

    Returns ``None`` when nothing changed. The offset posting is the negated
    sum of the asset postings, so the entry balances by construction and the
    asset postings sum to exactly ``nav(after) − nav(before)``.
    """
    postings: list[Posting] = []
    for account in sorted(set(before) | set(after)):
        delta = after.get(account, Decimal("0")) - before.get(account, Decimal("0"))
        if delta != 0:
            postings.append(Posting(account, delta))
    if not postings:
        return None
    net = sum((p.amount for p in postings), Decimal("0"))
    if net != 0:
        postings.append(Posting(offset_account, -net))
    nav_before = sum(before.values(), Decimal("0"))
    nav_after = sum(after.values(), Decimal("0"))
    ev = {
        "nav_before_usd": str(nav_before),
        "nav_after_usd": str(nav_after),
        "nav_change_usd": str(nav_after - nav_before),
        **(dict(evidence) if evidence else {}),
    }
    return JournalEntry(
        effective_date=effective_date,
        source=SOURCE_DAILY_AUTO_CORRECT,
        kind=kind,
        reason=reason,
        postings=tuple(postings),
        evidence=ev,
    )


def opening_balance_entry(
    statement: Mapping[str, object],
    *,
    settled_closes: Mapping[str, float | int | str | Decimal | None],
    closes_date: str,
    close_source: str,
    reason: str | None = None,
) -> JournalEntry:
    """The ledger's opening balance, anchored on ONE broker statement.

    ``statement`` is a stored crucible-trader broker statement document
    (``broker_statement.v1``/``v2``: ``account``, ``trading_day``, ``source``,
    ``positions`` as whole non-zero shares, ``cash_usd``, ``accrued_cash_usd``).
    Ground truth (alpha-engine-config-I9638 / I10052): IB for shares and cash,
    Polygon settled data for prices. So each position is valued at
    ``shares × settled_closes[ticker]``, and ``closes_date`` must be the
    statement's own trading day.

    Refused rather than guessed (no balancing plugs, policy Rule 8):

    * a document that is not a broker read (``source`` outside
      :data:`ANCHOR_STATEMENT_SOURCES`) or not a known statement schema;
    * a statement without a recorded accrual (v1, or v2 with ``None``): an
      opening book that silently sets accrued interest to zero misstates NAV;
    * a held position with no finite, positive settled close;
    * closes from another day than the statement's.

    The share counts and closes go in ``evidence`` (a posting carries no
    quantity in the aec#12510 shape). The entry balances by construction: the
    offset to ``equity:opening_balance`` is the negated sum of the asset lines.
    """
    version = statement.get("schema_version")
    if version not in ANCHOR_STATEMENT_SCHEMAS:
        raise ValueError(
            f"opening balance needs a broker statement {ANCHOR_STATEMENT_SCHEMAS}, "
            f"got schema_version {version!r}"
        )
    source = statement.get("source")
    if source not in ANCHOR_STATEMENT_SOURCES:
        raise ValueError(
            f"statement source {source!r} is not a broker read {ANCHOR_STATEMENT_SOURCES}; "
            "a book rebuilt from fills or from the ledger cannot anchor the ledger"
        )
    account = statement.get("account")
    statement_date = statement.get("trading_day")
    if not isinstance(account, str) or not account:
        raise ValueError("statement carries no account")
    if not isinstance(statement_date, str):
        raise ValueError("statement carries no trading_day")
    datetime.strptime(statement_date, "%Y-%m-%d")
    if closes_date != statement_date:
        raise ValueError(
            f"settled closes are for {closes_date}, the statement is for {statement_date}; "
            "the anchor values the statement's own day"
        )
    if not close_source or not close_source.strip():
        raise ValueError("close_source must name where the settled closes came from")
    accrued = statement.get("accrued_cash_usd")
    if not isinstance(accrued, (int, float, str, Decimal)):
        raise ValueError(
            f"{version} statement for {account} {statement_date} has no recorded accrual "
            "(accrued_cash_usd); anchor on a broker_statement.v2 that records one"
        )
    cash = statement.get("cash_usd")
    if not isinstance(cash, (int, float, str, Decimal)):
        raise ValueError("statement carries no cash_usd")

    raw_positions: object = statement.get("positions")
    if raw_positions is None:
        raw_positions = {}
    if not isinstance(raw_positions, Mapping):
        raise ValueError("statement positions are not a ticker -> shares mapping")
    positions = cast("Mapping[object, object]", raw_positions)
    shares_by_ticker: dict[str, int] = {}
    for ticker, shares in positions.items():
        if (
            not isinstance(ticker, str)
            or not ticker
            or not isinstance(shares, int)
            or isinstance(shares, bool)
            or shares == 0
        ):
            raise ValueError(f"position {ticker!r}={shares!r} is not whole non-zero shares")
        shares_by_ticker[ticker] = shares
    missing = sorted(t for t in shares_by_ticker if settled_closes.get(t) is None)
    if missing:
        raise ValueError(
            f"no settled close on {closes_date} for held position(s): {', '.join(missing)}"
        )

    closes: dict[str, str] = {}
    postings: list[Posting] = []
    for ticker in sorted(shares_by_ticker):
        raw = settled_closes[ticker]
        if raw is None:  # pragma: no cover - refused above as missing
            raise ValueError(f"no settled close for {ticker}")
        close = Decimal(repr(raw)) if isinstance(raw, float) else Decimal(str(raw))
        if not close.is_finite() or close <= 0:
            raise ValueError(f"settled close for {ticker} is {raw!r}, not a positive price")
        closes[ticker] = str(close)
        mv = to_cents(close * shares_by_ticker[ticker])
        if mv != 0:
            postings.append(Posting(ACCOUNT_SECURITIES_PREFIX + ticker, mv))
    for account_name, value in ((ACCOUNT_CASH, cash), (ACCOUNT_ACCRUED, accrued)):
        amount = to_cents(value)
        if amount != 0:
            postings.append(Posting(account_name, amount))
    net = sum((p.amount for p in postings), Decimal("0"))
    if net == 0:
        raise ValueError(
            f"statement for {account} {statement_date} nets to zero; "
            "there is nothing to anchor"
        )
    postings.append(Posting(ACCOUNT_OPENING_BALANCE, -net))
    return JournalEntry(
        effective_date=statement_date,
        source=SOURCE_OPENING_BALANCE,
        kind=KIND_STATEMENT_ANCHOR,
        reason=reason
        or f"opening balance anchored on the {statement_date} {source} statement for {account}",
        postings=tuple(postings),
        evidence={
            "account": account,
            "statement_date": statement_date,
            "statement_schema_version": version,
            "statement_source": source,
            "shares": dict(sorted(shares_by_ticker.items())),
            "settled_closes": closes,
            "closes_date": closes_date,
            "close_source": close_source,
            "nav_before_usd": "0.00",
            "nav_after_usd": str(net),
            "nav_change_usd": str(net),
        },
    )


# ── Storage: trades.db (append-only) + one immutable S3 object per entry ─────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_journal_entries (
    entry_id          TEXT PRIMARY KEY,
    effective_date    TEXT NOT NULL,
    posted_at         TEXT NOT NULL,
    source            TEXT NOT NULL,
    kind              TEXT NOT NULL,
    reason            TEXT NOT NULL,
    reverses_entry_id TEXT,
    evidence_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_postings (
    entry_id  TEXT NOT NULL REFERENCES ledger_journal_entries(entry_id),
    line_no   INTEGER NOT NULL,
    account   TEXT NOT NULL,
    amount    TEXT NOT NULL,
    currency  TEXT NOT NULL,
    PRIMARY KEY (entry_id, line_no)
);
CREATE INDEX IF NOT EXISTS ledger_entries_by_date
    ON ledger_journal_entries(effective_date);
CREATE TRIGGER IF NOT EXISTS ledger_entries_no_update
    BEFORE UPDATE ON ledger_journal_entries
    BEGIN SELECT RAISE(ABORT, 'ledger_journal_entries is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_entries_no_delete
    BEFORE DELETE ON ledger_journal_entries
    BEGIN SELECT RAISE(ABORT, 'ledger_journal_entries is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_postings_no_update
    BEFORE UPDATE ON ledger_postings
    BEGIN SELECT RAISE(ABORT, 'ledger_postings is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_postings_no_delete
    BEFORE DELETE ON ledger_postings
    BEGIN SELECT RAISE(ABORT, 'ledger_postings is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_postings_sealed_once_posted
    BEFORE INSERT ON ledger_postings
    WHEN EXISTS (SELECT 1 FROM ledger_journal_entries WHERE entry_id = NEW.entry_id)
    BEGIN SELECT RAISE(ABORT, 'ledger entry is already posted; its postings are sealed'); END;
CREATE TRIGGER IF NOT EXISTS ledger_entries_balanced_at_post
    BEFORE INSERT ON ledger_journal_entries
    WHEN (SELECT COUNT(*) FROM ledger_postings WHERE entry_id = NEW.entry_id) < 2
      OR EXISTS (
        SELECT 1 FROM ledger_postings WHERE entry_id = NEW.entry_id
        GROUP BY currency
        HAVING SUM(CAST(ROUND(CAST(amount AS REAL) * 100) AS INTEGER)) != 0
      )
    BEGIN SELECT RAISE(ABORT, 'ledger entry does not balance: postings must sum to zero per currency'); END;
CREATE TRIGGER IF NOT EXISTS ledger_single_opening_balance
    BEFORE INSERT ON ledger_journal_entries
    WHEN NEW.source = 'opening_balance'
     AND EXISTS (SELECT 1 FROM ledger_journal_entries WHERE source = 'opening_balance')
    BEGIN SELECT RAISE(ABORT, 'ledger already has an opening balance; a ledger has one anchor'); END;
"""

#: The triggers that make the store, not the caller, refuse an unbalanced or
#: partial entry. Named so a test can assert each one is installed.
STORE_GUARD_TRIGGERS: tuple[str, ...] = (
    "ledger_postings_sealed_once_posted",
    "ledger_entries_balanced_at_post",
)

#: The trigger that keeps the ledger on ONE anchor (invariant 7).
OPENING_BALANCE_GUARD_TRIGGER = "ledger_single_opening_balance"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def record_entry(conn: sqlite3.Connection, entry: JournalEntry) -> tuple[dict, bool]:
    """Append ``entry`` to ``trades.db``. Returns ``(document, newly_written)``.

    Idempotent on ``entry_id``. The entry and its postings commit together or
    not at all. Postings go in first and the header last, because the header
    is what the store checks for balance (invariant 5). The stored ledger's
    trial balance and every entry's own balance are re-checked inside the same
    transaction, so an unbalanced ledger can never be committed.
    """
    ensure_schema(conn)
    doc = entry.to_document()
    existing = conn.execute(
        "SELECT posted_at FROM ledger_journal_entries WHERE entry_id = ?",
        (doc["entry_id"],),
    ).fetchone()
    if existing:
        doc["posted_at"] = existing[0]
        return doc, False
    try:
        conn.execute("SAVEPOINT ledger_append")
        conn.executemany(
            "INSERT INTO ledger_postings (entry_id, line_no, account, amount, currency) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (doc["entry_id"], i, p.account, str(p.amount), p.currency)
                for i, p in enumerate(entry.postings)
            ],
        )
        # The header LAST: its insert is the posting boundary the store checks.
        conn.execute(
            "INSERT INTO ledger_journal_entries (entry_id, effective_date, posted_at, "
            "source, kind, reason, reverses_entry_id, evidence_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc["entry_id"], doc["effective_date"], doc["posted_at"], doc["source"],
                doc["kind"], doc["reason"], doc["reverses_entry_id"],
                json.dumps(doc["evidence"], sort_keys=True, default=str),
            ),
        )
        trial_balance(conn)
        verify_entries(conn)
        conn.execute("RELEASE SAVEPOINT ledger_append")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT ledger_append")
        conn.execute("RELEASE SAVEPOINT ledger_append")
        raise
    conn.commit()
    return doc, True


def trial_balance(conn: sqlite3.Connection) -> dict[str, Decimal]:
    """Σ every posting per currency. Raises unless every total is exactly zero.

    Summed as Decimal from the stored text, never as SQLite REAL. Never runs
    DDL itself: it is called inside :func:`record_entry`'s savepoint, and
    ``executescript`` would commit that savepoint away.
    """
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_postings'"
    ).fetchone():
        return {}
    totals: dict[str, Decimal] = {}
    for amount, currency in conn.execute("SELECT amount, currency FROM ledger_postings"):
        totals[currency] = totals.get(currency, Decimal("0")) + Decimal(amount)
    broken = {c: t for c, t in totals.items() if t != 0}
    if broken:
        raise LedgerIntegrityError(
            "money ledger trial balance is not zero: "
            + ", ".join(f"{c} {t:+}" for c, t in sorted(broken.items()))
        )
    return totals


def verify_entries(conn: sqlite3.Connection) -> int:
    """Check every stored entry on its own. Returns the number of entries.

    Raises :class:`LedgerIntegrityError` naming every offender when:

    * an entry's postings do not sum to exactly zero in some currency;
    * an entry has fewer than two postings (a header with none is a partial
      entry);
    * a posting names an entry with no header (an orphan line, which only a
      write bypassing :func:`record_entry` can produce).

    The trial balance alone cannot see two broken entries that offset each
    other; this can. Summed as Decimal from the stored text, never as REAL.
    Like :func:`trial_balance`, it never runs DDL, so it is safe inside
    :func:`record_entry`'s savepoint.
    """
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_postings'"
    ).fetchone():
        return 0
    headers = {row[0] for row in conn.execute("SELECT entry_id FROM ledger_journal_entries")}
    lines: dict[str, int] = {}
    sums: dict[tuple[str, str], Decimal] = {}
    for entry_id, amount, currency in conn.execute(
        "SELECT entry_id, amount, currency FROM ledger_postings"
    ):
        lines[entry_id] = lines.get(entry_id, 0) + 1
        sums[(entry_id, currency)] = sums.get((entry_id, currency), Decimal("0")) + Decimal(amount)
    offenders: list[str] = []
    for entry_id in sorted(set(lines) - headers):
        offenders.append(f"{entry_id}: {lines[entry_id]} posting(s) with no entry header")
    for entry_id in sorted(headers):
        if lines.get(entry_id, 0) < 2:
            offenders.append(f"{entry_id}: {lines.get(entry_id, 0)} posting(s), needs two or more")
    for (entry_id, currency), total in sorted(sums.items()):
        if total != 0 and entry_id in headers:
            offenders.append(f"{entry_id}: {currency} {total:+} does not balance")
    if offenders:
        raise LedgerIntegrityError(
            f"money ledger has {len(offenders)} broken entr(y/ies): " + "; ".join(offenders)
        )
    return len(headers)


def entries_for_date(conn: sqlite3.Connection, effective_date: str) -> list[dict]:
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT entry_id, kind, reason, posted_at FROM ledger_journal_entries "
        "WHERE effective_date = ? ORDER BY posted_at",
        (effective_date,),
    ).fetchall()
    return [
        {"entry_id": r[0], "kind": r[1], "reason": r[2], "posted_at": r[3]} for r in rows
    ]


def publish_entry(s3_client: Any, bucket: str, doc: Mapping[str, Any]) -> str:
    """Write the immutable S3 copy. An existing object is the same entry."""
    key = S3_KEY_TEMPLATE.format(
        effective_date=doc["effective_date"], entry_id=doc["entry_id"]
    )
    body = json.dumps(doc, indent=2, sort_keys=True, default=str).encode()
    try:
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=body,
            ContentType="application/json", IfNoneMatch="*",
        )
    except Exception as exc:  # noqa: BLE001 — classified below, re-raised otherwise
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
            return key
        raise
    return key


def post(
    conn: sqlite3.Connection,
    entry: JournalEntry | None,
    *,
    s3_client: Any = None,
    bucket: str | None = None,
) -> dict | None:
    """Record ``entry`` in ``trades.db`` and publish its S3 copy.

    The database write is the primary deliverable and raises on failure. The S3
    copy is retried by the next run (``entry_id`` is stable), so its failure is
    logged at ERROR and returned on the document, never swallowed silently.
    """
    if entry is None:
        return None
    doc, new = record_entry(conn, entry)
    doc["newly_written"] = new
    if s3_client is not None and bucket:
        try:
            doc["s3_key"] = publish_entry(s3_client, bucket, doc)
        except Exception as exc:  # noqa: BLE001 — recorded on the doc and logged
            logger.error("money ledger: S3 copy of %s failed: %s", doc["entry_id"], exc)
            doc["s3_error"] = f"{exc.__class__.__name__}: {exc}"
    logger.info(
        "money ledger: %s %s %s for %s, %d posting(s), NAV %s -> %s",
        "posted" if new else "already had",
        doc["entry_id"], doc["kind"], doc["effective_date"], len(doc["postings"]),
        doc["evidence"].get("nav_before_usd"), doc["evidence"].get("nav_after_usd"),
    )
    return doc


def describe(entry: JournalEntry, *, top: int = 5) -> str:
    """One line for a report: the NAV move and the largest lines."""
    lines = sorted(
        (p for p in entry.postings if p.account.startswith("assets:")),
        key=lambda p: -abs(p.amount),
    )[:top]
    return (
        f"{entry.kind} {entry.effective_date}: NAV "
        f"{entry.evidence.get('nav_before_usd')} -> {entry.evidence.get('nav_after_usd')} "
        f"({entry.evidence.get('nav_change_usd')}); "
        + ", ".join(f"{p.account.rsplit(':', 1)[-1]} {p.amount:+}" for p in lines)
    )


def iter_postings(entries: Iterable[JournalEntry]) -> Iterable[Posting]:
    for e in entries:
        yield from e.postings
