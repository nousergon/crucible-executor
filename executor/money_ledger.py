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
from typing import Any

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = "money_ledger.v1"
CURRENCY = "USD"
CENT = Decimal("0.01")

ACCOUNT_CASH = "assets:cash:USD"
ACCOUNT_ACCRUED = "assets:accrued_interest:USD"
ACCOUNT_SECURITIES_PREFIX = "assets:securities:"
ACCOUNT_BROKER_MARK = "assets:broker_mark_adjustment"
ACCOUNT_VALUATION_CORRECTION = "income:unrealized_gain:valuation_correction"

#: The closed set of journal sources. A new source is a code change on purpose.
SOURCES: tuple[str, ...] = ("daily_auto_correct",)

#: The closed set of correction kinds a ``daily_auto_correct`` entry carries.
KIND_IB_MARK_TO_SETTLED = "ib_mark_to_settled_close"
KIND_SETTLED_RESTRIKE = "settled_restrike"
KINDS: tuple[str, ...] = (KIND_IB_MARK_TO_SETTLED, KIND_SETTLED_RESTRIKE)

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
        if self.kind not in KINDS:
            raise ValueError(f"journal kind {self.kind!r} not in {KINDS}")
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
        """Content hash: the same correction is the same entry."""
        canonical = json.dumps(
            {
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
            },
            sort_keys=True,
        )
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
        source="daily_auto_correct",
        kind=kind,
        reason=reason,
        postings=tuple(postings),
        evidence=ev,
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
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def record_entry(conn: sqlite3.Connection, entry: JournalEntry) -> tuple[dict, bool]:
    """Append ``entry`` to ``trades.db``. Returns ``(document, newly_written)``.

    Idempotent on ``entry_id``. The entry and its postings commit together or
    not at all, and the stored ledger's trial balance is re-checked inside the
    same transaction, so an unbalanced ledger can never be committed.
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
        conn.executemany(
            "INSERT INTO ledger_postings (entry_id, line_no, account, amount, currency) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (doc["entry_id"], i, p.account, str(p.amount), p.currency)
                for i, p in enumerate(entry.postings)
            ],
        )
        trial_balance(conn)
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
