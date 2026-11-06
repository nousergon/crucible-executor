"""The v2 trader's fills, as trade rows for the v1 EOD attribution (alpha-engine-config-I13228).

**Why this exists.** Since 2026-11-03 the v2 trader places this account's
orders, and it writes its fills to its own store, not to the executor's
``trades`` table. The EOD P&L attribution prices the rotation and new-entry
sleeves from that table, so on 2026-11-04 (11 sells, 2 buys, all v2) it saw no
trades: every sold name was priced at its prior close, the realized P&L of the
whole rotation fell into the unattributed residual (+$70,995, 6.7% of NAV) and
the integrity gate failed the post-close reconcile pipeline.

**Where the fills come from: the v2 ledger journal.** One document per tenant
per calendar day, ``trader/ledger/journal/{tenant}/{run_date}.json``, written
by ``trader.reconcile`` from IB's own executions (``kind: fill``) and
commission reports (``kind: commission``, joined to its fill by
``event_ref``). It is the trader's book of record and it is keyed by the day
the fills happened, so no lookup by another day's name is needed.

The first version of this module (crucible-executor-PR649) read
``trader/execution_shortfall/`` instead. The trader role this box reads the
store as holds ``s3:PutObject`` there but neither ``GetObject`` nor
``ListBucket``, so every read was AccessDenied and the 2026-11-05 run added
no rows (+$33,154 unattributed, the same failure). The role's ledger grants
(``ReadItsOwnLedger`` / ``ListItsOwnLedger``) already cover the journal, and
the role's inline policies sit within 15 bytes of IAM's 10,240-byte aggregate
ceiling, so reading what is already granted is the change that can land.

**What it returns.** One row per ticker and side, in the shape
:func:`executor.trade_logger.get_todays_trades` returns (``ticker``,
``action``, ``shares``, ``fill_price`` as the share-weighted fill, and
``commission_usd`` when every fill in the row has its commission report). The
journal holds no arrival price, so ``price_at_order`` is absent and the
row's slippage is reported as not measured, never as zero.

**What it never does.** It never invents a row: a missing env file, an
absent or unreadable journal, or a journal for another day gives no rows,
and the integrity gate then fails exactly as before. A ticker the executor
itself logged a trade for today is skipped, so a fill is never counted twice.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from executor.v2_trade_alert import TRADER_ENV, Store, open_store, read_env_file

logger = logging.getLogger(__name__)

#: The paper fund's ledger tenant (crucible-trader
#: ``exception_record.PAPER_FUND_TENANT``).
LEDGER_TENANT = "nous-ergon-paper"
JOURNAL_SCHEMA_VERSION = "ledger_journal.v1"
_SECURITY_ACCOUNT = "assets:securities:"
_SOURCE = "crucible-trader-ledger"

Row = dict[str, object]


def journal_key(run_date: str, tenant: str = LEDGER_TENANT) -> str:
    """The journal's key relative to the store (crucible-trader ``ledger.journal_key``)."""
    return f"trader/ledger/journal/{tenant}/{run_date}.json"


def _docs(doc: Mapping[str, object] | None, key: str) -> list[dict[str, object]]:
    """A list of JSON objects under ``key``; anything else reads as empty."""
    value = None if doc is None else doc.get(key)
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in cast(list[object], value) if isinstance(item, dict)]


def _cents(posting: Mapping[str, object]) -> int | None:
    value = posting.get("amount_cents")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _security_leg(entry: Mapping[str, object]) -> tuple[str, float, int] | None:
    """``(ticker, signed quantity, signed cents)`` of a fill's securities posting."""
    for posting in _docs(entry, "postings"):
        account = str(posting.get("account", ""))
        quantity = posting.get("quantity")
        cents = _cents(posting)
        if not account.startswith(_SECURITY_ACCOUNT) or cents is None:
            continue
        if not isinstance(quantity, (int, float)) or isinstance(quantity, bool) or quantity == 0:
            continue
        return account[len(_SECURITY_ACCOUNT) :], float(quantity), cents
    return None


def _commission_cents(entry: Mapping[str, object]) -> int | None:
    for posting in _docs(entry, "postings"):
        cents = _cents(posting)
        if posting.get("account") == "expenses:commissions" and cents is not None:
            return cents
    return None


def rows_from_journal(doc: Mapping[str, object], run_date: str) -> list[Row]:
    """The fills of one day's ledger journal, one row per ticker and side."""
    if doc.get("schema_version") != JOURNAL_SCHEMA_VERSION or doc.get("trading_day") != run_date:
        logger.warning(
            "v2 ledger journal for %s not read as fills: schema %r, trading_day %r",
            run_date,
            doc.get("schema_version"),
            doc.get("trading_day"),
        )
        return []
    entries = _docs(doc, "entries")
    commissions: dict[str, int] = {}
    for entry in entries:
        cents = _commission_cents(entry) if entry.get("kind") == "commission" else None
        if cents is not None:
            ref = str(entry.get("event_ref"))
            commissions[ref] = commissions.get(ref, 0) + cents

    # (ticker, side) -> [shares, cents, commission cents, every fill has a commission report]
    groups: dict[tuple[str, str], list[float]] = {}
    for entry in entries:
        if entry.get("kind") != "fill":
            continue
        leg = _security_leg(entry)
        if leg is None:
            continue
        ticker, quantity, cents = leg
        side = "BUY" if quantity > 0 else "SELL"
        group = groups.setdefault((ticker, side), [0.0, 0.0, 0.0, 1.0])
        group[0] += abs(quantity)
        group[1] += abs(cents)
        commission = commissions.get(str(entry.get("event_ref")))
        if commission is None:
            group[3] = 0.0
        else:
            group[2] += commission

    rows: list[Row] = []
    for (ticker, side), (shares, cents, commission, complete) in groups.items():
        row: Row = {
            "ticker": ticker,
            "action": side,
            "shares": shares,
            "fill_price": cents / 100.0 / shares,
            "status": "Filled",
            "source": _SOURCE,
        }
        if complete:
            row["commission_usd"] = commission / 100.0
        rows.append(row)
    return rows


def v2_trade_rows(
    run_date: str,
    existing: Sequence[Mapping[str, object]] = (),
    *,
    store: Store | None = None,
    env_file: Path = TRADER_ENV,
) -> list[Row]:
    """The v2 fills on ``run_date`` that ``existing`` does not already hold."""
    key = journal_key(run_date)
    try:
        if store is None:
            if not env_file.exists():
                return []
            store = open_store(read_env_file(env_file))
        doc = store.read_json(key)
    except Exception as exc:  # noqa: BLE001 - no rows; the integrity gate still decides
        logger.warning("v2 fills for %s unreadable: %s: %s", run_date, type(exc).__name__, exc)
        return []
    if doc is None:
        logger.warning("v2 fills for %s: no ledger journal at %s (absent or not readable)", run_date, key)
        return []
    logged = {str(t.get("ticker")) for t in existing}
    rows = rows_from_journal(doc, run_date)
    skipped = sorted({str(r["ticker"]) for r in rows if r["ticker"] in logged})
    if skipped:
        logger.warning("v2 fills for %s already in the trades table, not added: %s", run_date, skipped)
    kept = [r for r in rows if r["ticker"] not in logged]
    logger.info("v2 fills for %s: %d row(s) added to the attribution from %s", run_date, len(kept), key)
    return kept
