"""The v2 trader's fills, in the row shape the EOD attribution prices
(alpha-engine-config-I13227).

**Why this exists.** Paper orders moved from this executor to the v2 trader
on 2026-11-03, and the v2 trader's first fills landed on 2026-11-04. The EOD
attribution in :mod:`executor.eod_reconcile` reads today's fills from v1's own
``trades`` table only, so it saw none of them: ``rotation_realized_usd`` read
$0 while eleven names were sold, an add to a held name was valued from the
prior close instead of its fill, and the whole of the day's trading P&L landed
in ``unattributed_true_usd``. NAV was right; the residual bounds gate failed on
P&L that was earned and simply had no sleeve.

**Where the fills come from.** The v2 ledger journal, the book of record:
``{store}/trader/ledger/journal/{tenant}/{day}.json``, written create-only by
``trader.reconcile`` from IB's executions. Each ``fill`` entry posts the
consideration to ``assets:securities:<TICKER>`` with the signed share count,
and a ``commission`` entry under the same ``event_ref`` carries its commission.
The journal is read under the trader's own role, through the same env file and
profile :mod:`executor.v2_trade_alert` uses, so this adds no grant.

**What it does not do.** It never writes the v2 store, never touches an order,
and never raises into the reconcile: a journal it cannot read is returned as a
``data_warnings`` line naming why, and the residual bounds gate then judges the
day exactly as it did before this module existed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from executor.v2_trade_alert import Store

logger = logging.getLogger(__name__)

#: The first session the v2 trader could have filled in. A reconcile of an
#: earlier date never reads the v2 store.
V2_FIRST_SESSION = "2026-11-03"
#: The fund whose book this executor reconciles (crucible-trader
#: ``exception_record.PAPER_FUND_TENANT``).
LEDGER_TENANT = "nous-ergon-paper"
JOURNAL_SCHEMA_VERSION = "ledger_journal.v1"
_SECURITIES = "assets:securities:"
_COMMISSIONS = "expenses:commissions"
#: ``crucible_trader.v1_fills_export`` addresses a v1 fill as ``v1:<trade_id>``.
#: Such a fill is already a row in v1's ``trades`` table.
_V1_REF_PREFIX = "v1:"
SOURCE = "v2_ledger_journal"

Row = dict[str, object]


class V2FillsError(ValueError):
    """A journal this module refuses to read as fills."""


def journal_key(day: str, tenant: str = LEDGER_TENANT) -> str:
    return f"trader/ledger/journal/{tenant}/{day}.json"


def _entries(document: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = document.get("entries")
    if not isinstance(raw, list):
        raise V2FillsError("journal has no entries list")
    return [cast(Mapping[str, object], e) for e in cast(list[object], raw) if isinstance(e, dict)]


def _postings(entry: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = entry.get("postings")
    if not isinstance(raw, list):
        return []
    return [cast(Mapping[str, object], p) for p in cast(list[object], raw) if isinstance(p, dict)]


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2FillsError(f"{what} is not an integer: {value!r}")
    return value


def fills_from_journal(document: Mapping[str, object], run_date: str) -> list[Row]:
    """Every fill in one day's v2 journal, as attribution rows.

    A row carries ``action`` ``BUY``/``SELL``, unsigned ``shares`` and
    ``filled_shares``, the gross ``fill_price`` (consideration / shares, before
    commission), ``commission_usd`` (``None`` when the ledger did not separate
    it, ``0.0`` when it did and IB charged nothing) and ``status`` ``Filled``.
    ``price_at_order`` is ``None``: the journal holds no arrival price, so
    these fills count toward notional and commission but not toward slippage.
    """
    if document.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise V2FillsError(f"journal schema is {document.get('schema_version')!r}")
    if document.get("trading_day") != run_date:
        raise V2FillsError(
            f"journal trading_day {document.get('trading_day')!r} is not {run_date}"
        )
    entries = _entries(document)

    commission_cents: dict[str, int] = {}
    for entry in entries:
        if entry.get("kind") != "commission":
            continue
        ref = str(entry.get("event_ref") or "")
        for posting in _postings(entry):
            if posting.get("account") == _COMMISSIONS:
                cents = _int(posting.get("amount_cents"), f"commission {ref}")
                commission_cents[ref] = commission_cents.get(ref, 0) + cents

    rows: list[Row] = []
    for entry in entries:
        if entry.get("kind") != "fill":
            continue
        ref = str(entry.get("event_ref") or "")
        if ref.startswith(_V1_REF_PREFIX):
            continue
        securities = [
            p for p in _postings(entry) if str(p.get("account", "")).startswith(_SECURITIES)
        ]
        if len(securities) != 1:
            raise V2FillsError(f"fill {ref} has {len(securities)} securities postings, not 1")
        posting = securities[0]
        ticker = str(posting["account"])[len(_SECURITIES) :]
        quantity = _int(posting.get("quantity"), f"fill {ref} quantity")
        amount_cents = _int(posting.get("amount_cents"), f"fill {ref} amount")
        if quantity == 0 or amount_cents == 0 or (quantity > 0) != (amount_cents > 0):
            raise V2FillsError(
                f"fill {ref} ({ticker}) is not a trade: {quantity} shares for {amount_cents} cents"
            )
        shares = abs(quantity)
        price = abs(amount_cents) / 100.0 / shares
        evidence = entry.get("evidence")
        separated = isinstance(evidence, dict) and bool(
            cast(Mapping[str, object], evidence).get("commission_separated")
        )
        commission = commission_cents.get(ref)
        commission_usd = (
            commission / 100.0 if commission is not None else (0.0 if separated else None)
        )
        rows.append(
            {
                "ticker": ticker,
                "action": "BUY" if quantity > 0 else "SELL",
                "shares": shares,
                "filled_shares": shares,
                "fill_price": price,
                "price": price,
                "price_at_order": None,
                "commission_usd": commission_usd,
                "status": "Filled",
                "source": SOURCE,
                "exec_ref": ref,
            }
        )
    return rows


def merge_for_attribution(
    v1_rows: Sequence[Row], v2_rows: Sequence[Row]
) -> tuple[list[Row], list[Row], list[str]]:
    """``(merged, v2_kept, overlap)``.

    v1's rows are kept exactly as they are. A v2 fill in a ticker v1 ALSO
    recorded a trade for today is dropped and its ticker returned in
    ``overlap``: the v2 reconcile reads the whole account, so on such a day its
    journal can carry v1's own executions, and v1 stores no execution id to
    match them by. Counting both would double the sleeve; dropping v1's would
    change what every pre-v2 session already reports.
    """
    v1_tickers = {str(r.get("ticker")) for r in v1_rows if r.get("ticker")}
    kept = [r for r in v2_rows if str(r.get("ticker")) not in v1_tickers]
    overlap = sorted({str(r.get("ticker")) for r in v2_rows} & v1_tickers)
    return [*v1_rows, *kept], kept, overlap


def _default_store() -> Store:
    from executor.v2_trade_alert import TRADER_ENV, open_store, read_env_file

    return open_store(read_env_file(TRADER_ENV))


def load_v2_fills(
    run_date: str,
    *,
    env_path: Path | None = None,
    store_factory: Callable[[], Store] | None = None,
) -> tuple[list[Row], str | None]:
    """``(rows, warning)`` for ``run_date``. Never raises.

    ``warning`` is a ``data_warnings`` line whenever the day's v2 fills could
    not be read; any v2 trading P&L that day then remains in
    ``unattributed_true_usd``, and the line says so.
    """
    if run_date < V2_FIRST_SESSION:
        return [], None
    if env_path is None:
        from executor.v2_trade_alert import TRADER_ENV

        env_path = TRADER_ENV
    suffix = (
        f"Any v2 trading P&L on {run_date} is inside unattributed_true_usd, "
        "not in the rotation or position sleeves (alpha-engine-config-I13227)."
    )
    if store_factory is None:
        if not env_path.is_file():
            return [], f"v2 fills not read for {run_date}: trader env {env_path} is absent. {suffix}"
        store_factory = _default_store
    key = journal_key(run_date)
    try:
        store = store_factory()
        document = store.read_json(key)
        if document is None:
            return [], (
                f"v2 fills not read for {run_date}: no ledger journal at {key} "
                f"(trader.reconcile writes it after the close). {suffix}"
            )
        rows = fills_from_journal(document, run_date)
    except Exception as exc:  # reported as a data_warnings line, never raised into the reconcile
        logger.warning("v2 fills unreadable for %s: %s", run_date, exc)
        return [], f"v2 fills not read for {run_date}: {type(exc).__name__}: {exc}. {suffix}"
    return rows, None
