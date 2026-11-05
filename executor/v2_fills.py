"""The v2 trader's fills, as trade rows for the v1 EOD attribution (alpha-engine-config-I13228).

**Why this exists.** Since 2026-11-03 the v2 trader places this account's
orders, and it writes its fills to its own store, not to the executor's
``trades`` table. The EOD P&L attribution prices the rotation and new-entry
sleeves from that table, so on 2026-11-04 (11 sells, 2 buys, all v2) it saw no
trades: every sold name was priced at its prior close, the realized P&L of the
whole rotation fell into the unattributed residual (+$70,995, 6.7% of NAV) and
the integrity gate failed the post-close reconcile pipeline.

**What it does.** Returns the filled orders of the v2 session that traded on
``run_date`` in the row shape :func:`executor.trade_logger.get_todays_trades`
returns (``ticker``, ``action``, ``shares``, ``fill_price``,
``price_at_order``), read from ``trader/execution_shortfall/{book day}.json``.
The session trades the book of the previous trading day, so the file is chosen
by its ``calendar_date``, not by its name.

**What it never does.** It never invents a row: a missing file, an unreadable
store or a box with no v2 env file gives no rows, and the gate then fails
exactly as before. A ticker the executor itself logged a trade for today is
skipped, so a fill is never counted twice.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from executor.v2_trade_alert import TRADER_ENV, Store, _docs, _num, open_store, read_env_file

logger = logging.getLogger(__name__)

SHORTFALL_PREFIX = "trader/execution_shortfall/"
#: How many of the newest shortfall files are read to find ``run_date``'s.
LOOKBACK_FILES = 5

Row = dict[str, object]


def _shortfall_keys(store: Store) -> list[str]:
    keys: list[str] = []
    paginator = store.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=store.bucket, Prefix=store.key(SHORTFALL_PREFIX)):
        for obj in _docs(page, "Contents"):
            key = str(obj.get("Key", ""))
            if key.endswith(".json"):
                keys.append(key)
    return sorted(keys)[-LOOKBACK_FILES:]


def rows_from_shortfall(doc: Mapping[str, object]) -> list[Row]:
    """The filled orders of one shortfall document, as trade rows."""
    rows: list[Row] = []
    for order in _docs(doc, "orders"):
        filled = _num(order, "filled_quantity")
        price = _num(order, "fill_price")
        side = str(order.get("side", "")).lower()
        if filled <= 0 or price <= 0 or side not in ("buy", "sell"):
            continue
        row: Row = {
            "ticker": str(order.get("symbol")),
            "action": side.upper(),
            "shares": filled,
            "fill_price": price,
            "status": "Filled",
            "source": "crucible-trader",
        }
        if _num(order, "decision_price") > 0:
            row["price_at_order"] = _num(order, "decision_price")
        rows.append(row)
    return rows


def v2_trade_rows(
    run_date: str,
    existing: Sequence[Mapping[str, object]] = (),
    *,
    store: Store | None = None,
    env_file: Path = TRADER_ENV,
) -> list[Row]:
    """The v2 session's fills on ``run_date`` that ``existing`` does not already hold."""
    try:
        if store is None:
            if not env_file.exists():
                return []
            store = open_store(read_env_file(env_file))
        doc = None
        for key in reversed(_shortfall_keys(store)):
            relative = key[len(store.key("")) :] if store.prefix else key
            candidate = store.read_json(relative)
            if candidate is not None and candidate.get("calendar_date") == run_date:
                doc = candidate
                break
    except Exception as exc:  # noqa: BLE001 - no rows; the integrity gate still decides
        logger.warning("v2 fills for %s unreadable: %s: %s", run_date, type(exc).__name__, exc)
        return []
    if doc is None:
        return []
    logged = {str(t.get("ticker")) for t in existing}
    rows = rows_from_shortfall(doc)
    skipped = sorted({str(r["ticker"]) for r in rows if r["ticker"] in logged})
    if skipped:
        logger.warning("v2 fills for %s already in the trades table, not added: %s", run_date, skipped)
    kept = [r for r in rows if r["ticker"] not in logged]
    logger.info("v2 fills for %s: %d row(s) added to the attribution", run_date, len(kept))
    return kept
