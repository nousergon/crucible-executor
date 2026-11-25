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

**Splits (alpha-engine-config-I14106).** A corporate-action split is a
``kind: split`` journal entry that restates quantity at unchanged cents (its
securities posting carries the share DELTA, ``evidence.ratio`` the ratio). The
v1 attribution values the prior day's position from its own snapshot, which is
pre-split, so on 2026-11-20 (SNDK 8:1, 262 -> 2,096 shares, all 2,096 then sold
at about $89) the sale was priced against 262 shares at the pre-split close and
$159,821 of realized P&L fell into the residual. :func:`v2_splits` reads those
entries and :func:`apply_splits_to_prior_positions` restates the prior
positions (shares x ratio, per-share prices / ratio, market value unchanged)
before the attribution runs. A split with no matching prior position raises
:class:`V2SplitError` by name; it is never dropped.

**What it never does.** It never invents a row: a missing env file, an
absent or unreadable journal, or a journal for another day gives no rows,
and the integrity gate then fails exactly as before. A ticker the executor
itself logged a trade for today is skipped, so a fill is never counted twice.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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


def _cash_cents(entry: Mapping[str, object]) -> int | None:
    """The fill's cash posting: what actually changed hands.

    A sale with IB's realized P&L books the securities posting at COST (the
    holding relieved at IB's cost, the difference to ``income:realized_gains``),
    so the securities cents are not the execution price; the cash posting is
    (alpha-engine-config-I14106: 2026-11-20 SNDK booked at $104.33 cost against
    $89.30 proceeds, $31k of realized P&L in the residual).
    """
    for posting in _docs(entry, "postings"):
        cents = _cents(posting)
        if posting.get("account") == "assets:cash:USD" and cents is not None and cents != 0:
            return cents
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
        cash = _cash_cents(entry)
        if cash is not None:
            cents = cash
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


class V2SplitError(RuntimeError):
    """A journal split that cannot be applied to the v1 prior positions."""


@dataclass(frozen=True)
class Split:
    """One journal ``split`` entry: ``ticker`` gained ``delta_shares`` (``ratio`` when stated)."""

    ticker: str
    delta_shares: float
    ratio: float | None
    event_ref: str | None = None


def _ratio(entry: Mapping[str, object]) -> float | None:
    evidence = entry.get("evidence")
    value = cast(Mapping[str, object], evidence).get("ratio") if isinstance(evidence, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ratio = float(value)
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def splits_from_journal(doc: Mapping[str, object], run_date: str) -> list[Split]:
    """The ``kind: split`` entries of one day's ledger journal (same day/schema guard as the fills)."""
    if doc.get("schema_version") != JOURNAL_SCHEMA_VERSION or doc.get("trading_day") != run_date:
        return []
    splits: list[Split] = []
    for entry in _docs(doc, "entries"):
        if entry.get("kind") != "split":
            continue
        leg = _security_leg(entry)
        if leg is None:
            raise V2SplitError(f"v2 journal for {run_date}: split entry {entry.get('event_ref')!r} has no securities posting")
        ticker, delta, _ = leg
        ref = entry.get("event_ref")
        splits.append(Split(ticker, delta, _ratio(entry), ref if isinstance(ref, str) else None))
    return splits


def v2_splits(run_date: str, *, store: Store | None = None, env_file: Path = TRADER_ENV) -> list[Split]:
    """The v2 journal's splits on ``run_date``; an unreadable or absent journal gives none (as the fills do)."""
    try:
        if store is None:
            if not env_file.exists():
                return []
            store = open_store(read_env_file(env_file))
        doc = store.read_json(journal_key(run_date))
    except Exception as exc:  # noqa: BLE001 - the fills read logs the same failure
        logger.warning("v2 splits for %s unreadable: %s: %s", run_date, type(exc).__name__, exc)
        return []
    return [] if doc is None else splits_from_journal(doc, run_date)


@dataclass(frozen=True)
class ForwardSplit:
    """A journaled split whose effective (journal) day is AFTER the run date being re-run.

    ``journal_ref`` names the journal object and entry it came from, so a
    restatement that used it can say where the evidence is.
    """

    ticker: str
    effective: date
    ratio: float
    journal_ref: str


#: Forward-scan results by ``(run_date, through)``, default store only: the audit
#: asks once per held ticker per day and each answer is the same journal scan.
_FORWARD_SPLIT_CACHE: dict[tuple[str, str], list[ForwardSplit]] = {}


def v2_forward_splits(
    run_date: str,
    *,
    through: date | None = None,
    store: Store | None = None,
    env_file: Path = TRADER_ENV,
) -> list[ForwardSplit]:
    """Journaled splits effective AFTER ``run_date``, read through ``through`` (default today).

    A deliberate lookahead (alpha-engine-config-I14285): re-running a past day
    needs to know whether a split booked LATER makes the reference bars (restated
    to the post-split basis) differ from the share count the broker held that day.
    A split with no stated ratio is skipped. An unreadable store, a bad run date or an unreadable
    journal day gives no split for that day, which leaves the caller on the unrestated
    bars exactly as before.
    """
    try:
        start = date.fromisoformat(run_date)
    except (TypeError, ValueError):
        return []
    end = through if through is not None else datetime.now(UTC).date()
    if end <= start:
        return []
    cache_key = (run_date, end.isoformat())
    if store is None:
        if cache_key in _FORWARD_SPLIT_CACHE:
            return list(_FORWARD_SPLIT_CACHE[cache_key])
        try:
            if not env_file.exists():
                return []
            store = open_store(read_env_file(env_file))
        except Exception as exc:  # noqa: BLE001 - no evidence leaves the bars unrestated
            logger.warning("v2 forward splits for %s: store unreadable: %s: %s", run_date, type(exc).__name__, exc)
            return []
        cache = True
    else:
        cache = False
    found: list[ForwardSplit] = []
    day = start + timedelta(days=1)
    while day <= end:
        if day.weekday() < 5:
            key = journal_key(day.isoformat())
            try:
                doc = store.read_json(key)
                splits = [] if doc is None else splits_from_journal(doc, day.isoformat())
            except Exception as exc:  # noqa: BLE001 - one bad day must not hide the others
                logger.warning("v2 forward splits: journal %s unreadable: %s: %s", key, type(exc).__name__, exc)
                splits = []
            for split in splits:
                if split.ratio is None:
                    logger.warning("v2 forward splits: %s %s states no ratio; not used", key, split.event_ref)
                    continue
                found.append(ForwardSplit(split.ticker, day, split.ratio, f"{key}#{split.event_ref}"))
        day += timedelta(days=1)
    if cache:
        _FORWARD_SPLIT_CACHE[cache_key] = found
    return list(found)


def apply_splits_to_prior_positions(
    prior_positions: MutableMapping[str, MutableMapping[str, object]],
    splits: Sequence[Split],
) -> list[str]:
    """Restate ``prior_positions`` in place to the post-split share basis; returns the tickers restated.

    Shares x ratio; ``closing_price`` and ``avg_cost`` / ratio; ``market_value``
    and the IB mark are unchanged (a split moves no value). Raises
    :class:`V2SplitError` for a split whose ticker has no prior position, or
    whose ratio is neither stated nor derivable.
    """
    restated: list[str] = []
    for split in splits:
        pos = prior_positions.get(split.ticker)
        try:
            prior_shares = float(cast(float, pos.get("shares", 0) or 0)) if pos is not None else 0.0
        except (TypeError, ValueError):
            prior_shares = 0.0
        if pos is None or prior_shares <= 0:
            raise V2SplitError(
                f"v2 journal split for {split.ticker} (+{split.delta_shares:g} shares) has no matching "
                "prior-day position in the v1 snapshot; refusing to drop it"
            )
        derived = (prior_shares + split.delta_shares) / prior_shares
        ratio = split.ratio if split.ratio is not None else derived
        if not math.isfinite(ratio) or ratio <= 0:
            raise V2SplitError(f"v2 journal split for {split.ticker}: no usable ratio (delta {split.delta_shares:g})")
        if abs(prior_shares * ratio - (prior_shares + split.delta_shares)) > 0.5:
            logger.warning(
                "v2 split %s: v1 prior shares %g x ratio %g != ledger post-split %g; v1 snapshot and ledger disagree",
                split.ticker, prior_shares, ratio, prior_shares + split.delta_shares,
            )
        pos["shares"] = prior_shares * ratio
        for field in ("closing_price", "avg_cost"):
            value = pos.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pos[field] = float(value) / ratio
        pos["split_ratio_applied"] = ratio
        restated.append(split.ticker)
        logger.info("v1 prior position %s restated for the %g:1 split (%g -> %g shares)", split.ticker, ratio, prior_shares, prior_shares * ratio)
    return restated
