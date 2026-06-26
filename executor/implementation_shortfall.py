"""Implementation-shortfall TCA, stratified by entry trigger (L4583 · #656, G11).

Per-order implementation shortfall (IS) measures the cost of *acting* on a
decision: the slippage between the price that was true when the decision was
made and the price actually paid. At retail size the arrival-price benchmark
suffices (we don't move the market), so this decomposes IS into two legs in
basis points:

  * **delay** — decision price (``signal_price``, the mid when the signal
    fired) → arrival price (``price_at_order``, the mid when the order was
    placed). This is the cost of the entry-trigger *waiting* for its
    condition (pullback / VWAP / support / expiry) instead of crossing at
    decision time.
  * **execution** — arrival price (``price_at_order``) → fill price
    (``fill_price``). The cost of crossing the spread once the order is live.
  * **total IS** = delay + execution = decision → fill.

All legs are signed so a *positive* value is always a *cost* (worse price)
for the order's side: for a BUY, paying more than the reference is a cost;
for a SELL, receiving less is a cost.

The payoff (gap G11): aggregating per-order IS by **entry trigger** directly
tests whether the entry-trigger layer earns its delay — e.g. a pullback
trigger should show *negative* delay (it waited and got a better price); if a
trigger's mean delay is a consistent positive cost it isn't beating a
market-open fill and should be questioned.

This module is pure compute (no I/O): ``order_shortfall`` for one order,
``aggregate_by_trigger`` for a batch, and ``build_tca_summary`` for the
report/S3 payload. The thin DB read (``load_entry_orders``) is the only
side-effecting helper.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

# Entry-trigger labels are free text; group anything unset under this bucket so
# it's visible rather than silently dropped.
_UNLABELLED = "unlabelled"


@dataclass(frozen=True)
class OrderShortfall:
    """Per-order implementation shortfall, all legs signed as cost (bps)."""

    ticker: str
    side: str  # "BUY" or "SELL"
    entry_trigger: str
    decision_price: float
    arrival_price: float
    fill_price: float
    delay_bps: float  # decision -> arrival, signed as cost
    execution_bps: float  # arrival -> fill, signed as cost
    total_is_bps: float  # decision -> fill, signed as cost


def _side_sign(side: str) -> int:
    """+1 for a BUY (paying more is a cost), -1 for a SELL (receiving less is
    a cost). Raises on an unrecognised side so a bad row fails loud rather
    than silently flipping the sign of the cost."""
    s = (side or "").strip().upper()
    if s in {"BUY", "B"}:
        return 1
    if s in {"SELL", "S"}:
        return -1
    raise ValueError(f"unrecognised order side {side!r} (expected BUY/SELL)")


def _bps(reference: float, executed: float, sign: int) -> float:
    """Signed slippage of ``executed`` vs ``reference`` in basis points.

    ``sign`` orients the cost by side. For a BUY (sign +1) executing above
    the reference is a positive cost; for a SELL (sign -1) executing below
    the reference is a positive cost."""
    if reference is None or executed is None or reference == 0:
        raise ValueError("non-zero reference and a fill price are required")
    return sign * (executed - reference) / reference * 10_000.0


def order_shortfall(
    *,
    ticker: str,
    side: str,
    entry_trigger: str | None,
    decision_price: float | None,
    arrival_price: float | None,
    fill_price: float | None,
) -> OrderShortfall | None:
    """Compute IS legs for one entry order, or ``None`` if inputs are missing.

    Returns ``None`` (skip, not a crash) when any of the three prices is
    absent — older rows predate the signal_price/fill capture and shouldn't
    be guessed at. A present-but-zero reference still raises (a zero mid is a
    data error, not a missing field)."""
    if decision_price is None or arrival_price is None or fill_price is None:
        return None
    sign = _side_sign(side)
    delay = _bps(decision_price, arrival_price, sign)
    execution = _bps(arrival_price, fill_price, sign)
    return OrderShortfall(
        ticker=ticker,
        side=side.strip().upper(),
        entry_trigger=(entry_trigger or _UNLABELLED).strip() or _UNLABELLED,
        decision_price=decision_price,
        arrival_price=arrival_price,
        fill_price=fill_price,
        delay_bps=delay,
        execution_bps=execution,
        total_is_bps=delay + execution,
    )


@dataclass(frozen=True)
class TriggerTCA:
    """IS aggregated over all orders sharing one entry trigger."""

    entry_trigger: str
    n_orders: int
    mean_delay_bps: float
    mean_execution_bps: float
    mean_total_is_bps: float
    total_delay_bps: float
    total_execution_bps: float
    total_is_bps: float


def aggregate_by_trigger(orders: list[OrderShortfall]) -> list[TriggerTCA]:
    """Group per-order IS by entry trigger; sorted by mean total IS desc
    (worst execution cost first, so the report leads with the problem)."""
    buckets: dict[str, list[OrderShortfall]] = {}
    for o in orders:
        buckets.setdefault(o.entry_trigger, []).append(o)

    out: list[TriggerTCA] = []
    for trigger, group in buckets.items():
        n = len(group)
        sum_delay = sum(o.delay_bps for o in group)
        sum_exec = sum(o.execution_bps for o in group)
        sum_total = sum(o.total_is_bps for o in group)
        out.append(
            TriggerTCA(
                entry_trigger=trigger,
                n_orders=n,
                mean_delay_bps=sum_delay / n,
                mean_execution_bps=sum_exec / n,
                mean_total_is_bps=sum_total / n,
                total_delay_bps=sum_delay,
                total_execution_bps=sum_exec,
                total_is_bps=sum_total,
            )
        )
    out.sort(key=lambda t: t.mean_total_is_bps, reverse=True)
    return out


def load_entry_orders(
    conn: sqlite3.Connection, *, since_date: str | None = None
) -> list[OrderShortfall]:
    """Read filled BUY/SELL entry orders from the trades table and compute IS.

    Entry orders only (``entry_trade_id IS NULL`` — exits carry the exit
    reason in trigger_type and a different cost model). Rows missing any of
    the three prices are skipped by ``order_shortfall``. ``since_date`` (ISO)
    bounds the window for the weekly summary."""
    sql = (
        "SELECT ticker, action, "
        "COALESCE(entry_trigger, trigger_type) AS entry_trigger, "
        "signal_price, price_at_order, fill_price "
        "FROM trades "
        "WHERE entry_trade_id IS NULL "
        "AND fill_price IS NOT NULL AND signal_price IS NOT NULL"
    )
    params: tuple = ()
    if since_date:
        sql += " AND date >= ?"
        params = (since_date,)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    orders: list[OrderShortfall] = []
    for r in rows:
        try:
            o = order_shortfall(
                ticker=r["ticker"],
                side=r["action"],
                entry_trigger=r["entry_trigger"],
                decision_price=r["signal_price"],
                arrival_price=r["price_at_order"],
                fill_price=r["fill_price"],
            )
        except ValueError:
            # Bad side / zero reference on a single row shouldn't sink the
            # whole weekly summary — skip it (it's surfaced by being absent
            # from n_orders rather than crashing the report).
            continue
        if o is not None:
            orders.append(o)
    return orders


def build_tca_summary(
    orders: list[OrderShortfall], *, since_date: str | None = None
) -> dict:
    """Assemble the weekly IS-TCA summary payload (S3 / console tile)."""
    by_trigger = aggregate_by_trigger(orders)
    n = len(orders)
    return {
        "since_date": since_date,
        "n_orders": n,
        "overall_mean_total_is_bps": (
            sum(o.total_is_bps for o in orders) / n if n else 0.0
        ),
        "overall_mean_delay_bps": (
            sum(o.delay_bps for o in orders) / n if n else 0.0
        ),
        "overall_mean_execution_bps": (
            sum(o.execution_bps for o in orders) / n if n else 0.0
        ),
        "by_trigger": [asdict(t) for t in by_trigger],
    }
