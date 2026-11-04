"""Explicit ex-date dividend accrual, and total-return SPY.

WHY (alpha-engine-config-I8188, defect 3). Two legs of the same comparison were
on different return definitions, and the direction of the bias was unknown
until it was traced:

* ``dividend_usd`` summed to **$0.00 across all 115 live sessions** while the
  book held LMT for 34 sessions, CTAS for 27, plus MA, COST, AXP, BRO and FAST
  — all payers. The source was ``IBKR.get_accrued_dividends_by_symbol()``,
  whose own docstring says "paper accounts often populate nothing" and which
  returns ``{}`` on that path. Measured: **0 non-zero ``accrued_dividend``
  values across 114 persisted snapshots.** The dividend line was structurally
  incapable of being non-zero.

* ``spy_return_pct`` was computed from ``spy_close`` alone, so SPY's
  distributions were excluded — understating the benchmark by ≈0.50pp over the
  live window.

WHICH DIRECTION THE BIAS RAN — established, not assumed. Per-position P&L is
``shares × (settled close − prior settled close)``, and the stored ArcticDB
closes are split-adjusted but explicitly NOT dividend-adjusted (alpha-engine-
data's ``corporate_actions.get_dividends``: dividends "are NEVER folded into
the stored split-adjusted price level"; ``total_return_close`` is absent from
every live library — verified against the ``macro`` and ``universe`` libraries,
which carry ``Close`` only). So position P&L is price return and excludes
dividends. The headline NAV, however, is IB NetLiquidation, which rises when
the dividend cash lands. Therefore:

    dividends were NOT dropped from the return — they were dropped from the
    ATTRIBUTION and silently credited to ``unattributed_usd``.

Corroborated in the data: on the 17 sessions with no position change, the
residual after lifting rotation out averages **+$168/session (+$2,848 total)**,
against an expected ≈$1,940 of dividends for a ~$1M book at ~55% equity
exposure and a ~1.2% yield over that many sessions. Same sign, same order of
magnitude, on exactly the days where nothing else could produce a positive
plug.

Consequences of the fix, stated plainly because it moves a published number:

* Portfolio total return is UNCHANGED — it was already total return, via NAV.
  What changes is that the dividend moves out of the unnamed plug into a named
  ``dividend_usd`` line.
* The benchmark leg RISES to total return, so reported ``daily_alpha_pct``
  FALLS by the SPY distribution yield — ≈0.50pp over 2026-03-09 → 2026-08-21.
  There is no offsetting uplift on the portfolio leg.

SOURCE. Polygon ``/v3/reference/dividends`` (ex_dividend_date + cash_amount),
via the ``PolygonClient`` already vendored in this repo and already called from
the EOD path for same-day splits (``reconciliation_audit.fetch_same_day_split_ratios``).
This mirrors that function's shape deliberately rather than inventing a second
corporate-action fetch idiom.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# SPY is the benchmark leg; it is fetched alongside the holdings so both legs of
# the comparison come from one source on one call path.
SPY_TICKER = "SPY"


def _in_interval(ex_date: str, after: str | None, through: str) -> bool:
    """True when ``ex_date`` falls in the half-open interval (after, through].

    The interval — not equality with ``through`` — is what keeps the accrual
    correct across a skipped session: the NAV baseline spans from the prior
    persisted eod_pnl row, so the dividend leg must span the same window or the
    two sides of the reconciliation measure different intervals.
    """
    try:
        ex = date.fromisoformat(ex_date)
        end = date.fromisoformat(through)
    except (TypeError, ValueError):
        return False
    if ex > end:
        return False
    if after is None:
        return ex == end
    try:
        start = date.fromisoformat(after)
    except (TypeError, ValueError):
        return ex == end
    return ex > start


def fetch_ex_dividends(
    tickers: Iterable[str],
    run_date: str,
    *,
    prior_date: str | None = None,
    client: Any | None = None,
) -> tuple[dict[str, float], dict[str, str | None], bool, str | None]:
    """Per-share cash dividends going ex in ``(prior_date, run_date]``.

    Returns ``({ticker: cash_amount_per_share}, {ticker: pay_date}, available,
    warning)``. The pay date is carried because NAV on this account is
    cash-basis: the ex-date accrual and the cash arrival are different days,
    and the accrual ledger needs to know when to release the receivable.

    ONE date-filtered Polygon query (``/v3/reference/dividends?
    ex_dividend_date.gte=<prior_date>&ex_dividend_date.lte=<run_date>``)
    returns every dividend going ex in the window across all tickers; the held
    book is intersected against it locally. Inject ``client`` (a
    ``PolygonClient``-shaped object exposing ``get_dividends_for_window``) in
    tests.

    **Why the per-ticker loop is gone** (alpha-engine-config-I10047). The
    client is constructed with ``calls_per_min=5`` (Polygon free tier) and the
    loop issued one ``get_dividends`` per held ticker — thirteen per run date
    (12 held names + SPY). A postclose runs two dates back to back, so on
    2026-09-04 that was 28 requests in ~5 minutes: the log carries 16 ``Rate
    limited (429)`` lines, CRUS and MU each failed after three retries and were
    recorded as having no ex-dividend, and the split query that followed failed
    outright. The varying names across runs are the signature of a rate
    limiter, not of a per-ticker data problem. This mirrors what
    ``reconciliation_audit.fetch_same_day_split_ratios`` did for splits (I9646).

    ``available`` is False when the window query could not be performed at all
    (no ``POLYGON_API_KEY``, client construction failure, or the query
    failing). It is NOT False for "no dividends today" — an empty dict with
    ``available=True`` is a real, positive statement that nothing went ex.
    Collapsing those two into the same ``{}`` is precisely the defect this
    module exists to fix, so they are returned as distinct facts and the caller
    persists ``dividend_accrual_available`` alongside ``dividend_usd``.

    **Failure is ALL-OR-NOTHING**, as it is for splits after I9630. One query
    covers the whole book, so either it succeeded — in which case every held
    name is established, a name's absence from the result being a POSITIVE
    finding of no ex-dividend — or it failed, in which case NO name is, and the
    warning says so naming the date and the count. There is no longer a
    per-ticker "treating as no ex-dividend" path: that was the shape that let
    two rate-limited names be silently scored as non-payers.

    Deviation from the fail-loud default is deliberate and bounded (a) the
    failure swallowed is a third-party HTTP/credential/rate-limit error on the
    single dividend-window query, whose consequence is that NO held ticker's
    ex-dividend status is established for this date — never a subset silently
    rendered as clean; (b) the primary deliverable — the NAV row, the positions
    snapshot and the EOD email — survives, and hard-failing the trading day's
    reconciliation on a vendor outage would be a worse trade than carrying a
    named degradation; (c) the recording surface is threefold: the returned
    ``warning`` (→ ``data_warnings`` → EOD email + console), the persisted
    ``dividend_accrual_available=0`` column, and the residual bound in
    ``pnl_integrity`` — with the accrual absent, the dividend sits back in the
    residual, which is now bounded and raises.
    """
    tickers = sorted({t for t in (tickers or []) if t} | {SPY_TICKER})
    if not tickers:
        return {}, {}, True, None

    if client is None:
        try:
            from polygon_client import PolygonClient  # lazy: optional dep / key

            client = PolygonClient()
        except Exception as exc:  # noqa: BLE001 — key absent / construction failure
            logger.error(
                "[dividends] Polygon client unavailable (%s) — dividend accrual "
                "NOT performed for %s; the accrual stays inside the bounded "
                "residual and dividend_accrual_available is recorded False",
                exc, run_date,
            )
            return {}, {}, False, (
                f"Dividend accrual unavailable for {run_date} (Polygon client "
                "could not be constructed) — dividends remain inside the "
                "unattributed residual for this session."
            )

    # The lower bound is sent to Polygon as an INCLUSIVE `.gte` because the
    # endpoint has no exclusive form; `_in_interval` re-applies the half-open
    # `(prior_date, run_date]` edge locally, so a dividend going ex exactly on
    # `prior_date` — already counted in the prior session — is dropped here.
    window_start = prior_date or run_date
    try:
        events = client.get_dividends_for_window(window_start, run_date)
    except Exception:  # noqa: BLE001 — see the recording surface in the docstring
        # EVERY held ticker is unestablished, not zero of them. One query
        # covers the whole book, so its failure leaves the whole book
        # unmeasured (alpha-engine-config-I10047).
        logger.error(
            "[dividends] ex-dividend window query failed for (%s, %s] "
            "(rate limit, HTTP or credential error) — NO held ticker's "
            "ex-dividend status was established for %s; all %d name(s) stay "
            "inside the bounded residual and dividend_accrual_available is "
            "recorded False", prior_date, run_date, run_date, len(tickers),
            exc_info=True,
        )
        return {}, {}, False, (
            f"Dividend accrual unavailable for {run_date}: the Polygon "
            f"ex-dividend window query for ({prior_date}, {run_date}] failed, "
            f"so NO held ticker's ex-dividend status was established for "
            f"{run_date} ({len(tickers)} name(s): {', '.join(tickers)}) — "
            "dividends remain inside the unattributed residual for this "
            "session."
        )

    held = set(tickers)
    out: dict[str, float] = {}
    pay_dates: dict[str, str | None] = {}
    for ev in events or []:
        ticker = ev.get("ticker")
        if ticker not in held:
            continue
        ex_date = ev.get("ex_dividend_date")
        amount = ev.get("cash_amount")
        if not ex_date or amount in (None, ""):
            continue
        if not _in_interval(str(ex_date), prior_date, run_date):
            continue
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        out[ticker] = out.get(ticker, 0.0) + amount
        # Two distributions in one interval is rare (a special alongside a
        # regular); the LATER pay date wins so the receivable is released
        # once, late rather than early.
        ev_pay = ev.get("pay_date") or None
        prior_pay = pay_dates.get(ticker)
        if ev_pay and (prior_pay is None or str(ev_pay) > prior_pay):
            pay_dates[ticker] = str(ev_pay)
        else:
            pay_dates.setdefault(ticker, prior_pay)

    logger.info(
        "[dividends] %s: %d of %d held ticker(s) went ex in (%s, %s] — "
        "established by ONE window query",
        run_date, len(out), len(tickers), prior_date, run_date,
    )
    return out, pay_dates, True, None


def accrue_position_dividends(
    positions: Mapping[str, dict],
    prior_positions: Mapping[str, Mapping[str, Any]] | None,
    ex_dividends: Mapping[str, float],
) -> list[dict]:
    """Accrue ex-date dividends onto each position; return one record each.

    Each record is ``{ticker, per_share, shares, amount_usd}`` — the caller
    writes them to the ``dividend_accruals`` ledger with their pay dates, which
    is what lets the receivable be released when the cash actually lands.

    Share count is taken from the PRIOR close, not today's, because entitlement
    is settled before the ex-date opens — a name bought ON the ex-date does not
    receive that dividend, and a name sold on the ex-date still does. Falls
    back to today's shares only when there is no prior snapshot for the name.

    Mutates each position in place with ``dividend_usd`` (dollars accrued) and
    folds it into ``daily_return_usd``/``daily_return_pct`` so the dividend
    lands in position attribution instead of the residual.

    SPY is accrued here like any other holding when the book holds it (it is
    the enhanced-index core since 2026-05-13). That is NOT a double count
    against :func:`spy_total_return_pct`: this leg is the dollar dividend the
    BOOK received on the shares it owns, the other is the BENCHMARK index's
    total return. They are the two sides of the comparison, not the same
    number counted twice — and omitting one of them is precisely what made the
    comparison apples-to-oranges.
    """
    accruals: list[dict] = []
    for ticker, pos in positions.items():
        per_share = ex_dividends.get(ticker)
        if not per_share:
            continue
        if pos.get("dividend_usd"):
            # The IB accrual path (eod_reconcile._apply_dividend_delta) already
            # credited this name today. It has produced a non-zero value on
            # zero of 114 live sessions, but if the broker feed ever does
            # populate, the broker is the settlement authority and the Polygon
            # ex-date accrual must not be added on top of it.
            logger.info(
                "[dividends] %s already credited $%.2f by the broker accrual "
                "path — skipping the ex-date accrual to avoid double-counting",
                ticker, pos["dividend_usd"],
            )
            continue
        prior = (prior_positions or {}).get(ticker) or {}
        shares = prior.get("shares", pos.get("shares", 0))
        try:
            shares = float(shares or 0)
        except (TypeError, ValueError):
            shares = 0.0
        if shares <= 0:
            continue
        amount = per_share * shares
        pos["dividend_usd"] = pos.get("dividend_usd", 0.0) + amount
        pos["dividend_per_share"] = per_share
        pos["daily_return_usd"] = pos.get("daily_return_usd", 0.0) + amount
        prior_mv = None
        prior_price = pos.get("prior_price")
        if prior_price:
            try:
                prior_mv = float(prior_price) * shares
            except (TypeError, ValueError):
                prior_mv = None
        if prior_mv:
            pos["daily_return_pct"] = pos.get("daily_return_usd", 0.0) / prior_mv * 100.0
        accruals.append({
            "ticker": ticker,
            "per_share": per_share,
            "shares": shares,
            "amount_usd": amount,
        })
    return accruals


def spy_total_return_pct(
    *,
    spy_close: float | None,
    prior_spy_close: float | None,
    spy_dividend_per_share: float = 0.0,
) -> float | None:
    """SPY TOTAL return over the interval, not price return.

    ``(close + distributions going ex in the interval) / prior_close − 1``.

    The portfolio leg is NAV-based and therefore already total return (the
    dividend cash is inside NetLiquidation). Leaving the benchmark on price
    return understated it by ≈0.50pp over 2026-03-09 → 2026-08-21 and
    overstated reported alpha by the same amount.
    """
    if not spy_close or not prior_spy_close:
        return None
    return ((spy_close + (spy_dividend_per_share or 0.0)) / prior_spy_close - 1.0) * 100.0
