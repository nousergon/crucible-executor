"""Trade-ledger completeness check (alpha-engine-config-I12411).

Asserts that the ``trades`` ledger and the EOD ``positions_snapshot`` agree on
WHICH tickers are held, on the snapshot date:

* every ticker with net open shares in ``trades`` must appear in that day's
  ``positions_snapshot`` (``ledger_open_not_in_snapshot``), and
* every snapshot position must be backed by net open shares in ``trades``
  (``snapshot_position_not_in_ledger``).

Share counts of tickers present on BOTH sides are not compared here — that is
the reconcile audit's job, and partial-fill timing makes it a different check.

Net shares are ``ENTER``/``COVER`` minus ``EXIT``/``REDUCE``, each counted as
``COALESCE(filled_shares, shares)`` exactly as ``backfill_eod_pnl`` does: on a
partial fill ``shares`` holds only what filled inside the daemon's poll window.

Declared exceptions
-------------------
A known, ruled divergence is excused by a record in
``config/ledger_declared_exceptions.json`` — never by editing a ledger row.
The ruling that created the first seven is ``nousergon/alpha-engine-config#12411``
(Brian, 2026-10-21 16:34Z, "Declare all"). An exception excuses exactly one
divergence: it matches on ticker AND the exact signed ``delta_shares`` (ledger
net minus snapshot shares). Anything else still fails:

``undeclared_gap``
    A divergence no valid exception covers. Includes a declared ticker whose
    delta is no longer the declared one: that is a NEW gap.
``declared_exception_without_ruling``
    The entry has no ruling id (``owner/repo#N`` or ``<repo>-I<N>``) or no
    reason; it excuses nothing.
``declared_exception_malformed``
    Not a mapping, no ticker, no integer ``delta_shares``, no ISO ``date``, or
    a ticker declared twice; it excuses nothing.
``declared_exception_not_a_gap``
    The entry matches no current divergence. An exception for a gap that does
    not exist is a registry nobody can trust.
``unknown_action``
    A ledger action this check cannot sign; it fails closed.

This module is read-only: it opens the database ``mode=ro`` and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from executor.pnl_integrity import ACCEPTED_GAP_RULING_RE

DECLARED_EXCEPTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "ledger_declared_exceptions.json"
)

_ACTION_SIGN = {"ENTER": 1, "COVER": 1, "EXIT": -1, "REDUCE": -1}
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_declared_exceptions(path: Path | str = DECLARED_EXCEPTIONS_PATH) -> list[Any]:
    """Return the raw ``exceptions`` list. A missing or unreadable file yields
    ``[]``, which excuses nothing — every gap then fails."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    raw = data.get("exceptions") if isinstance(data, dict) else None
    return raw if isinstance(raw, list) else []


def _net_shares(
    trades: Iterable[Mapping[str, Any]], as_of: str, breaches: list[dict[str, Any]],
) -> dict[str, int]:
    net: dict[str, int] = {}
    for t in trades:
        if str(t.get("date", "")) > as_of:
            continue
        sign = _ACTION_SIGN.get(str(t.get("action")))
        ticker = t.get("ticker")
        if sign is None:
            breaches.append({
                "kind": "unknown_action", "ticker": ticker, "date": t.get("date"),
                "message": f"Ledger row {t.get('trade_id')!r} has action {t.get('action')!r}; "
                           "this check cannot sign it.",
            })
            continue
        qty = t.get("filled_shares")
        if qty is None:
            qty = t.get("shares")
        net[ticker] = net.get(ticker, 0) + sign * int(qty or 0)
    return net


def _validate_exceptions(
    entries: Sequence[Any], breaches: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return ``{ticker: entry}`` for entries that may excuse a gap."""
    valid: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for e in entries:
        if isinstance(e, Mapping) and isinstance(e.get("ticker"), str):
            counts[e["ticker"]] = counts.get(e["ticker"], 0) + 1
    for e in entries:
        ok_shape = (
            isinstance(e, Mapping)
            and isinstance(e.get("ticker"), str) and e["ticker"]
            and isinstance(e.get("delta_shares"), int) and not isinstance(e.get("delta_shares"), bool)
            and e["delta_shares"] != 0
            and isinstance(e.get("date"), str) and _ISO_DATE_RE.match(e["date"])
        )
        if not ok_shape or counts.get(e["ticker"], 0) > 1:
            breaches.append({
                "kind": "declared_exception_malformed",
                "ticker": e.get("ticker") if isinstance(e, Mapping) else None,
                "entry": e,
                "message": f"Declared exception {e!r} is malformed or duplicated; it excuses nothing.",
            })
            continue
        ruling, reason = e.get("ruling"), e.get("reason")
        if not (isinstance(ruling, str) and ACCEPTED_GAP_RULING_RE.match(ruling)
                and isinstance(reason, str) and reason.strip()):
            breaches.append({
                "kind": "declared_exception_without_ruling",
                "ticker": e["ticker"],
                "message": f"Declared exception for {e['ticker']} lacks a ruling id or a reason; "
                           "it excuses nothing.",
            })
            continue
        valid[e["ticker"]] = dict(e)
    return valid


def check_ledger_completeness(
    trades: Iterable[Mapping[str, Any]],
    positions_snapshot: Mapping[str, Any],
    *,
    as_of: str,
    declared_exceptions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compare ledger net open shares with ``positions_snapshot`` on ``as_of``.

    ``trades`` rows need ``date, ticker, action, shares`` and optionally
    ``filled_shares``; rows dated after ``as_of`` are ignored.
    ``positions_snapshot`` is ``{ticker: {"shares": n, ...}}``.
    """
    breaches: list[dict[str, Any]] = []
    net = _net_shares(trades, as_of, breaches)
    snap = {
        t: int(v.get("shares") or 0) if isinstance(v, Mapping) else int(v or 0)
        for t, v in positions_snapshot.items()
    }
    snap = {t: s for t, s in snap.items() if s != 0}

    findings: list[dict[str, Any]] = []
    for ticker in sorted(set(net) | set(snap)):
        n, s = net.get(ticker, 0), snap.get(ticker, 0)
        if (n != 0) == (s != 0):
            continue
        findings.append({
            "ticker": ticker,
            "divergence": "ledger_open_not_in_snapshot" if n != 0 else "snapshot_position_not_in_ledger",
            "ledger_net_shares": n,
            "snapshot_shares": s,
            "delta_shares": n - s,
        })

    valid = _validate_exceptions(list(declared_exceptions or []), breaches)
    declared: list[dict[str, Any]] = []
    matched: set[str] = set()
    for f in findings:
        ex = valid.get(f["ticker"])
        if ex is not None and ex["delta_shares"] == f["delta_shares"]:
            matched.add(f["ticker"])
            declared.append({**f, "date": ex["date"], "ruling": ex["ruling"], "reason": ex["reason"]})
            continue
        breaches.append({
            "kind": "undeclared_gap", **f,
            "message": (
                f"{f['ticker']}: ledger net {f['ledger_net_shares']} vs snapshot "
                f"{f['snapshot_shares']} on {as_of} (delta {f['delta_shares']:+d}, {f['divergence']}) "
                "and no valid declared exception covers that delta."
            ),
        })
    for ticker, ex in valid.items():
        if ticker not in matched:
            breaches.append({
                "kind": "declared_exception_not_a_gap", "ticker": ticker,
                "delta_shares": ex["delta_shares"],
                "message": f"Declared exception for {ticker} (delta {ex['delta_shares']:+d}) "
                           f"matches no divergence on {as_of}.",
            })

    status = "breach" if breaches else "ok"
    return {
        "status": status,
        "as_of": as_of,
        "n_findings": len(findings),
        "declared": declared,
        "breaches": breaches,
        "message": (
            f"Ledger and positions_snapshot agree on held tickers on {as_of}"
            + (f", with {len(declared)} declared exception(s): "
               + ", ".join(f"{d['ticker']} ({d['ruling']})" for d in declared) if declared else "")
            + "."
            if status == "ok" else
            f"Ledger completeness: {len(breaches)} breach(es) on {as_of} — "
            + "; ".join(b["message"] for b in breaches[:8])
            + ("" if len(breaches) <= 8 else f" (+{len(breaches) - 8} more)")
        ),
    }


def run_on_db(
    db_path: Path | str, *, as_of: str | None = None,
    declared_exceptions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Read ``trades`` and ``eod_pnl`` from ``db_path`` (read-only) and check."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if as_of is None:
            row = conn.execute(
                "SELECT MAX(date) FROM eod_pnl WHERE positions_snapshot IS NOT NULL"
            ).fetchone()
            as_of = row[0]
        if not as_of:
            raise ValueError("no eod_pnl positions_snapshot to check against")
        snap_row = conn.execute(
            "SELECT positions_snapshot FROM eod_pnl WHERE date = ?", (as_of,)
        ).fetchone()
        if snap_row is None or not snap_row[0]:
            raise ValueError(f"no positions_snapshot for {as_of}")
        trades = [dict(r) for r in conn.execute(
            "SELECT trade_id, date, ticker, action, shares, filled_shares "
            "FROM trades WHERE date <= ? ORDER BY date, created_at", (as_of,))]
    finally:
        conn.close()
    if declared_exceptions is None:
        declared_exceptions = load_declared_exceptions()
    return check_ledger_completeness(
        trades, json.loads(snap_row[0]), as_of=as_of, declared_exceptions=declared_exceptions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", required=True, help="path to a trades.db copy (opened read-only)")
    p.add_argument("--as-of", help="snapshot date (default: latest eod_pnl row)")
    p.add_argument("--exceptions", default=str(DECLARED_EXCEPTIONS_PATH))
    p.add_argument("--no-exceptions", action="store_true", help="ignore declared exceptions")
    a = p.parse_args(argv)
    ex = [] if a.no_exceptions else load_declared_exceptions(a.exceptions)
    res = run_on_db(a.db, as_of=a.as_of, declared_exceptions=ex)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
