"""Post the 2026-03-12 money-ledger journal entry through the normal path (alpha-engine-config-I12580).

The aec#9613 chain's step 2 (``aec11087-backfill-0312-1023c.sh``) backfilled the synthesized
2026-03-12 ``eod_pnl`` row with ``post_journal=False``, because the journal's S3 copy is
create-only and the chain could not roll it back. Brian ruled "skip, add later" (2026-10-22
01:42Z). This script is the "later". It runs on the trading box as ec2-user, in the executor venv,
AFTER Brian's chain has finished and verified. Brian runs it. Nobody else does.

It posts with the SAME code the live reconcile uses for ``post_journal=True``:
``executor.eod_reconcile._post_mark_to_settled_entry``. Its inputs are read from the stored 03-12
row: ``nav_ib_raw_usd``, ``portfolio_nav``, ``total_cash``, ``accrued_interest`` and
``positions_snapshot`` (``ib_market_value`` and ``market_value`` per name). Nothing is hand-written.

**What the normal path posts for 03-12 is probably NOTHING, and that is a finding, not a bug.**
The ledger journals CORRECTIONS to a published NAV: the broker's view minus the published view, on
the same date. It does not journal the day-over-day NAV change. The 03-12 row is a flat book
(``positions_snapshot`` ``{}``) whose IB NetLiquidation and settled NAV were both set to the ruled
estimate, so the two views are equal and ``correction_entry`` returns ``None``. ``check`` reports
that as ``status: no_entry`` with both views. ``apply`` then writes nothing and says so. If
``check`` instead shows an entry, the backfill landed a different row than the ruled one, and that
needs reading before anything is applied.

Phases (``python scripts/aec12580_journal_0312_box.py <phase> [--apply]``):

``check``  The default and read-only. It copies trades.db into memory (SQLite online backup) and
           runs the posting there with no S3 client. The real database and S3 are never touched.
           It reports the planned entry (or ``no_entry``), the existing 03-12 entries, and the trial
           balance before and after on the copy.
``apply``  Needs ``--apply``. It refuses on a weekday between 07:00 and 16:30 ET (the ruled
           market window plus the preopen buffer). It then checks the preconditions, writes the
           backup file and its sha256, posts through the normal path (trades.db, plus the
           create-only S3 copy), and verifies the result. The trial balance must be zero, the
           entry's postings must sum to zero, and its asset postings must equal the row's
           correction, ``portfolio_nav - nav_ib_raw_usd``.

The result is printed as one ``RESULT_JSON=`` line on stdout. Logs go to stderr. Exit 0 means OK or
``no_entry``, 1 means a refusal or failure, and 2 means usage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sqlite3
import sys
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

RUN_DATE = "2026-03-12"
#: The ruled roll-forward estimate the backfill wrote (aec#11087 / aec#9613 chain step 2).
RULED_NAV = 1_008_072.58
NAV_TOLERANCE_USD = 0.01
REPO = "/home/ec2-user/alpha-engine"
BACKUP_DIR = "/home/ec2-user/aec12580"
ET = ZoneInfo("America/New_York")

ROW_COLS = (
    "date",
    "portfolio_nav",
    "total_cash",
    "accrued_interest",
    "positions_snapshot",
    "nav_basis",
    "nav_ib_raw_usd",
)


class Refuse(Exception):
    """A precondition failed; nothing was written."""


def window_refusal(now: dt.datetime) -> str | None:
    """Why ``now`` is inside the refused window, or None."""
    et = now.astimezone(ET)
    if et.weekday() < 5 and dt.time(7, 0) <= et.time() < dt.time(16, 30):
        return f"weekday {et:%a %H:%M} ET is inside 07:00-16:30 ET (market window + preopen buffer)"
    return None


def read_row(conn: sqlite3.Connection, run_date: str = RUN_DATE) -> dict[str, Any] | None:
    cur = conn.execute(
        "SELECT date, portfolio_nav, total_cash, accrued_interest, positions_snapshot, "
        "nav_basis, nav_ib_raw_usd FROM eod_pnl WHERE date = ?",
        (run_date,),
    )
    r = cur.fetchone()
    return None if r is None else dict(zip(ROW_COLS, r, strict=True))


def inputs_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """The arguments ``_post_mark_to_settled_entry`` takes, rebuilt from the stored row."""
    raw = row.get("positions_snapshot")
    positions = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
    return {
        "nav_ib_raw": row.get("nav_ib_raw_usd"),
        "nav": row.get("portfolio_nav"),
        "cash": row.get("total_cash"),
        "accrued": row.get("accrued_interest"),
        "ib_raw_market_values": {t: p.get("ib_market_value") for t, p in positions.items()},
        "positions": positions,
    }


def preconditions(
    conn: sqlite3.Connection, *, expect_nav: float, run_date: str = RUN_DATE, writable: bool = False
) -> dict[str, Any]:
    """Raise Refuse unless the row is the one the chain verified. Returns the row.

    On a read-only connection the ledger checks are left to :func:`plan_on_copy`, which runs
    them on the in-memory copy.
    """
    from executor import money_ledger
    from executor.eod_reconcile import NAV_BASIS_SETTLED_CLOSE

    row = read_row(conn, run_date)
    if row is None:
        raise Refuse(f"no eod_pnl row for {run_date}: run Brian's chain (step 2) first")
    if row["portfolio_nav"] is None or abs(float(row["portfolio_nav"]) - expect_nav) > NAV_TOLERANCE_USD:
        raise Refuse(f"{run_date} portfolio_nav {row['portfolio_nav']} is not the ruled {expect_nav:,.2f}")
    if row["nav_basis"] != NAV_BASIS_SETTLED_CLOSE:
        raise Refuse(
            f"{run_date} nav_basis is {row['nav_basis']!r}; the live path journals only "
            f"{NAV_BASIS_SETTLED_CLOSE!r} rows (eod_reconcile.run, post_journal branch)"
        )
    if row["nav_ib_raw_usd"] is None or row["total_cash"] is None:
        raise Refuse(f"{run_date} row has no nav_ib_raw_usd / total_cash to strike the views from")
    if writable:
        money_ledger.ensure_schema(conn)
        money_ledger.trial_balance(conn)  # raises LedgerIntegrityError on a broken ledger
    return row


def plan_on_copy(conn: sqlite3.Connection, row: dict[str, Any], run_date: str = RUN_DATE) -> dict[str, Any]:
    """Run the normal posting on an in-memory COPY of ``conn``; report what it would write."""
    from executor import money_ledger
    from executor.eod_reconcile import _post_mark_to_settled_entry

    mem = sqlite3.connect(":memory:")
    try:
        conn.backup(mem)
        money_ledger.ensure_schema(mem)
        money_ledger.trial_balance(mem)  # the ledger as it stands must already balance
        existing = money_ledger.entries_for_date(mem, run_date)
        doc = _post_mark_to_settled_entry(mem, run_date=run_date, trades_bucket=None, **inputs_from_row(row))
        tb = money_ledger.trial_balance(mem)
    finally:
        mem.close()
    out: dict[str, Any] = {
        "existing_entries": [e.get("entry_id") for e in existing],
        "trial_balance_after_on_copy": {k: str(v) for k, v in tb.items()},
    }
    if doc is None:
        out.update(
            status="no_entry",
            why=(
                "the broker view and the published view of the stored row are equal, so "
                "correction_entry returns None: the normal path journals nothing for this day"
            ),
        )
    else:
        out.update(status="entry", entry={k: doc.get(k) for k in ("entry_id", "kind", "postings", "evidence")})
    return out


def backup(db_path: str, stamp: str) -> dict[str, Any]:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dest = os.path.join(BACKUP_DIR, f"trades-pre-{stamp}.db")
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    d = sqlite3.connect(dest)
    try:
        src.backup(d)
    finally:
        d.close()
        src.close()
    c = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    ok = c.execute("PRAGMA integrity_check").fetchone()[0]
    c.close()
    if ok != "ok":
        raise Refuse(f"backup integrity_check: {ok}")
    with open(dest, "rb") as f:
        return {"backup_path": dest, "backup_sha256": hashlib.sha256(f.read()).hexdigest()}


def verify(conn: sqlite3.Connection, doc: dict[str, Any], row: dict[str, Any]) -> list[str]:
    from executor import money_ledger

    bad: list[str] = []
    money_ledger.trial_balance(conn)  # raises on non-zero
    amounts = [Decimal(str(p["amount"])) for p in doc["postings"]]
    if sum(amounts, Decimal("0")) != 0:
        bad.append(f"entry {doc['entry_id']} postings sum to {sum(amounts)}")
    assets = sum(
        (Decimal(str(p["amount"])) for p in doc["postings"] if str(p["account"]).startswith("assets:")), Decimal("0")
    )
    want = money_ledger.to_cents(row["portfolio_nav"]) - money_ledger.to_cents(row["nav_ib_raw_usd"])
    if assets != want:
        bad.append(f"asset postings {assets} != the row's correction portfolio_nav - nav_ib_raw_usd = {want}")
    return bad


def run(
    phase: str,
    *,
    apply: bool,
    db_path: str,
    trades_bucket: str | None,
    expect_nav: float = RULED_NAV,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"phase": phase, "run_date": RUN_DATE, "db_path": db_path}
    now = now or dt.datetime.now(dt.UTC)
    if phase == "check":
        ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = preconditions(ro, expect_nav=expect_nav)
            out["row"] = {k: row[k] for k in ROW_COLS if k != "positions_snapshot"}
            out.update(plan_on_copy(ro, row))
        finally:
            ro.close()
        return out

    if not apply:
        raise Refuse("apply needs --apply (check is the default and writes nothing)")
    why = window_refusal(now)
    if why:
        raise Refuse(why)

    from executor.eod_reconcile import _post_mark_to_settled_entry

    conn = sqlite3.connect(db_path, timeout=60)
    try:
        row = preconditions(conn, expect_nav=expect_nav, writable=True)
        plan = plan_on_copy(conn, row)
        out["plan"] = plan
        if plan["status"] == "no_entry":
            out.update(status="no_entry", written=False)
            return out
        out.update(backup(db_path, now.strftime("%Y%m%dT%H%M%SZ")))
        # The live path, unchanged: trades.db first, then the create-only S3 copy.
        doc = _post_mark_to_settled_entry(conn, run_date=RUN_DATE, trades_bucket=trades_bucket, **inputs_from_row(row))
        conn.commit()
        if doc is None:  # cannot differ from the plan on the same row, but never assume it
            out.update(status="no_entry", written=False)
            return out
        bad = verify(conn, doc, row)
        out.update(
            status="posted" if not bad else "verify_failed",
            written=True,
            entry_id=doc["entry_id"],
            s3_key=doc.get("s3_key"),
            s3_error=doc.get("s3_error"),
            newly_written=doc.get("newly_written"),
            problems=bad,
        )
        return out
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post the 2026-03-12 journal entry (aec#12580).")
    ap.add_argument("phase", nargs="?", default="check", choices=["check", "apply"])
    ap.add_argument("--apply", action="store_true", help="required with the apply phase")
    ap.add_argument("--db-path", default=None, help="default: the executor config's db_path")
    args = ap.parse_args(argv)
    sys.path.insert(0, REPO)
    result: dict[str, Any] = {}
    rc = 0
    try:
        db_path, bucket = args.db_path, None
        if db_path is None:
            from executor.config_loader import load_config

            cfg = load_config()
            db_path, bucket = cfg["db_path"], cfg["trades_bucket"]
        result = run(args.phase, apply=args.apply, db_path=db_path, trades_bucket=bucket)
        rc = 0 if result.get("status") in ("posted", "no_entry", "entry") else 1
    except Refuse as exc:
        result.update(refused=str(exc))
        rc = 1
    except Exception as exc:  # noqa: BLE001 -- one fatal line for the operator
        logging.exception("aec12580 journal 03-12 failed")
        result.update(fatal=f"{type(exc).__name__}: {exc}")
        rc = 1
    print("RESULT_JSON=" + json.dumps(result, default=str, sort_keys=True))
    return rc


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
