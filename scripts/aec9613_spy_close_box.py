"""Box-side driver for ``scripts/aec9613_spy_close_restate.py`` (alpha-engine-config-I9613).

It runs on the trading box as ec2-user, in the executor venv, and is shipped over SSM by the
laptop script. It does nothing on import. Brian runs the laptop script; nobody runs this by hand.

Phases (``python aec9613_spy_close_box.py <phase> [arg]``):

``check``   read-only. The eod_pnl state, the current ``spy_close`` of the nine target cells,
            the value the 2026-03-12 backfill wrote that cell from (``eod_reconcile._spy_close``,
            ArcticDB ``macro/SPY``), the vendor raw closes the audit anchors on
            (``pnl_measurement_backfill.fetch_vendor_closes``, Polygon), and the two benchmark-leg
            gates on the current rows and with the restatement applied in memory.
``state``   read-only. The eod_pnl digest and export hash only (the rollback path).
``backup``  SQLite online backup of trades.db, uploaded under the run's backup prefix.
``apply``   ONE transaction of compare-and-set UPDATEs of ``spy_close`` (any miss rolls back all
            of them), then the eod_pnl.csv / trades_latest.db re-export with a conditional PUT.

The result is printed as one ``RESULT_JSON=`` line on stdout; logs go to stderr.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any

W_FIRST = "2026-03-09"
BACKFILL_DATE = "2026-03-12"
DIGEST_COLS = ("portfolio_nav", "daily_return_pct", "spy_return_pct", "daily_alpha_pct", "spy_close")
REPO = "/home/ec2-user/alpha-engine"


class Refuse(Exception):
    """A precondition failed on the box; nothing was written."""


def fnum(v: Any) -> float | None:
    return None if v in (None, "") else float(v)


def digest(rows: list[dict[str, Any]]) -> str:
    """Hash of (date + the benchmark/NAV columns) over every row. The laptop computes the same
    hash from the CSV, so box and S3 are compared on values, not on text."""
    canon = [[str(r["date"])] + [fnum(r.get(k)) for k in DIGEST_COLS]
             for r in sorted(rows, key=lambda r: str(r["date"]))]
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def all_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM eod_pnl ORDER BY date")]
    finally:
        conn.row_factory = None


def render_csv(conn: sqlite3.Connection) -> bytes:
    import pandas as pd  # the same export eod_reconcile does

    return pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn).to_csv(index=False).encode()


def apply_cas(conn: sqlite3.Connection, plan: list[dict[str, Any]], expect_digest: str) -> dict[str, Any]:
    """Write ``to_close`` into each planned row's ``spy_close`` only if it still holds
    ``from_close``. One transaction: any row that does not match rolls back every row.

    ``conn`` must be opened with ``isolation_level=None``. Returns a result dict; on a refusal it
    carries ``refused`` and the table is untouched.
    """
    pre = all_rows(conn)
    if digest(pre) != expect_digest:
        return {"refused": "eod_pnl changed on the box since the check (digest differs); nothing written"}
    tgt = {c["date"] for c in plan}
    conn.execute("BEGIN IMMEDIATE")
    try:
        for c in plan:
            cur = conn.execute(
                "UPDATE eod_pnl SET spy_close=? WHERE date=? AND spy_close IS ?",
                (c["to_close"], c["date"], c["from_close"]),
            )
            if cur.rowcount != 1:
                raise Refuse(f"{c['date']}: spy_close does not hold the expected pre-restatement value "
                             f"{c['from_close']!r} (rowcount {cur.rowcount})")
        conn.execute("COMMIT")
    except Exception as e:  # noqa: BLE001 -- rolled back and reported; the caller stops on "refused"
        conn.execute("ROLLBACK")
        return {"refused": f"transaction rolled back, nothing written: {type(e).__name__}: {e}"}
    post = all_rows(conn)
    pb = {str(r["date"]): r for r in post}
    pa = {str(r["date"]): r for r in pre}
    wrong = [c["date"] for c in plan if fnum(pb[c["date"]]["spy_close"]) != c["to_close"]]
    others_same = all(pa[d] == pb.get(d) for d in pa if d not in tgt) and set(pa) == set(pb)
    other_cols_same = all({k: v for k, v in pa[d].items() if k != "spy_close"}
                          == {k: v for k, v in pb[d].items() if k != "spy_close"} for d in tgt)
    return {"committed": len(plan), "db_rows_wrong": wrong, "db_other_rows_unchanged": others_same,
            "db_target_other_columns_unchanged": other_cols_same, "digest_post": digest(post)}


def put_conditional(s3: Any, bucket: str, key: str, body: bytes, etag: str) -> dict[str, Any]:
    """PUT only if the object still has the ETag recorded before the backup."""
    import botocore.exceptions as bx

    try:
        r = s3.put_object(Bucket=bucket, Key=key, Body=body, IfMatch=etag)
        mode = "If-Match"
    except bx.ParamValidationError:  # boto3 too old for IfMatch: compare-then-put (narrow race, recorded)
        cur = s3.head_object(Bucket=bucket, Key=key)["ETag"]
        if cur != etag:
            raise Refuse(f"s3://{bucket}/{key} changed since the backup (ETag {cur} != {etag})") from None
        r = s3.put_object(Bucket=bucket, Key=key, Body=body)
        mode = "head-compare (boto3 has no IfMatch)"
    except bx.ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
            raise Refuse(f"s3://{bucket}/{key} changed since the backup (If-Match failed)") from e
        raise
    return {"key": key, "etag": r.get("ETag"), "version_id": r.get("VersionId"), "mode": mode}


def services() -> dict[str, str]:
    s = {}
    for u in ("alpha-engine-eod-reconcile-standalone.service", "alpha-engine-trader-reconcile.service",
              "alpha-engine-daemon.service", "boot-pull.service"):
        s[u] = subprocess.run(["/usr/bin/systemctl", "is-active", u], capture_output=True, text=True).stdout.strip()
    return s


def bench_gates(pmb: Any, rows: list[dict[str, Any]], closes: dict[str, float],
                divs: dict[str, float]) -> dict[str, Any]:
    """The two gates that read ``spy_close``. Nothing else reads the column."""
    from executor.pnl_integrity import check_benchmark_vendor_anchor, verify_benchmark_chain_closes

    kw: dict[str, Any] = {}
    if "declared_non_sessions" in inspect.signature(check_benchmark_vendor_anchor).parameters:
        with contextlib.suppress(Exception):
            from executor.accepted_gaps import load_declared_non_sessions
            kw["declared_non_sessions"] = load_declared_non_sessions()
    anc = check_benchmark_vendor_anchor(rows, vendor_closes=closes, vendor_dividends=divs, **kw)
    kinds: dict[str, int] = {}
    for b in anc:
        kinds[b.get("kind")] = kinds.get(b.get("kind"), 0) + 1
    div_dates = sorted(b["run_date"] for b in anc if b.get("kind") == "benchmark_close_divergence")
    bc = verify_benchmark_chain_closes(rows, ex_dividends=divs, prior_session_of=pmb._prior_session_of)
    return {"anchor_breaches_n": len(anc), "anchor_kinds": kinds, "close_divergence_dates": div_dates,
            "close_divergence_n": len(div_dates), "bench_closes": bc.get("closes"),
            "bench_drift_bps": round(bc["drift_bps"], 3) if bc.get("drift_bps") is not None else None,
            "bench_offenders_n": len(bc.get("offenders") or []),
            "bench_offender_dates": sorted(o["date"] for o in (bc.get("offenders") or []))}


def phase_check(cfg: dict[str, Any], targets: list[str], out: dict[str, Any], *, full: bool) -> None:
    from executor import pnl_measurement_backfill as pmb

    db = cfg["db_path"]
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = all_rows(conn)
        out.update(n_rows=len(rows), first=rows[0]["date"], last=rows[-1]["date"], digest=digest(rows),
                   csv_sha256=hashlib.sha256(render_csv(conn)).hexdigest())
    finally:
        conn.close()
    by = {str(r["date"]): r for r in rows}
    out["cells"] = {d: (fnum(by[d]["spy_close"]) if d in by else None) for d in targets}
    out["has_0312"] = BACKFILL_DATE in by
    if not full:
        return
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from executor.eod_reconcile import _spy_close
            out["arctic_0312"] = _spy_close(BACKFILL_DATE, cfg)
        except Exception as e:  # noqa: BLE001 -- reported; the laptop refuses without it
            out["arctic_0312_error"] = f"{type(e).__name__}: {e}"
        start, end = str(rows[0]["date"]), str(rows[-1]["date"])
        closes = pmb.fetch_vendor_closes(start, end)
        divs = pmb.map_ex_dividends_to_sessions(rows, pmb._fetch_spy_events(start))
    out["vendor_closes"] = {d: closes.get(d) for d in targets}
    sim = []
    for r in rows:
        r = dict(r)
        d = str(r["date"])
        if d in targets and closes.get(d) is not None:
            r["spy_close"] = closes[d]
        sim.append(r)
    with contextlib.redirect_stdout(sys.stderr):
        out["gates_pre"] = bench_gates(pmb, rows, closes, divs)
        out["gates_post_simulated"] = bench_gates(pmb, sim, closes, divs)


def phase_backup(cfg: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
    db = cfg["db_path"]
    dest = f"/home/ec2-user/aec9613/trades-pre-{os.path.basename(prefix)}.db"
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    d = sqlite3.connect(dest)
    try:
        src.backup(d)
    finally:
        d.close()
        src.close()
    c = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    ok = c.execute("PRAGMA integrity_check").fetchone()[0]
    n = c.execute("SELECT COUNT(*) FROM eod_pnl").fetchone()[0]
    c.close()
    with open(dest, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    import boto3

    key = f"{prefix}/box-trades.db"
    boto3.client("s3", region_name="us-east-1").upload_file(dest, cfg["trades_bucket"], key)
    out.update(backup_path=dest, s3_key=key, sha256=h, integrity=ok, eod_rows=n)


def phase_apply(cfg: dict[str, Any], payload_path: str, out: dict[str, Any]) -> None:
    with open(payload_path) as f:
        pl = json.load(f)
    out["direction"] = pl["direction"]
    db, bucket = cfg["db_path"], cfg["trades_bucket"]
    conn = sqlite3.connect(db, timeout=60, isolation_level=None)
    try:
        out.update(apply_cas(conn, pl["plan"], pl["expect_digest"]))
        if "refused" in out:
            return
        body = render_csv(conn)
    finally:
        conn.close()
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    try:
        out["csv_put"] = put_conditional(s3, bucket, "trades/eod_pnl.csv", body, pl["csv_etag"])
    except Exception as e:  # noqa: BLE001 -- reported; the laptop marks verification FAILED
        out["csv_put_error"] = f"{type(e).__name__}: {e}"
    # trades_latest.db is the boot-time restore source; keep it consistent with the box db.
    try:
        fd, snap = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s, d = sqlite3.connect(f"file:{db}?mode=ro", uri=True), sqlite3.connect(snap)
        try:
            s.backup(d)
        finally:
            d.close()
            s.close()
        with open(snap, "rb") as f:
            out["latest_put"] = put_conditional(s3, bucket, "trades/trades_latest.db", f.read(), pl["latest_etag"])
        os.unlink(snap)
    except Exception as e:  # noqa: BLE001 -- reported; the next box backup refreshes it
        out["latest_put_error"] = f"{type(e).__name__}: {e}"


def main(argv: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    phase = argv[0]
    from executor.config_loader import load_config

    cfg = load_config()
    out.update(phase=phase, db_path=cfg["db_path"], trades_bucket=cfg["trades_bucket"], services=services(),
               git_head=subprocess.run(["/usr/bin/git", "-C", REPO, "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip())
    if phase in ("check", "state"):
        phase_check(cfg, json.loads(argv[1]), out, full=phase == "check")
    elif phase == "backup":
        phase_backup(cfg, argv[1], out)
    elif phase == "apply":
        phase_apply(cfg, argv[1], out)
    else:
        out["fatal"] = f"unknown phase {phase!r}"
    return out


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result: dict[str, Any] = {}
    rc = 0
    try:
        result = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001 -- the laptop reads "fatal" and stops
        logging.exception("aec9613 spy_close driver failed")
        result["fatal"] = f"{type(exc).__name__}: {exc}"
        rc = 1
    sys.stdout.write("RESULT_JSON=" + json.dumps(result, default=str) + "\n")
    sys.stdout.flush()
    sys.exit(rc)
