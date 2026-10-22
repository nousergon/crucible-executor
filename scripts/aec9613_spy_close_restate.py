"""Restate the pre-2026-03-20 ``eod_pnl.spy_close`` cells, and the 2026-03-12 backfill cell, to the
vendor's RAW close (alpha-engine-config-I9613, the ``spy_close`` decision card).

RED LINE 4 (production data write). Brian runs this. Nobody else runs ``--apply``.

Ruling: Brian ruled Restate 2026-10-22 15:37Z, card aec9613 spy_close. It is the THIRD step of the
chain, run in the same session after the two existing ones:

  1. the aec#9613 restatement   (``aec9613-spy-restate.sh --apply --start-box``)
  2. the aec#11087 03-12 backfill (``aec11087-backfill-0312.sh --apply --start-box --i-confirm-9613-ruled``)
  3. this script                (``--apply --start-box --i-confirm-9613-ruled``)

What is wrong
  * The 8 ``spy_close`` cells 2026-03-09..03-19 (03-09, 10, 11, 13, 16, 17, 18, 19) are
    dividend-back-adjusted: every one sits exactly -27.2bp under the vendor close, which is
    $1.797 / $659.80, SPY's 2026-03-20 distribution. A close cannot be adjusted for a dividend that
    has not gone ex, so they were rewritten after 03-20 from an adjusted series.
  * The 03-12 backfill writes ``spy_close`` from ArcticDB ``macro/SPY``
    (``eod_reconcile._spy_close``), which is now back-adjusted through the June and September
    distributions too: about 660.90 against the vendor's 666.06, -77.5bp.
  * Only those 9 cells change. No ``spy_return_pct``, ``daily_alpha_pct`` or NAV cell is touched,
    and no row is added or removed.

Where the vendor close comes from
  ``executor.pnl_measurement_backfill.fetch_vendor_closes``: Polygon ``/v2/aggs`` daily closes,
  split-adjusted and NOT dividend-adjusted. It is the series ``pnl_integrity.check_benchmark_vendor_anchor``
  anchors on, so a restated cell is by construction a 0.0bp ``benchmark_close_divergence``. The box
  fetches it (the plan source) and this laptop fetches it independently; the two must agree exactly
  on every target date before ``--apply`` writes. The dry run prints every vendor close.

Compare-and-set (each cell is written only if it still holds its expected pre-restatement value)
  * The 8 pre-03-20 cells: the as-published values pinned in ``PRE_0320_AS_PUBLISHED`` below
    (s3 ``trades/eod_pnl.csv`` version ``1jZ1kYi7y9k77RHByqEvd0H5iaIb4iUD``, 2026-10-21 00:37Z). The
    aec#9613 restatement writes only ``spy_return_pct``/``daily_alpha_pct`` and the backfill only
    adds a row, so neither moves these values.
  * The 03-12 cell only exists after the backfill. Its expected value is the backfill's own output:
    the value ``eod_reconcile._spy_close("2026-03-12")`` returns on the box (the exact call the
    backfill's reconcile wrote the cell from), which must also lie within $0.50 of the 660.90
    measured on 2026-10-22. If the row is absent, the run is a NO-GO: run the backfill first.

NO-GO, never overridable: a compare-and-set mismatch; the 03-12 row absent; no aec#9613 restatement
record in S3 (ruled order); box trades.db differs from s3 eod_pnl.csv; the box and laptop vendor
closes disagree; the time window; a running pipeline; ``--apply`` without ``--i-confirm-9613-ruled``.
NO-GO, ``--force-plan`` overrides: a cell's divergence from the vendor is not the diagnosed one
(-27.2bp +/- 1bp before 03-20, -77.5bp +/- 3bp on 03-12); a target cell still diverges after the
restatement is simulated; a benchmark-leg gate goes newly red when it is simulated.

``--apply`` refuses unless all of these hold:
  1. Time (America/New_York), the same windows as the two scripts before it: it refuses on a weekday
     07:00-16:30 and 16:30-21:30 (market, preopen buffer, postclose and the 17:05 EOD timer), and on
     any day 00:45-01:15 (stop-trading backstop). Under EDT that leaves weekdays 01:30-11:00Z, less
     04:45-05:15Z. Inside a window a dry run also skips the vendor and box calls.
  2. ``--i-confirm-9613-ruled`` is passed: Brian's attestation of the 2026-10-22 15:37Z ruling.
  3. No RUNNING preopen, postclose, postclose-reconcile or EOD-data pipeline; on the box neither
     reconcile unit is active.
  4. The gates above pass.
  5. Backups first, each read back and sha256-verified: s3 ``trades/eod_pnl.csv`` and
     ``trades/trades_latest.db`` (by version id) and a SQLite online backup of the box's trades.db,
     plus the as-published record (each cell's old and new value, the vendor close, the ruling),
     all under ``s3://alpha-engine-research/ops/daily-summaries/backups/aec9613-spyclose-restate-<STAMP>/``.
The write is ONE SQLite transaction of compare-and-set UPDATEs on the box (any miss rolls back all
of them), then the eod_pnl.csv and trades_latest.db re-export with a conditional PUT (If-Match on the
ETag read before the backup). The script then re-reads the CSV and checks that exactly the 9
``spy_close`` cells changed, to the vendor closes, and that every other byte is unchanged. It prints
the rollback command (``--rollback <STAMP> --apply``: compare-and-set back to the as-published values).

Usage (repo root, repo venv; AWS from the environment):
  AWS_PROFILE=ne-admin AWS_REGION=us-east-1 python scripts/aec9613_spy_close_restate.py --start-box
  AWS_PROFILE=ne-admin AWS_REGION=us-east-1 python scripts/aec9613_spy_close_restate.py --apply --start-box --i-confirm-9613-ruled
  AWS_PROFILE=ne-admin AWS_REGION=us-east-1 python scripts/aec9613_spy_close_restate.py --rollback <STAMP> [--apply] [--start-box]
``--start-box`` boots the stopped trading box and stops it at the end, only if this script started it.
Booting starts its units (daemon, IB Gateway) as on any boot.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = Path(__file__).resolve().parent / "aec9613_spy_close_box.py"

RULING = "Brian ruled Restate 2026-10-22 15:37Z, card aec9613 spy_close"
ISSUE = "https://github.com/nousergon/alpha-engine-config/issues/9613"
BUCKET = "alpha-engine-research"
INSTANCE = "i-018eb3307a21329bf"
ACCOUNT = "711398986525"
SFNS = ("ne-preopen-trading-pipeline", "ne-postclose-trading-pipeline",
        "ne-postclose-reconcile-pipeline", "ne-data-collection-eod")
CSV_KEY, LATEST_KEY = "trades/eod_pnl.csv", "trades/trades_latest.db"
RESTATE_RECORD_PREFIX = "ops/daily-summaries/backups/aec9613-spy-restate-"
BACKUP_ROOT = "ops/daily-summaries/backups/aec9613-spyclose"

# The as-published dividend-back-adjusted cells: s3 trades/eod_pnl.csv version
# 1jZ1kYi7y9k77RHByqEvd0H5iaIb4iUD (2026-10-21 00:37Z), read 2026-10-22. Exact float repr.
PRE_0320_AS_PUBLISHED: dict[str, float] = {
    "2026-03-09": 676.4227294921875,
    "2026-03-10": 675.3356323242188,
    "2026-03-11": 674.4879760742188,
    "2026-03-13": 660.4862060546875,
    "2026-03-16": 667.2078857421875,
    "2026-03-17": 668.9630126953125,
    "2026-03-18": 659.6285400390625,
    "2026-03-19": 658.0029907226562,
}
BACKFILL_DATE = "2026-03-12"
BACKFILL_MEASURED = 660.90  # ArcticDB macro/SPY for 03-12, measured 2026-10-22 (C6)
BACKFILL_BAND_USD = 0.50
TARGETS = tuple(sorted([*PRE_0320_AS_PUBLISHED, BACKFILL_DATE]))
DIAGNOSED_BP = {"pre_0320": (-27.2, 1.0), "backfill": (-77.5, 3.0)}

SCRIPT = "python scripts/aec9613_spy_close_restate.py"


class NoGo(Exception):
    """The run stops. Nothing past the failing step happened."""


# ───────────────────────────── pure helpers (tested) ─────────────────────────────

def refusal_reason(now: datetime) -> str | None:
    """The refusal windows of aec9613-spy-restate.sh / aec11087-backfill-0312.sh, in New York time."""
    et = now.astimezone(ZoneInfo("America/New_York"))
    hm = et.hour * 60 + et.minute
    weekday = et.weekday() < 5
    if 45 <= hm < 75:
        return "00:45-01:15 ET: alpha-engine-stop-trading backstop window"
    if weekday and 7 * 60 <= hm < 16 * 60 + 30:
        return "weekday 07:00-16:30 ET (market hours 13:30-20:30Z plus the preopen buffer)"
    if weekday and 16 * 60 + 30 <= hm < 21 * 60 + 30:
        return "weekday 16:30-21:30 ET: postclose pipelines and the 17:05 EOD timer"
    return None


def fnum(v: Any) -> float | None:
    return None if v in (None, "") else float(v)


def parse_csv(body: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    csv.field_size_limit(10**9)
    r = csv.reader(io.StringIO(body.decode()))
    hdr = next(r)
    return hdr, [dict(zip(hdr, x, strict=False), __raw=x) for x in r]


def digest(rows: list[dict[str, Any]]) -> str:
    """Same hash as the box driver's ``digest`` (date + NAV/benchmark columns, as floats)."""
    cols = ("portfolio_nav", "daily_return_pct", "spy_return_pct", "daily_alpha_pct", "spy_close")
    canon = [[x["date"]] + [fnum(x.get(k)) for k in cols] for x in sorted(rows, key=lambda x: x["date"])]
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def expected_pre_values(by: dict[str, dict[str, Any]], arctic_0312: float | None) -> tuple[dict[str, float], list[str]]:
    """The value each target cell must hold before the write (the compare-and-set operand).

    Returns ``(expected, hard)``. A date is in ``expected`` only if its expected value is defined;
    ``hard`` lists the never-overridable refusals (a cell that does not hold its expected value, the
    03-12 row absent, the backfill's value unavailable).
    """
    expected: dict[str, float] = {}
    hard: list[str] = []
    for d, v in PRE_0320_AS_PUBLISHED.items():
        row = by.get(d)
        if row is None:
            hard.append(f"{d}: no eod_pnl row")
            continue
        expected[d] = v
        if fnum(row.get("spy_close")) != v:
            hard.append(f"{d}: spy_close is {row.get('spy_close')!r}, not the as-published {v!r} "
                        "(compare-and-set: it changed since 2026-10-21, so nothing is written)")
    row = by.get(BACKFILL_DATE)
    if row is None:
        hard.append(f"{BACKFILL_DATE}: no eod_pnl row yet, so the aec#11087 backfill has not run. Ruled order: "
                    "the restatement, then the 03-12 backfill, then this script")
    elif arctic_0312 is None:
        hard.append(f"{BACKFILL_DATE}: the backfill's own value (eod_reconcile._spy_close on the box) is not "
                    "available, so the cell's expected value is undefined; rerun with --start-box")
    elif abs(arctic_0312 - BACKFILL_MEASURED) > BACKFILL_BAND_USD:
        hard.append(f"{BACKFILL_DATE}: ArcticDB macro/SPY now gives {arctic_0312!r}, more than "
                    f"${BACKFILL_BAND_USD:.2f} from the {BACKFILL_MEASURED} measured 2026-10-22; the series moved")
    else:
        expected[BACKFILL_DATE] = arctic_0312
        if fnum(row.get("spy_close")) != arctic_0312:
            hard.append(f"{BACKFILL_DATE}: spy_close is {row.get('spy_close')!r}, not {arctic_0312!r}, the value the "
                        "backfill wrote it from (compare-and-set)")
    return expected, hard


def build_plan(expected: dict[str, float], vendor: dict[str, float | None]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """One correction per target cell whose expected value is defined: ``from_close`` (the
    compare-and-set operand) -> ``to_close`` (the vendor raw close). Returns ``(plan, hard, soft)``."""
    plan: list[dict[str, Any]] = []
    hard: list[str] = []
    soft: list[str] = []
    for d in TARGETS:
        if d not in expected:
            continue
        v = vendor.get(d)
        if not v:
            hard.append(f"{d}: the vendor has no close")
            continue
        bp = (expected[d] / v - 1.0) * 1e4
        kind = "backfill" if d == BACKFILL_DATE else "pre_0320"
        centre, tol = DIAGNOSED_BP[kind]
        if abs(bp - centre) > tol:
            soft.append(f"{d}: divergence {bp:+.2f}bp is not the diagnosed {centre:+.1f}bp +/- {tol}bp")
        plan.append({"date": d, "from_close": expected[d], "to_close": float(v), "vendor_close": float(v),
                     "divergence_bp": bp,
                     "basis": ("ArcticDB macro/SPY via the aec#11087 backfill" if kind == "backfill"
                               else "dividend-back-adjusted for the 2026-03-20 distribution")})
    return plan, hard, soft


def cross_check(box: dict[str, float | None], laptop: dict[str, float | None]) -> list[str]:
    return [f"{d}: box vendor close {box.get(d)!r} != laptop {laptop.get(d)!r}"
            for d in TARGETS if box.get(d) != laptop.get(d)]


def gate_findings(pre: dict[str, Any], post: dict[str, Any]) -> list[str]:
    """Soft NO-GO findings from the benchmark-leg gates simulated on the box."""
    bad = []
    still = sorted(set(post.get("close_divergence_dates") or []) & set(TARGETS))
    if still:
        bad.append(f"target cell(s) still diverge from the vendor after the simulated restatement: {still}")
    for k in ("anchor_breaches_n", "close_divergence_n", "bench_offenders_n"):
        if (post.get(k) or 0) > (pre.get(k) or 0):
            bad.append(f"{k} goes up after the simulated restatement: {pre.get(k)} -> {post.get(k)}")
    if pre.get("bench_closes") is True and post.get("bench_closes") is False:
        bad.append("verify_benchmark_chain_closes goes from closing to not closing")
    return bad


def postcheck(pre_body: bytes, post_body: bytes, plan: list[dict[str, Any]]) -> list[str]:
    """Exactly the planned spy_close cells changed, to the planned values; every other byte is the same."""
    h0, r0 = parse_csv(pre_body)
    h1, r1 = parse_csv(post_body)
    bad = []
    if h0 != h1:
        return ["eod_pnl.csv header changed"]
    a = {x["date"]: x["__raw"] for x in r0}
    b = {x["date"]: x["__raw"] for x in r1}
    if set(a) != set(b):
        bad.append(f"row set changed: +{sorted(set(b) - set(a))[:5]} -{sorted(set(a) - set(b))[:5]}")
    i = h0.index("spy_close")
    tgt = {c["date"]: c for c in plan}
    for d in sorted(set(a) & set(b)):
        x, y = a[d], b[d]
        if d not in tgt:
            if x != y:
                bad.append(f"{d}: a non-target row changed")
            continue
        other = [h0[j] for j in range(len(x)) if j != i and x[j] != y[j]]
        if other:
            bad.append(f"{d}: columns other than spy_close changed: {other}")
        if fnum(y[i]) != tgt[d]["to_close"]:
            bad.append(f"{d}: spy_close {y[i]} != planned {tgt[d]['to_close']}")
    return bad


def as_published_record(plan: list[dict[str, Any]], box: dict[str, Any], csvmeta: dict[str, Any],
                        prefix: str, forced: list[str]) -> dict[str, Any]:
    return {
        "record": "as-published eod_pnl.spy_close before the aec#9613 spy_close restatement",
        "ruling": RULING, "issue": ISSUE,
        "approver": "Brian (Director decision card aec9613 spy_close, option 'Restate', 2026-10-22 15:37Z)",
        "written_at": datetime.now(UTC).isoformat(),
        "method": ("spy_close := the vendor raw close: Polygon /v2/aggs daily close, split-adjusted and NOT "
                   "dividend-adjusted (executor.pnl_measurement_backfill.fetch_vendor_closes, the anchor of "
                   "executor.pnl_integrity.check_benchmark_vendor_anchor), fetched on the trading box and "
                   "cross-checked from the laptop. No other column is written."),
        "box_git_head": box.get("git_head"), "backup_prefix": f"s3://{BUCKET}/{prefix}/", "s3_before": csvmeta,
        "rows": [{"date": c["date"], "as_published": {"spy_close": c["from_close"]},
                  "restated": {"spy_close": c["to_close"]}, "vendor_close": c["vendor_close"],
                  "divergence_bp_before": c["divergence_bp"], "basis_before": c["basis"]} for c in plan],
        "gates_pre": box.get("gates_pre"), "gates_post_simulated": box.get("gates_post_simulated"),
        "forced_past": forced,
    }


def rollback_plan(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"date": x["date"], "from_close": x["restated"]["spy_close"], "to_close": x["as_published"]["spy_close"]}
            for x in record["rows"]]


def ssm_parameters(driver_src: bytes, args: list[str], payload: bytes | None) -> dict[str, Any]:
    b64 = base64.b64encode(driver_src).decode()
    cmds = ["set -u", "F=$(mktemp /tmp/aec9613c-XXXXXX.py)", f"echo '{b64}' | base64 -d > \"$F\"", "chmod 644 \"$F\""]
    argv = [shlex.quote(x) for x in args]
    if payload is not None:
        pb = base64.b64encode(payload).decode()
        cmds += ["J=$(mktemp /tmp/aec9613c-XXXXXX.json)", f"echo '{pb}' | base64 -d > \"$J\"", "chmod 644 \"$J\""]
        argv.append('"$J"')
    cmds += [
        "sudo -u ec2-user -H mkdir -p /home/ec2-user/aec9613",
        f"LOG=/home/ec2-user/aec9613/spyclose-{args[0]}-$(date -u +%Y%m%dT%H%M%SZ).log",
        "cd /home/ec2-user/alpha-engine",
        "sudo -u ec2-user -H env PYTHONPATH=/home/ec2-user/alpha-engine AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 "
        f"/home/ec2-user/alpha-engine/.venv/bin/python \"$F\" {' '.join(argv)} 2>\"$LOG\"",
        "rc=$?",
        "echo \"box log: $LOG\" >&2; tail -c 6000 \"$LOG\" >&2",
        "rm -f \"$F\" ${J:+\"$J\"}",
        "exit $rc",
    ]
    return {"commands": cmds, "executionTimeout": ["3600"]}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ───────────────────────────── AWS (boto3, credentials from the environment) ─────────────────────────────

class Aws:
    """Every AWS call the script makes. Tests replace it with a recorder."""

    def __init__(self) -> None:
        import boto3

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_REGION", region)
        sess = boto3.Session(region_name=region)
        self.s3, self.ssm, self.ec2 = sess.client("s3"), sess.client("ssm"), sess.client("ec2")
        self.sfn, self.sts = sess.client("stepfunctions"), sess.client("sts")
        self.inflight: str | None = None
        self.started_box = False

    def caller_arn(self) -> str:
        return self.sts.get_caller_identity()["Arn"]

    def running_executions(self, sm: str) -> int:
        arn = f"arn:aws:states:us-east-1:{ACCOUNT}:stateMachine:{sm}"
        return len(self.sfn.list_executions(stateMachineArn=arn, statusFilter="RUNNING")["executions"])

    def head(self, key: str) -> tuple[str, str]:
        h = self.s3.head_object(Bucket=BUCKET, Key=key)
        return h["ETag"], h["VersionId"]

    def get(self, key: str, version: str | None = None) -> bytes:
        kw = {"VersionId": version} if version else {}
        return self.s3.get_object(Bucket=BUCKET, Key=key, **kw)["Body"].read()

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        return keys

    def put(self, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)

    def box_state(self) -> str:
        r = self.ec2.describe_instances(InstanceIds=[INSTANCE])
        return r["Reservations"][0]["Instances"][0]["State"]["Name"]

    def start_box(self) -> None:
        self.ec2.start_instances(InstanceIds=[INSTANCE])
        self.ec2.get_waiter("instance_running").wait(InstanceIds=[INSTANCE])
        for _ in range(60):
            info = self.ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [INSTANCE]}])
            lst = info.get("InstanceInformationList") or []
            if lst and lst[0].get("PingStatus") == "Online":
                break
            time.sleep(10)
        else:
            raise NoGo("box not SSM-Online after ~10 min")
        print("   waiting for boot-pull to finish (90s settle)")
        time.sleep(90)

    def stop_box(self) -> None:
        if self.inflight:
            st = self.ssm.get_command_invocation(CommandId=self.inflight, InstanceId=INSTANCE)["Status"]
            if st in ("Pending", "InProgress", "Delayed"):
                print(f"NOT stopping {INSTANCE}: SSM command {self.inflight} is still {st} on it. Stop it yourself "
                      f"once that finishes: aws ec2 stop-instances --instance-ids {INSTANCE}", file=sys.stderr)
                return
        print(f"== stopping {INSTANCE} (this script started it)")
        self.ec2.stop_instances(InstanceIds=[INSTANCE])

    def ssm_run(self, args: list[str], payload: bytes | None = None) -> dict[str, Any]:
        params = ssm_parameters(DRIVER.read_bytes(), args, payload)
        cid = self.ssm.send_command(InstanceIds=[INSTANCE], DocumentName="AWS-RunShellScript",
                                    Comment=f"aec9613 spy_close restate: {args[0]}", Parameters=params)["Command"]["CommandId"]
        self.inflight = cid
        print(f"   ssm command {cid} ({args[0]}) sent; polling")
        inv: dict[str, Any] = {}
        for _ in range(400):
            time.sleep(10)
            try:
                inv = self.ssm.get_command_invocation(CommandId=cid, InstanceId=INSTANCE)
            except self.ssm.exceptions.InvocationDoesNotExist:  # right after send; keep polling
                continue
            if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
                break
        else:
            raise NoGo(f"ssm {cid} still running after ~66 min")
        self.inflight = None
        lines = [x for x in (inv.get("StandardOutputContent") or "").splitlines() if x.startswith("RESULT_JSON=")]
        if not lines:
            print("---- box stderr (tail) ----\n" + (inv.get("StandardErrorContent") or "")[-4000:], file=sys.stderr)
            raise NoGo(f"ssm {cid} ({args[0]}) ended {inv.get('Status')} with no result")
        try:
            res = json.loads(lines[-1][len("RESULT_JSON="):])
        except ValueError as e:
            raise NoGo(f"ssm {cid} ({args[0]}): result is not valid JSON (truncated?)") from e
        print(f"   ssm {cid} ended {inv['Status']}")
        return res


def laptop_vendor_closes(start: str, end: str) -> dict[str, float]:
    """The same anchor the box uses, fetched from this laptop (Polygon key from SSM via the profile)."""
    sys.path.insert(0, str(REPO_ROOT))
    from executor.pnl_measurement_backfill import fetch_vendor_closes

    return fetch_vendor_closes(start, end)


# ───────────────────────────── the run ─────────────────────────────

def _print_plan(plan: list[dict[str, Any]], vendor_src: str) -> None:
    print(f"\nPLAN: spy_close -> vendor raw close ({vendor_src}); no other column is written")
    print(f"  {'date':10s} {'as published':>18s} {'vendor raw close':>17s} {'div bp':>8s}  basis of the stored value")
    for c in plan:
        print(f"  {c['date']:10s} {c['from_close']!r:>18} {c['to_close']!r:>17} {c['divergence_bp']:+8.2f}  {c['basis']}")


def run(argv: list[str], *, aws: Any = None, now: Callable[[], datetime] | None = None,
        vendor_fn: Callable[[str, str], dict[str, float]] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aec9613_spy_close_restate.py",
                                 description="Restate the 9 aec#9613 spy_close cells to the vendor raw close.")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--start-box", action="store_true", help="boot the stopped trading box, stop it at the end")
    ap.add_argument("--i-confirm-9613-ruled", dest="confirm", action="store_true",
                    help="Brian's attestation of the 2026-10-22 15:37Z ruling; required with --apply")
    ap.add_argument("--force-plan", action="store_true", help="override the plan-shape checks only")
    ap.add_argument("--rollback", metavar="STAMP", help="put the as-published values of run STAMP back")
    a = ap.parse_args(argv)
    now = now or (lambda: datetime.now(UTC))
    mode = "APPLY" if a.apply else "DRY-RUN"
    kind = "rollback" if a.rollback else "restate"
    stamp = now().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{BACKUP_ROOT}-{kind}-{stamp}"
    notes: list[str] = []
    print(f"=================== aec9613 spy_close {kind} — {mode} ===================")
    print(f"ruling: {RULING} ({ISSUE})")

    # Refusals that need no AWS call come first, so a refused --apply touches nothing.
    why = refusal_reason(now())
    print(f"now {now().astimezone(ZoneInfo('America/New_York')):%a %Y-%m-%d %H:%M %Z}: "
          + (f"REFUSE, {why}" if why else "OK (outside the refusal windows)"))
    if a.apply and not a.rollback and not a.confirm:
        print("NO-GO: --apply needs --i-confirm-9613-ruled (Brian's attestation of the 2026-10-22 15:37Z ruling)",
              file=sys.stderr)
        return 1
    if a.apply and why:
        print(f"NO-GO: inside a refusal window: {why}", file=sys.stderr)
        return 1

    aws = aws or Aws()
    try:
        return _run(a, aws, now, vendor_fn or laptop_vendor_closes, why, prefix, stamp, notes)
    except NoGo as e:
        print(f"NO-GO: {e}", file=sys.stderr)
        return 1
    finally:
        if getattr(aws, "started_box", False):
            try:
                aws.stop_box()
            except Exception as e:  # noqa: BLE001 -- the operator gets the exact command instead
                print(f"WARN: stop failed ({e}); run: aws ec2 stop-instances --instance-ids {INSTANCE}", file=sys.stderr)


def _run(a: argparse.Namespace, aws: Any, now: Callable[[], datetime], vendor_fn: Callable[[str, str], dict[str, float]],
         why: str | None, prefix: str, stamp: str, notes: list[str]) -> int:
    print("== 1/9 identity, pipelines idle")
    print(f"   {aws.caller_arn()}")
    busy = 0
    for sm in SFNS:
        n = aws.running_executions(sm)
        print(f"   {sm} RUNNING={n}")
        busy += n
    if a.apply and busy:
        raise NoGo("a trading/EOD pipeline is RUNNING")

    print("== 2/9 s3 state (eod_pnl.csv and trades_latest.db, pinned by version id) and the ruled order")
    csv_etag, csv_ver = aws.head(CSV_KEY)
    db_etag, db_ver = aws.head(LATEST_KEY)
    pre_body = aws.get(CSV_KEY, csv_ver)
    csvmeta = {"eod_pnl.csv": {"etag": csv_etag, "version_id": csv_ver},
               "trades_latest.db": {"etag": db_etag, "version_id": db_ver}}
    _, rows = parse_csv(pre_body)
    by = {x["date"]: x for x in rows}
    print(f"   eod_pnl.csv version {csv_ver} etag {csv_etag}: {len(rows)} rows {rows[0]['date']} -> {rows[-1]['date']}")
    print(f"   trades_latest.db version {db_ver} etag {db_etag}")
    order_hard: list[str] = []
    records = [k for k in aws.list_keys(RESTATE_RECORD_PREFIX) if k.endswith("/as-published.json")]
    print(f"   aec#9613 restatement record: {records[-1] if records else 'NONE'}")
    print(f"   aec#11087 03-12 row: {'present' if BACKFILL_DATE in by else 'ABSENT'}")
    if not a.rollback:
        if not records:
            order_hard.append("no aec#9613 restatement as-published record in S3: the restatement has not been applied. "
                              "Ruled order: the restatement, then the 03-12 backfill, then this script")

    if a.rollback:
        rec_key = f"{BACKUP_ROOT}-restate-{a.rollback}/as-published.json"
        print(f"== 3/9 rollback record s3://{BUCKET}/{rec_key}")
        try:
            record = json.loads(aws.get(rec_key))
        except Exception as e:  # noqa: BLE001 -- a missing record is the refusal
            raise NoGo(f"no as-published record under that STAMP: {e}") from e
        laptop_vendor: dict[str, float] | None = None
    else:
        print("== 3/9 laptop vendor closes (independent of the box)")
        laptop_vendor = None
        if why:
            print("   skipped: inside a refusal window (the Polygon key is shared with the trading day)")
            notes.append("vendor closes not evaluated: run outside the refusal windows")
        else:
            try:
                laptop_vendor = vendor_fn(rows[0]["date"], rows[-1]["date"])
                print(f"   laptop vendor: {len(laptop_vendor)} sessions {min(laptop_vendor)} -> {max(laptop_vendor)}")
            except Exception as e:  # noqa: BLE001 -- apply refuses without it
                print(f"   laptop vendor fetch FAILED: {type(e).__name__}: {e}")
                notes.append("laptop vendor fetch FAILED (apply would refuse)")

    print("== 4/9 trading box")
    state = aws.box_state()
    print(f"   {INSTANCE} is {state}")
    usable = False
    if why:
        print("   box commands skipped: inside a refusal window")
    elif state == "running":
        usable = True
    elif a.start_box:
        if state != "stopped":
            raise NoGo(f"box is {state}; wait until it is stopped or running")
        print(f"   starting {INSTANCE} (--start-box)")
        aws.started_box = True
        aws.start_box()
        usable = True
    else:
        notes.append(f"box-side check SKIPPED (box {state}); rerun with --start-box to include it")
    if a.apply and not usable:
        raise NoGo(f"box is {state}; rerun with --start-box")

    box: dict[str, Any] = {}
    if usable:
        phase = "state" if a.rollback else "check"
        print(f"== 5/9 box {phase}: read-only state" + ("" if a.rollback else ", backfill value, vendor closes, simulated gates"))
        box = aws.ssm_run([phase, json.dumps(list(TARGETS))])
        if "fatal" in box:
            raise NoGo(f"box check failed: {box['fatal']}")
        print(f"   box git HEAD {box.get('git_head')}  db {box.get('db_path')}  eod_pnl rows {box.get('n_rows')} "
              f"{box.get('first')} -> {box.get('last')}  03-12 row {'PRESENT' if box.get('has_0312') else 'absent'}")
        print(f"   services {box.get('services')}")
        svc = box.get("services") or {}
        busy_units = [u for u in ("alpha-engine-eod-reconcile-standalone.service", "alpha-engine-trader-reconcile.service")
                      if svc.get(u) in ("active", "activating", "reloading")]
        if svc.get("boot-pull.service") == "activating":
            busy_units.append("boot-pull.service")
        if busy_units:
            raise NoGo(f"box units busy: {busy_units}")
    else:
        print("== 5/9 box check: skipped")

    hard = list(order_hard)
    if box:
        if box.get("digest") != digest(rows):
            hard.append("box trades.db eod_pnl (date, NAV, returns, spy_close) != s3 eod_pnl.csv")
        if box.get("csv_sha256") != sha(pre_body):
            hard.append("the box's export of eod_pnl is not byte-identical to s3 eod_pnl.csv (a re-export would rewrite other rows)")

    if a.rollback:
        print("== 6/9 rollback plan")
        plan = rollback_plan(record)
        for c in plan:
            cur = fnum(by.get(c["date"], {}).get("spy_close"))
            if cur != c["from_close"]:
                hard.append(f"{c['date']}: spy_close is {cur!r}, not the restated {c['from_close']!r} (compare-and-set)")
            print(f"  {c['date']}  spy_close {c['from_close']!r} -> {c['to_close']!r}")
        for x in hard:
            print("NO-GO (not overridable):", x)
        if not a.apply:
            print("\n=================== SUMMARY (ROLLBACK DRY RUN, nothing written) ===================")
            for n in notes:
                print("note:", n)
            print(f"to write: {SCRIPT} --rollback {a.rollback} --apply --start-box")
            return 1 if (hard and box) else 0
        if hard:
            raise NoGo("rollback preconditions failed (above)")
        record_out = None
        forced: list[str] = []
    else:
        print("== 6/9 expected pre-restatement values, vendor closes, plan, gates")
        expected, h = expected_pre_values(by, box.get("arctic_0312") if box else None)
        hard += h
        if box.get("arctic_0312_error"):
            print(f"   box _spy_close({BACKFILL_DATE}) failed: {box['arctic_0312_error']}")
        elif box:
            print(f"   backfill value for {BACKFILL_DATE} (eod_reconcile._spy_close on the box, ArcticDB macro/SPY): "
                  f"{box.get('arctic_0312')!r}; stored: {by.get(BACKFILL_DATE, {}).get('spy_close', 'NO ROW')}")
        box_vendor = box.get("vendor_closes") if box else None
        if box_vendor and laptop_vendor is not None:
            dif = cross_check(box_vendor, laptop_vendor)
            print(f"   cross-check box vs laptop vendor closes on the {len(TARGETS)} target dates: "
                  + ("identical" if not dif else f"{len(dif)} differ"))
            hard += dif
        vendor = box_vendor or laptop_vendor or {}
        src = "box: executor.pnl_measurement_backfill.fetch_vendor_closes (Polygon)" if box_vendor else \
            ("laptop: the same fetch_vendor_closes" if laptop_vendor else "NOT EVALUATED")
        print(f"   vendor raw closes ({src}):")
        for d in TARGETS:
            stored = by.get(d, {}).get("spy_close", "NO ROW")
            print(f"     {d}  vendor {vendor.get(d)!r:>18}  stored {stored}")
        plan, h, soft = build_plan(expected, vendor) if vendor else ([], [], [])
        hard += h
        if plan:
            _print_plan(plan, src)
        if box.get("gates_pre"):
            pre, post = box["gates_pre"], box["gates_post_simulated"]
            print("\nBENCHMARK-LEG GATES on the box (full history); after = restatement applied in memory:")
            for k in sorted(pre):
                print(f"  {k:28s} {json.dumps(pre[k])[:80]:80s} -> {json.dumps(post.get(k))[:80]}")
            soft += gate_findings(pre, post)
        print()
        for x in hard:
            print("NO-GO (not overridable):", x)
        for x in soft:
            print(("FORCED past: " if a.force_plan else "NO-GO: ") + x)
        evaluated = bool(vendor) and bool(box)
        go = evaluated and not hard and (not soft or a.force_plan) and len(plan) == len(TARGETS)
        print("PLAN VERDICT:", "GO" if go else ("NO-GO" if evaluated or hard else "NOT EVALUATED (needs the box and the vendor)"))
        if not a.apply:
            print("\n=================== SUMMARY (DRY RUN, nothing written) — paste back ===================")
            print(f"aec9613 spy_close dry run {now():%Y-%m-%dT%H:%MZ}: {len(plan)}/{len(TARGETS)} cells planned; "
                  f"time window {'REFUSED' if why else 'OK'}; --i-confirm-9613-ruled {'given' if a.confirm else 'NOT given'}")
            for c in plan:
                print(f"  {c['date']} {c['from_close']!r} -> {c['to_close']!r} ({c['divergence_bp']:+.2f}bp)")
            for n in notes:
                print("note:", n)
            print(f"verdict: {'GO' if go else 'NO-GO'}")
            print(f"to write: {SCRIPT} --apply --start-box --i-confirm-9613-ruled   (outside the refusal windows)")
            return 1 if (hard or (evaluated and not go)) else 0
        if not go:
            raise NoGo("plan gates failed (above)")
        if laptop_vendor is None:
            raise NoGo("the laptop vendor closes are required for --apply (cross-check)")
        forced = soft if a.force_plan else []
        record_out = as_published_record(plan, box, csvmeta, prefix, forced)

    payload = json.dumps({"direction": "rollback" if a.rollback else "apply", "expect_digest": box["digest"],
                          "csv_etag": csv_etag, "latest_etag": db_etag,
                          "plan": [{"date": c["date"], "from_close": c["from_close"], "to_close": c["to_close"]}
                                   for c in plan]}).encode()

    print(f"== 7/9 backups and as-published record (verified before any write) -> s3://{BUCKET}/{prefix}/")
    if refusal_reason(now()):
        raise NoGo("time window closed during the run")
    manifest = []
    for key, ver in ((CSV_KEY, csv_ver), (LATEST_KEY, db_ver)):
        body = aws.get(key, ver)
        if key == CSV_KEY and body != pre_body:
            raise NoGo("eod_pnl.csv backup differs from the copy the plan was made from")
        bk = f"{prefix}/{key.replace('/', '__')}"
        aws.put(bk, body)
        if sha(aws.get(bk)) != sha(body):
            raise NoGo(f"backup sha mismatch for {key}")
        manifest.append(f"{key} version={ver} sha256={sha(body)}")
        print("  ", manifest[-1])
    print("   box trades.db online backup")
    bb = aws.ssm_run(["backup", prefix])
    if "sha256" not in bb or bb.get("integrity") != "ok" or bb.get("eod_rows") != box.get("n_rows"):
        raise NoGo(f"box backup failed or does not verify: {bb}")
    if sha(aws.get(f"{prefix}/box-trades.db")) != bb["sha256"]:
        raise NoGo("box backup sha mismatch")
    manifest.append(f"box-trades.db path={bb['backup_path']} sha256={bb['sha256']}")
    print("  ", manifest[-1])
    rec_sha = None
    if record_out is not None:
        rec = json.dumps(record_out, indent=1).encode()
        aws.put(f"{prefix}/as-published.json", rec, "application/json")
        rec_sha = sha(rec)
        if sha(aws.get(f"{prefix}/as-published.json")) != rec_sha:
            raise NoGo("as-published record sha mismatch")
        manifest.append(f"as-published.json sha256={rec_sha}")
        print("  ", manifest[-1])
    aws.put(f"{prefix}/payload.json", payload, "application/json")
    aws.put(f"{prefix}/MANIFEST.txt", ("\n".join(manifest) + "\n").encode(), "text/plain")

    print("== 8/9 WRITE on the box: one transaction of compare-and-set UPDATEs, then conditional re-export")
    if refusal_reason(now()):
        raise NoGo("time window closed during the run")
    res = aws.ssm_run(["apply"], payload)
    for k in ("refused", "fatal", "csv_put_error", "latest_put_error"):
        if k in res:
            print(f"   box {k}: {res[k]}")
    print(f"   committed rows: {res.get('committed')}; wrong: {res.get('db_rows_wrong')}; other rows unchanged: "
          f"{res.get('db_other_rows_unchanged')}; target rows' other columns unchanged: {res.get('db_target_other_columns_unchanged')}")
    print(f"   eod_pnl.csv put: {res.get('csv_put')}\n   trades_latest.db put: {res.get('latest_put')}")
    if "refused" in res:
        raise NoGo("box refused before writing (nothing written)")
    if "fatal" in res and "committed" not in res:
        raise NoGo("box failed before the commit (nothing written)")

    print("== 9/9 verify")
    verify = "OK"
    if "csv_put_error" in res or res.get("db_rows_wrong") or not res.get("db_other_rows_unchanged") \
            or not res.get("db_target_other_columns_unchanged"):
        verify = "FAILED"
    new_ver = (res.get("csv_put") or {}).get("version_id")
    _, now_ver = aws.head(CSV_KEY)
    if not new_ver or new_ver != now_ver:
        print(f"   current eod_pnl.csv version {now_ver} != the one written ({new_ver})", file=sys.stderr)
        verify = "FAILED"
    bad = postcheck(pre_body, aws.get(CSV_KEY, now_ver), plan)
    for x in bad[:20]:
        print("  -", x, file=sys.stderr)
    if bad:
        verify = "FAILED"
    print(f"   post-write eod_pnl.csv: {len(plan) - len([x for x in bad if 'planned' in x])}/{len(plan)} cells hold the "
          f"planned values; other rows byte-identical: {not any('non-target' in x for x in bad)}")
    if "latest_put_error" in res:
        notes.append("trades_latest.db not re-exported (see above); the box's next backup refreshes it")

    print(f"\n=================== SUMMARY ({'ROLLBACK' if a.rollback else 'RESTATE'} APPLY) — paste back ===================")
    print(f"aec9613 spy_close {'rollback' if a.rollback else 'restate'} applied {now():%Y-%m-%dT%H:%MZ} by {aws.caller_arn()}")
    print(f"ruling: {RULING}")
    print(f"verification: {verify}")
    for c in plan:
        print(f"  {c['date']} spy_close {c['from_close']!r} -> {c['to_close']!r}")
    if rec_sha:
        print(f"as-published record: s3://{BUCKET}/{prefix}/as-published.json sha256={rec_sha}")
    print(f"backups: s3://{BUCKET}/{prefix}/ (MANIFEST.txt; box copy also at {bb['backup_path']})")
    print(f"eod_pnl.csv version: {csv_ver} -> {now_ver}")
    for n in notes:
        print("note:", n)
    if not a.rollback:
        print("ROLLBACK (only if Brian decides to; compare-and-set back to the as-published values):")
        print(f"  {SCRIPT} --rollback {stamp} --apply --start-box")
        print(f"  full restore instead: stop the executor, restore the box db from {bb['backup_path']}, then")
        print(f"  aws s3api copy-object --bucket {BUCKET} --key {CSV_KEY} --copy-source '{BUCKET}/{CSV_KEY}?versionId={csv_ver}'")
    return 0 if verify == "OK" else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
