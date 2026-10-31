"""Tests for scripts/aec9613_spy_close_restate.py and its box driver (alpha-engine-config-I9613).

The script restates the 8 pre-2026-03-20 ``eod_pnl.spy_close`` cells (dividend-back-adjusted, -27.2bp)
and the 2026-03-12 backfill cell (ArcticDB-adjusted, -77.5bp) to the vendor raw close. It is a red-line-4
production write Brian runs, so what is pinned here is the refusals: a dry run writes nothing, a cell that
does not hold its expected pre-restatement value stops the run, and so do the time window, a missing
``--i-confirm-9613-ruled`` and a 2026-03-12 row the backfill has not written yet.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import aec9613_spy_close_box as box_driver  # noqa: E402
from scripts import aec9613_spy_close_restate as s  # noqa: E402

OK_NOW = datetime(2026, 10, 23, 3, 0, tzinfo=UTC)  # Thu 23:00 ET: outside every refusal window
MARKET_NOW = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)  # Thu 11:00 ET
ARCTIC_0312 = 660.9012451171875
RESTATE_RECORD = "ops/daily-summaries/backups/aec9613-spy-restate-20261023T020000Z/as-published.json"
HDR = ["date", "portfolio_nav", "daily_return_pct", "spy_return_pct", "daily_alpha_pct", "positions_snapshot", "spy_close"]


def _vendor() -> dict[str, float]:
    v = {d: x / (1 - 27.2e-4) for d, x in s.PRE_0320_AS_PUBLISHED.items()}
    v["2026-03-12"] = ARCTIC_0312 / (1 - 77.5e-4)
    v["2026-03-20"] = 648.5700073242188
    v["2026-03-23"] = 655.3800048828125
    return v


def _csv(with_0312: bool = True, override: dict[str, str] | None = None) -> bytes:
    cells = {d: repr(x) for d, x in s.PRE_0320_AS_PUBLISHED.items()}
    if with_0312:
        cells["2026-03-12"] = repr(ARCTIC_0312)
    cells["2026-03-20"] = "648.5700073242188"
    cells["2026-03-23"] = "655.3800048828125"
    cells.update(override or {})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HDR)
    for i, d in enumerate(sorted(cells)):
        w.writerow([d, f"{1007985.6 + i}", "0.01", "0.1", "-0.09", "{}", cells[d]])
    return buf.getvalue().encode()


def _db_from_csv(body: bytes) -> sqlite3.Connection:
    _, rows = s.parse_csv(body)
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, portfolio_nav REAL, daily_return_pct REAL, "
                 "spy_return_pct REAL, daily_alpha_pct REAL, positions_snapshot TEXT, spy_close REAL)")
    for r in rows:
        conn.execute("INSERT INTO eod_pnl VALUES (?,?,?,?,?,?,?)",
                     (r["date"], float(r["portfolio_nav"]), float(r["daily_return_pct"]), float(r["spy_return_pct"]),
                      float(r["daily_alpha_pct"]), r["positions_snapshot"], float(r["spy_close"])))
    return conn


class FakeAws:
    """Records every call; ``writes`` holds every call that changes AWS state."""

    def __init__(self, body: bytes, *, box_state: str = "running", arctic: float | None = ARCTIC_0312,
                 records: tuple[str, ...] = (RESTATE_RECORD,)) -> None:
        self.store: dict[str, bytes] = {s.CSV_KEY: body, s.LATEST_KEY: b"sqlite-bytes"}
        self.store.update(dict.fromkeys(records, b"{}"))
        self.versions = {s.CSV_KEY: "v-csv-1", s.LATEST_KEY: "v-db-1"}
        self.state = box_state
        self.arctic = arctic
        self.started_box = False
        self.calls: list[str] = []
        self.writes: list[str] = []
        self.phases: list[str] = []
        self.db = _db_from_csv(body)

    def caller_arn(self):
        self.calls.append("sts")
        return "arn:aws:sts::711398986525:assumed-role/ne-admin/brian"

    def running_executions(self, sm):
        self.calls.append(f"sfn:{sm}")
        return 0

    def head(self, key):
        self.calls.append(f"head:{key}")
        return f'"etag-{self.versions[key]}"', self.versions[key]

    def get(self, key, version=None):
        self.calls.append(f"get:{key}")
        return self.store[key]

    def list_keys(self, prefix):
        self.calls.append(f"list:{prefix}")
        return sorted(k for k in self.store if k.startswith(prefix))

    def put(self, key, body, content_type="application/octet-stream"):
        self.writes.append(f"put:{key}")
        self.store[key] = body

    def box_state(self):
        self.calls.append("ec2:describe")
        return self.state

    def start_box(self):
        self.writes.append("ec2:start")
        self.state = "running"

    def stop_box(self):
        self.writes.append("ec2:stop")

    def ssm_run(self, args, payload=None):
        phase = args[0]
        self.phases.append(phase)
        body = self.store[s.CSV_KEY]
        _, rows = s.parse_csv(body)
        base = {"git_head": "abc123", "db_path": "/home/ec2-user/alpha-engine/trades.db",
                "services": {"alpha-engine-eod-reconcile-standalone.service": "inactive",
                             "alpha-engine-trader-reconcile.service": "inactive", "boot-pull.service": "inactive"}}
        if phase in ("check", "state"):
            v = _vendor()
            base.update(n_rows=len(rows), first=rows[0]["date"], last=rows[-1]["date"],
                        digest=box_driver.digest(box_driver.all_rows(self.db)),
                        csv_sha256=hashlib.sha256(body).hexdigest(), has_0312=any(r["date"] == "2026-03-12" for r in rows),
                        arctic_0312=self.arctic, vendor_closes={d: v.get(d) for d in s.TARGETS},
                        gates_pre={"close_divergence_dates": list(s.TARGETS), "close_divergence_n": 9,
                                   "anchor_breaches_n": 10, "bench_offenders_n": 3, "bench_closes": False},
                        gates_post_simulated={"close_divergence_dates": [], "close_divergence_n": 0,
                                              "anchor_breaches_n": 1, "bench_offenders_n": 0, "bench_closes": True})
            return base
        self.writes.append(f"ssm:{phase}")
        if phase == "backup":
            self.store[f"{args[1]}/box-trades.db"] = b"box-db"
            base.update(sha256=hashlib.sha256(b"box-db").hexdigest(), integrity="ok", eod_rows=len(rows),
                        backup_path="/home/ec2-user/aec9613/trades-pre.db")
            return base
        pl = json.loads(payload)
        res = box_driver.apply_cas(self.db, pl["plan"], pl["expect_digest"])
        if "refused" not in res:  # the box re-exports: same text, spy_close cells re-rendered
            new = {r["date"]: r["spy_close"] for r in box_driver.all_rows(self.db)}
            hdr, rows = s.parse_csv(body)
            i = hdr.index("spy_close")
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(hdr)
            for r in rows:
                raw = list(r["__raw"])
                if float(raw[i]) != new[r["date"]]:
                    raw[i] = repr(new[r["date"]])
                w.writerow(raw)
            self.store[s.CSV_KEY] = buf.getvalue().encode()
            self.versions[s.CSV_KEY] = "v-csv-2"
            res["csv_put"] = {"version_id": "v-csv-2"}
        base.update(res)
        return base


def _run(fake, *argv, now=OK_NOW):
    return s.run(list(argv), aws=fake, now=lambda: now, vendor_fn=lambda a, b: _vendor())


# ───────────────────────────── time window ─────────────────────────────

@pytest.mark.parametrize(("utc", "refused"), [
    (datetime(2026, 10, 22, 1, 29, tzinfo=UTC), True),    # Wed 21:29 ET: postclose / EOD timer
    (datetime(2026, 10, 22, 1, 30, tzinfo=UTC), False),   # the 01:30Z opening
    (datetime(2026, 10, 22, 4, 50, tzinfo=UTC), True),    # 00:50 ET stop-trading backstop
    (datetime(2026, 10, 22, 10, 59, tzinfo=UTC), False),  # 06:59 ET
    (datetime(2026, 10, 22, 11, 0, tzinfo=UTC), True),    # 07:00 ET preopen buffer: the 11:00Z close
    (datetime(2026, 10, 22, 15, 0, tzinfo=UTC), True),    # market hours
    (datetime(2026, 10, 24, 15, 0, tzinfo=UTC), False),   # Saturday
    # EST from Sun 2026-11-01: the same wall-clock windows sit an hour later in UTC.
    (datetime(2026, 11, 2, 11, 30, tzinfo=UTC), False),   # Mon 06:30 EST (07:30 under EDT)
    (datetime(2026, 11, 2, 12, 0, tzinfo=UTC), True),     # Mon 07:00 EST preopen buffer
    (datetime(2026, 11, 2, 20, 45, tzinfo=UTC), True),    # Mon 15:45 EST, in session
    (datetime(2026, 11, 3, 2, 29, tzinfo=UTC), True),     # Mon 21:29 EST: postclose / EOD timer
    (datetime(2026, 11, 3, 2, 30, tzinfo=UTC), False),    # the 02:30Z opening under EST
])
def test_refusal_window_is_the_0130_1100z_run_window(utc, refused):
    assert (s.refusal_reason(utc) is not None) is refused


def test_apply_inside_the_window_is_refused_before_any_aws_call(capsys):
    fake = FakeAws(_csv())
    rc = _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled", now=MARKET_NOW)
    assert rc == 1
    assert fake.calls == [] and fake.writes == []
    assert "refusal window" in capsys.readouterr().err


def test_dry_run_inside_the_window_skips_vendor_and_box(capsys):
    fake = FakeAws(_csv())
    assert _run(fake, "--start-box", now=MARKET_NOW) == 1  # 03-12 value undefined without the box
    assert fake.writes == [] and fake.phases == []
    assert "NOT EVALUATED" in capsys.readouterr().out


# ───────────────────────────── confirm flag ─────────────────────────────

def test_apply_without_the_confirm_flag_is_refused_before_any_aws_call(capsys):
    fake = FakeAws(_csv())
    rc = _run(fake, "--apply", "--start-box")
    assert rc == 1
    assert fake.calls == [] and fake.writes == []
    assert "--i-confirm-9613-ruled" in capsys.readouterr().err


# ───────────────────────────── dry run ─────────────────────────────

def test_dry_run_makes_no_writes_and_prints_the_vendor_closes(capsys):
    fake = FakeAws(_csv())
    assert _run(fake) == 0
    out = capsys.readouterr().out
    assert fake.writes == []
    assert fake.phases == ["check"]
    for d, v in _vendor().items():
        if d in s.TARGETS:
            assert repr(v) in out
    assert "9/9 cells planned" in out and "verdict: GO" in out
    assert "--apply --start-box --i-confirm-9613-ruled" in out


def test_dry_run_with_start_box_only_boots_and_stops_the_box():
    fake = FakeAws(_csv(), box_state="stopped")
    assert _run(fake, "--start-box") == 0
    assert fake.writes == ["ec2:start", "ec2:stop"]
    assert fake.phases == ["check"]
    assert not any(w.startswith(("put:", "ssm:")) for w in fake.writes)


# ───────────────────────────── compare-and-set ─────────────────────────────

def test_a_pre_0320_cell_that_moved_refuses_the_whole_run(capsys):
    fake = FakeAws(_csv(override={"2026-03-16": "668.9"}))
    rc = _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled")
    assert rc == 1
    assert fake.writes == []
    assert "2026-03-16: spy_close is '668.9'" in capsys.readouterr().out


def test_the_0312_cell_must_hold_the_backfill_value(capsys):
    fake = FakeAws(_csv(override={"2026-03-12": "666.06"}))  # not what eod_reconcile._spy_close gave
    assert _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled") == 1
    assert fake.writes == []
    assert "the value the backfill wrote it from (compare-and-set)" in capsys.readouterr().out


def test_box_apply_rolls_back_every_row_when_one_cell_misses():
    conn = _db_from_csv(_csv())
    before = box_driver.all_rows(conn)
    v = _vendor()
    plan = [{"date": d, "from_close": s.PRE_0320_AS_PUBLISHED.get(d, ARCTIC_0312), "to_close": v[d]} for d in s.TARGETS]
    plan[-1]["from_close"] = 1.0  # 03-19 no longer holds this value
    res = box_driver.apply_cas(conn, plan, box_driver.digest(before))
    assert "refused" in res and "2026-03-19" in res["refused"]
    assert box_driver.all_rows(conn) == before


def test_box_apply_refuses_when_the_table_moved_since_the_check():
    conn = _db_from_csv(_csv())
    res = box_driver.apply_cas(conn, [], "not-the-digest")
    assert "digest differs" in res["refused"]


def test_box_apply_writes_only_spy_close():
    conn = _db_from_csv(_csv())
    before = {r["date"]: r for r in box_driver.all_rows(conn)}
    v = _vendor()
    plan = [{"date": d, "from_close": s.PRE_0320_AS_PUBLISHED.get(d, ARCTIC_0312), "to_close": v[d]} for d in s.TARGETS]
    res = box_driver.apply_cas(conn, plan, box_driver.digest(list(before.values())))
    assert res["committed"] == 9 and res["db_rows_wrong"] == []
    assert res["db_other_rows_unchanged"] and res["db_target_other_columns_unchanged"]
    after = {r["date"]: r for r in box_driver.all_rows(conn)}
    for d in s.TARGETS:
        assert after[d]["spy_close"] == v[d]
    assert after["2026-03-20"] == before["2026-03-20"]


# ───────────────────────────── 03-12 sequencing ─────────────────────────────

def test_the_0312_cell_is_absent_before_the_backfill():
    _, rows = s.parse_csv(_csv(with_0312=False))
    expected, hard = s.expected_pre_values({r["date"]: r for r in rows}, ARCTIC_0312)
    assert "2026-03-12" not in expected
    assert any("aec#11087 backfill has not run" in h for h in hard)
    plan, _, _ = s.build_plan(expected, _vendor())
    assert [c["date"] for c in plan] == sorted(s.PRE_0320_AS_PUBLISHED)


def test_apply_before_the_backfill_is_refused_and_writes_nothing(capsys):
    fake = FakeAws(_csv(with_0312=False))
    assert _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled") == 1
    assert fake.writes == []
    assert "aec#11087 03-12 row: ABSENT" in capsys.readouterr().out


def test_apply_before_the_9613_restatement_is_refused(capsys):
    fake = FakeAws(_csv(), records=())
    assert _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled") == 1
    assert fake.writes == []
    assert "the restatement has not been applied" in capsys.readouterr().out


# ───────────────────────────── plan and apply ─────────────────────────────

def test_plan_divergences_are_the_diagnosed_ones():
    _, rows = s.parse_csv(_csv())
    expected, hard = s.expected_pre_values({r["date"]: r for r in rows}, ARCTIC_0312)
    plan, h, soft = s.build_plan(expected, _vendor())
    assert hard == [] and h == [] and soft == []
    by = {c["date"]: c for c in plan}
    assert by["2026-03-19"]["divergence_bp"] == pytest.approx(-27.2)
    assert by["2026-03-12"]["divergence_bp"] == pytest.approx(-77.5)


def test_laptop_and_box_digest_agree():
    body = _csv()
    _, rows = s.parse_csv(body)
    assert s.digest(rows) == box_driver.digest(box_driver.all_rows(_db_from_csv(body)))


def test_apply_backs_up_first_then_writes_exactly_the_nine_cells(capsys):
    fake = FakeAws(_csv())
    pre = fake.store[s.CSV_KEY]
    assert _run(fake, "--apply", "--start-box", "--i-confirm-9613-ruled") == 0
    out = capsys.readouterr().out
    assert "verification: OK" in out
    i_apply = fake.writes.index("ssm:apply")
    puts = [w for w in fake.writes[:i_apply] if w.startswith("put:")]
    assert any(w.endswith("trades__eod_pnl.csv") for w in puts)
    assert any(w.endswith("trades__trades_latest.db") for w in puts)
    assert fake.writes.index("ssm:backup") < i_apply
    rec_key = next(w[4:] for w in puts if w.endswith("as-published.json"))
    rec = json.loads(fake.store[rec_key])
    assert rec["ruling"] == "Brian ruled Restate 2026-10-22 15:37Z, card aec9613 spy_close"
    assert [r["date"] for r in rec["rows"]] == list(s.TARGETS)
    assert s.postcheck(pre, fake.store[s.CSV_KEY], [
        {"date": r["date"], "to_close": r["restated"]["spy_close"]} for r in rec["rows"]]) == []


def test_box_check_phase_runs_the_real_gates_on_a_db(tmp_path, monkeypatch):
    """The driver's read-only check against a real sqlite file, with the vendor and ArcticDB stubbed:
    the executor's own gates see 9 close divergences now and none once the plan is simulated."""
    from executor import eod_reconcile
    from executor import pnl_measurement_backfill as pmb

    db = tmp_path / "trades.db"
    disk = sqlite3.connect(db)
    _db_from_csv(_csv()).backup(disk)
    disk.close()
    v = _vendor()
    monkeypatch.setattr(pmb, "fetch_vendor_closes", lambda a, b: dict(v))
    monkeypatch.setattr(pmb, "_fetch_spy_events", lambda start: [])
    monkeypatch.setattr(eod_reconcile, "_spy_close", lambda d, cfg=None: ARCTIC_0312)
    out: dict = {}
    box_driver.phase_check({"db_path": str(db), "trades_bucket": "b"}, list(s.TARGETS), out, full=True)
    assert out["has_0312"] and out["arctic_0312"] == ARCTIC_0312
    assert out["cells"]["2026-03-19"] == s.PRE_0320_AS_PUBLISHED["2026-03-19"]
    assert out["vendor_closes"] == {d: v[d] for d in s.TARGETS}
    assert out["gates_pre"]["close_divergence_dates"] == list(s.TARGETS)
    assert out["gates_post_simulated"]["close_divergence_n"] == 0
    assert s.gate_findings(out["gates_pre"], out["gates_post_simulated"]) == []
    _, rows = s.parse_csv(_csv())
    assert out["digest"] == s.digest(rows)
