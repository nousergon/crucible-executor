"""Fill reconciliation — rows logged before their broker fill arrived
(alpha-engine-config-I10800).

The 2026-09-14 shape, verbatim from the daemon log: a REDUCE 775 PBF market
sell placed at 13:30:37Z, still ``Submitted`` when the 30 s poll window
closed, logged as ``Working`` with the $77.33 estimate standing in for the
fill; nine executions at $74.00 arrived 13:31:48–13:32:xx and were never
written back. The rotation sleeve then priced the sale at $77.33 and the
residual bounds gate failed the postclose pipeline on the $2,581 gap.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from executor.fill_reconciliation import (
    NON_TERMINAL_STATUSES,
    fills_by_order_from_ib,
    parse_daemon_log_executions,
    reconcile_from_daemon_log_file,
    reconcile_from_ib,
    reconcile_unfilled_trades,
    sweep_unresolved_sessions,
    unresolved_session_dates,
    unresolved_trades,
)
from executor.trade_logger import init_db, log_trade


@pytest.fixture
def conn(tmp_path):
    return init_db(str(tmp_path / "trades.db"))


def _log(conn, **over):
    row = {
        "date": "2026-09-14",
        "ticker": "PBF",
        "action": "REDUCE",
        "shares": 775,
        "price_at_order": 77.33,
        "fill_price": None,
        "fill_time": None,
        "filled_shares": None,
        "ib_order_id": 690,
        "status": "Working",
        "exit_reason": "optimizer_scale_down",
    }
    row.update(over)
    return log_trade(conn, row)


def _ex(order_id, symbol, side, shares, price, exec_id, commission=None, t="2026-09-14T13:31:48+00:00"):
    return {
        "symbol": symbol, "side": side, "shares": float(shares), "price": float(price),
        "time": t, "commission": commission, "exec_id": exec_id,
    }


# ── The 2026-09-14 case ───────────────────────────────────────────────────────


class TestReconcileUnfilled:
    def test_working_reduce_is_patched_to_the_broker_vwap(self, conn):
        tid = _log(conn)
        fills = {690: [
            _ex(690, "PBF", "SLD", 100, 74.0, "a", commission=0.5),
            _ex(690, "PBF", "SLD", 675, 74.0, "b", commission=3.0, t="2026-09-14T13:32:10+00:00"),
        ]}
        res = reconcile_unfilled_trades(conn, "2026-09-14", fills)
        assert len(res["patched"]) == 1 and not res["unresolved"]
        row = conn.execute(
            "SELECT status, fill_price, filled_shares, fill_time, commission_usd FROM trades WHERE trade_id=?",
            (tid,),
        ).fetchone()
        assert row == ("Filled", 74.0, 775, "2026-09-14T13:32:10+00:00", 3.5)
        assert unresolved_trades(conn, "2026-09-14") == []

    def test_partial_broker_quantity_stays_partial_fill(self, conn):
        _log(conn)
        res = reconcile_unfilled_trades(conn, "2026-09-14", {690: [_ex(690, "PBF", "SLD", 400, 74.0, "a")]})
        assert res["patched"][0]["after"]["status"] == "PartialFill"
        assert res["patched"][0]["after"]["filled_shares"] == 400
        # still eligible for a later, fuller reconciliation
        assert len(unresolved_trades(conn, "2026-09-14")) == 1

    def test_no_executions_for_the_order_is_reported_not_guessed(self, conn):
        _log(conn)
        res = reconcile_unfilled_trades(conn, "2026-09-14", {999: [_ex(999, "PBF", "SLD", 775, 74.0, "a")]})
        assert res["patched"] == []
        assert [r["ticker"] for r in res["unresolved"]] == ["PBF"]
        assert conn.execute("SELECT status, fill_price FROM trades").fetchone() == ("Working", None)

    def test_symbol_and_side_mismatch_does_not_cross_wire_orders(self, conn):
        _log(conn)
        wrong = {690: [_ex(690, "DUOL", "SLD", 775, 147.3, "a"), _ex(690, "PBF", "BOT", 775, 74.0, "b")]}
        res = reconcile_unfilled_trades(conn, "2026-09-14", wrong)
        assert res["patched"] == [] and len(res["unresolved"]) == 1

    def test_terminal_rows_are_never_touched(self, conn):
        _log(conn, status="Filled", fill_price=77.33, filled_shares=775)
        _log(conn, status="Rejected", ib_order_id=691)
        res = reconcile_unfilled_trades(conn, "2026-09-14", {690: [_ex(690, "PBF", "SLD", 775, 74.0, "a")]})
        assert res == {"patched": [], "unresolved": []}

    def test_idempotent_second_pass_is_a_no_op(self, conn):
        _log(conn)
        fills = {690: [_ex(690, "PBF", "SLD", 775, 74.0, "a")]}
        reconcile_unfilled_trades(conn, "2026-09-14", fills)
        assert reconcile_unfilled_trades(conn, "2026-09-14", fills) == {"patched": [], "unresolved": []}

    def test_dry_run_reports_without_writing(self, conn):
        _log(conn)
        res = reconcile_unfilled_trades(
            conn, "2026-09-14", {690: [_ex(690, "PBF", "SLD", 775, 74.0, "a")]}, dry_run=True
        )
        assert res["patched"][0]["after"]["fill_price"] == 74.0
        assert conn.execute("SELECT status FROM trades").fetchone() == ("Working",)

    def test_other_dates_are_out_of_scope(self, conn):
        _log(conn, date="2026-09-10", ib_order_id=650, shares=76)
        res = reconcile_unfilled_trades(conn, "2026-09-14", {650: [_ex(650, "PBF", "SLD", 76, 78.0, "a")]})
        assert res == {"patched": [], "unresolved": []}

    def test_all_non_terminal_statuses_are_eligible(self, conn):
        for i, st in enumerate(NON_TERMINAL_STATUSES):
            _log(conn, status=st, ib_order_id=700 + i, ticker=f"T{i}")
        fills = {700 + i: [_ex(700 + i, f"T{i}", "SLD", 775, 10.0, f"e{i}")] for i in range(len(NON_TERMINAL_STATUSES))}
        res = reconcile_unfilled_trades(conn, "2026-09-14", fills)
        assert len(res["patched"]) == len(NON_TERMINAL_STATUSES)

    def test_commission_none_when_no_execution_reported_one(self, conn):
        _log(conn)
        reconcile_unfilled_trades(conn, "2026-09-14", {690: [_ex(690, "PBF", "SLD", 775, 74.0, "a")]})
        assert conn.execute("SELECT commission_usd FROM trades").fetchone() == (None,)


class TestRoundtripRecompute:
    def test_sell_row_realized_columns_follow_the_real_fill(self, conn):
        entry = log_trade(conn, {
            "date": "2026-08-26", "ticker": "PBF", "action": "ENTER", "shares": 1485,
            "price_at_order": 69.75, "fill_price": 73.5183881, "status": "Filled",
        })
        tid = _log(conn, entry_trade_id=entry, realized_pnl=1175.9, realized_return_pct=5.18,
                   spy_return_during_hold=1.0, realized_alpha_pct=4.18)
        reconcile_unfilled_trades(conn, "2026-09-14", {690: [_ex(690, "PBF", "SLD", 775, 74.0, "a")]})
        rpnl, rpct, ralpha = conn.execute(
            "SELECT realized_pnl, realized_return_pct, realized_alpha_pct FROM trades WHERE trade_id=?", (tid,)
        ).fetchone()
        assert rpnl == pytest.approx((74.0 - 73.5183881) * 775)
        assert rpct == pytest.approx((74.0 / 73.5183881 - 1) * 100)
        assert ralpha == pytest.approx(rpct - 1.0)

    def test_buy_row_has_no_roundtrip_to_recompute(self, conn):
        _log(conn, action="ENTER", ticker="DOCS", ib_order_id=695, realized_pnl=None)
        res = reconcile_unfilled_trades(conn, "2026-09-14", {695: [_ex(695, "DOCS", "BOT", 775, 26.44, "a")]})
        assert "realized_pnl" not in res["patched"][0]["after"]


# ── Sources ───────────────────────────────────────────────────────────────────


def _daemon_lines():
    """Three real line shapes from the 2026-09-14 daemon log (account id redacted)."""
    exec_repr = (
        "Execution(execId='00025b49.6aa7b5bf.01.01', time=datetime.datetime(2026, 9, 14, 13, 31, 48, "
        "tzinfo=datetime.timezone.utc), acctNumber='DU0000000', exchange='NYSE', side='SLD', shares=100.0, "
        "price=74.0, permId=973605159, clientId=2, orderId=690, liquidation=0, cumQty=100.0, avgPrice=74.0, "
        "orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1)"
    )
    exec2 = exec_repr.replace("00025b49.6aa7b5bf.01.01", "0000dc8f.6b9c082e.01.01").replace(
        "cumQty=100.0", "cumQty=200.0").replace("13, 31, 48", "13, 31, 50").replace("shares=100.0", "shares=675.0")
    duol = (
        "Execution(execId='00025b44.6aa7ad30.01.01', time=datetime.datetime(2026, 9, 14, 13, 31, 47, "
        "tzinfo=datetime.timezone.utc), acctNumber='DU0000000', exchange='NASDAQ', side='SLD', shares=369.0, "
        "price=147.3182, permId=973605157, clientId=2, orderId=684, liquidation=0, cumQty=369.0, "
        "avgPrice=147.3182, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1)"
    )
    rows = [
        {"module": "wrapper", "func": "execDetails", "msg": f"execDetails {exec_repr}"},
        {"module": "wrapper", "func": "execDetails", "msg": (
            "execDetails: Fill(contract=Stock(conId=118786560, symbol='PBF', exchange='SMART', "
            f"primaryExchange='NYSE', currency='USD', localSymbol='PBF', tradingClass='PBF'), execution={exec_repr}, "
            "commissionReport=CommissionReport(execId='', commission=0.0, currency='', realizedPNL=0.0, "
            "yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2026, 9, 14, 13, 31, 48, "
            "tzinfo=datetime.timezone.utc))")},
        {"module": "wrapper", "func": "commissionReport", "msg": (
            "commissionReport: CommissionReport(execId='00025b49.6aa7b5bf.01.01', commission=0.5, "
            "currency='USD', realizedPNL=48.16, yield_=0.0, yieldRedemptionDate=0)")},
        {"module": "wrapper", "func": "execDetails", "msg": f"execDetails {exec2}"},
        {"module": "wrapper", "func": "execDetails", "msg": (
            "execDetails: Fill(contract=Stock(conId=118786560, symbol='PBF', exchange='SMART', "
            f"primaryExchange='NYSE', currency='USD', localSymbol='PBF', tradingClass='PBF'), execution={exec2}, "
            "commissionReport=CommissionReport(execId='', commission=0.0, currency='', realizedPNL=0.0, "
            "yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2026, 9, 14, 13, 31, 50, "
            "tzinfo=datetime.timezone.utc))")},
        {"module": "wrapper", "func": "execDetails", "msg": f"execDetails {duol}"},
        {"module": "daemon", "func": "run_daemon", "msg": "URGENT REDUCE PBF: SELL 775 shares | reason: optimizer_scale_down"},
    ]
    return [json.dumps({"ts": "2026-09-14T13:31:48+00:00", "level": "INFO", **r}) for r in rows]


class TestDaemonLogSource:
    def test_parses_and_dedupes_executions_by_exec_id(self):
        fills = parse_daemon_log_executions(_daemon_lines())
        assert set(fills) == {690, 684}
        pbf = sorted(fills[690], key=lambda e: e["time"])
        assert [(e["shares"], e["price"], e["symbol"], e["side"]) for e in pbf] == [
            (100.0, 74.0, "PBF", "SLD"), (675.0, 74.0, "PBF", "SLD"),
        ]
        assert pbf[0]["commission"] == 0.5 and pbf[1]["commission"] is None
        assert pbf[0]["time"] == "2026-09-14T13:31:48+00:00"
        # bare Execution line only — symbol unknown, still usable by order id
        assert fills[684][0]["symbol"] is None and fills[684][0]["price"] == 147.3182

    def test_end_to_end_repair_of_the_2026_09_14_rows(self, conn):
        _log(conn)
        _log(conn, ticker="DUOL", shares=369, price_at_order=147.25, ib_order_id=684)
        res = reconcile_unfilled_trades(conn, "2026-09-14", parse_daemon_log_executions(_daemon_lines()))
        assert {r["ticker"]: r["after"]["fill_price"] for r in res["patched"]} == {"PBF": 74.0, "DUOL": 147.3182}
        assert res["unresolved"] == []

    def test_an_execution_on_a_whole_minute_is_not_dropped(self, conn):
        # datetime repr omits a zero second: the 2026-09-14 PBF 75-share
        # execution at 13:32:00 was skipped and the row stayed PartialFill 700/775.
        whole_minute = (
            "execDetails Execution(execId='0000dc8f.6b9c094e.01.01', time=datetime.datetime(2026, 9, 14, 13, 32, "
            "tzinfo=datetime.timezone.utc), acctNumber='DU0000000', exchange='IBKRATS', side='SLD', shares=75.0, "
            "price=74.0, permId=973605159, clientId=2, orderId=690, liquidation=0)"
        )
        fills = parse_daemon_log_executions([*_daemon_lines(), whole_minute])
        minute = [e for e in fills[690] if e["exec_id"] == "0000dc8f.6b9c094e.01.01"]
        assert [(e["shares"], e["time"]) for e in minute] == [(75.0, "2026-09-14T13:32:00+00:00")]
        _log(conn, shares=850)
        res = reconcile_unfilled_trades(conn, "2026-09-14", fills)
        assert res["patched"][0]["after"]["filled_shares"] == 850
        assert res["patched"][0]["after"]["status"] == "Filled"

    def test_placeholder_commission_report_is_unknown_not_zero(self):
        fills = parse_daemon_log_executions(_daemon_lines())
        # the second PBF execution only ever appeared with the empty placeholder report
        assert [e["commission"] for e in fills[690] if e["exec_id"].startswith("0000dc8f")] == [None]


class TestIbSource:
    @staticmethod
    def _fill(order_id, symbol, side, shares, price, exec_id, commission=None, reported=True):
        return SimpleNamespace(
            contract=SimpleNamespace(symbol=symbol),
            execution=SimpleNamespace(
                orderId=order_id, side=side, shares=shares, price=price, execId=exec_id,
                time=datetime(2026, 9, 14, 13, 31, 48, tzinfo=UTC),
            ),
            commissionReport=SimpleNamespace(
                execId=exec_id if reported else "", commission=commission if reported else 0.0
            ),
        )

    def test_groups_session_fills_by_order_id(self):
        ib = SimpleNamespace(fills=lambda: [
            self._fill(690, "PBF", "SLD", 100.0, 74.0, "a", 0.5),
            self._fill(690, "PBF", "SLD", 675.0, 74.0, "b", reported=False),
            self._fill(684, "DUOL", "SLD", 369.0, 147.3182, "c", 1.2),
        ])
        fills = fills_by_order_from_ib(ib)
        assert [e["commission"] for e in fills[690]] == [0.5, None]
        assert fills[690][0]["time"] == "2026-09-14T13:31:48+00:00"
        assert fills[684][0]["symbol"] == "DUOL"

    def test_reconcile_from_ib_skips_the_broker_read_when_nothing_is_open(self, conn):
        _log(conn, status="Filled", fill_price=77.33)
        ib = SimpleNamespace(fills=lambda: (_ for _ in ()).throw(AssertionError("must not be read")))
        assert reconcile_from_ib(conn, "2026-09-14", ib) == {"patched": [], "unresolved": []}

    def test_reconcile_from_ib_patches_open_rows(self, conn):
        _log(conn)
        ib = SimpleNamespace(fills=lambda: [self._fill(690, "PBF", "SLD", 775.0, 74.0, "a", 3.5)])
        res = reconcile_from_ib(conn, "2026-09-14", ib)
        assert res["patched"][0]["after"]["status"] == "Filled"


# ── Historical sweep (alpha-engine-config-I10807) ────────────────────────────


class _FakeS3:
    """Minimal S3 double for the sweep: an object store plus a call log.

    `logs` maps run_date -> list of log lines; `markers` maps run_date -> dict.
    `fetched` records which archived logs were actually downloaded, which is
    what the negative-cache tests assert on.
    """

    def __init__(self, logs=None, markers=None):
        self.logs = logs or {}
        self.markers = dict(markers or {})
        self.fetched: list[str] = []
        self.put: list[str] = []

    class _Missing(Exception):
        pass

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg spelling
        if Key.startswith("trades/fill_reconciliation/"):
            d = Key.split("/")[-1].removesuffix(".json")
            if d not in self.markers:
                raise self._Missing(f"NoSuchKey {Key}")
            return {"Body": SimpleNamespace(read=lambda: json.dumps(self.markers[d]).encode())}
        d = Key.split("/")[-2]
        if d not in self.logs:
            raise self._Missing(f"NoSuchKey {Key}")
        self.fetched.append(d)
        lines = [line.encode() for line in self.logs[d]]
        return {"Body": SimpleNamespace(iter_lines=lambda: iter(lines))}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
        d = Key.split("/")[-1].removesuffix(".json")
        self.markers[d] = json.loads(Body.decode())
        self.put.append(d)


@pytest.fixture
def fake_s3(monkeypatch):
    holder = {}

    def _install(s3):
        holder["s3"] = s3
        monkeypatch.setattr("executor.fill_reconciliation.boto3.client", lambda *a, **k: s3)
        return s3

    return _install


class TestUnresolvedSessionDates:
    def test_distinct_prior_dates_ascending(self, conn):
        _log(conn, date="2026-09-10", ib_order_id=650)
        _log(conn, date="2026-08-31", ib_order_id=640)
        _log(conn, date="2026-08-31", ib_order_id=641, ticker="DUOL")
        _log(conn, date="2026-09-14")
        assert unresolved_session_dates(conn, before="2026-09-14") == ["2026-08-31", "2026-09-10"]

    def test_the_current_session_is_never_swept_from_the_archive(self, conn):
        _log(conn, date="2026-09-14")
        assert unresolved_session_dates(conn, before="2026-09-14") == []

    def test_terminal_rows_do_not_put_a_date_in_the_sweep(self, conn):
        _log(conn, date="2026-09-10", ib_order_id=650, status="Filled", fill_price=78.0)
        assert unresolved_session_dates(conn, before="2026-09-14") == []


class TestSweep:
    def test_repairs_a_prior_session_from_its_archived_log(self, conn, fake_s3):
        # The historical shape: the estimate was written INTO fill_price, which
        # is what made a Working row indistinguishable from a filled one.
        _log(conn, date="2026-09-11", ib_order_id=690, fill_price=77.33)
        s3 = fake_s3(_FakeS3(logs={"2026-09-11": _daemon_lines()}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")
        assert res["changed"] == ["2026-09-11"] and res["swept"] == ["2026-09-11"]
        assert conn.execute("SELECT status, fill_price FROM trades").fetchone() == ("Filled", 74.0)
        assert s3.markers["2026-09-11"]["patched"][0]["fill_price"] == 74.0
        assert s3.markers["2026-09-11"]["patched"][0]["was"] == 77.33

    def test_an_already_swept_date_is_not_downloaded_again(self, conn, fake_s3):
        _log(conn, date="2026-09-11", ib_order_id=999)  # no execution will ever match
        s3 = fake_s3(_FakeS3(logs={"2026-09-11": _daemon_lines()},
                             markers={"2026-09-11": {"run_date": "2026-09-11", "patched": []}}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")
        assert res["skipped"] == ["2026-09-11"] and res["swept"] == []
        assert s3.fetched == []

    def test_force_re_sweeps_a_marked_date(self, conn, fake_s3):
        _log(conn, date="2026-09-11", ib_order_id=690)
        s3 = fake_s3(_FakeS3(logs={"2026-09-11": _daemon_lines()},
                             markers={"2026-09-11": {"run_date": "2026-09-11", "patched": []}}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14", force=True)
        assert res["changed"] == ["2026-09-11"] and s3.fetched == ["2026-09-11"]

    def test_a_row_the_log_cannot_resolve_is_marked_not_guessed(self, conn, fake_s3):
        _log(conn, date="2026-09-11", ib_order_id=999)
        s3 = fake_s3(_FakeS3(logs={"2026-09-11": _daemon_lines()}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")
        assert res["changed"] == [] and res["still_unresolved"] == {"2026-09-11": 1}
        assert conn.execute("SELECT status, fill_price FROM trades").fetchone() == ("Working", None)
        # the marker is still written — that is the negative cache that stops a
        # ~100 MB log being re-downloaded every night for a row that can never resolve
        assert s3.markers["2026-09-11"]["unresolved"][0]["ib_order_id"] == 999

    def test_a_missing_archived_log_is_reported_not_raised(self, conn, fake_s3):
        _log(conn, date="2026-05-26", ib_order_id=690)
        s3 = fake_s3(_FakeS3(logs={}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")
        assert res["unavailable"] == ["2026-05-26"] and res["swept"] == []
        # no marker: the date was never actually examined, so it must stay eligible
        assert s3.markers == {}

    def test_max_dates_caps_one_run_and_leaves_the_rest(self, conn, fake_s3):
        for d in ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"):
            _log(conn, date=d, ib_order_id=690)
        fake_s3(_FakeS3(logs={d: _daemon_lines() for d in
                              ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11")}))
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14", max_dates=2)
        assert res["swept"] == ["2026-09-08", "2026-09-09"]
        assert unresolved_session_dates(conn, before="2026-09-14") == ["2026-09-10", "2026-09-11"]

    def test_no_unresolved_rows_touches_s3_not_at_all(self, conn, fake_s3):
        s3 = fake_s3(_FakeS3(logs={"2026-09-11": _daemon_lines()}))
        assert sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")["swept"] == []
        assert s3.fetched == [] and s3.put == []

    def test_a_failed_marker_write_does_not_undo_the_repair(self, conn, fake_s3):
        _log(conn, date="2026-09-11", ib_order_id=690)
        s3 = _FakeS3(logs={"2026-09-11": _daemon_lines()})
        s3.put_object = lambda **kw: (_ for _ in ()).throw(RuntimeError("AccessDenied"))
        fake_s3(s3)
        res = sweep_unresolved_sessions(conn, bucket="b", before="2026-09-14")
        assert res["changed"] == ["2026-09-11"]
        assert conn.execute("SELECT fill_price FROM trades").fetchone() == (74.0,)


# ── Local daemon log (the snapshot-capture backstop's second source) ─────────


class TestDaemonLogFile:
    """The snapshot session is on a different clientId than the daemon, so on
    the 2026-09-14 replay it held none of the daemon's executions; the box's
    own daemon log held all of them."""

    @staticmethod
    def _write(tmp_path, lines):
        path = tmp_path / "daemon.log"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_repairs_the_open_rows_from_the_log_on_disk(self, conn, tmp_path):
        _log(conn)
        _log(conn, ticker="DUOL", shares=369, price_at_order=147.25, ib_order_id=684)
        res = reconcile_from_daemon_log_file(conn, "2026-09-14", self._write(tmp_path, _daemon_lines()))
        assert {r["ticker"]: r["after"]["fill_price"] for r in res["patched"]} == {"PBF": 74.0, "DUOL": 147.3182}
        assert unresolved_trades(conn, "2026-09-14") == []

    def test_an_execution_from_another_day_cannot_resolve_todays_row(self, conn, tmp_path):
        # IB order ids are per-client counters and the live log spans sessions.
        _log(conn, date="2026-09-15")
        res = reconcile_from_daemon_log_file(conn, "2026-09-15", self._write(tmp_path, _daemon_lines()))
        assert res["patched"] == [] and len(res["unresolved"]) == 1
        assert conn.execute("SELECT status, fill_price FROM trades").fetchone() == ("Working", None)

    def test_a_missing_log_raises_for_the_caller_to_judge(self, conn, tmp_path):
        _log(conn)
        with pytest.raises(FileNotFoundError):
            reconcile_from_daemon_log_file(conn, "2026-09-14", str(tmp_path / "absent.log"))

    def test_nothing_open_never_reads_the_log(self, conn, tmp_path):
        _log(conn, status="Filled", fill_price=74.0)
        assert reconcile_from_daemon_log_file(conn, "2026-09-14", str(tmp_path / "absent.log")) == {
            "patched": [], "unresolved": [],
        }
