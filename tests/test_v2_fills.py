"""`executor.v2_fills`: the v2 trader's fills feed the v1 EOD attribution (alpha-engine-config-I13228)."""

from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path
from typing import Any

from executor import v2_fills
from executor import v2_trade_alert as alert
from executor.eod_reconcile import _compute_daily_return
from executor.eod_report import _buy_entry_prices, compute_rotation_realized
from tests.test_v2_trade_alert import _S3, _ClientError

AT = dt.datetime(2026, 11, 5, 21, 45, tzinfo=dt.UTC)
DAY = "2026-11-05"


def _fill(ref: str, ticker: str, quantity: int, cents: int) -> dict[str, Any]:
    """A ``kind: fill`` entry: cash one way, the security the other (cents signed with quantity)."""
    return {
        "event_ref": ref,
        "kind": "fill",
        "postings": [
            {"account": "assets:cash:USD", "amount_cents": -cents, "quantity": None},
            {"account": f"assets:securities:{ticker}", "amount_cents": cents, "quantity": quantity},
        ],
    }


def _commission(ref: str, cents: int) -> dict[str, Any]:
    return {
        "event_ref": ref,
        "kind": "commission",
        "postings": [
            {"account": "expenses:commissions", "amount_cents": cents, "quantity": None},
            {"account": "assets:cash:USD", "amount_cents": -cents, "quantity": None},
        ],
    }


def _journal(entries: list[dict[str, Any]], day: str = DAY) -> dict[str, Any]:
    return {"schema_version": "ledger_journal.v1", "tenant": "nous-ergon-paper", "trading_day": day, "entries": entries}


# The 2026-11-05 session as the real journal booked it, aggregated to two
# executions per side: sold 1,608 PLTR for $249,428.48, bought 294 SNDK for
# $245,161.36 (the same fills the execution_shortfall file records).
JOURNAL_1105 = _journal(
    [
        _fill("a.01", "PLTR", -1000, -15511000),
        _commission("a.01", 520),
        _fill("a.02", "PLTR", -608, -9431848),
        _commission("a.02", 312),
        _fill("b.01", "SNDK", 200, 16677600),
        _commission("b.01", 104),
        _fill("b.02", "SNDK", 94, 7838536),
        {
            "event_ref": "accrual/2026-11-05",
            "kind": "interest_accrual",
            "postings": [
                {"account": "assets:interest_receivable:USD", "amount_cents": 184, "quantity": None},
                {"account": "income:interest", "amount_cents": -184, "quantity": None},
            ],
        },
    ]
)


class _TraderS3(_S3):
    """The trader role as measured on 2026-11-05: the ledger is readable,
    ``trader/execution_shortfall/`` is write-only (GET and LIST both denied)."""

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if "/ledger/" not in Key:
            raise _ClientError("AccessDenied")
        return super().get_object(Bucket=Bucket, Key=Key)

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        raise _ClientError("AccessDenied")


def _store(docs: dict[str, dict[str, Any]], prefix: str = "crucible") -> alert.Store:
    head = f"{prefix}/" if prefix else ""
    objects = {f"{head}{v2_fills.journal_key(day)}": (AT, doc) for day, doc in docs.items()}
    objects[f"{head}trader/execution_shortfall/2026-11-04.json"] = (AT, {"calendar_date": DAY, "orders": []})
    return alert.Store(_TraderS3(objects), "bucket", prefix)


def test_the_days_journal_gives_one_row_per_ticker_and_side() -> None:
    rows = v2_fills.v2_trade_rows(DAY, store=_store({DAY: JOURNAL_1105}))
    by = {r["ticker"]: r for r in rows}
    assert set(by) == {"PLTR", "SNDK"}
    assert by["PLTR"]["action"] == "SELL" and by["PLTR"]["shares"] == 1608
    assert abs(float(by["PLTR"]["fill_price"]) - 249428.48 / 1608) < 1e-9  # type: ignore[arg-type]
    assert by["PLTR"]["commission_usd"] == 8.32
    assert by["SNDK"]["action"] == "BUY" and by["SNDK"]["shares"] == 294
    assert abs(float(by["SNDK"]["fill_price"]) - 245161.36 / 294) < 1e-9  # type: ignore[arg-type]
    assert "price_at_order" not in by["SNDK"]


def test_a_fill_without_its_commission_report_leaves_the_rows_commission_unmeasured() -> None:
    rows = v2_fills.v2_trade_rows(DAY, store=_store({DAY: JOURNAL_1105}))
    sndk = next(r for r in rows if r["ticker"] == "SNDK")
    assert "commission_usd" not in sndk


def test_the_write_only_shortfall_prefix_is_never_needed() -> None:
    """The 2026-11-05 failure: listing execution_shortfall/ was AccessDenied, so no rows."""
    rows = v2_fills.v2_trade_rows(DAY, store=_store({DAY: JOURNAL_1105}))
    assert len(rows) == 2


def test_an_unprefixed_store_reads_the_same() -> None:
    assert len(v2_fills.v2_trade_rows(DAY, store=_store({DAY: JOURNAL_1105}, prefix=""))) == 2


def test_no_journal_on_that_day_gives_no_rows() -> None:
    assert v2_fills.v2_trade_rows("2026-11-06", store=_store({DAY: JOURNAL_1105})) == []


def test_a_journal_for_another_day_or_schema_gives_no_rows() -> None:
    wrong_day = _journal(JOURNAL_1105["entries"], day="2026-11-04")
    assert v2_fills.v2_trade_rows(DAY, store=_store({DAY: wrong_day})) == []
    wrong_schema = {**JOURNAL_1105, "schema_version": "ledger_journal.v2"}
    assert v2_fills.v2_trade_rows(DAY, store=_store({DAY: wrong_schema})) == []


def test_a_ticker_the_executor_logged_is_not_counted_twice() -> None:
    rows = v2_fills.v2_trade_rows(DAY, [{"ticker": "PLTR", "action": "SELL"}], store=_store({DAY: JOURNAL_1105}))
    assert [r["ticker"] for r in rows] == ["SNDK"]


def test_an_unreadable_store_gives_no_rows() -> None:
    class _Broken(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise RuntimeError("throttled")

    assert v2_fills.v2_trade_rows(DAY, store=alert.Store(_Broken({}), "bucket", "crucible")) == []


def test_a_box_without_the_trader_env_file_gives_no_rows(tmp_path: Path) -> None:
    assert v2_fills.v2_trade_rows(DAY, env_file=tmp_path / "missing.env") == []


def test_the_env_file_opens_the_trader_store(tmp_path: Path, monkeypatch: Any) -> None:
    env = tmp_path / "trader.env"
    env.write_text("CRUCIBLE_TRADER_STORE_URI=s3://bucket/crucible\n")
    seen: dict[str, str] = {}

    def _open(values: dict[str, str]) -> alert.Store:
        seen.update(values)
        return _store({DAY: JOURNAL_1105})

    monkeypatch.setattr(v2_fills, "open_store", _open)
    assert len(v2_fills.v2_trade_rows(DAY, env_file=env)) == 2
    assert seen["CRUCIBLE_TRADER_STORE_URI"] == "s3://bucket/crucible"


def test_the_2026_11_05_residual_is_explained_by_the_journal() -> None:
    """Replay of the failed 2026-11-05 run's own persisted numbers.

    eod_pnl for 11-05: nav_change -$94,258.91, positions -$127,415.22, interest
    $1.84, rotation $0, unattributed +$33,154.47 against a $5,000 bound. With
    the journal's fills the 294 SNDK bought today are priced from their fill,
    not the prior close, and the 1,608 PLTR sold get their realized P&L.
    """
    prior = {"PLTR": {"shares": 1852, "closing_price": 171.70}, "SNDK": {"shares": 240, "closing_price": 1037.40}}
    today = {"PLTR": {"shares": 244}, "SNDK": {"shares": 534}}
    close = {"PLTR": 154.99, "SNDK": 806.43}
    nav_change, interest = -94258.91, 1.84
    rows = v2_fills.v2_trade_rows(DAY, store=_store({DAY: JOURNAL_1105}))
    entry = _buy_entry_prices(rows)

    positions = 0.0
    for ticker, pos in today.items():
        _, usd, _, na = _compute_daily_return(
            ticker,
            pos,
            prior[ticker],
            close[ticker],
            pos["shares"],
            prior[ticker]["closing_price"],
            date(2026, 11, 4),
            date(2026, 11, 4),
            add_entry_px=entry.get(ticker),
        )
        assert na is None
        positions += usd
    rotation = compute_rotation_realized(today, prior, rows)
    unattributed = nav_change - positions - interest - rotation

    assert round(rotation, 2) == -26665.12
    assert abs(unattributed) < 50.0, unattributed  # was +$33,154; the bound is $5,000
