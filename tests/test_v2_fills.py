"""`executor.v2_fills`: the v2 trader's fills feed the v1 EOD attribution (alpha-engine-config-I13228)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from executor import v2_fills
from executor import v2_trade_alert as alert
from tests.test_v2_trade_alert import _S3

AT = dt.datetime(2026, 11, 4, 18, 8, tzinfo=dt.UTC)


def _order(symbol: str, side: str, qty: float, filled: float | None = None, **kw: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "filled_quantity": qty if filled is None else filled,
        "decision_price": 100.0,
        "fill_price": 100.5,
        **kw,
    }


def _store(docs: dict[str, dict[str, Any]], prefix: str = "crucible") -> alert.Store:
    head = f"{prefix}/" if prefix else ""
    objects = {f"{head}trader/execution_shortfall/{name}.json": (AT, doc) for name, doc in docs.items()}
    objects[f"{head}trader/execution_shortfall/README"] = (AT, {})
    return alert.Store(_S3(objects), "bucket", prefix)


# The 11-04 session traded the 11-03 book: the file is named by book day.
SESSION = {
    "calendar_date": "2026-11-04",
    "orders": [
        _order("ANF", "sell", 255, fill_price=170.14, decision_price=170.43),
        _order("PLTR", "buy", 1465),
        _order("SNDK", "buy", 240, filled=0),
        _order("XXX", "hold", 10),
        _order("YYY", "buy", 10, fill_price=0),
        _order("ZZZ", "sell", 5, decision_price=None),
    ],
}


def test_the_session_is_found_by_calendar_date_not_file_name() -> None:
    store = _store({"2026-10-30": {"calendar_date": "2026-10-31", "orders": []}, "2026-11-03": SESSION})
    rows = v2_fills.v2_trade_rows("2026-11-04", store=store)
    assert [r["ticker"] for r in rows] == ["ANF", "PLTR", "ZZZ"]
    assert rows[0] == {
        "ticker": "ANF",
        "action": "SELL",
        "shares": 255,
        "fill_price": 170.14,
        "status": "Filled",
        "source": "crucible-trader",
        "price_at_order": 170.43,
    }
    assert "price_at_order" not in rows[2]


def test_an_unprefixed_store_reads_the_same() -> None:
    rows = v2_fills.v2_trade_rows("2026-11-04", store=_store({"2026-11-03": SESSION}, prefix=""))
    assert len(rows) == 3


def test_no_session_on_that_day_gives_no_rows() -> None:
    assert v2_fills.v2_trade_rows("2026-11-05", store=_store({"2026-11-03": SESSION})) == []


def test_a_ticker_the_executor_logged_is_not_counted_twice() -> None:
    rows = v2_fills.v2_trade_rows(
        "2026-11-04", [{"ticker": "PLTR", "action": "BUY"}], store=_store({"2026-11-03": SESSION})
    )
    assert [r["ticker"] for r in rows] == ["ANF", "ZZZ"]


def test_an_unreadable_store_gives_no_rows() -> None:
    class _Broken(_S3):
        def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
            raise RuntimeError("AccessDenied")

    store = alert.Store(_Broken({}), "bucket", "crucible")
    assert v2_fills.v2_trade_rows("2026-11-04", store=store) == []


def test_a_box_without_the_trader_env_file_gives_no_rows(tmp_path: Path) -> None:
    assert v2_fills.v2_trade_rows("2026-11-04", env_file=tmp_path / "missing.env") == []


def test_the_env_file_opens_the_trader_store(tmp_path: Path, monkeypatch: Any) -> None:
    env = tmp_path / "trader.env"
    env.write_text("CRUCIBLE_TRADER_STORE_URI=s3://bucket/crucible\n")
    seen: dict[str, str] = {}

    def _open(values: dict[str, str]) -> alert.Store:
        seen.update(values)
        return _store({"2026-11-03": SESSION})

    monkeypatch.setattr(v2_fills, "open_store", _open)
    assert len(v2_fills.v2_trade_rows("2026-11-04", env_file=env)) == 3
    assert seen["CRUCIBLE_TRADER_STORE_URI"] == "s3://bucket/crucible"
