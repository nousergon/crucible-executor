"""Consumer contract test: ArcticDB library `universe`.

alpha-engine-config-I10796 / data_collection_plan_260914.md P-27 — crucible-
trader as a first-class consumer of the data collector. Pins
``contracts/arctic_universe.schema.json`` against:

1. the schema itself being loadable JSON and declaring the columns this
   repo actually reads;
2. a producer-shaped frame (the pinned column/dtype/index shape) round-
   tripping through ``executor.price_cache.load_price_histories`` and
   ``load_atr_14_pct`` — the two live read paths — via the same
   ``price_cache._arcticdb.Arctic`` mock point the repo's existing ATR
   test suite (``tests/test_price_cache_atr.py``) uses, so this test
   exercises the real ``_open_universe_library`` wrapper rather than
   bypassing it.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if "arcticdb" not in sys.modules:
    sys.modules["arcticdb"] = MagicMock()

from executor import price_cache  # noqa: E402
from executor.price_cache import load_atr_14_pct, load_price_histories  # noqa: E402

_CONTRACT_PATH = Path(__file__).parent.parent / "contracts" / "arctic_universe.schema.json"


@pytest.fixture()
def schema() -> dict:
    return json.loads(_CONTRACT_PATH.read_text())


def _mock_arctic_lib(ticker_rows: dict[str, pd.DataFrame]):
    """Same mock shape as tests/test_price_cache_atr.py::_mock_arctic_lib —
    a fake Arctic() -> get_library() -> read(ticker).data chain."""
    lib = MagicMock()

    def _read(ticker):
        if ticker not in ticker_rows:
            raise KeyError(f"no such symbol: {ticker}")
        result = MagicMock()
        result.data = ticker_rows[ticker]
        return result

    lib.read.side_effect = _read
    lib.list_symbols.return_value = list(ticker_rows.keys())

    arctic = MagicMock()
    arctic.get_library.return_value = lib
    return arctic


def _pinned_frame(schema: dict, last_date: date, n: int = 3) -> pd.DataFrame:
    """A frame carrying every column contracts/arctic_universe.schema.json
    declares, over the DatetimeIndex shape the contract pins."""
    columns = schema["symbol_shape"]["columns"]
    index = pd.bdate_range(end=pd.Timestamp(last_date), periods=n)
    data = {}
    for i, col in enumerate(columns):
        if col == "Volume":
            data[col] = [1_000_000 + i * 100 for _ in range(n)]
        elif col == "atr_14_pct":
            data[col] = [0.02 + 0.001 * k for k in range(n)]
        else:
            data[col] = [100.0 + k for k in range(n)]
    return pd.DataFrame(data, index=index)


class TestContractIsValid:
    def test_schema_file_parses_as_json(self, schema):
        assert schema["library"] == "universe"

    def test_schema_pins_version_1(self, schema):
        assert schema["version"] == 1

    def test_schema_declares_every_column_price_cache_reads(self, schema):
        columns = schema["symbol_shape"]["columns"]
        for expected in ("Open", "High", "Low", "Close", "atr_14_pct"):
            assert expected in columns, f"contract missing column read by price_cache.py: {expected}"


class TestProducerShapedFrameRoundTripsThroughLoadPriceHistories:
    def test_ohlc_columns_extracted(self, schema):
        ref = date(2026, 9, 11)
        rows = {"AAPL": _pinned_frame(schema, last_date=ref)}
        with patch.object(price_cache._arcticdb, "Arctic", return_value=_mock_arctic_lib(rows)):
            result = load_price_histories(["AAPL"], "test-bucket")

        assert set(result.keys()) == {"AAPL"}
        for col in ("open", "high", "low", "close"):
            assert col in result["AAPL"].columns


class TestProducerShapedFrameRoundTripsThroughLoadAtr14Pct:
    def test_atr_column_extracted(self, schema):
        ref = date(2026, 9, 11)
        rows = {"AAPL": _pinned_frame(schema, last_date=ref)}
        with patch.object(price_cache, "is_trading_day", return_value=True):
            with patch.object(price_cache._arcticdb, "Arctic", return_value=_mock_arctic_lib(rows)):
                result = load_atr_14_pct(
                    tickers=["AAPL"], signals_bucket="test-bucket", reference_date=ref,
                )
        assert "AAPL" in result
        assert result["AAPL"] > 0

    def test_frame_missing_pinned_atr_column_hard_fails(self, schema):
        """A producer that drops atr_14_pct from the frame is the exact
        drift this contract exists to catch — the read path must still
        hard-fail, per the schema's own consumer_contract.hard_fail_on."""
        assert "any requested ticker missing the atr_14_pct column (load_atr_14_pct)" in (
            schema["consumer_contract"]["hard_fail_on"]
        )
        ref = date(2026, 9, 11)
        frame = _pinned_frame(schema, last_date=ref).drop(columns=["atr_14_pct"])
        rows = {"AAPL": frame}
        with patch.object(price_cache, "is_trading_day", return_value=True):
            with patch.object(price_cache._arcticdb, "Arctic", return_value=_mock_arctic_lib(rows)):
                with pytest.raises(RuntimeError):
                    load_atr_14_pct(
                        tickers=["AAPL"], signals_bucket="test-bucket", reference_date=ref,
                    )
