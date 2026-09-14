"""Consumer contract test: market_data/weekly/{date}/constituents.json.

alpha-engine-config-I10796 / data_collection_plan_260914.md P-27 — crucible-
trader as a first-class consumer of the data collector. Pins
``contracts/constituents.schema.json`` against:

1. the schema itself being a loadable, valid JSON Schema;
2. a producer-shaped fixture (nousergon-data collectors/constituents.py's
   ``result`` dict shape) validating against it;
3. this repo's actual read path (``_load_constituents_sector_map``,
   exercised through the pre-open call chain
   ``signal_reader.patch_unknown_sectors_with_constituents``) extracting
   the expected sector map from a schema-valid fixture — so a producer
   field rename is caught here, not only by the narrower behavioral test
   in test_eod_reconcile_logic.py::TestLoadConstituentsSectorMap.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from executor.eod_reconcile import _load_constituents_sector_map
from executor.signal_reader import patch_unknown_sectors_with_constituents

_CONTRACT_PATH = Path(__file__).parent.parent / "contracts" / "constituents.schema.json"


@pytest.fixture()
def schema() -> dict:
    return json.loads(_CONTRACT_PATH.read_text())


def _producer_fixture(**overrides) -> dict:
    """One full constituents.json body, shaped per collectors/constituents.py::collect()."""
    tickers = ["MSFT", "VRTX", "XOM"]
    fixture = {
        "date": "2026-09-13",
        "tickers": tickers,
        "sp500_tickers": tickers[:2],
        "sp400_tickers": tickers[2:],
        "sector_map": {
            "MSFT": "Information Technology",
            "VRTX": "Health Care",
            "XOM": "Energy",
        },
        "sector_etf_map": {"Information Technology": "XLK", "Health Care": "XLV", "Energy": "XLE"},
        "sub_industry_map": {"MSFT": "Systems Software"},
        "sub_sector_etf_map": {},
        "sp500_count": 2,
        "sp400_count": 1,
        "total_count": 3,
        "fetched_at": "2026-09-13T09:05:11.123456+00:00",
    }
    fixture.update(overrides)
    return fixture


class TestContractIsValid:
    def test_schema_file_parses_as_json_schema(self, schema):
        jsonschema.Draft7Validator.check_schema(schema)

    def test_schema_pins_version_1(self, schema):
        assert schema["version"] == 1


class TestProducerShapedFixtureValidates:
    def test_full_producer_body_validates(self, schema):
        jsonschema.validate(_producer_fixture(), schema)

    def test_body_without_sp_index_split_validates(self, schema):
        """config-I6946: sp500_tickers/sp400_tickers are OMITTED (not present
        as empty lists) when the counts don't describe the ticker list —
        both must be valid producer shapes."""
        fixture = _producer_fixture()
        del fixture["sp500_tickers"]
        del fixture["sp400_tickers"]
        jsonschema.validate(fixture, schema)

    def test_body_missing_sector_map_is_rejected(self, schema):
        fixture = _producer_fixture()
        del fixture["sector_map"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(fixture, schema)


class TestConsumerExtractsFromSchemaValidFixture:
    """The field this repo actually reads, from a full producer-shaped body."""

    def _mock_s3_with_body(self, body: dict):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "market_data/weekly/2026-09-13/constituents.json"}],
        }
        s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(body).encode())}
        return s3

    @patch("executor.eod_reconcile.boto3")
    def test_load_constituents_sector_map_reads_full_producer_fixture(self, mock_boto3, schema):
        fixture = _producer_fixture()
        jsonschema.validate(fixture, schema)  # the fixture itself is schema-valid
        mock_boto3.client.return_value = self._mock_s3_with_body(fixture)

        result = _load_constituents_sector_map("alpha-engine-research")

        assert result == fixture["sector_map"]

    @patch("executor.eod_reconcile.boto3")
    def test_preopen_sector_patch_consumes_the_same_fixture(self, mock_boto3, schema):
        """executor.signal_reader.patch_unknown_sectors_with_constituents is
        the pre-open call path (invoked from executor/main.py's morning
        planner) that shares _load_constituents_sector_map under the hood."""
        fixture = _producer_fixture()
        jsonschema.validate(fixture, schema)
        mock_boto3.client.return_value = self._mock_s3_with_body(fixture)

        signals_raw = {
            "buy_candidates": [{"ticker": "MSFT", "signal": "ENTER", "sector": "Unknown"}],
            "universe": [],
        }
        patched = patch_unknown_sectors_with_constituents(signals_raw, "alpha-engine-research")

        assert patched == 1
        assert signals_raw["buy_candidates"][0]["sector"] == "Information Technology"
