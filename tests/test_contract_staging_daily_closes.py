"""Consumer contract test: staging/daily_closes/{trading_day}.parquet.

alpha-engine-config-I10796 / data_collection_plan_260914.md P-27 — crucible-
trader as a first-class consumer of the data collector. Pins
``contracts/staging_daily_closes.schema.json`` against:

1. the schema itself being a loadable, valid JSON Schema;
2. a producer-shaped fixture (one PriceBar.to_record() row,
   nousergon-data sources/contract.py) validating against it;
3. ``executor.upstream_artifact_gate.EXECUTOR_UPSTREAM_SPECS`` still
   declaring the same S3 key template this contract pins — so a rename
   on either side is caught here rather than at the next live preopen.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from executor.upstream_artifact_gate import EXECUTOR_UPSTREAM_SPECS

_CONTRACT_PATH = Path(__file__).parent.parent / "contracts" / "staging_daily_closes.schema.json"


@pytest.fixture()
def schema() -> dict:
    return json.loads(_CONTRACT_PATH.read_text())


class TestContractIsValid:
    def test_schema_file_parses_as_json_schema(self, schema):
        jsonschema.Draft7Validator.check_schema(schema)

    def test_schema_pins_version_1(self, schema):
        assert schema["version"] == 1


class TestProducerShapedFixtureValidates:
    """One PriceBar.to_record() row (nousergon-data sources/contract.py)."""

    def _price_bar_record(self, **overrides) -> dict:
        record = {
            "ticker": "MSFT",
            "date": "2026-09-11",
            "Open": 512.30,
            "High": 515.10,
            "Low": 510.00,
            "Close": 513.75,
            "Adj_Close": 513.75,
            "Volume": 18_204_552,
            "VWAP": 513.02,
            "source": "polygon_only",
        }
        record.update(overrides)
        return record

    def test_full_record_validates(self, schema):
        jsonschema.validate(self._price_bar_record(), schema)

    def test_null_vwap_validates(self, schema):
        """FRED-sourced / VWAP-less bars carry VWAP: null (PriceBar.vwap: Optional[float])."""
        jsonschema.validate(self._price_bar_record(VWAP=None), schema)

    def test_zero_volume_validates(self, schema):
        """Sources with no volume (e.g. FRED) persist Volume: 0, not absent."""
        jsonschema.validate(self._price_bar_record(Volume=0), schema)

    def test_record_missing_required_field_is_rejected(self, schema):
        record = self._price_bar_record()
        del record["Close"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, schema)


class TestRegistryKeyParity:
    """The pinned schema and the live gate spec must name the same artifact."""

    def test_gate_spec_key_template_matches_contract_title(self, schema):
        spec = next(
            s for s in EXECUTOR_UPSTREAM_SPECS if s.artifact_id == "daily_closes_parquet"
        )
        assert spec.s3_key_template == "staging/daily_closes/{trading_day}.parquet"
        assert spec.s3_key_template in schema["title"]

    def test_gate_spec_is_critical(self):
        """Trader-blocking today (predictor inference reads this too) —
        matches ARTIFACT_REGISTRY.yaml's daily_closes_parquet severity."""
        spec = next(
            s for s in EXECUTOR_UPSTREAM_SPECS if s.artifact_id == "daily_closes_parquet"
        )
        assert spec.severity == "critical"
