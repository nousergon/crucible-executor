"""Consumer contract test: staging/daily_closes/{trading_day}.parquet.

alpha-engine-config-I10796 / data_collection_plan_260914.md P-27 — crucible-
trader as a first-class consumer of the data collector. Pins
``contracts/staging_daily_closes.schema.json`` against:

1. the schema itself being a loadable, valid JSON Schema;
2. the pin being a byte copy of the producer's own
   ``nousergon-data/contracts/staging_daily_closes.schema.json`` (mirrored
   under ``tests/contracts/producer/``) — the data gate's
   ``data.D17.schema_contract`` / ``data.D19.schema_contract`` clauses
   compare this file's validation shape with the producer's on every run,
   so a hand-edited pin reads STALE there;
3. a producer-shaped fixture (one PriceBar.to_record() row,
   nousergon-data sources/contract.py) validating against it;
4. ``executor.upstream_artifact_gate.EXECUTOR_UPSTREAM_SPECS`` still
   declaring the same S3 key the contract's ``x-key-pattern`` names — so a
   rename on either side is caught here rather than at the next live preopen.

To re-pin when the producer moves: copy the producer file again into both
``contracts/`` and ``tests/contracts/producer/``, then update the fixtures
here until this file passes. Never hand-edit the copy to match.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from executor.upstream_artifact_gate import EXECUTOR_UPSTREAM_SPECS

_REPO = Path(__file__).parent.parent
_CONTRACT_PATH = _REPO / "contracts" / "staging_daily_closes.schema.json"
_PRODUCER_COPY_PATH = Path(__file__).parent / "contracts" / "producer" / "staging_daily_closes.schema.json"


@pytest.fixture()
def schema() -> dict:
    return json.loads(_CONTRACT_PATH.read_text())


class TestContractIsValid:
    def test_schema_file_parses_as_json_schema(self, schema):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_pin_is_a_byte_copy_of_the_producer_schema(self):
        """The pin and the producer mirror are the same bytes. A difference
        means one was refreshed without the other, or the pin was edited."""
        assert _CONTRACT_PATH.read_bytes() == _PRODUCER_COPY_PATH.read_bytes()


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
            "source": "polygon",
            "revision": 1,
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
        jsonschema.validate(self._price_bar_record(Volume=0, source="fred"), schema)

    def test_null_ohlc_validates(self, schema):
        """A gap-day row with no vendor value carries null prices; the
        producer's source-priority coalesce treats a null Close as missing
        (alpha-engine-config-I10853)."""
        row = self._price_bar_record(Open=None, High=None, Low=None, Close=None, Adj_Close=None, Volume=None)
        jsonschema.validate(row, schema)

    def test_record_missing_required_field_is_rejected(self, schema):
        record = self._price_bar_record()
        del record["Close"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, schema)

    def test_record_missing_revision_is_rejected(self, schema):
        record = self._price_bar_record()
        del record["revision"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, schema)

    def test_unknown_vendor_is_rejected(self, schema):
        """``source`` is the closed set of vendors the producer can name."""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(self._price_bar_record(source="polygon_only"), schema)


class TestRegistryKeyParity:
    """The pinned schema and the live gate spec must name the same artifact."""

    def test_gate_spec_key_template_matches_contract_key_pattern(self, schema):
        spec = next(s for s in EXECUTOR_UPSTREAM_SPECS if s.artifact_id == "daily_closes_parquet")
        assert spec.s3_key_template == "staging/daily_closes/{trading_day}.parquet"
        # The producer names the placeholder {date}; the gate spec names it
        # {trading_day}. Same key, same single date segment.
        assert schema["x-key-pattern"] == "staging/daily_closes/{date}.parquet"
        assert spec.s3_key_template.replace("{trading_day}", "{date}") == schema["x-key-pattern"]

    def test_gate_spec_is_critical(self):
        """Trader-blocking today (predictor inference reads this too) —
        matches ARTIFACT_REGISTRY.yaml's daily_closes_parquet severity."""
        spec = next(s for s in EXECUTOR_UPSTREAM_SPECS if s.artifact_id == "daily_closes_parquet")
        assert spec.severity == "critical"
