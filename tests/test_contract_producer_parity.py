"""Producer-parity for the pinned consumer contracts (alpha-engine-config-I10796).

`contracts/*.schema.json` in this repo were written on 2026-09-14, when P-07
(`data_collection_plan_260914.md` §8.3) had not yet landed and nousergon-data
had NO `contracts/` directory. Each carried the same standing instruction in
its own `description`:

    "no producer-side contracts/ copy of this schema exists yet in
     nousergon-data ... must be reconciled against the producer's own
     contracts/ file once P-07 lands there."

P-07 landed as `nousergon-data-PR1711` (merged 2026-09-14, `2ef120fb`),
publishing `contracts/constituents.schema.json` and
`contracts/arctic_universe.schema.json`. This file is that reconciliation,
and it is a TEST rather than a one-time edit because the instruction it
discharges recurs: the producer's schema will move again, and a consumer pin
that agreed with the producer once and silently diverged afterwards is the
`avg_volume_20d` failure mode with a schema file next to it.

`nousergon-data-PR1712` (merged 2026-09-15, `3cd646b`) subsequently
published `contracts/staging_daily_closes.schema.json`, closing the third
gap this file's own docstring named. Reconciling it surfaced a genuine,
still-open divergence rather than agreement: the producer types
`Open`/`High`/`Low`/`Close`/`Adj_Close`/`Volume` as nullable (a documented,
legitimate producer state — "Null Close is treated as missing by the
source-priority coalesce"), while this repo's consumer pin types the same
fields as plain `number`/`integer` with no null option. A producer-valid row
carrying a real null price fails this repo's own contract. It is pinned as
a passing, documented-divergence test below (not silently loosened) —
tracked as `alpha-engine-config-I10853`.

`tests/contracts/producer/*.schema.json` are byte-verbatim copies of the
producer's files at that SHA. Nothing here edits them, and no code path reads
them at runtime — `contracts/` stays the repo's own consumer pin (it has to:
it carries the read-path facts the producer's copy does not model, see the
hard-dependency check below). These copies exist so the DIFFERENCE between the
two is asserted rather than assumed.

What is pinned:

  1. Both producer copies are valid JSON Schema.
  2. Every field this repo's consumer pin marks REQUIRED is a field the
     producer actually publishes. A consumer requiring something the producer
     never declared is a contract this repo invented.
  3. A producer-conformant document validates against this repo's consumer
     pin. A consumer pin stricter than the producer rejects real data.
  4. The producer still declares every column this repo hard-depends on (see
     `_ARCTIC_COLUMNS_THIS_REPO_DEPENDS_ON`) — closed alpha-engine-config-I10828,
     asserted going forward so a future producer regression is caught here.

Refresh procedure when the producer's schema moves: copy the file again from
`nousergon-data` `main`, run this test, and reconcile `contracts/` against
whatever it reports — per `data_collection_plan_260914.md` §4.3 that is a
cross-repo PR with the consumer side merged first.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).parent.parent
_CONSUMER = _REPO / "contracts"
_PRODUCER = Path(__file__).parent / "contracts" / "producer"

# The producer SHA these copies were taken at. Cited so a reviewer can diff
# them without guessing which revision they mirror.
PRODUCER_SHA = "nousergon-data-PR1726"  # atr_14_pct + VWAP added, alpha-engine-config-I10828
STAGING_DAILY_CLOSES_PRODUCER_SHA = "nousergon-data-PR1712"  # source+revision stamp, alpha-engine-config-I10783

# ── The divergence is closed ─────────────────────────────────────────────────
#
# `executor/price_cache.py` HARD-FAILS on a `universe` frame missing
# `atr_14_pct` (`load_atr_14_pct`: absent column, non-finite value, or a most-
# recent row older than `_ATR_MAX_STALENESS_TRADING_DAYS` all abort the morning
# planner), and reads `VWAP` in `load_prior_day_vwap`. The producer's
# `arctic_universe.schema.json` used to declare neither while being
# `additionalProperties: false` — a frame VALID against the producer's own
# published contract was one this repo refused to trade on
# (alpha-engine-config-I10828). The producer now declares both (nullable,
# additive — VWAP null on yfinance/FRED rows, atr_14_pct null during ATR
# warmup), closing the gap. The columns this repo's read path actually
# depends on for a hard-fail path.
_ARCTIC_COLUMNS_THIS_REPO_DEPENDS_ON = ("atr_14_pct", "VWAP")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def producer_constituents() -> dict:
    return _load(_PRODUCER / "constituents.schema.json")


@pytest.fixture(scope="module")
def producer_arctic() -> dict:
    return _load(_PRODUCER / "arctic_universe.schema.json")


@pytest.fixture(scope="module")
def consumer_constituents() -> dict:
    return _load(_CONSUMER / "constituents.schema.json")


@pytest.fixture(scope="module")
def consumer_arctic() -> dict:
    return _load(_CONSUMER / "arctic_universe.schema.json")


@pytest.fixture(scope="module")
def producer_staging_daily_closes() -> dict:
    return _load(_PRODUCER / "staging_daily_closes.schema.json")


@pytest.fixture(scope="module")
def consumer_staging_daily_closes() -> dict:
    return _load(_CONSUMER / "staging_daily_closes.schema.json")


# ── 1. The producer copies are schemas ──────────────────────────────────────


@pytest.mark.parametrize("name", ["constituents", "arctic_universe"])
def test_producer_copy_is_a_valid_json_schema(name):
    schema = _load(_PRODUCER / f"{name}.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    # Verbatim copies keep the producer's own $id — that is how a reviewer
    # tells a mirrored file from a locally-authored one.
    assert schema["$id"] == f"nousergon-data/contracts/{name}.schema.json"


def test_staging_daily_closes_producer_copy_is_a_valid_json_schema(
    producer_staging_daily_closes,
):
    # Unlike constituents/arctic_universe, the producer's staging_daily_closes
    # schema (nousergon-data-PR1712) carries no `$id` — a fact about that
    # repo's own file, not something to invent here. Draft is 2020-12 per its
    # own `$schema`.
    jsonschema.Draft202012Validator.check_schema(producer_staging_daily_closes)
    assert "$id" not in producer_staging_daily_closes


# ── 2/3. constituents: the one key this repo actually parses ────────────────


def test_consumer_requires_nothing_the_producer_does_not_publish(
    producer_constituents, consumer_constituents,
):
    """Every field this repo marks required must be a field the producer
    declares. A consumer requirement the producer never publishes is a
    contract this repo invented, and it will be discovered at 09:00 ET."""
    producer_fields = set(producer_constituents["properties"])
    invented = set(consumer_constituents["required"]) - producer_fields
    assert invented == set(), (
        f"consumer pin requires {sorted(invented)}, which nousergon-data's "
        f"constituents.schema.json ({PRODUCER_SHA}) does not declare"
    )


def test_a_producer_conformant_constituents_doc_passes_the_consumer_pin(
    producer_constituents, consumer_constituents,
):
    """The other direction: a document built to satisfy the PRODUCER's
    contract must satisfy this repo's. A consumer pin stricter than the
    producer rejects real data — the failure reads as a data defect and is a
    contract defect."""
    doc = {
        "date": "2026-09-14",
        "tickers": ["AAPL", "MSFT"],
        "sp500_tickers": ["AAPL"],
        "sp400_tickers": ["MSFT"],
        "sector_map": {"AAPL": "Information Technology", "MSFT": "Information Technology"},
        "sector_etf_map": {"AAPL": "XLK", "MSFT": "XLK"},
        "sub_industry_map": {"AAPL": "Technology Hardware", "MSFT": "Systems Software"},
        "sub_sector_etf_map": {"AAPL": "XLK", "MSFT": "IGV"},
        "sp500_count": 1,
        "sp400_count": 1,
        "total_count": 2,
        "fetched_at": "2026-09-14T13:05:00+00:00",
    }
    jsonschema.validate(instance=doc, schema=producer_constituents)
    jsonschema.validate(instance=doc, schema=consumer_constituents)


def test_sector_map_is_required_on_both_sides(
    producer_constituents, consumer_constituents,
):
    """`sector_map` is THE field `_load_constituents_sector_map` reads. It is
    required by both contracts, so a producer change that made it optional is
    a red test here rather than an "Unknown"-sector EOD report."""
    assert "sector_map" in producer_constituents["required"]
    assert "sector_map" in consumer_constituents["required"]


# ── 2/3. arctic universe: OHLCV parity + hard-dependency coverage ───────────


def test_consumer_ohlcv_columns_match_the_producer_contract(
    producer_arctic, consumer_arctic,
):
    """The five OHLCV columns `load_price_histories` reads must be exactly the
    ones the producer publishes as required."""
    producer_required = set(producer_arctic["required"]) - {"symbol", "index_date"}
    consumer_columns = set(consumer_arctic["symbol_shape"]["columns"])
    assert producer_required <= consumer_columns, (
        f"producer requires {sorted(producer_required - consumer_columns)} which "
        f"this repo's pin does not document"
    )
    assert producer_required == {"Open", "High", "Low", "Close", "Volume"}


@pytest.mark.parametrize("column", _ARCTIC_COLUMNS_THIS_REPO_DEPENDS_ON)
def test_producer_declares_the_columns_this_repo_hard_depends_on(
    producer_arctic, consumer_arctic, column,
):
    """`atr_14_pct` and `VWAP` are read by `executor/price_cache.py`
    (`load_atr_14_pct` hard-fails without the first). Closed
    alpha-engine-config-I10828: the producer now declares both, nullable, on
    its `additionalProperties: false` contract, so a frame valid against the
    producer's own schema is one this repo can actually trade on.

    If the producer ever drops either column again, this test goes red —
    that is the intended signal to re-open the gap and re-add a declared
    divergence entry rather than let the drop pass silently."""
    assert producer_arctic["additionalProperties"] is False
    assert column in producer_arctic["properties"], (
        f"{column} is no longer declared in the producer contract "
        f"({PRODUCER_SHA}) — executor/price_cache.py still hard-depends on "
        f"it; re-open alpha-engine-config-I10828"
    )
    assert column in consumer_arctic["symbol_shape"]["columns"], (
        f"{column} is read by executor/price_cache.py but is documented by "
        f"neither contract — the trader's dependency would be written down "
        f"nowhere"
    )


def test_bitemporal_columns_are_additive_not_required(producer_arctic):
    """config#2459's `settled`/`as_of`/`source_tier`/`valid_date`/
    `knowledge_time` are ADDITIVE: absent on older rows. This repo must never
    grow a hard dependency on one, so their optionality is pinned here."""
    additive = {"settled", "as_of", "source_tier", "valid_date", "knowledge_time"}
    assert additive <= set(producer_arctic["properties"])
    assert additive.isdisjoint(producer_arctic["required"])


# ── 2/3. staging_daily_closes: freshness-only reader, no invented columns ──
#
# executor/upstream_artifact_gate.py::EXECUTOR_UPSTREAM_SPECS is the only
# code path that touches this artifact today, and it is a freshness probe
# (S3 HEAD/LIST via nousergon_lib.artifact_freshness.check_freshness) — it
# never parses a row. The set of columns "this repo's staging_daily_closes
# reader uses" is therefore the pinned consumer contract's own `required`
# list (contracts/README.md: "the row schema below is pinned so a future
# content-parsing consumer ... [has] one place to change"), not a set read
# out of live code. That list must still be a set the producer actually
# declares — the same invented-requirement check as constituents.

_STAGING_DAILY_CLOSES_COLUMNS_THIS_REPO_PINS = (
    "ticker", "date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "source",
)


def test_consumer_requires_nothing_the_producer_does_not_publish_for_staging_daily_closes(
    producer_staging_daily_closes, consumer_staging_daily_closes,
):
    producer_fields = set(producer_staging_daily_closes["properties"])
    invented = set(consumer_staging_daily_closes["required"]) - producer_fields
    assert invented == set(), (
        f"consumer pin requires {sorted(invented)}, which nousergon-data's "
        f"staging_daily_closes.schema.json ({STAGING_DAILY_CLOSES_PRODUCER_SHA}) "
        f"does not declare"
    )


@pytest.mark.parametrize("column", _STAGING_DAILY_CLOSES_COLUMNS_THIS_REPO_PINS)
def test_producer_declares_every_column_this_repos_pin_names(
    producer_staging_daily_closes, column,
):
    """Every column named in this repo's own consumer pin (the future
    content-parsing consumer's contract, per contracts/README.md) must still
    be one the producer actually publishes."""
    assert column in producer_staging_daily_closes["properties"], (
        f"{column} is pinned by crucible-executor's staging_daily_closes "
        f"consumer contract but is no longer declared by the producer "
        f"({STAGING_DAILY_CLOSES_PRODUCER_SHA})"
    )


def _producer_valid_row(**overrides) -> dict:
    row = {
        "ticker": "MSFT",
        "date": "2026-09-14",
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
    row.update(overrides)
    return row


def test_a_producer_conformant_staging_daily_closes_row_passes_the_consumer_pin(
    producer_staging_daily_closes, consumer_staging_daily_closes,
):
    """The happy path: a fully-populated producer-shaped row (no nulls) does
    satisfy both contracts. This is the case the freshness-only reader never
    exercises today, but it is the case a future content-parsing consumer
    would see on every normal trading day."""
    row = _producer_valid_row()
    jsonschema.validate(row, producer_staging_daily_closes)
    jsonschema.validate(row, consumer_staging_daily_closes)


def test_producer_allows_null_ohlc_but_consumer_pin_rejects_it(
    producer_staging_daily_closes, consumer_staging_daily_closes,
):
    """OPEN DIVERGENCE — alpha-engine-config-I10853. The producer types
    Open/High/Low/Close/Adj_Close/Volume as nullable
    (`nousergon-data-PR1712`'s own schema description: "Null Close is
    treated as missing by the source-priority coalesce") — a real, legitimate
    producer state (a gap day with no vendor value), not an edge case. This
    repo's consumer pin types the same fields as plain `number`/`integer`
    with no null option, so a producer-conformant row carrying a real null
    price is REJECTED by this repo's own contract.

    Not fixed here: fixing it means loosening a contract this repo owns
    without a ruling on whether a future content-parsing consumer needs the
    stricter (non-null) shape enforced upstream instead. Pinned as a passing,
    documented-divergence test so it cannot silently drift further in either
    direction — a fix on either side must touch this test.
    """
    for field in ("Open", "High", "Low", "Close", "Adj_Close"):
        assert producer_staging_daily_closes["properties"][field]["type"] == [
            "number", "null",
        ]
        assert consumer_staging_daily_closes["properties"][field]["type"] == "number"
    assert producer_staging_daily_closes["properties"]["Volume"]["type"] == [
        "integer", "null",
    ]
    assert consumer_staging_daily_closes["properties"]["Volume"]["type"] == "integer"

    null_row = _producer_valid_row(
        Open=None, High=None, Low=None, Close=None, Adj_Close=None, Volume=0,
        source="fred",
    )
    jsonschema.validate(null_row, producer_staging_daily_closes)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(null_row, consumer_staging_daily_closes)


def test_producer_source_enum_is_a_subset_of_the_consumers_free_string(
    producer_staging_daily_closes, consumer_staging_daily_closes,
):
    """The producer restricts `source` to a closed enum; the consumer pin
    types it as a free string. That direction is safe (stricter producer,
    looser consumer) — pinned so a future consumer tightening does not
    silently start rejecting a real vendor name."""
    assert producer_staging_daily_closes["properties"]["source"]["enum"] == [
        "polygon", "fred", "yfinance",
    ]
    assert consumer_staging_daily_closes["properties"]["source"]["type"] == "string"
    assert "enum" not in consumer_staging_daily_closes["properties"]["source"]


def test_revision_is_producer_required_but_additive_on_the_consumer_pin(
    producer_staging_daily_closes, consumer_staging_daily_closes,
):
    """`revision` (alpha-engine-config-I10783) is required by the producer
    but the consumer pin neither declares nor requires it. That is fine only
    because the consumer pin is `additionalProperties: true` — a real
    producer row carrying `revision` still validates. Pinned so a future
    tightening to `additionalProperties: false` on the consumer side is
    forced to address this field explicitly rather than silently reject
    every producer row."""
    assert "revision" in producer_staging_daily_closes["required"]
    assert "revision" not in consumer_staging_daily_closes.get("properties", {})
    assert consumer_staging_daily_closes["additionalProperties"] is True


# ── The standing instruction the consumer pins carried ──────────────────────


@pytest.mark.parametrize(
    "name", ["constituents", "arctic_universe", "staging_daily_closes"],
)
def test_consumer_pin_no_longer_claims_the_producer_copy_is_absent(name):
    """Each consumer pin shipped with "no producer-side contracts/ copy of
    this schema exists yet in nousergon-data". That is now false, and a
    load-bearing file asserting a false fact about the fleet drives the next
    reader to re-derive the reconciliation this file performs."""
    description = _load(_CONSUMER / f"{name}.schema.json")["description"]
    assert "no producer-side" not in description.lower()
    assert "contracts/" in description
