# Pinned consumer contracts

crucible-trader (this repo) is a first-class consumer of the data collector
(`nousergon-data`), per `alpha-engine-config-I10796` /
`data_collection_plan_260914.md` §3 and amendment 2. Each file here pins a
copy of one producer artifact's shape, for every collector key this repo
reads:

| File | Producer key | Read by |
|---|---|---|
| `staging_daily_closes.schema.json` | `staging/daily_closes/{trading_day}.parquet` | `executor/upstream_artifact_gate.py` (freshness only) |
| `constituents.schema.json` | `market_data/weekly/{date}/constituents.json` | `executor/eod_reconcile.py::_load_constituents_sector_map`, via `executor/signal_reader.py::patch_unknown_sectors_with_constituents` on the pre-open path |
| `arctic_universe.schema.json` | ArcticDB library `universe` | `executor/price_cache.py` (`load_price_histories`, `load_atr_14_pct`), `executor/signal_reader.py::filter_buy_candidates_to_universe` |

Each is exercised by a contract test in `tests/test_contract_*.py`, asserting
both that a producer-shaped fixture validates against the pin AND that this
repo's own read path extracts the expected fields from it — so a producer
change that reshapes the artifact is caught here even before a live run.

**Change rule** (`data_collection_plan_260914.md` §4.3): a schema version
change is a cross-repo PR naming its consumers, with the consumer PR (this
one, for a future bump) merged first.

## Producer parity

P-07 landed in `nousergon-data` as `nousergon-data-PR1711` (`2ef120fb`),
publishing the producer's own `contracts/`, and `nousergon-data-PR1712`
(`3cd646b`) subsequently added `contracts/staging_daily_closes.schema.json`.
All three keys now have a producer-side copy, mirrored byte-verbatim under
`tests/contracts/producer/` and held in agreement with the files here by
`tests/test_contract_producer_parity.py`:

| This pin | Producer copy | Status |
|---|---|---|
| `constituents.schema.json` | `nousergon-data/contracts/constituents.schema.json` | reconciled; no divergence |
| `arctic_universe.schema.json` | `nousergon-data/contracts/arctic_universe.schema.json` | reconciled; no divergence (closed alpha-engine-config-I10828) |
| `staging_daily_closes.schema.json` | `nousergon-data/contracts/staging_daily_closes.schema.json` (`nousergon-data-PR1936`) | byte copy of the producer; divergence closed (alpha-engine-config-I10853) |

For constituents and arctic_universe the two files stay SEPARATE rather than
one replacing the other: the producer's models the write path
(`additionalProperties: false`), this repo's records what the trader's read
path consumes and what it hard-fails without. staging_daily_closes is the
exception: its only reader here is a freshness probe that never parses a row,
so the pin carries no read-path fact and IS the producer's schema, byte for
byte. The data gate's `data.D17.schema_contract` and `data.D19.schema_contract`
clauses compare its validation shape with the producer's on every run.

**Closed: the arctic_universe divergence (alpha-engine-config-I10828).** The
producer's `arctic_universe.schema.json` used to be `additionalProperties:
false` while declaring neither `atr_14_pct` nor `VWAP`, both of which
`executor/price_cache.py` reads and the first of which `load_atr_14_pct`
hard-fails without — a frame valid against the producer's own published
contract was one the morning planner refused to trade on. The producer now
declares both (nullable, additive). Ongoing coverage:
`test_producer_declares_the_columns_this_repo_hard_depends_on`, which goes red
if the producer ever drops either column again.

**Closed: the staging_daily_closes divergence (alpha-engine-config-I10853).**
The producer types `Open`/`High`/`Low`/`Close`/`Adj_Close`/`Volume` as
nullable (a gap day with no vendor value), and this repo's pin used to type
them non-null, so a producer-conformant row failed this repo's own contract.
Since 2026-10-02 the pin is a byte copy of the producer's schema, which also
picks up the producer's `source` vendor enum and required `revision`.
`executor/upstream_artifact_gate.py` still only probes freshness, so no
runtime path changed. Ongoing coverage:
`test_staging_daily_closes_pin_is_the_producer_schema` and
`tests/test_contract_staging_daily_closes.py::TestContractIsValid::test_pin_is_a_byte_copy_of_the_producer_schema`.

**Refreshing a producer copy**: copy the file again from `nousergon-data`
`main` into `tests/contracts/producer/`, bump `PRODUCER_SHA` in the parity
test, run it, and reconcile whatever it reports. For
`staging_daily_closes.schema.json`, copy it into `contracts/` as well, and
never hand-edit either copy.
