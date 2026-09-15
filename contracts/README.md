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
publishing the producer's own `contracts/`. Two of the three keys now have a
producer-side copy, mirrored byte-verbatim under
`tests/contracts/producer/` and held in agreement with the files here by
`tests/test_contract_producer_parity.py`:

| This pin | Producer copy | Status |
|---|---|---|
| `constituents.schema.json` | `nousergon-data/contracts/constituents.schema.json` | reconciled; no divergence |
| `arctic_universe.schema.json` | `nousergon-data/contracts/arctic_universe.schema.json` | reconciled; no divergence (closed alpha-engine-config-I10828) |
| `staging_daily_closes.schema.json` | *(none)* | producer has not published one; this pin stays sourced from `sources/contract.py::PriceBar` |

The two files per key stay SEPARATE rather than one replacing the other: the
producer's models the write path (`additionalProperties: false`), this repo's
records what the trader's read path consumes and what it hard-fails without.

**Closed: the arctic_universe divergence (alpha-engine-config-I10828).** The
producer's `arctic_universe.schema.json` used to be `additionalProperties:
false` while declaring neither `atr_14_pct` nor `VWAP`, both of which
`executor/price_cache.py` reads and the first of which `load_atr_14_pct`
hard-fails without — a frame valid against the producer's own published
contract was one the morning planner refused to trade on. The producer now
declares both (nullable, additive). Ongoing coverage:
`test_producer_declares_the_columns_this_repo_hard_depends_on`, which goes red
if the producer ever drops either column again.

**Refreshing a producer copy**: copy the file again from `nousergon-data`
`main` into `tests/contracts/producer/`, bump `PRODUCER_SHA` in the parity
test, run it, and reconcile whatever it reports.
