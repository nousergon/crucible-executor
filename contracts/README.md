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
| `arctic_universe.schema.json` | `nousergon-data/contracts/arctic_universe.schema.json` | reconciled; **one declared divergence** |
| `staging_daily_closes.schema.json` | *(none)* | producer has not published one; this pin stays sourced from `sources/contract.py::PriceBar` |

The two files per key stay SEPARATE rather than one replacing the other: the
producer's models the write path (`additionalProperties: false`), this repo's
records what the trader's read path consumes and what it hard-fails without.

**The declared divergence.** The producer's `arctic_universe.schema.json` is
`additionalProperties: false` and declares neither `atr_14_pct` nor `VWAP`.
`executor/price_cache.py` reads both, and `load_atr_14_pct` hard-fails without
the first — so a frame that is valid against the producer's own published
contract is one the morning planner refuses to trade on. The gap is in the
producer's contract; it is asserted in both directions by
`test_declared_divergence_is_still_true`, which goes red the day the producer
adds either column. Do not close it by editing `nousergon-data/contracts/`
from this repo — that is the parallel contract the P-07 pattern exists to
prevent.

**Refreshing a producer copy**: copy the file again from `nousergon-data`
`main` into `tests/contracts/producer/`, bump `PRODUCER_SHA` in the parity
test, run it, and reconcile whatever it reports.
