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
one, for a future bump) merged first. As of this writing (2026-09-14) no
producer-side `contracts/` copy of any of these three schemas exists yet in
`nousergon-data` — each file here is sourced directly from the producer's
current write path (cited in its own `description`) and must be reconciled
against the producer's own copy once P-07 (`data_collection_plan_260914.md`
§8.3) lands there.
