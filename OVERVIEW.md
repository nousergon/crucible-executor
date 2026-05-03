# alpha-engine (executor) — Code Index

> Index of entry points, key files, and data contracts. Companion to [README.md](README.md). System overview lives in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs).

## Module purpose

Risk-gated trade executor — reads signals + predictions, sizes positions, routes orders through IB Gateway via a morning planner + intraday daemon split.

## Entry points

| File | What it does |
|---|---|
| [`executor/main.py`](executor/main.py) | Morning order-book planner — never places orders |
| [`executor/daemon.py`](executor/daemon.py) | Sole order executor — urgent exits + intraday entry triggers |
| [`executor/eod_reconcile.py`](executor/eod_reconcile.py) | 1:20 PM PT — NAV, alpha, positions, EOD email |
| [`executor/connection_test.py`](executor/connection_test.py) | IB Gateway connection check |
| [`executor/liquidate_all.py`](executor/liquidate_all.py) | Emergency manual liquidation |

## Where things live

| Concept | File |
|---|---|
| Signal reader (signals.json + predictions.json from S3) | [`executor/signal_reader.py`](executor/signal_reader.py) |
| Hard risk-rule enforcement + graduated drawdown | [`executor/risk_guard.py`](executor/risk_guard.py) |
| Position sizer (equal-weight + adjustments) | [`executor/position_sizer.py`](executor/position_sizer.py) |
| IB Gateway wrapper (NAV, positions, prices, orders) | [`executor/ibkr.py`](executor/ibkr.py) |
| Trade logger (SQLite + S3 backup) | [`executor/trade_logger.py`](executor/trade_logger.py) |
| Order book (intraday JSON state) | [`executor/order_book.py`](executor/order_book.py) |
| Bracket orders (BUY + trailing stop as parent/child) | [`executor/bracket_orders.py`](executor/bracket_orders.py) |
| Intraday entry triggers (pullback / VWAP / support / expiry) | [`executor/entry_triggers.py`](executor/entry_triggers.py) |
| Intraday exit manager (trail / profit-take / collapse) | [`executor/intraday_exit_manager.py`](executor/intraday_exit_manager.py) |
| Strategy layer — ATR stops + time decay | [`executor/strategies/exit_manager.py`](executor/strategies/exit_manager.py) |
| Strategy config (YAML override) | [`executor/strategies/config.py`](executor/strategies/config.py) |
| 15-min delayed price subscriptions | [`executor/price_monitor.py`](executor/price_monitor.py) |
| Price cache loader (ArcticDB slim → ATR) | [`executor/price_cache.py`](executor/price_cache.py) |
| Feature lookup (ArcticDB) | [`executor/feature_lookup.py`](executor/feature_lookup.py) |
| Decision deciders (entry / hold / reduce / exit logic) | [`executor/deciders.py`](executor/deciders.py) |
| EOD email builder (HTML + plain) | [`executor/eod_emailer.py`](executor/eod_emailer.py) |
| Telegram push notifications for trades | [`executor/notifier.py`](executor/notifier.py) |
| Snapshot capturer (intraday state) | [`executor/snapshot_capturer.py`](executor/snapshot_capturer.py) |
| Market hours helpers | [`executor/market_hours.py`](executor/market_hours.py) |
| Uptime tracker | [`executor/uptime_tracker.py`](executor/uptime_tracker.py) |
| Emergency shutdown handler | [`executor/emergency_shutdown.py`](executor/emergency_shutdown.py) |
| Config loader | [`executor/config_loader.py`](executor/config_loader.py) |

## Inputs / outputs

### Reads
| Source | Path |
|---|---|
| Research signals | `s3://alpha-engine-research/signals/{date}/signals.json` |
| Predictor predictions | `s3://alpha-engine-research/predictor/predictions/{date}.json` |
| Predictor veto threshold + auto-tuned params | `s3://alpha-engine-research/config/predictor_params.json` |
| Executor risk + sizing params | `s3://alpha-engine-research/config/executor_params.json` |
| Price slim cache (ATR + intraday lookups) | `s3://alpha-engine-research/arcticdb/universe_slim/` |

### Writes
| Destination | Path |
|---|---|
| Trade audit log | `s3://alpha-engine-research/trades/trades_full.csv` + SQLite `trades.db` |
| Daily NAV / alpha / positions | `s3://alpha-engine-research/trades/eod_pnl.csv` + SQLite `eod_pnl` table |
| Per-day intraday snapshots | `s3://alpha-engine-research/trades/snapshots/{date}/` |
| Health status (planner + daemon completion) | `s3://alpha-engine-research/health/executor_*.json` |

## Run modes

| Mode | Where | Command |
|---|---|---|
| Production planner | `ae-trading` EC2 (boot via systemd, ~6:15 AM PT) | Weekday SF starts the EC2 instance |
| Production daemon | `ae-trading` EC2 (boot via systemd, ~6:20 AM PT) | Boot-triggered; safety-net systemd timer at 9:29 AM ET |
| Production EOD | `ae-trading` EC2 (SSM, 1:20 PM PT) | EOD Step Function |
| Local dry run | venv | `python executor/main.py --dry-run` |
| Connection test | venv or EC2 | `python executor/connection_test.py` |
| Manual liquidation | venv or EC2 | `python executor/liquidate_all.py` |

Deploy: `git push origin main && ae-trading "cd ~/alpha-engine && git pull"`. IB Gateway runs in Docker (gnzsnz/ib-gateway-docker) with TOTP-based 2FA on paper account port 4002.

## Tests

`pytest tests/` covers risk-guard rules, position sizing math, signal-reader fallbacks, trade-logger SQLite roundtrips, intraday exit manager state machines, entry-trigger evaluation, EOD reconciliation, and bracket-order construction.
