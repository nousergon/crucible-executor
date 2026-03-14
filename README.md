# Alpha Engine — Executor

Execution module for the Alpha Engine trading system. Reads AI-generated signals from S3 and places paper trades via Interactive Brokers.

This repo is the **executor** only — infrastructure for order routing, risk enforcement, and trade logging. The research pipeline that generates signals lives separately and is not included here.

---

## Architecture

```
[Research pipeline] → signals.json → S3
                                       ↓
                              executor/main.py
                                       ↓
                         ┌─────────────────────────┐
                         │  signal_reader.py        │  reads S3 signals
                         │  position_sizer.py       │  sizes positions
                         │  risk_guard.py           │  hard rule enforcement
                         │  ibkr.py                 │  ib_insync wrapper
                         │  trade_logger.py         │  SQLite + S3 backup
                         └─────────────────────────┘
                                       ↓
                              IB Gateway (paper)
```

EOD reconciliation (`eod_reconcile.py`) runs after market close, records daily P&L vs SPY, and emails a summary via AWS SES.

---

## Prerequisites

- **IB Gateway** running in paper mode on port 4002 (tested with v10.44)
- **IBC** for headless/automated login
- **AWS credentials** with:
  - Read access to your signals S3 bucket (`signals/` prefix)
  - Read/write access to your trades S3 bucket
  - SES send permission for the EOD email
- Python 3.11+

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/YOUR_USERNAME/alpha-engine.git
cd alpha-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/risk.yaml.example config/risk.yaml
# Edit config/risk.yaml — fill in your S3 bucket names, email, and db path
```

`config/risk.yaml` is gitignored and never committed.

### 3. Test IB Gateway connection

```bash
python executor/connection_test.py
```

### 4. Run

```bash
# Dry run — prints orders without placing them
python executor/main.py --dry-run

# Live paper trading
python executor/main.py
```

---

## Signals format

The executor expects a `signals/{YYYY-MM-DD}/signals.json` file in S3:

```json
{
  "date": "2026-03-06",
  "market_regime": "bull",
  "sector_ratings": {
    "Technology": {"rating": "overweight", "modifier": 1.2, "rationale": "..."}
  },
  "universe": [
    {
      "ticker": "PLTR",
      "sector": "Technology",
      "signal": "ENTER",
      "rating": "BUY",
      "score": 78.5,
      "conviction": "rising",
      "price_target_upside": 0.18,
      "thesis_summary": "..."
    }
  ],
  "buy_candidates": []
}
```

Valid `signal` values: `ENTER`, `EXIT`, `REDUCE`, `HOLD`.

---

## Risk rules

All parameters are in `config/risk.yaml`. Key defaults:

| Rule | Default |
|---|---|
| Max position size | 5% NAV |
| Max sector exposure | 25% NAV |
| Max total equity | 90% NAV |
| Graduated drawdown (0% to -3%) | Full sizing |
| Graduated drawdown (-3% to -5%) | 50% sizing |
| Graduated drawdown (-5% to -8%) | 25% sizing |
| Drawdown circuit breaker (beyond -8%) | Full halt |
| Bear regime position cap | 2.5% NAV |
| Min signal score to enter | 70 |

---

## Strategy layer

The executor includes a strategy layer (`executor/strategies/`) that adds backtestable quantitative exit and risk rules. These operate on OHLCV price data and require no LLM calls.

### ATR trailing stop

Exits a position when price falls below `highest_high_since_entry - ATR(14) * 3.0`. The stop adapts to each stock's volatility — wide for volatile names, tight for calm ones.

### Time-based exit decay

- After 5 trading days with signal = HOLD: reduces position 50%
- After 10 trading days with signal = HOLD: full exit
- Resets if Research reaffirms the position (signal = ENTER)

### Graduated drawdown response

Replaces the original binary -8% circuit breaker with tiered sizing:

| Drawdown | Sizing multiplier |
|---|---|
| 0% to -3% | 1.0 (full) |
| -3% to -5% | 0.5 (half) |
| -5% to -8% | 0.25 (quarter) |
| Beyond -8% | 0.0 (full halt) |

All strategy parameters are configurable in `config/risk.yaml` under the `strategy` key.

---

## Cron schedule (EC2)

```
# Morning trading run (6:30am PT = 9:30am ET market open)
30 6 * * 1-5  python /path/to/alpha-engine/executor/main.py >> /var/log/executor.log 2>&1

# EOD reconciliation (4:05pm ET)
5 21 * * 1-5  python /path/to/alpha-engine/executor/eod_reconcile.py >> /var/log/eod.log 2>&1
```

---

## Repository structure

```
executor/
  main.py           # daily trading loop
  signal_reader.py  # reads signals.json from S3
  risk_guard.py     # hard rule enforcement + graduated drawdown response
  position_sizer.py # equal-weight base sizing with sector/conviction/upside/drawdown adjustments
  ibkr.py           # ib_insync wrapper (NAV, positions, prices, orders)
  trade_logger.py   # SQLite trades.db + S3 backup + entry_date lookup
  price_cache.py    # loads OHLCV from predictor S3 slim cache for ATR
  eod_reconcile.py  # daily P&L vs SPY, emails summary via SES
  eod_emailer.py    # HTML/plain email builder
  connection_test.py # quick IB Gateway connection test
  strategies/       # strategy layer (added 2026-03-14)
    __init__.py     # module docstring
    config.py       # strategy defaults + YAML override loader
    exit_manager.py # ATR trailing stops + time-based exit decay
config/
  risk.yaml.example # template — copy to risk.yaml and fill in values
infrastructure/
  deploy.sh         # AWS infrastructure setup
  s3-policy.json    # IAM S3 permissions policy
  trust-policy.json # IAM role trust policy
requirements.txt
```

---

## Stack

| Layer | Tool |
|---|---|
| Broker | Interactive Brokers (paper account via IB Gateway) |
| IBKR client | ib_insync |
| Signal source | S3 (JSON, written by separate research pipeline) |
| Trade log | SQLite + S3 backup |
| Cloud | AWS (S3, SES) |
| EOD email | AWS SES |

---

## License

MIT
