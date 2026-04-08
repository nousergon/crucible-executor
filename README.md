# Nous Ergon: Alpha Engine

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-152_passing-brightgreen.svg)]()

**[Nous Ergon](https://nousergon.ai)** (νοῦς ἔργον — "intelligence at work") is a fully autonomous trading system that combines AI-driven research, quantitative prediction, and rule-based execution to generate market alpha.

```
Alpha = Portfolio Return − SPY Return
```

**[Website](https://nousergon.ai)** | **[Blog](https://nousergon.ai/blog)** | **[Full Documentation Index](https://github.com/cipher813/alpha-engine-docs#readme)**

## Table of Contents

- [System Architecture](#system-architecture)
- [Modules](#modules)
- [S3 Layout](#s3-layout)
- [Key Metrics](#key-metrics)
- [Stack](#stack)
- [Executor Module](#executor)

## System Architecture

Six modules run on AWS, connected through a shared S3 bucket. Each module reads its inputs from S3 and writes its outputs back — no shared state beyond the bucket.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEEKLY CADENCE (Saturday)                         │
│                                                                     │
│  Data Phase 1 ──── prices (ArcticDB), macro, constituents          │
│  RAG Ingestion ──── SEC filings, 8-Ks, earnings → research KB     │
│  Research ──── scan 900 tickers, rotate population, write signals  │
│  Data Phase 2 ──── alternative data for promoted tickers           │
│  Predictor Training ──── meta-model retrain, walk-forward validate │
│  Backtester ──── signal quality + evaluation + param optimization  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DAILY CADENCE (Mon–Fri)                           │
│                                                                     │
│  Daily Data (6:05 AM PT) ──── daily closes + macro refresh         │
│  Predictor (6:07 AM PT) ──── reads signals, predicts 5d alpha     │
│       │  predictions.json                                           │
│  Executor (6:15 AM PT) ──── order book ───► entries + exits        │
│       │  order book (approved entries + urgent exits + stops)       │
│  Intraday Daemon (6:20 AM – 1:15 PM PT) ──── sole order executor  │
│       │  (market close)                                             │
│  EOD Reconcile (1:20 PM PT) ──── NAV, alpha ───► email + EC2 stop │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    ALWAYS-ON                                        │
│                                                                     │
│  Dashboard (Streamlit) ──── read-only monitoring via S3             │
└─────────────────────────────────────────────────────────────────────┘
```

Orchestrated by three AWS Step Functions (Saturday, weekday, EOD). S3 as the communication bus means any module can be replaced, rewritten, or tested independently.

## Modules

| # | Module | Repo | Role |
|---|--------|------|------|
| 1 | **Research** | [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) | 6 LLM sector teams + CIO scan ~900 stocks weekly, produce composite scores (0-100) as `signals.json` |
| 2 | **Predictor** | [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) | Meta-model (4 GBMs + ridge) predicts 5-day sector-relative alpha; veto gate blocks high-confidence DOWN |
| 3 | **Executor** | [`alpha-engine`](https://github.com/cipher813/alpha-engine) *(this repo)* | Morning planner + intraday daemon: risk rules, position sizing, technical entry triggers, trailing stops |
| 4 | **Backtester** | [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) | Weekly evaluation (component grades, P/R/F1) + 6 autonomous optimizers → S3 config feedback loop |
| 5 | **Dashboard** | [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) | Streamlit monitoring: portfolio performance, signal quality, system report card, execution evaluation |
| 6 | **Data** | [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) | Centralized data collection: ArcticDB price store (909 tickers), macro, alternative data |

Each module has its own README with Quick Start, Architecture, and S3 Contract sections.

## S3 Layout

All inter-module communication flows through a single S3 bucket:

```
s3://alpha-engine-research/
├── signals/{date}/signals.json          ← Research → Predictor + Executor
├── arcticdb/
│   ├── universe/                        ← 10y OHLCV, 909 tickers (ArcticDB)
│   └── universe_slim/                   ← 2y slices for inference (ArcticDB)
├── predictor/
│   ├── daily_closes/{date}.parquet      ← Daily OHLCV archive
│   ├── feature_store/{date}/            ← Pre-computed features (54 x 903 tickers)
│   ├── weights/                         ← GBM + meta-model weights
│   ├── predictions/{date}.json          ← Daily predictions
│   └── metrics/latest.json              ← Model performance
├── trades/
│   ├── trades_full.csv                  ← Complete trade audit log
│   └── eod_pnl.csv                      ← Daily NAV, return, alpha
├── backtest/{date}/                     ← Weekly: report, grades, trigger/shadow/exit/veto analysis
├── backtest/grade_history.json          ← 52-week rolling component grades
├── config/                              ← Auto-optimized by backtester
│   ├── scoring_weights.json             ← → Research
│   ├── executor_params.json             ← → Executor
│   ├── predictor_params.json            ← → Predictor
│   └── research_params.json             ← → Research (deferred)
├── health/{module}.json                 ← Module completion markers
└── research.db                          ← SQLite (signal history, theses, macro)
```

## Key Metrics

| Metric | What It Measures |
|--------|-----------------|
| Total alpha | Portfolio cumulative return − SPY cumulative return |
| Sharpe ratio | Risk-adjusted return (annualized) |
| Signal accuracy | % of BUY signals beating SPY over configurable windows |
| GBM IC | Rank correlation of predicted vs actual forward returns |
| Max drawdown | Peak-to-trough portfolio decline |
| System grade | Weekly A-F scorecard across all components (backtester) |

## Stack

| Component | Technology |
|-----------|------------|
| LLM provider | Anthropic Claude (Haiku-4.5 per-ticker, Sonnet-4.6 synthesis) |
| ML framework | LightGBM (meta-model: 4 specialized GBMs + ridge) |
| Agent orchestration | LangGraph |
| Price data | ArcticDB (S3-backed) + Polygon.io + yfinance fallback |
| Broker | Interactive Brokers (paper account via IB Gateway) |
| Cloud | AWS (Lambda, S3, SES, EC2, Step Functions, SSM, EventBridge) |
| Dashboard | Streamlit + Plotly |
| Secrets | AWS SSM Parameter Store (24 params under /alpha-engine/*) |
| Config | Private config repo (alpha-engine-config) with per-module search |

---

# Executor

> Reads signals and predictions from S3, applies hard risk rules, sizes positions, and executes orders via IB Gateway. The morning planner writes the order book; the daemon is the sole order executor.

**Part of the [Nous Ergon](https://nousergon.ai) autonomous trading system.**

## Quick Start

### Prerequisites

- Python 3.11+
- IB Gateway running in paper mode on port 4002
- AWS credentials with S3 read/write and SES send permission

### Setup

```bash
git clone https://github.com/cipher813/alpha-engine.git
cd alpha-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config/risk.yaml.example config/risk.yaml
# Edit config/risk.yaml — set S3 bucket names, email addresses, risk parameters

python executor/connection_test.py   # verify IB Gateway
python executor/main.py --dry-run    # full loop, no orders placed
```

### Key Files

```
executor/main.py              # Morning order-book planner (no orders placed)
executor/daemon.py            # Sole order executor — urgent exits + technical entry triggers
executor/order_book.py        # JSON-based intraday order book
executor/signal_reader.py     # Reads signals.json from S3
executor/risk_guard.py        # 9-rule enforcement + graduated drawdown
executor/position_sizer.py    # Equal-weight base with 9 sizing adjustments
executor/ibkr.py              # IB Gateway wrapper (ib_insync)
executor/entry_triggers.py    # Pullback, VWAP, support, graduated expiry
executor/intraday_exit_manager.py  # Trail, profit-take, collapse exits
executor/eod_reconcile.py     # EOD P&L vs SPY + email (Step Function orchestrated)
executor/strategies/          # ATR stops, time-decay, profit-take, momentum
config/risk.yaml.example      # Safe template — copy to config/risk.yaml
```

## Architecture

### Daily Execution Flow

| Step | Time (PT) | What happens |
|------|-----------|--------------|
| **Morning Planner** | ~6:15 AM | Read signals, evaluate exits, apply risk rules, size positions → write order book (no orders) |
| **Intraday Daemon** | ~6:20 AM – 1:15 PM | Execute urgent exits at open. Monitor entries for technical triggers. Monitor stops for exit rules. |
| **EOD Reconcile** | 1:20 PM | Capture NAV, compute daily return vs SPY, log alpha, send email. Step Function stops EC2 after. |

### Decision Pipeline

Every ENTER signal flows through this pipeline:

```
signals.json (S3)
       │
Signal Reader ──── read today's signals; fall back up to 5 prior trading days
       │
Exit Manager ──── evaluate held positions against 5 exit strategies
       │
Risk Guard ──── 9 rule layers (all must pass for entry)
       │
Position Sizer ──── compute shares (base × 9 adjustments, capped)
       │
Order Book ──── write approved entry with trigger levels + metadata
       │
Daemon ──── waits for technical trigger, places bracket order, logs trade
```

### Entry Triggers (Intraday, OR Logic)

| Trigger | Default | Fires When |
|---------|---------|------------|
| Pullback | 2% | (day_high - current) / day_high >= 2% |
| VWAP Discount | 0.5% | (vwap - current) / vwap >= 0.5% (previous day's VWAP from daily_closes) |
| Support Bounce | 1% | 0 <= (current - support) / support <= 1% |
| Graduated Expiry | 3-tier | Technical-only before 2 PM → graduated (price <= open+1%) 2-3:30 PM → unconditional 3:55 PM |

### Risk Guard Rules

1. Score minimum (score >= min_score_to_enter)
2. Conviction gate (blocks "declining" conviction)
3. Momentum gate (20d return must exceed threshold)
4. Graduated drawdown (tiered sizing reduction → halt at circuit breaker)
5. Max single position (% of NAV, adjustable by regime)
6. Bear regime block (blocks entries in underweight sectors)
7. Sector exposure limit (default 25% NAV)
8. Cross-ticker correlation (alerts on concentrated correlated holdings)
9. Predictor veto (high-confidence DOWN prediction overrides BUY)

### S3 Contract

**Reads:**
| Path | Source | Content |
|------|--------|---------|
| `signals/{date}/signals.json` | Research | Per-ticker signal, score, conviction, sector |
| `predictor/predictions/{date}.json` | Predictor | Direction, confidence, predicted alpha |
| `config/executor_params.json` | Backtester | Auto-tuned risk parameters |
| ArcticDB `universe_slim` | Data | 2y OHLCV for ATR computation |

**Writes:**
| Path | Content |
|------|---------|
| `trades/trades_full.csv` | Complete trade audit log |
| `trades/eod_pnl.csv` | Daily NAV, return, alpha |
| `trades/shadow_book.csv` | Risk guard blocked entries |
| `health/executor.json` | Module health marker |

## Testing

```bash
# Simulate mode: real signals from S3, synthetic IB positions, no orders placed
python executor/main.py --simulate

# Dry run on EC2: real IB prices + positions, no order book written
python executor/main.py --dry-run

# Full test suite (152 tests)
pytest tests/ -v
```

## Deployment

- **EC2 trading instance** (t3.small, market hours only): Started by weekday Step Function, stopped by EOD Step Function
- **Boot sequence** (systemd): boot-pull → IB Gateway → morning planner → daemon
- **Deploy gate**: boot-pull runs `--dry-run` after code update, auto-rollback on failure
- **Config**: Reads from alpha-engine-config (private repo) first, falls back to local risk.yaml
- **Secrets**: SSM Parameter Store (`/alpha-engine/*`), loaded at runtime

## Related Modules

- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — Meta-model predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis, evaluation framework, and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard
- [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) — Centralized data collection and ArcticDB price store
- [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs) — Documentation index and system audits

## License

MIT — see [LICENSE](LICENSE).
