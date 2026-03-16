# Nous Ergon: Alpha Engine
__See Nous Ergon blog series on [Hashnode](https://nous-ergon.hashnode.dev/)__

**Nous Ergon** (νοῦς ἔργον — "intelligence at work") is a fully autonomous trading system that combines AI-driven research, quantitative prediction, and rule-based execution to generate market alpha.

```
Alpha = Portfolio Return − SPY Return
```

The system targets sustained outperformance against the S&P 500 by splitting the problem into three layers, each matched to the right tool:

| Layer | Tool | Role |
|-------|------|------|
| **Research** | LLM agents (Claude) | Judgment over unstructured data — news, analyst reports, macro context |
| **Prediction** | Machine learning ensemble (LightGBM) | Pattern recognition over structured numerical features |
| **Execution** | Deterministic rules | Hard risk constraints that never get creative |

---

## Architecture

Five modules run on AWS, connected through a shared S3 bucket. Each module reads its inputs from S3 and writes its outputs back — no shared state beyond the bucket.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEEKLY CADENCE (Sunday/Monday)                    │
│                                                                     │
│  Research ──── scan 900 tickers, rotate population, write signals  │
│  Predictor Training ──── retrain on 10y history, promote if IC >   │
│  Backtester ──── signal quality + weight optimization + param sweep │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DAILY CADENCE (Mon–Fri)                           │
│                                                                     │
│  Predictor (6:15 AM PT) ──── reads latest signals.json from S3     │
│       │                                                             │
│       ▼  predictions.json                                           │
│  Executor (6:30 AM PT) ──── trades ───► Interactive Brokers        │
│       │                                                             │
│       ▼  (market close)                                             │
│  EOD Reconcile (1:05 PM PT) ──── NAV, return, alpha ───► email     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    ALWAYS-ON                                        │
│                                                                     │
│  Dashboard (Streamlit) ──── read-only monitoring via S3             │
└─────────────────────────────────────────────────────────────────────┘
```

S3 as the communication bus means any module can be replaced, rewritten, or tested independently. They agree on a JSON schema, and S3 handles the rest.

---

## Modules

### 1. Research — [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research)

Autonomous investment research pipeline. Five LLM agents orchestrated by LangGraph maintain rolling investment theses on ~20 tracked stocks and scan ~900 S&P 500 and S&P 400 tickers weekly for the top buy candidates.

- Quantitative filter reduces ~900 tickers to ~50 candidates (no LLM calls)
- Ranking agent (Sonnet) selects the top ~35 from the filtered set
- Per-ticker agents (news sentiment + analyst research) run independently on every candidate (Haiku)
- Macro agent (Sonnet) assesses market environment and sector conditions
- Consolidator agent (Sonnet) synthesizes all analyses into a research brief
- Outputs composite attractiveness scores (0–100) per ticker as `signals.json`

### 2. Predictor — [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor)

LightGBM gradient-boosted model that predicts 5-day market-relative returns for each ticker. Produces directional predictions (UP/FLAT/DOWN) with confidence scores.

- 29 engineered features across technical indicators, macro context, volume analysis, and cross-sectional measures
- Trains on sector-neutral labels (stock returns minus sector ETF returns)
- Weekly retraining with 10 years of price history; new weights promote only if IC > 0.03
- Veto gate: high-confidence DOWN predictions override BUY signals from Research

### 3. Executor — [`alpha-engine`](https://github.com/cipher813/alpha-engine) *(this repo)*

Reads signals and predictions from S3, applies hard risk rules, sizes positions, and executes market orders on Interactive Brokers (paper trading).

- Graduated drawdown response: full → half → quarter sizing → full halt at -8%
- ATR-based trailing stops (volatility-adaptive) with time-decay exit rules
- Position caps (5% NAV, 2.5% in bear), sector concentration limits (25% NAV)
- Deterministic execution — no reasoning, no prediction, just parameter application

### 4. Backtester — [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester)

The system's learning mechanism. Validates signal quality, runs attribution analysis, and recommends parameter updates that flow back to upstream modules.

- Signal quality: measures BUY signal accuracy at 10-day and 30-day horizons
- Attribution: correlates sub-scores (news vs. research) with outperformance outcomes
- Weight optimization: adjusts Research scoring weights with conservative guardrails
- Parameter sweep: randomized search across executor parameters, ranked by Sharpe ratio
- Veto threshold calibration: sweeps predictor confidence thresholds against historical outcomes

### 5. Dashboard — [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard)

Read-only Streamlit application for monitoring the full system: portfolio performance vs SPY, signal quality trends, per-ticker research timelines, backtester results, and predictor metrics.

---

## Key Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Daily alpha | Portfolio return − SPY return | Positive |
| Signal accuracy (10d) | % of BUY signals beating SPY over 10 days | > 55% |
| Signal accuracy (30d) | % of BUY signals beating SPY over 30 days | > 55% |
| GBM IC | Rank correlation of predicted vs actual 5d returns | > 0.03 |
| Max drawdown | Peak-to-trough portfolio decline | < −8% |
| Sharpe ratio | Risk-adjusted return (annualized) | > 1.0 |

---

## S3 Layout

All inter-module communication flows through a single S3 bucket:

```
s3://alpha-engine-research/
├── signals/{date}/signals.json          ← Research → Predictor + Executor
├── archive/universe/{TICKER}/           ← Research theses
├── archive/candidates/{TICKER}/         ← Buy candidate theses
├── archive/macro/                       ← Macro environment reports
├── predictor/
│   ├── price_cache/*.parquet            ← 10y OHLCV (weekly refresh)
│   ├── price_cache_slim/*.parquet       ← 2y slice for inference
│   ├── daily_closes/{date}.parquet      ← Daily OHLCV archive
│   ├── weights/gbm_latest.txt           ← Active GBM model
│   ├── predictions/{date}.json          ← Daily predictions
│   └── metrics/latest.json              ← Model performance
├── trades/
│   ├── trades_full.csv                  ← Complete trade audit log
│   └── eod_pnl.csv                      ← Daily NAV, return, alpha
├── backtest/{date}/                     ← Weekly backtester outputs
├── config/scoring_weights.json          ← Auto-updated by backtester
└── research.db                          ← SQLite (signal history, theses)
```

---

## Stack

| Component | Technology |
|-----------|------------|
| LLM provider | Anthropic Claude (Haiku for per-ticker, Sonnet for synthesis) |
| ML framework | LightGBM |
| Agent orchestration | LangGraph |
| Broker | Interactive Brokers (paper account via IB Gateway) |
| Cloud | AWS (Lambda, S3, SES, EC2) |
| Dashboard | Streamlit + Plotly |
| Databases | SQLite per-module (backed up to S3) |

---

## Executor Setup (this repo)

### Prerequisites

- IB Gateway running in paper mode on port 4002
- AWS credentials with S3 read/write and SES send permission
- Python 3.11+

### Quick start

```bash
git clone https://github.com/cipher813/alpha-engine.git
cd alpha-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/risk.yaml.example config/risk.yaml
# Edit config/risk.yaml with your S3 bucket names and email

python executor/connection_test.py   # verify IB Gateway
python executor/main.py --dry-run    # full loop, no orders
```

---

## License

MIT
