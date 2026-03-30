# Nous Ergon: Alpha Engine

**See the Nous Ergon public dashboard at [nousergon.ai](https://nousergon.ai) and blog series on [Hashnode](https://nousergon.ai/blog)**

**Nous Ergon** (νοῦς ἔργον — "intelligence at work") is a fully autonomous trading system that combines AI-driven research, quantitative prediction, and rule-based execution to generate market alpha.

```
Alpha = Portfolio Return − SPY Return
```

The system targets sustained outperformance against the S&P 500 by splitting the problem into three layers, each matched to the right tool:

| Layer | Tool | Role |
|-------|------|------|
| **Research** | LLM agents (Claude) | Judgment over unstructured data — news, analyst reports, macro context |
| **Prediction** | Machine learning (LightGBM) | Pattern recognition over structured numerical features |
| **Execution** | Deterministic rules | Hard risk constraints that never get creative |

---

## System Architecture

Five modules run on AWS, connected through a shared S3 bucket. Each module reads its inputs from S3 and writes its outputs back — no shared state beyond the bucket.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEEKLY CADENCE (Monday)                          │
│                                                                     │
│  Research ──── scan 900 tickers, rotate population, write signals  │
│  Predictor Training ──── retrain on multi-year history, promote    │
│  Backtester ──── signal quality + weight optimization + param sweep │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DAILY CADENCE (Mon–Fri)                           │
│                                                                     │
│  Predictor (6:15 AM PT) ──── reads latest signals.json from S3     │
│       │                                                             │
│       ▼  predictions.json                                           │
│  Executor (6:20 AM PT) ──── order book ───► entries + exits        │
│       │                                                             │
│       ▼  order book (approved entries + urgent exits + stops)       │
│  Intraday Daemon (6:25 AM – 1:00 PM PT) ──── sole order executor  │
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

Autonomous investment research pipeline. Five LLM agents orchestrated by LangGraph maintain rolling investment theses on a configurable universe of tracked stocks and scan ~900 S&P 500/400 tickers weekly for top buy candidates.

- Quantitative filter reduces ~900 tickers to a shortlist (no LLM calls)
- Ranking agent (Sonnet) selects top candidates from the filtered set
- Per-ticker agents (news + research) run independently on every candidate (Haiku)
- Macro agent (Sonnet) assesses market environment and sector conditions
- Consolidator (Sonnet) synthesizes into a morning research brief via email
- Outputs composite attractiveness scores (0–100) per ticker as `signals.json`

### 2. Predictor — [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor)

LightGBM model that predicts 5-day market-relative returns for each ticker. Produces directional predictions (UP/FLAT/DOWN) with confidence scores.

- Engineered features across technical indicators, macro context, volume, and cross-sectional measures
- Trains on sector-neutral labels (stock returns minus sector ETF returns)
- Weekly retraining with walk-forward validation; weights promote only if IC gate passes
- Veto gate: high-confidence DOWN predictions override BUY signals from Research

### 3. Executor — [`alpha-engine`](https://github.com/cipher813/alpha-engine) *(this repo)*

Reads signals and predictions from S3, applies hard risk rules, sizes positions, and writes the intraday order book. The daemon is the sole order executor — no orders are placed at market open. Entry timing is optimized via technical triggers; exits execute immediately.

- Morning planner: signal ingestion, risk evaluation, position sizing → writes order book (no orders placed)
- Intraday daemon: sole order executor using 15-min delayed streaming via IB Gateway
- Graduated drawdown response with configurable halt threshold
- ATR-based trailing stops (volatility-adaptive) with time-decay exit rules
- Broker-side trailing stops via bracket orders for crash protection
- Telegram push notifications for all intraday trades
- Auto-tuned by backtester via S3-delivered `config/executor_params.json`

### 4. Backtester — [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester)

The system's learning mechanism. Validates signal quality, runs attribution analysis, and autonomously recommends parameter updates that flow back to upstream modules.

- Signal quality: measures BUY signal accuracy at configurable horizons
- Attribution: correlates sub-scores with outperformance outcomes
- Weight optimization: adjusts Research scoring weights with conservative guardrails
- Parameter sweep: randomized search across executor parameters, ranked by Sharpe ratio
- Veto threshold calibration: sweeps predictor confidence thresholds

### 5. Dashboard — [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard)

Read-only Streamlit application for monitoring the full system: portfolio performance vs SPY, signal quality trends, per-ticker research timelines, backtester results, and predictor metrics.

---

## Getting Started

Each module has its own README with a Quick Start section. The table below shows what you need to configure for each:

| Module | Config Files to Create | First Command |
|--------|----------------------|---------------|
| [Research](https://github.com/cipher813/alpha-engine-research) | `.env`, `config/universe.yaml`, `config/scoring.yaml`, `config/prompts/` + 13 proprietary source files | `python3 main.py --dry-run --skip-scanner` |
| [Predictor](https://github.com/cipher813/alpha-engine-predictor) | `config/predictor.yaml` | `python train_gbm.py --data-dir data/cache` |
| [Executor](https://github.com/cipher813/alpha-engine) | `config/risk.yaml` | `python executor/main.py --dry-run` |
| [Backtester](https://github.com/cipher813/alpha-engine-backtester) | `config.yaml` | `python backtest.py --mode signal-quality` |
| [Dashboard](https://github.com/cipher813/alpha-engine-dashboard) | None (works with defaults) | `streamlit run app.py` |

---

## Executor Quick Start (This Repo)

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
executor/order_book.py        # JSON-based intraday order book (entries, urgent exits, stops)
executor/signal_reader.py     # Reads signals.json from S3
executor/risk_guard.py        # 8-rule enforcement + graduated drawdown
executor/position_sizer.py    # Equal-weight base with 9 sizing adjustments
executor/ibkr.py              # IB Gateway wrapper (ib_insync)
executor/bracket_orders.py    # BUY + trailing stop as parent/child IB orders
executor/strategies/          # ATR trailing stops, time-decay, profit-take, momentum exits
executor/price_monitor.py     # 15-min delayed streaming subscriptions
executor/intraday_exit_manager.py  # Intraday exit rules (trail, profit-take, collapse)
executor/entry_triggers.py    # Intraday entry triggers (pullback, VWAP, support, expiry)
executor/notifier.py          # Telegram push notifications for trades
executor/eod_reconcile.py     # EOD P&L vs SPY + email
executor/price_cache.py       # OHLCV from predictor S3 slim cache for ATR
config/risk.yaml.example      # Safe template — copy to config/risk.yaml
```

---

## Executor Architecture

### Daily Execution Flow

The executor operates in three phases: a morning planner that writes the order book, an intraday daemon that executes all trades, and EOD reconciliation.

| Step | Time (PT) | What happens |
|------|-----------|--------------|
| **Morning Planner** | ~6:20 AM | Read signals, evaluate exits, apply risk rules, size positions → write order book (no orders) |
| **Intraday Daemon** | ~6:25 AM – 1:00 PM | Wait for market open (6:30 AM PT). Execute urgent exits immediately. Monitor entries for technical triggers. Monitor stops for exit rules. |
| **EOD Reconcile** | 1:05 PM | Capture final NAV, compute daily return vs SPY, log alpha, send email report |

The morning planner is the **decision-maker** (what to buy/sell and how much). The daemon is the **sole executor** (when to buy/sell during the day, with a hard safety check that the connected account is paper). Both write to the same SQLite trade log and S3 backup.

### Decision Pipeline

Every ENTER signal flows through this deterministic pipeline:

```
signals.json (S3)
       │
       ▼
Signal Reader ──── read today's signals; fall back up to 5 prior trading days
       │
       ▼
Exit Manager ──── evaluate held positions against 5 exit strategies:
  1. ATR trailing stop (volatility-adaptive)
  2. Fallback stop (fixed % when ATR unavailable)
  3. Time-based decay (reduce/exit stale HOLD positions)
  4. Profit-taking (trim at configurable gain threshold)
  5. Momentum exit (20d momentum flip + RSI below threshold)
  + Sector-relative veto (skip exit if stock outperforms sector)
       │
       ▼
Risk Guard ──── 8 rule layers (all must pass):
  1. Score minimum (score >= min_score_to_enter)
  2. Conviction gate (blocks "declining" conviction)
  3. Momentum gate (20d return must exceed threshold)
  4. Graduated drawdown (tiered sizing reduction → halt at circuit breaker)
  5. Max single position (% of NAV, adjustable by regime)
  6. Bear regime block (blocks new entries in underweight sectors)
  7. Sector exposure limit (default 25% NAV)
  8. Cross-ticker correlation (alerts on concentrated correlated holdings)
       │
       ▼
Position Sizer ──── compute shares:
  base_weight = 1 / n_enter_signals
  × sector_adj (overweight/market/underweight)
  × conviction_adj (declining → reduced)
  × upside_adj (low price target → reduced)
  × confidence_adj (from GBM p_up, continuous scaling)
  × atr_adj (inverse volatility sizing)
  × drawdown_multiplier (from graduated drawdown tier)
  × staleness_adj (decay for aged signals)
  × earnings_adj (reduced sizing near earnings)
  → capped at max_position_pct
       │
       ▼
Order Book ──── write approved entry to order_book.json with:
  - shares, sizing factors, ATR value
  - entry triggers (pullback %, VWAP discount, support level)
  - full metadata for daemon's trade logging
       │
       ▼
Daemon ──── waits for technical trigger, then places bracket order + logs trade
```

### Intraday Daemon

The daemon is the **sole order executor**. It starts after the morning planner completes, waits for market open (9:30 AM ET / 6:30 AM PT), then runs until market close. It connects to IB Gateway on a separate `clientId` (2) to avoid conflicts with the planner (clientId 1). A runtime safety check verifies the connected account starts with "D" (paper prefix) before placing any orders.

**Execution phases:**

1. **Phase 0 — Urgent exits** (immediately at market open): EXIT and REDUCE signals from Research and strategy rules execute as market orders with no trigger delay. Risk reduction is never deferred.
2. **Phase 1+2 — Entry triggers + stop monitoring** (polling loop until market close): Pending entries wait for technical triggers. Active stops are monitored for exit rules.

**Price monitoring:** Subscribes to IB Gateway's free 15-minute delayed streaming data (`reqMarketDataType(3)`) for all tracked tickers. Prices update via `pendingTickersEvent` callbacks.

**Exit rules** (evaluated on each price update):

| Rule | Trigger | Action |
|------|---------|--------|
| ATR trailing stop | Price drops below `high_water - ATR × multiple` | SELL full position |
| Profit-taking | Price up > 8% from entry | SELL 50% position |
| Intraday collapse | Price drops > 5% from intraday high | SELL full position |
| Time-based tightening | After 3+ days held, tighten trail to 1.5× ATR | Update stop |

**Entry triggers** (any one fires — OR logic):

| Trigger | Logic | Default |
|---------|-------|---------|
| Pullback entry | Price drops ≥ 2% from intraday high | `pullback_pct: 0.02` |
| VWAP discount | Price < VWAP by ≥ 0.5% | `vwap_discount_pct: 0.005` |
| Support bounce | Price within 1% of 20-day support level | `support_pct: 0.01` |
| Time expiry | No trigger by 3:30 PM ET → execute at market | `expiry_time: "15:30"` |

**Notifications:** Every intraday trade sends a Telegram push notification via bot API (fire-and-forget, never blocks execution).

### Data Sources

The executor consumes six data streams — all read-only, no feedback during execution:

| Source | What | Updated |
|--------|------|---------|
| `signals/{date}/signals.json` | Per-ticker signal (ENTER/EXIT/REDUCE/HOLD), score, conviction, price target, sector rating | Weekly (Monday, by Research) |
| `predictor/predictions/{date}.json` | Per-ticker predicted direction, confidence, predicted alpha | Daily (by Predictor) |
| IBKR account state | Live NAV, positions, current prices | Real-time via IB Gateway |
| `predictor/price_cache_slim/*.parquet` | 2-year OHLCV per ticker (for ATR computation) | Weekly |
| `trades.db` (SQLite) | Peak NAV, entry dates, trade history | After each execution |
| `config/executor_params.json` (S3) | Backtester-tuned parameters | Weekly (Monday, by Backtester) |

EXIT and REDUCE signals from Research always bypass all risk rules — reducing exposure is never blocked.

### EC2 Infrastructure (Two Instances)

The system runs on two EC2 instances to separate always-on hosting from market-hours trading:

**Micro instance (t3.micro, 24/7) — dashboard host:**

| Process | Type | Port |
|---------|------|------|
| Nginx (reverse proxy + SSL) | Always-on | 80, 443 |
| nousergon.ai (Streamlit) | Always-on | 8502 |
| dashboard.nousergon.ai (Streamlit) | Always-on | 8501 |

**Trading instance (t3.small, market hours only) — started/stopped by EventBridge:**

| Process | Type | Trigger |
|---------|------|---------|
| Xvfb (virtual display) | systemd | On boot |
| IB Gateway (paper account, IBC) | systemd | After Xvfb |
| Morning planner (`main.py`) | systemd oneshot | After IB Gateway ready (port 4002 poll) |
| Intraday daemon (`daemon.py`) | systemd | After morning planner |
| EOD reconcile (`eod_reconcile.py`) | systemd timer | 1:05 PM PT |

**EventBridge Scheduler (serverless, no cron):**

| Schedule | Time | Target |
|----------|------|--------|
| Start trading instance | Weekdays 6:15 AM PT | EC2 StartInstances |
| Stop trading instance | Weekdays 1:30 PM PT | EC2 StopInstances |
| Backtester spot launch | Mondays 08:00 UTC | SSM RunCommand → micro |

**IB Gateway:** Uses IBC (IB Controller) with paper account credentials stored in `~/ibc/config.ini`. The paper account has its own dedicated username/password, separate from the live brokerage account. Paper accounts do not require 2FA. The daemon includes a runtime safety check that hard-exits if connected to a non-paper account.

---

## Local Testing

Test executor changes locally before deploying to EC2. No IB Gateway connection required.

```bash
# Simulate mode: real signals from S3, synthetic IB positions, no orders placed
python executor/main.py --simulate

# Dry run on EC2: real IB prices + positions, no order book written
python executor/main.py --dry-run

# Validate signals.json for executor compatibility (local file or S3)
python tests/validate_signals.py --s3 2026-03-24
python tests/validate_signals.py path/to/signals.json
```

**Preprod workflow:**
1. Make code changes
2. `python executor/main.py --simulate` — verify no crashes, review planned orders
3. `--dry-run` on EC2 (after market close) — verify with real IB data
4. Deploy live on next trading day

### Portfolio Reset

Reset the portfolio to a clean $1M state (e.g., after tuning is complete and you want a fresh track record).

```bash
# Preview what will happen (no changes made)
bash infrastructure/reset-portfolio.sh --dry-run

# Execute the full reset (DESTRUCTIVE — archives all trade history first)
bash infrastructure/reset-portfolio.sh --live
```

**Reset process:**
1. Reset IB paper account balance at [IB Account Management](https://www.interactivebrokers.com) (manual — no API)
2. Stop the trading instance
3. Run `reset-portfolio.sh --live` — archives trade history to S3, writes clean db/csv
4. Start the trading instance — first EOD reconcile creates the new inception row
5. Dashboard automatically picks up the new inception date

All historical data is preserved in `s3://alpha-engine-executor/trades/archive/{timestamp}/`.

---

## Auto-Optimization

The backtester writes three S3 config files that upstream modules read on cold-start, closing the feedback loop automatically:

| S3 Key | Written By | Read By | Controls |
|--------|-----------|---------|----------|
| `config/scoring_weights.json` | Backtester | Research | Sub-score composite weights |
| `config/executor_params.json` | Backtester | Executor | Risk parameters and sizing |
| `config/predictor_params.json` | Backtester | Predictor | Veto confidence threshold |
| `config/research_params.json` | Backtester | Research | Signal boost parameters (deferred until 200+ samples) |

**Propagation timing:** Backtester runs Saturday 08:00 UTC and writes configs to S3. Lambda-based modules (Research, Predictor) pick up new configs on next cold-start (typically their next scheduled invocation). EC2-based modules (Executor) pick up configs when `main.py` runs on next trading day. There is no mid-day hot-reload — config changes take effect on the next run cycle.

---

## Key Metrics

| Metric | What It Measures |
|--------|-----------------|
| Total alpha | Portfolio cumulative return − SPY cumulative return |
| Sharpe ratio | Risk-adjusted return (annualized) |
| Daily alpha | Portfolio daily return − SPY daily return |
| Signal accuracy | % of BUY signals beating SPY over configurable windows |
| GBM IC | Rank correlation of predicted vs actual forward returns |
| Max drawdown | Peak-to-trough portfolio decline |

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
├── config/scoring_weights.json          ← Auto-updated by Backtester → Research
├── config/executor_params.json          ← Auto-updated by Backtester → Executor
├── config/predictor_params.json         ← Auto-updated by Backtester → Predictor
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
| Notifications | Telegram Bot API |

---

## Cross-Module Opportunities

### Stale Predictions Propagation

If `daily_closes/{date}.parquet` is not written before inference runs, predictions use the slim cache only (potentially 1-2 days stale). The predictor tracks this via `price_freshness.max_age_days` in `predictor/metrics/latest.json`. The Dashboard Predictor page surfaces this metric. If `max_age_days > 2`, investigate whether `save_daily_closes()` is failing or the inference Lambda is running before market close.

### S3 Retry Standardization

S3 retry logic varies across modules. Current state (2026-03-30):

| Module | Retry | Pattern |
|--------|-------|---------|
| Research | Yes | Custom `retry.py` decorator (3 attempts, exponential backoff) |
| Predictor | No | One-shot S3 calls |
| Executor | No | One-shot S3 calls (signals read, trades backup) |
| Backtester | No | One-shot S3 calls (config writes, report upload) |
| Dashboard | Yes | `_s3_get_object()` retry loop (3 attempts, backoff, transient-only) |

Future: extract research's `retry.py` into a shared package or copy the pattern into predictor/backtester. Dashboard's retry is inline (appropriate for its single-file S3 loader). Priority is low — S3 throttling is rare at current request volumes.

---

## Related Modules

- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
