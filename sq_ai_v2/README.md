# SimpleQuant AI v2

A world-class probabilistic trading system for Indian equity markets.

## Architecture Overview

```
sq_ai_v2/
├── config/           — Central settings (pydantic-settings, .env)
├── data/
│   ├── ingestion/    — Kite Connect + Yahoo Finance + synthetic fallback
│   ├── storage/      — QuestDB (time-series), PostgreSQL (fundamentals), Redis (cache/streams)
│   └── feature_store/— Point-in-time safe feature computation (35+ features)
├── models/
│   ├── ensemble/     — LightGBM, 1D-CNN, BiLSTM+Attention, GNN, HMM
│   ├── meta_learner.py    — Logistic Regression stacker
│   ├── calibration.py     — Platt / Isotonic probability calibration
│   └── train_pipeline.py  — End-to-end training orchestrator
├── signals/
│   ├── composite_signal.py— ML(60%) + Sentiment(20%) + Fundamentals(10%) + Regime(10%)
│   ├── sentiment.py        — FinBERT news scoring (NewsAPI / Finnhub)
│   ├── fundamentals.py     — Earnings/revenue surprise + macro index
│   └── feature_generation.py — GP feature engineering + drift detection (PSI)
├── risk/
│   ├── position_sizer.py  — Fractional Kelly + volatility targeting
│   ├── correlation.py     — Dynamic correlation overlay
│   └── stop_loss.py       — ATR trailing + hard + time stops
├── execution/
│   ├── broker.py          — PaperBroker / KiteLiveBroker abstraction
│   ├── order_manager.py   — Full order lifecycle + portfolio state
│   └── slippage_model.py  — Almgren-style market impact model
├── backtest/
│   ├── engine.py          — Event-driven, strict no-lookahead
│   └── walk_forward.py    — Rolling 3-year train / 3-month test
├── api/              — FastAPI REST + WebSocket (Prometheus metrics endpoint)
├── dashboard/        — Streamlit real-time dashboard
├── monitoring/       — Prometheus config + Grafana dashboard JSON
└── scripts/          — ingest_data.py, train_models.py, run_backtest.py
```

---

## Quick Start (Offline / Synthetic Data)

All components run **fully offline** using synthetic GBM price data when no API keys are set.

```bash
cd ~/0to100/sq_ai_v2

# 1. Install dependencies
make install
source venv/bin/activate

# 2. Copy env file (edit later with real keys)
cp .env.example .env

# 3. Ingest data (synthetic if no Kite key)
make ingest

# 4. Train all models
make train

# 5. Run backtest
make backtest

# 6. Launch API
make api
# → http://localhost:8000/docs

# 7. Launch dashboard (separate terminal)
make dashboard
# → http://localhost:8501
```

---

## Docker Setup (Databases + Monitoring)

```bash
# Start QuestDB, PostgreSQL, Redis, Prometheus, Grafana
make docker-up

# View logs
make docker-logs

# Stop all services
make docker-down
```

Service URLs after `make docker-up`:
| Service    | URL                        | Default Credentials |
|------------|----------------------------|---------------------|
| QuestDB    | http://localhost:9000      | none                |
| PostgreSQL | localhost:5432             | sqai / sqai_secret  |
| Redis      | localhost:6379             | no password         |
| Prometheus | http://localhost:9090      | none                |
| Grafana    | http://localhost:3000      | admin / admin       |

---

## Getting a Kite Connect API Key

1. Go to https://developers.kite.trade and create an app.
2. Note your **API Key** and **API Secret**.
3. Set them in `.env`:
   ```
   KITE_API_KEY=your_key
   KITE_API_SECRET=your_secret
   ```
4. Generate an access token (valid for 1 day):
   ```python
   from data.ingestion.kite_client import KiteClient
   client = KiteClient()
   print(client.generate_login_url())
   # Visit the URL, login, copy the request_token from the redirect URL
   token = client.complete_login("YOUR_REQUEST_TOKEN")
   # Add to .env: KITE_ACCESS_TOKEN=token
   ```

---

## Optional Data Sources

| Source        | Key Variable       | Free Tier  | What it adds                  |
|---------------|--------------------|------------|-------------------------------|
| Alpha Vantage | `ALPHA_VANTAGE_KEY`| 25 req/day | GDP, CPI, Fed rate (macro)    |
| NewsAPI       | `NEWSAPI_KEY`      | 100 req/day| Financial news for sentiment  |
| Finnhub       | `FINNHUB_KEY`      | 60 req/min | Company news + earnings data  |

Set these in `.env`. The system works without them (uses neutral 0.5 fallback).

---

## Running a Backtest

```bash
# Default: 2020-01-01 to 2023-12-31 on all symbols in UNIVERSE
make backtest

# Custom date range and specific symbols
python scripts/run_backtest.py --start 2021-01-01 --end 2022-12-31 --symbols RELIANCE,INFY,TCS

# Walk-forward validation (takes longer — trains on each fold)
python scripts/train_models.py --walk-forward
```

Results saved to `logs/`:
- `equity_curve.parquet` — equity value at each bar
- `trades.parquet` — full trade journal
- `backtest_stats.json` — Sharpe, win rate, max drawdown, etc.

---

## Live Paper Trading

```bash
# Start all infrastructure
make docker-up

# Ingest latest data (today's bars)
make ingest

# In terminal 1: start API
make api

# In terminal 2: start dashboard
make dashboard

# In terminal 3: run live loop (paper trading, no real orders)
python -c "
import sys; sys.path.insert(0, '.')
from data.ingestion.realtime import RealTimeIngestion
rt = RealTimeIngestion()
rt.start()
print('Synthetic tick generator running. Press Ctrl+C to stop.')
import time
while True: time.sleep(1)
"
```

Open the dashboard at http://localhost:8501 to see live signals.

---

## Signal Architecture

```
Raw OHLCV
    │
    ▼
FeatureStore (35 features: RSI, MACD, ATR, BB, ADX, CCI, OBV, VWAP...)
    │
    ├──► LightGBM ──────┐
    ├──► 1D CNN ─────────┤
    ├──► Bi-LSTM ────────┼──► MetaLearner ──► Calibration ──► base_prob
    └──► GNN ────────────┘
                         │
    HMM Regime ──────────┤  (×signal/risk multiplier)
                         │
    FinBERT Sentiment ───┤  (weight 20%)
                         │
    Earnings/Macro ──────┘  (weight 10%)
                         │
                         ▼
                  Final Probability [0,1]
                         │
              ┌──────────┼──────────┐
            <0.40      0.40–0.60  >0.60
             SELL        HOLD       BUY
                         │
                  Kelly Position Size
                         │
                  Volatility Targeting
                         │
                  ATR Trailing Stop
```

---

## Model Retraining

Models are saved to `models/saved/`. Retrain with:

```bash
# Quick retrain on full history
make train

# Walk-forward (production-grade, ~1 hour for 10 years of data)
python scripts/train_models.py --walk-forward
```

**Automated drift detection**: `FeatureGenerator.detect_drift()` computes Population Stability Index (PSI) per feature. PSI > 0.2 triggers a warning and should prompt retraining.

---

## Monitoring with Grafana

1. After `make docker-up`, open http://localhost:3000 (admin/admin).
2. The dashboard JSON at `monitoring/grafana_dashboard.json` is auto-provisioned.
3. Make sure the API is running (`make api`) so Prometheus can scrape `/metrics`.

Dashboard panels:
- **Portfolio Equity** — real-time equity curve
- **Sharpe Ratio gauge** — target > 1.5
- **Win Rate gauge** — alert below 45%
- **Signal Rate** — BUY/SELL/HOLD rates per 5 minutes
- **API Latency** — p50 and p99

---

## File-by-File Guide

| File | Purpose |
|------|---------|
| `config/settings.py` | All config loaded from `.env`. Single source of truth. |
| `data/ingestion/kite_client.py` | Kite Connect wrapper with synthetic fallback. |
| `data/ingestion/historical.py` | Downloads, caches (Parquet), writes to QuestDB. |
| `data/ingestion/realtime.py` | Kite WebSocket ticks → Redis Streams. |
| `data/storage/questdb_client.py` | ILP writes + REST reads for QuestDB. |
| `data/storage/postgres_client.py` | SQLAlchemy wrapper for earnings/macro/model registry. |
| `data/storage/redis_client.py` | Signal cache, pub/sub, tick stream reader. |
| `data/feature_store/store.py` | 35 point-in-time safe features. |
| `models/ensemble/lightgbm_model.py` | LightGBM binary classifier with incremental training. |
| `models/ensemble/cnn_model.py` | 1D CNN (stacked blocks + global avg pool). |
| `models/ensemble/lstm_model.py` | Bi-LSTM with attention. |
| `models/ensemble/gnn_model.py` | GCN over correlation graph (MLP fallback). |
| `models/ensemble/hmm_regime.py` | Gaussian HMM: bull/chop/bear detection. |
| `models/meta_learner.py` | LR stacker trained on OOS base-model predictions. |
| `models/calibration.py` | Platt scaling or Isotonic regression. |
| `models/train_pipeline.py` | End-to-end training orchestration. |
| `signals/feature_generation.py` | Normalised features + GP engineering + PSI drift. |
| `signals/sentiment.py` | FinBERT batch scoring + news API integration. |
| `signals/fundamentals.py` | Earnings/revenue/macro surprise signals. |
| `signals/composite_signal.py` | Blends all signals into final probability + action. |
| `risk/position_sizer.py` | Fractional Kelly + vol targeting + hard caps. |
| `risk/correlation.py` | Pairwise correlation matrix + position penalty. |
| `risk/stop_loss.py` | ATR trailing, hard, and time stops. |
| `execution/slippage_model.py` | Almgren volume-impact model + flat-rate option. |
| `execution/broker.py` | PaperBroker + KiteLiveBroker factory. |
| `execution/order_manager.py` | Order lifecycle, portfolio state, Redis publish. |
| `backtest/engine.py` | Event-driven backtester, no-lookahead. |
| `backtest/walk_forward.py` | Rolling fold training + testing + alerting. |
| `api/main.py` | FastAPI: REST endpoints + WebSocket + Prometheus. |
| `api/websocket_manager.py` | Redis pub/sub → WebSocket fan-out. |
| `dashboard/streamlit_app.py` | Real-time portfolio/signal/model/backtest UI. |

---

## Target Metrics (After Full Training)

| Metric | Target | v1 Baseline |
|--------|--------|------------|
| Sharpe ratio | > 1.5 | -0.30 |
| Win rate | > 55% | 60% |
| Max drawdown | < 15% | N/A |
| Profit factor | > 1.5 | 0.83 |
| Total return (annual) | > 15% | -1.21% |

---

## Known Limitations & Next Steps

1. **GNN requires torch_geometric** — falls back to MLP if not installed. Install with:
   ```bash
   pip install torch_geometric
   ```
2. **FinBERT download** — first run downloads ~400MB from HuggingFace. Set `HF_CACHE_DIR` to a persistent path.
3. **Live Kite trading** — requires daily access token refresh. Consider automating with APScheduler.
4. **Walk-forward takes ~1 hour** per 10 years of data with full model suite. Use `--symbols RELIANCE,INFY` during development.
