# sq_ai – unified Bloomberg + Screener.in + Moneycontrol cockpit (PRD v0.3)

## Original problem statement
Build a single-laptop platform that fuses:

* **Bloomberg-grade live cockpit**  (existing v0.2 dual-LLM trading loop)
* **Screener.in-style dynamic screener** (technical + fundamental + momentum)
* **Moneycontrol-style stock-research hub** (technicals, financials,
  earnings calls + Claude-extracted highlights, analyst estimates,
  shareholding, corporate actions, news, peers)
* **Daily PDF market report**           (auto at 17:30 IST)

Constraint: MacBook Air 2015, 8 GB RAM, no GPU. < 500 MB resident.

## Architecture
* FastAPI (port 8000) – cockpit + research + screener + reports + watchlist
* Streamlit multi-page web UI (port 8501) – Dashboard / Screener / Stock
  Research / Portfolio / Reports / Settings
* Textual TUI (low-mem alternative)
* APScheduler – 08:00 universe / 0:30 DeepSeek screen / 5-min Claude /
  17:30 PDF report / 23:00 nightly screener (tz Asia/Kolkata)
* SQLite cache (TTL `kv_cache` table + structured tables: prices, trades,
  signals, daily_equity, screener_results, screener_presets, user_watchlist,
  earnings_calls, llm_disagreements, instruments_cache, reports)
* LightGBM inference, Anthropic Claude, DeepSeek (OpenAI-compat),
  yfinance, Alpha Vantage, NewsAPI, Kite Connect
* PDF rendering via `reportlab` (Plotly charts saved as PNG)

## Critical regime fix (still preserved end-to-end)
`signals/composite_signal.py::compute_indicators` emits `regime ∈ {0,1,2}`;
`compute()` enforces a hard regime gate (no BUY in downtrend). Reused by
backtester, decision engine, walk-forward, screener, stock research.

## Tasks done (2026-01)
* v0.1 cockpit (FastAPI, scheduler, executor, signals, backtester, TUI,
  Colab notebook, walk-forward, README + run.sh).
* v0.2 dual-LLM upgrade (DeepSeek pre-filter + Claude decision +
  ensemble veto + dynamic Kite universe).
* **v0.3 unified platform**:
  - `backend/cache.py`          – SQLite TTL kv-cache + `@cached` decorator
  - `backend/financials.py`     – ratios + annual + quarterly (Alpha Vantage → yfinance fallback)
  - `backend/analyst_estimates.py` – EPS / target / rating distribution
  - `backend/shareholding.py`   – promoter/FII/DII history (yfinance + synth)
  - `backend/corporate_actions.py` – dividends + splits + buybacks
  - `backend/earnings_analyzer.py` – pdfplumber → Claude extracted highlights+guidance
  - `backend/screener_engine.py` – declarative JSON filter + score
  - `backend/stock_research.py` – aggregator for `/api/stock/profile/{sym}`
  - `backend/watchlist.py`      – CRUD service
  - `backend/report_scheduler.py` – daily snapshot + Claude narrative + reportlab PDF
  - tracker.py: added `user_watchlist`, `screener_presets`, `reports`,
    `earnings_calls` tables + helpers
  - api/app.py: 16 new routes (`/api/stock/*`, `/api/screener/run`,
    `/api/screener/presets`, `/api/watchlist`, `/api/reports/*`)
  - 6 Streamlit pages: dashboard, screener, stock_research, portfolio,
    reports, settings + shared `_api.py` helper
  - run.sh now boots FastAPI + Streamlit + Textual TUI together
  - 4 new test files (cache, screener_engine, watchlist_presets, report)
* **71/71 pytest pass**, ruff lint-clean, FastAPI smoke green.

## What works without keys
* All routes return 200 with deterministic fallbacks (no Kite, no NewsAPI,
  no Alpha Vantage, no Anthropic, no DeepSeek required).
* Stock-profile shows yfinance / synthesised data when keys are absent.
* Screener works fully offline as long as price history is available
  (LightGBM inference + composite ranking).

## What needs MacBook to verify
* Streamlit pages render (server starts, but rendering is a browser call).
* Live KiteConnect order placement (paper mode default).
* Real Anthropic + DeepSeek calls (placeholder keys → fallback paths).
* Daily PDF on real network (yfinance for index data — sandbox blocks it).

## Backlog
| P | item |
|---|------|
| P1 | Streamlit Plotly chart-rendering once Streamlit is reachable |
| P1 | Liquidity sort in `get_active_universe` (turnover-weighted) |
| P1 | Auto-refresh SSE for cockpit dashboard |
| P2 | LSTM head from Colab |
| P2 | Auto-retraining trigger when AUC < 0.55 for 30 d |
| P2 | Multi-broker abstraction |

## Performance targets
| metric | target |
|---|---|
| Cockpit + Streamlit RAM | < 500 MB |
| 2022 walk-forward Sharpe | > 1.0 |
| Profit factor | > 1.3 |
| LLM spend / month | < $5 |
| TUI refresh | 2 s |
| Decision cycle | 5 min IST |
| Screener cycle | 30 min IST |
| Report cadence | 17:30 IST |
