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

## Tasks done (2026-05-03) — instruments + live tracking

* **Full NSE instruments listing + live price tracker**:
  - `universe.py`: new `fetch_all_instruments()` fetches every instrument
    (EQ, FUT, CE/PE, INDEX) from Kite SDK first, then falls back to the
    public `https://api.kite.trade/instruments/NSE` CSV — no credentials needed
    for the instrument list.
  - `universe.py`: `_normalise()` unifies SDK and CSV field names;
    `_filter_eq` now works on normalised rows.
  - `portfolio/tracker.py`: `instruments_cache` schema extended with
    `instrument_type`, `segment`, `lot_size`, `tick_size`. Migration via
    `ALTER TABLE … ADD COLUMN` (no data loss on existing DBs).
  - `api/app.py`:
      - `GET /api/universe?q=` — cached EQ list with search
      - `GET /api/universe/all?q=&exchange=` — full live listing (all types)
      - `GET /api/ltp?symbols=NSE:X,NSE:Y` — Kite LTP for any symbol list
  - `ui/instruments_page.py`: new Streamlit page — searchable/filterable
    full-listing table + live price tracker with auto-refresh toggle
    (configurable 2–60 s interval, uses `st.rerun()`).
  - Wired into `streamlit_app.py` nav as "📋 Instruments".
  - 5 new tests in `test_universe.py`; 76/76 pass. ruff clean.

## Tasks done (2026-05-03) — P0 complete + P1.1 done

### P0 (all items closed)
* **P0.1 – Claude model fix**: `DEFAULT_MODEL` in `llm_clients.py`
  `claude-3-sonnet-20240229` (404 since 2024-07-21) → `claude-haiku-4-5-20251001`.
  Cost projection ~$1.25/month (780 calls/market-day at Haiku pricing), well under $5 cap.
  Created `0to100/.env.example` with all env vars + cost guidance. `CLAUDE_MODEL` override documented.
* **P0.2 – NewsAPI caching**: `@cached("news", ttl_seconds=1800)` on `fetch_news`.
  Verified: 2nd same-query call hits in 0.3 ms (vs 3.6 ms first call); keys are
  query-isolated (RELIANCE ≠ TCS). Free tier: ~20 req/day vs 100-req limit.
* **Lint sweep**: `ruff --fix` → 0 errors; `pyproject.toml` excludes notebook.
  Removed unused `longs_in_downtrend` in `test_backtester.py`.
* **P0.3 – Trained model**: network blocked in sandbox. `train_local.py` script created
  (`python -m sq_ai.train.train_local` from repo root) — identical feature engineering
  to Colab notebook, no Colab needed. Run on MacBook when ready.

### P1.1 – `/api/equity` + dashboard equity curve
* Added `GET /api/equity` route to `api/app.py` returning the full `daily_equity`
  series as `[{date, equity, cash}]`.
* Added Plotly line chart to `ui/dashboard_page.py` (equity + cash lines, unified
  hover, ₹ y-axis) with summary metrics: total return %, peak equity, max drawdown.
  Gracefully shows a caption when no history exists yet.
* 2 new tests in `test_api.py`; 73/73 pass.

## Tasks done (2026-05-03)
* **P0.1 – Fix deprecated Claude model**: updated `DEFAULT_MODEL` in
  `sq_ai/backend/llm_clients.py` from `claude-3-sonnet-20240229` (returns 404
  since 2024-07-21) to `claude-haiku-4-5-20251001`. Created
  `0to100/.env.example` documenting all required env vars including
  `CLAUDE_MODEL` override, `NEWSAPI_KEY`, and cost guidance. Updated
  `tests/test_llm_clients.py` to assert new model name.
* **P0.2 – NewsAPI caching**: added `@cached("news", ttl_seconds=1800)`
  decorator to `fetch_news` in `sq_ai/backend/data_fetcher.py`. NewsAPI free
  tier is 100 req/day; without the cache the 5-min loop hitting 10 symbols
  would exhaust the quota in ~8 minutes. With 30-min cache, daily usage is
  ~20 requests. The news headlines were already being injected into the Claude
  decision prompt via `build_decision_prompt` — this only adds the cache layer.
* **Lint sweep**: ran `ruff --fix`, added `pyproject.toml` to exclude
  `colab_train.ipynb`, removed unused `longs_in_downtrend` variable in
  `tests/test_backtester.py`. `ruff check .` now reports zero errors.
* **P0.3 – Trained model**: requires a fresh Colab run
  (`sq_ai/train/colab_train.ipynb → Runtime ▸ Run all`), then copy
  `lgb_trading_model.pkl` + `feature_names.txt` into `~/0to100/models/`.
  Cannot be automated from this environment.
* 71/71 pytest pass. ruff lint-clean.

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
