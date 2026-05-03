# sq_ai Quant Trading Cockpit — PRD (Emergent pod level)

> **Note**: This is the Emergent-pod-level PRD snapshot. The canonical,
> detailed product doc lives at `/app/0to100_fresh/memory/PRD.md` and
> `/app/0to100_fresh/memory/HANDOVER.md` inside the project repo.

## Original problem statement
A Bloomberg-terminal-grade quantitative trading system that runs locally
on an 8 GB MacBook Air (< 500 MB RAM, Colab for ML training). FastAPI +
SQLite + Streamlit + Textual TUI + dual-LLM (Claude + DeepSeek).

## Current state (Feb 2026)

### v0.3 baseline (pre-existing on GitHub, 86 tests passing)
* FastAPI cockpit + 6 Streamlit pages + Textual TUI.
* Dual-LLM decision engine with ensemble veto (`claude-haiku-4-5`).
* SQLite cache + portfolio + screener engine + stock-research aggregator.
* Daily auto-PDF report, full NSE instruments tracker, live LTP, equity curve.

### v0.5 Decision Terminal (this session)
* **Phase 0** — repo sanitization: flattened `0to100/` → root, deleted
  all pre-v0.1 legacy (analytics/, engine/, app.py, paper_trading, junk).
* **Phase 1** — deterministic decision layer:
  * `sq_ai/signals/profiles.py` (Conservative / Aggressive)
  * `sq_ai/signals/sector_strength.py`
  * `sq_ai/signals/conviction.py` (single source of truth for 0-100 score)
  * `sq_ai/signals/trade_setup.py` (ATR-based entry/stop/target/qty/RR)
  * `sq_ai/signals/news_sentiment.py` (deterministic keyword-based)
  * `sq_ai/signals/buzzing.py` (cross-sectional scanner)
  * `PortfolioTracker.buy_signals_today()`
* **Phase 2** — API routes, all take `?profile=` query param
  (fixes env-var hopping bug between Streamlit and FastAPI processes):
  * `GET /api/buzzing`
  * `GET /api/intelligence/{symbol}`
  * `GET /api/conviction/{symbol}`
  * `GET /api/trade_setup/{symbol}`
  * API version bumped to `0.5.0`.
* **Phase 3** — Streamlit `decision_terminal_page.py` landing page.
* **Phase 4** — 29 new tests across 5 files; total **115 passed**
  (target was ≥114).
* **Phase 5** — updated `memory/PRD.md` + `memory/HANDOVER.md` with
  full profile-decision matrix.

### Architecture / tech stack
* **Not** the standard Emergent React+FastAPI+MongoDB template.
* FastAPI on port 8000, SQLite (`./data/sq_ai.db`), Streamlit on port 8501.
* Runs locally on user's MacBook Air; `REACT_APP_BACKEND_URL` not used.

### 3rd-party integrations
* Anthropic Claude (`claude-haiku-4-5-20251001`) — user API key
* DeepSeek (via OpenAI SDK) — user API key
* Kite Connect, NewsAPI, Alpha Vantage — user API keys
* Graceful degradation when keys are missing (all routes return 200).

### Testing status
* `pytest -q tests/` → **115 passed** (from repo root `/app/0to100_fresh`).
* `ruff check .` → clean.
* Uvicorn smoke test passed (v0.5.0 reports, all 4 new routes reachable;
  returns 404 or `[]` in the sandbox because yfinance is blocked — expected
  graceful degradation).

## Working directory
All work lives under `/app/0to100_fresh/`. Do **not** use legacy `/app/`
root files — they were deleted in Phase 0.

## Backlog / next actions
| P  | item                                                                     |
|----|--------------------------------------------------------------------------|
| P1 | Streamlit Plotly chart-rendering when Streamlit is reachable             |
| P1 | Liquidity-sorted universe (turnover-weighted) in `get_active_universe`   |
| P1 | SSE auto-refresh for cockpit dashboard                                   |
| P2 | Position-level conviction decay (entry vs now) on dashboard              |
| P2 | Weekly journal-review script: journal.md ↔ closed_trades ↔ logs/decisions|
| P2 | LSTM head (Colab) blended with LightGBM                                  |
| P2 | Auto-retraining trigger when rolling AUC < 0.55 for 30 days              |
| P2 | Multi-broker abstraction                                                 |
