# sq_ai – dual-LLM quant cockpit (PRD v0.2)

## Original problem statement
Transform a broken backtester (–7 % in 2022) into a 100×-better live
trading cockpit on a MacBook Air 2015 (8 GB RAM, no GPU). v0.2 adds a
**dual-LLM hybrid**: DeepSeek as the cheap pre-filter (every 30 min,
universe up to 500), Claude as the decision engine on the top 10
(every 5 min), with an **ensemble veto** when Claude proposes
`size_pct > 10 %`.

## Architecture
* FastAPI (port 8000)
* APScheduler – 08:00 IST universe refresh • 0/30 min DeepSeek screener
  • every 5 min Claude decision • 23:00 IST nightly screener
* SQLite tables: prices, trades, signals, daily_equity,
  screener_results, **llm_disagreements**, **instruments_cache**
* Textual TUI (2-s polling)
* Anthropic + OpenAI SDKs (DeepSeek via `base_url`)
* LightGBM `.pkl` inference; training in Colab notebook
* Risk manager: Kelly + vol-target + 4 % daily-loss kill-switch + 50 % exposure
* Streamlit dashboard (optional)

## Critical fix
`signals/composite_signal.py::compute_indicators` emits `regime ∈ {0,1,2}`;
`compute()` enforces a hard regime gate (no BUY in downtrend). Reused by
`backtester.py`, `decision.py` and the Colab notebook → backtest = live.

## Tasks done (2026-01)
* v0.1 cockpit: FastAPI, scheduler, executor, signals, backtester, TUI,
  Colab notebook, walk-forward, README + run.sh.
* **v0.2 dual-LLM upgrade**:
  - `backend/llm_clients.py`           — ClaudeClient + DeepSeekClient (`generate(prompt, max_tokens, temperature)`)
  - `backend/screener.py`              — 30-min DeepSeek pre-filter, deterministic fallback
  - `backend/decision.py`              — 5-min Claude decision engine + ensemble veto wiring
  - `backend/ensemble.py`              — >10 % size veto, disagreement logging
  - `backend/universe.py`              — Kite instrument-master cache
  - `portfolio/tracker.py`             — added `llm_disagreements` + `instruments_cache` tables
  - `backend/executor.py`              — journal.md append + live broker stop/target follow-up
  - `api/app.py`                       — `/api/cycle/status`, `/api/screener/latest`, `/api/disagreements`, `/api/universe`, `/api/universe/refresh`
  - `config.yaml`                      — universe, cadences, ensemble threshold, risk knobs
  - `.env.template`, `run_hybrid.sh`
  - 4 new test files (`test_llm_clients`, `test_screener`, `test_ensemble`, `test_universe`)
* **52/52 pytest cases pass**, ruff lint-clean, FastAPI smoke green.

## What is NOT exercised locally
* Live KiteConnect order placement (paper mode default; broker stop/target code is `pragma: no cover`).
* Real Anthropic + DeepSeek calls (placeholder keys → graceful fallback paths).
* Textual TUI render (terminal-only widget).
* Colab notebook execution (designed for Google Colab T4).

## Backlog
| P | item |
|---|------|
| P1 | NewsAPI per-symbol headline injection in decision prompt (already wired, needs key) |
| P1 | Liquidity sort in `get_active_universe` (turnover-weighted) |
| P1 | `/api/equity` chart endpoint for Streamlit dashboard |
| P2 | LSTM head from Colab |
| P2 | Auto-retraining trigger when AUC < 0.55 for 30 d |
| P2 | Multi-broker abstraction (Upstox / Dhan) |

## Performance targets
| metric | target |
|---|---|
| Cockpit RAM | < 400 MB |
| 2022 walk-forward Sharpe | > 1.0 |
| Profit factor | > 1.3 |
| Anthropic + DeepSeek spend / month | < $5 |
| TUI refresh | 2 s |
| Decision cycle | 5 min IST |
| Screener cycle | 30 min IST |
