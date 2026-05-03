# sq_ai – Bloomberg-grade quant cockpit (PRD)

## Original problem statement
Transform a broken backtester (–7 % in 2022) into a 100×-better live
trading cockpit that runs on a MacBook Air 2015 (8 GB RAM, no GPU).
The laptop is the cockpit; Colab is the engine.

## Architecture
* FastAPI (port 8000) – read-only cockpit + manual `/api/trade`
* APScheduler – 5-min decision cycle + 23:00 IST screener (timezone Asia/Kolkata)
* SQLite (single file) – prices, trades, signals, daily_equity, screener_results
* Textual TUI – polls `/api/portfolio`, `/api/positions`, `/api/signals/latest`,
  `/api/cycle/last` every 2 s
* Anthropic Claude – assembled brief every 5 min, JSON response, ML+regime fallback
* LightGBM `.pkl` – inference only (training in `train/colab_train.ipynb`)
* Risk manager – Kelly + vol-target + 4 % daily loss kill-switch + 50 % exposure cap
* Streamlit dashboard (optional secondary view)

## Critical fix
`signals/composite_signal.py::compute_indicators` now emits `regime ∈ {0,1,2}`
and `_compute_ml_signal` consumes it from the feature vector. `compute()`
adds an explicit regime gate (no BUY when regime == 0).
Same engineering is reused by `backtest/backtester.py` and the Colab notebook
→ backtest = live.

## Tasks done (2026-01)
* Project scaffolded under `/app/0to100/` matching the requested layout.
* All Python files lint-clean (`ruff`, 0 errors).
* 28 / 28 unit tests pass (`pytest tests/`).
* End-to-end smoke test of FastAPI cockpit (`/api/health`, `/api/portfolio`,
  `/api/trade` BUY/SELL) succeeds.
* Regime fix verified end-to-end via composite-signal call.
* Colab notebook + walk-forward script exported under `sq_ai/train/`.
* `run.sh` one-command bootstrap; `.env` template + `.gitignore` written.
* README with macOS `launchd` and Linux `cron` scheduling instructions.

## What's NOT tested locally (needs MacBook)
* Textual TUI live render (terminal-only widget)
* Real KiteConnect order placement (paper mode default)
* Real Anthropic API calls (key placeholder in `.env`)
* Colab notebook execution (run on Google Colab)

## Backlog (P1)
* Wire NewsAPI → per-symbol headlines in the brief.
* Add WebSocket Kite tick stream to `data_fetcher`.
* Persist Claude raw responses for audit.
* Add equity-curve chart endpoint (`/api/equity`) for the Streamlit page.

## Backlog (P2)
* LSTM sequence model from Colab (LightGBM is enough for now).
* Auto-retraining trigger if AUC drops below 0.55 for 30 days.
* Multi-broker abstraction (Upstox, Dhan).

## Performance targets
| metric | target |
|---|---|
| Cockpit RAM | < 400 MB |
| 2022 walk-forward Sharpe | > 1.0 |
| Profit factor | > 1.3 |
| Anthropic spend / month | < $5 |
| TUI refresh | 2 s |
| Decision cycle | 5 min IST |
