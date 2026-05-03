# sq_ai – Bloomberg-grade quant cockpit on a MacBook Air

A zero-bloat, 8 GB-friendly trading cockpit:

* FastAPI backend (port **8000**)
* Textual TUI (terminal cockpit)
* SQLite cache
* APScheduler → 5-min decision loop + 23:00 IST screener
* LightGBM **inference only** (training happens on Colab)
* Anthropic Claude as the senior PM that signs off every trade

> **Architectural rule:** the laptop is the cockpit, not the engine.
> All training, walk-forward validation, and bulk downloads run in
> `sq_ai/train/colab_train.ipynb` on a free Colab T4 — never locally.

---

## 1. Prerequisites

* macOS / Linux, Python ≥ 3.10
* A Zerodha Kite developer app (for live; paper trading works without it)
* An Anthropic API key (https://console.anthropic.com)
* Optional: NewsAPI / Alpha Vantage keys for richer briefs

---

## 2. Setup

```bash
git clone https://github.com/rishabhdoshi09/0to100.git
cd 0to100
cp .env .env.local           # then fill in your real keys
```

`run.sh` will create `.venv`, install `requirements.txt`, start FastAPI on
`127.0.0.1:8000`, and launch the Textual TUI:

```bash
./run.sh
```

Press **q** in the TUI to quit (the FastAPI process is killed automatically).

---

## 3. CLI commands

```bash
# Backtest a single symbol on Yahoo data (no Colab needed)
python -m sq_ai.main backtest --symbol RELIANCE.NS --period 2y --out data/equity.csv

# Run one decision cycle (manual)
python -m sq_ai.main cycle

# Run the 23:00 screener once
python -m sq_ai.main screener

# Start the FastAPI cockpit (without TUI)
python -m sq_ai.main live
```

---

## 4. Training the model on Colab

1. Open `sq_ai/train/colab_train.ipynb` in Google Colab.
2. `Runtime ▸ Run all`.
3. The notebook downloads NSE500 history, engineers the **same features
   the live pipeline uses (including `regime`)**, trains LightGBM, and
   exports the artefacts to your Google Drive at
   `MyDrive/sq_ai_models/`.
4. Download:
   * `lgb_trading_model.pkl`
   * `feature_names.txt`
   into `~/0to100/models/` on the laptop.
5. Restart the cockpit – it picks up the new model on the next call.

The notebook also runs a 2022 out-of-sample walk-forward and prints
Sharpe / cumulative return per fold.

---

## 5. Scheduling the screener at 23:00 IST (laptop)

The scheduler runs in-process when you start `./run.sh`, but if you want
it to run *even when the cockpit is closed*, use `launchd` (macOS):

```bash
cat > ~/Library/LaunchAgents/com.sqai.screener.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.sqai.screener</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd ~/0to100 && source .venv/bin/activate && python -m sq_ai.main screener</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>/tmp/sqai-screener.log</string>
  <key>StandardErrorPath</key><string>/tmp/sqai-screener.err</string>
</dict>
</plist>
PLIST
launchctl load ~/Library/LaunchAgents/com.sqai.screener.plist
```

For Linux/cron:

```bash
crontab -e
# minute hour day month dow command
0 23 * * 1-5 cd ~/0to100 && source .venv/bin/activate && python -m sq_ai.main screener
```

---

## 6. Architecture (one-screen)

```
┌─ MACBOOK AIR (cockpit, <300 MB) ───────────────────────────────┐
│  Textual TUI ──polls 2s──▶ FastAPI :8000 ──▶ SQLite (cache)    │
│                                  │                              │
│                     APScheduler (every 5 min)                   │
└──────────────────────────────────┼──────────────────────────────┘
                  ┌────────────────┼────────────────┐
                  ▼                ▼                ▼
           ┌──────────┐    ┌──────────────┐  ┌──────────────┐
           │ Claude   │    │ Kite + yfin  │  │ Google Colab │
           │ Sonnet   │    │ live + hist. │  │ training T4  │
           └──────────┘    └──────────────┘  └──────────────┘
```

---

## 7. The critical fix (`regime`)

* `sq_ai/signals/composite_signal.py::compute_indicators` now emits
  `'regime'` (0 = down, 1 = sideways, 2 = up) computed from SMA20 vs
  SMA50.
* `_compute_ml_signal` consumes `features['regime']` — no more silent 0
  default.
* `compute()` enforces a **regime gate**: a bullish signal in
  `regime == 0` is forced to 0 (no BUY in a downtrend).
* The same engineering runs in `sq_ai/train/walk_forward.py` and the
  Colab notebook → backtest = live.

---

## 8. Performance targets

| metric                          | target              |
|---------------------------------|---------------------|
| 2022 walk-forward Sharpe        | > 1.0               |
| 2022 profit factor              | > 1.3               |
| RAM (cockpit + API + TUI)       | < 400 MB            |
| Anthropic spend / month         | < $5                |
| TUI refresh                     | every 2 s           |
| Decision cycle                  | every 5 min (IST)   |

---

## 9. Project layout

```
0to100/
├── sq_ai/
│   ├── api/app.py                 # FastAPI cockpit
│   ├── backend/{scheduler,data_fetcher,claude_client,executor,risk_manager}.py
│   ├── signals/{composite_signal,ml_model}.py    # ← regime fix lives here
│   ├── portfolio/tracker.py       # SQLite repo
│   ├── ui/{terminal,dashboard}.py # Textual + Streamlit
│   ├── backtest/{backtester,metrics}.py
│   ├── train/{colab_train.ipynb,walk_forward.py}
│   └── main.py
├── tests/
├── .env  .gitignore  requirements.txt  run.sh  README.md
```
