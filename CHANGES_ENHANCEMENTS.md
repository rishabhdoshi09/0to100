# SimpleQuant AI — Enhancement Session Changes

## Overview

Seven enhancements added to make `sq_ai` a powerful **manual trading assistant**.
All reuse existing code (IndicatorEngine, XGBoostSignalGenerator, FnOExecutor, etc.)
and integrate into the Streamlit dashboard as new tabs.

---

## Enhancement 1 — Multi-Timeframe Signal Aligner

**File:** `analysis/multi_timeframe.py`  
**Class:** `MultiTimeframeAligner`

Fetches OHLCV data for a symbol across multiple timeframes (5-min, 15-min, 1-h, daily),
runs the XGBoostSignalGenerator on each, and returns an alignment score:
- Score +1.0 = all timeframes BUY
- Score −1.0 = all timeframes SELL
- Consensus: BUY if >0.33, SELL if <-0.33, HOLD otherwise

**Dashboard tab:** "📐 Multi-Timeframe" — symbol selector, timeframe multiselect,
alignment score gauge, per-timeframe action/confidence table.

---

## Enhancement 2 — Market Regime Detector

**File:** `analysis/regime_detector.py`  
**Class:** `RegimeDetector`

Classifies the current NSE market regime using:
- **Hurst exponent** (custom R/S analysis) — TRENDING / MEAN_REVERTING / RANDOM
- **ADX** (14-period, computed from Nifty 50 data) — TRENDING / SIDEWAYS / MODERATE
- **India VIX** (via yfinance ^INDIAVIX) — LOW_VOL / NORMAL_VOL / HIGH_VOL
- **F&O expiry proximity** (FnOExecutor.should_rollover) — PRE_EXPIRY flag

Combined regime string e.g. `"TRENDING_HIGH_VOL"` with strategy recommendation.

**Dashboard tab:** "🌡️ Regime" — badge with current regime, VIX/Hurst/ADX metrics,
strategy recommendation, expiry warning.

---

## Enhancement 3 — Ensemble ML Signal

**File:** `ml/ensemble_signal.py`  
**Class:** `EnsembleSignalGenerator`

Combines three models with weighted voting:
- **XGBoost** (weight 0.4) — existing XGBoostSignalGenerator
- **LightGBM** (weight 0.4) — same features/labels, auto-trained, saved to `models/{symbol}_lgb.pkl`
- **Chronos-Bolt** (weight 0.2) — optional, skipped if HF transformers/torch not installed

Final action = majority vote; final confidence = weighted average.
Overrides to HOLD if weighted confidence < 0.60.

**CLI command:** `python main.py ensemble --symbol RELIANCE`

**Dashboard tab:** "🧠 Ensemble ML" — symbol selector, action badge, per-model breakdown table.

---

## Enhancement 4 — Risk Metrics per Symbol

**File:** `analytics/risk_metrics.py`  
**Class:** `RiskMetrics`

Computes from 2 years of daily log returns:
- VaR 95% / 99% (historical)
- CVaR 95% / 99%
- Sharpe ratio (annualised, RF = 6.5%)
- Sortino ratio (downside deviation)
- Calmar ratio (annual return / max drawdown)
- Max drawdown %
- Beta vs Nifty 50 (via yfinance)
- Annualised return and volatility
- **Risk score 1–10** (composite of VaR, drawdown, Sharpe)

**Dashboard tab:** "⚠️ Risk Metrics" — symbol selector, risk score badge (green/amber/red),
full metrics table.

---

## Enhancement 5 — What-If Trade Simulator

**File:** `simulator/whatif.py`  
**Class:** `WhatIfSimulator`

Estimates outcome distribution for a proposed trade using historical rolling windows:
- Probability of profit
- Probability of loss > 2%
- Expected return %
- 99% VaR in INR
- P10 / P50 / P90 return percentiles
- Entry-price adjustment (if entry ≠ current price, shifts distribution by implied gain/loss)

**Dashboard tab:** "🔮 What-If" — form with symbol/qty/price/holding days; outcome metrics;
disclaimer.

---

## Enhancement 6 — Correlation Matrix Heatmap

**File:** `analysis/correlation.py`  
**Class:** `CorrelationAnalyzer`

Computes pairwise daily-return correlations for up to 20 symbols:
- Full correlation matrix (DataFrame)
- Per-symbol average correlation to universe
- List of highly correlated pairs (> 0.7 threshold)

**Dashboard tab:** "🔗 Correlations" — symbol multiselect (up to 20), lookback-day slider,
Plotly interactive heatmap (RdYlGn, −1 to +1), high-pair table, avg-correlation table.

---

## Enhancement 7 — Alert System (Telegram + Background Monitor)

**File:** `notify/alerts.py`  
**Classes:** `AlertManager`, `SignalMonitor`

`AlertManager.send_alert(message, alert_type)`:
- Sends via Telegram (primary) if `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`
- Falls back to structured log entry
- `alert_type`: "signal", "risk", "regime_change", "error", "summary", "info"

`SignalMonitor.run()`:
- Polls `EnsembleSignalGenerator` every 5 minutes during NSE hours (09:15–15:30 IST)
- Sends alert when signal changes state for watchlist symbols
- Detects regime changes via `RegimeDetector`
- Sends daily summary at 15:30

**CLI command:** `python main.py alerts --start`

---

## New Files

| Path | Description |
|------|-------------|
| `analysis/__init__.py` | Package init |
| `analysis/multi_timeframe.py` | MultiTimeframeAligner |
| `analysis/regime_detector.py` | RegimeDetector |
| `analysis/correlation.py` | CorrelationAnalyzer |
| `ml/ensemble_signal.py` | EnsembleSignalGenerator |
| `analytics/risk_metrics.py` | RiskMetrics |
| `simulator/__init__.py` | Package init |
| `simulator/whatif.py` | WhatIfSimulator |
| `notify/__init__.py` | Package init |
| `notify/alerts.py` | AlertManager + SignalMonitor |

## Modified Files

| Path | Change |
|------|--------|
| `config.py` | Added: `multi_timeframe_periods`, `whatif_default_holding_days`, `telegram_bot_token`, `telegram_chat_id`, `alert_watchlist` (+ list properties) |
| `requirements.txt` | Added: `lightgbm>=4.0.0`, `plotly>=5.18.0`, `pytz>=2024.1`, `schedule>=1.2.0`, `yfinance>=0.2.36` |
| `main.py` | Added CLI commands: `ensemble --symbol`, `alerts --start` |
| `app.py` | Added 6 new Streamlit tabs: Multi-Timeframe, Regime, Ensemble ML, Risk Metrics, What-If, Correlations |

## New CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py ensemble --symbol RELIANCE` | Print XGBoost + LightGBM ensemble signal with per-model breakdown |
| `python main.py alerts --start` | Start background signal monitor; sends Telegram alerts on signal state changes |

## .env Variables Added

```
TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>
ALERT_WATCHLIST=RELIANCE,TCS,HDFCBANK
MULTI_TIMEFRAME_PERIODS=5min,15min,1h
WHATIF_DEFAULT_HOLDING_DAYS=5
```
