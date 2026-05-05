# SimpleQuant AI — Session Changes

## Files Created

### `backtest/walk_forward.py`
Full rewrite of the previous stub. Implements `WalkForwardValidator` — a rolling
walk-forward optimisation engine.  
- Runs an exhaustive grid search (81 combinations) over `zscore_entry`,
  `zscore_exit`, `rsi_entry_max`, `rsi_exit_min` on every in-sample (IS) window.  
- Applies the best IS parameter set to the next out-of-sample (OOS) window.  
- Aggregates OOS Sharpe, return, drawdown, win rate and IS/OOS overfitting ratio
  across all windows.  
- Strict no-lookahead: IS and OOS slices are constructed independently with
  `df.loc[df.index.isin(dates)]`.

### `execution/nse_guards.py`
New `NseOrderGuard` class with a single public method `validate()` that runs
four NSE/BSE market-rule checks before any order reaches Kite:  
1. Pre-open MARKET order block (09:00–09:15 IST)  
2. Circuit breaker (upper/lower price freeze)  
3. CNC short-sell block (can't sell CNC without holding the stock)  
4. Minimum order value (< ₹1 blocked)

### `ml/__init__.py`
Empty package initialiser for the new `ml/` module.

### `ml/xgboost_signal.py`
`XGBoostSignalGenerator` — per-symbol XGBoost multi-class classifier (BUY / HOLD / SELL).  
- 17 features: 13 from `IndicatorEngine.compute()` + 4 lagged return columns.  
- Labels derived from 5-day forward returns (thresholds ±0.5%).  
- HOLD rows dropped before training; requires ≥ 100 labelled samples.  
- `XGBClassifier(n_estimators=200, max_depth=4, lr=0.05, …)`.  
- Models persisted to `models/{symbol}_xgb.pkl` via joblib.  
- Auto-retrains when model is absent or > `XGBOOST_RETRAIN_DAYS` old.  
- Output dict is compatible with `TradingSignal` fields.

### `execution/fo_executor.py`
`FnOExecutor` — F&O (Futures & Options) execution layer for NFO segment.  
- `fetch_nfo_instruments()` — downloads NFO instrument list, caches to
  `data/nfo_instruments_{date}.csv`.  
- `get_front_month_future(equity_symbol)` — returns nearest-expiry futures contract.  
- `round_to_lot_size(quantity, lot_size)` — always returns ≥ 1 lot.  
- `check_margin(…)` — queries Kite order_margins API; fails safe on error.  
- `place_futures_order(…)` — margin check → place MARKET order on NFO.  
- `should_rollover(expiry)` — returns True when expiry ≤ `FNO_ROLLOVER_DAYS` away.  
- `rollover_position(…)` — closes front-month, opens next-month atomically.  
- `get_option_chain(equity_symbol, expiry)` — returns filtered DataFrame.  
- `place_option_order(…)` — margin check → place option MARKET order.

### `CHANGES.md`
This file.

---

## Files Modified

### `config.py`
Added six new Pydantic field definitions:
- `walkforward_is_days: int = 252`
- `walkforward_oos_days: int = 63`
- `xgboost_train_days: int = 252`
- `xgboost_retrain_days: int = 21`
- `enable_fno: bool = False`
- `fno_default_product: str = "NRML"`
- `fno_rollover_days: int = 3`

### `main.py`
Added two new CLI sub-commands and their handler functions:
- `walkforward` — downloads historical data and runs `WalkForwardValidator`,
  printing a per-window OOS report and overfitting ratio.
- `fnolive` — starts the live engine with F&O execution available; requires
  `ENABLE_FNO=true` in `.env` and explicit `"YES"` confirmation at startup.

### `execution/zerodha_broker.py`
- Imports `NseOrderGuard` at module level.
- Instantiates `self._nse_guard = NseOrderGuard()` in `__init__`.
- In `execute()`, calls `self._nse_guard.validate()` before the Kite API call;
  returns an `ERROR` `OrderResult` immediately if any guard check fails.
- `execute()` signature extended with optional `portfolio_positions` and `quote`
  keyword arguments (fully backwards-compatible — both default to `None`).

### `requirements.txt`
Added:
- `joblib>=1.3.0`
- `xgboost>=2.0.0`

### `.gitignore`
Added `*.pkl` to prevent trained model binaries from being committed.

---

## New CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py walkforward --from YYYY-MM-DD --to YYYY-MM-DD` | Run walk-forward parameter validation over the given date range. Prints per-window OOS Sharpe, total return, max drawdown, and an IS/OOS overfitting ratio (>2.0 = overfit warning). |
| `python main.py fnolive` | Start the live engine with F&O execution enabled. Requires `ENABLE_FNO=true` in `.env`. Prompts for explicit `YES` confirmation before any trading begins. |
