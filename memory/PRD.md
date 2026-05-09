# DevBloom / Prism Quant — PRD

## Original problem
Streamlit-based trading terminal that went **blank after a few seconds** (iter 1),
needed live trading + tab consolidation (iter 2), bulk multi-symbol simulation
(iter 3), and CLI integration with AlgoLab-saved strategies (iter 4).

## Iteration 1 — Stability hotfix (2026-05-08)
Fixed:
- Blocking `time.sleep(1)`×30 auto-refresh → replaced with `streamlit_autorefresh`
- `persist="disk"+ttl=…` on 10 caches → removed `persist`, TTL works again
- Deprecated `components.v1.html(MONACO_HTML)` → removed Monaco editor
- `tabs+[st.empty()]*14` placeholder hack → declared all 23 tabs properly
- Added stdlib structured logger → `/app/logs/streamlit_app.log`

## Iteration 2 — Live trading + tab consolidation (2026-05-08)

### A. AlgoLab Live Runner (`ui/live_runner.py`)
Manual flow: Evaluate → 5 risk gates → Confirm. Modes: Paper / Kite Live.
Order math: `qty = capital//LTP`, `SL = LTP ∓ 2×ATR`, `target = LTP ± 4×ATR`.
5 risk gates: Trading window · Max ₹/trade · Max open positions · Daily loss circuit · Tradable signal.

### B. Tab consolidation 23 → 6
| Top tab | Sub-tabs |
|---------|----------|
| 🏠 Command Center | (single) |
| 📊 Research | 📈 Charts · 📊 Pro Charts · 📐 Multi-TF · 📊 Fundamentals · 🔬 Deep Fund · 📊 Ownership |
| 🧠 Signals | 🔎 Decision · 🧠 Ensemble ML · 🌡️ Regime · ⚡ Co-Pilot · 🏦 Quant |
| 🧬 AlgoLab | Editor + Backtest + Live Runner + Bulk panels |
| ⚙️ Execute | Trading Cockpit · Paper Trading · Risk Metrics · What-If |
| 🔬 Tools | Screener · Quick Screener · Correlation · Journal · Pre-Market · Global |

Backwards-compat: `tabs[0..22]` legacy-mapping list kept ~1500 lines of body code untouched.

## Iteration 3 — Bulk multi-symbol simulator (2026-05-08)
New module `ui/bulk_simulator.py`:
- 4 universes (Nifty 50/100/500/Custom), parallel workers slider (2-16), timeout slider (5-60 s)
- 24 h disk-free TTL cache on yfinance OHLCV
- Bulk Backtest panel — table sorted by Sharpe + Heatmap toggle
- Bulk Live Scan — BUY/SELL/HOLD count chips + Heatmap toggle
- Verified: Nifty 50 → 48/50 succeeded; live scan 34 BUY · 14 SELL

## Iteration 4 — CLI ↔ AlgoLab DB bridge (2026-05-08)

### Goal
Let `main.py backtest --strategy NAME` run the same Python strategy that the user
edited & saved in the AlgoLab UI — without duplicating the existing event-driven
ML+LLM `Backtester` class and without touching any UI files.

### Reality check (flagged before implementing)
The existing `Backtester` does NOT accept a `generate_signals(df) -> Series`
callable. It runs an internal `IndicatorEngine + ConvictionScorer + DualLLMEngine`
pipeline. So `--strategy` mode **bypasses** the heavy Backtester and uses the
same vectorised simulation that AlgoLab UI's backtest button uses.

### New module `utils/strategy_loader.py`
Two pure-Python helpers (no Streamlit imports):

| Function | Purpose |
|----------|---------|
| `load_strategy_from_db(name)` | Auto-detects schema (`strateg*` table, `name` + `code/script/python_code/source` columns), `exec`'s the saved code in a namespace exposing `pd`, `np`, full `__builtins__`, returns the `generate_signals` callable. Raises with available-names list on miss. |
| `simulate_strategy(fn, df, capital)` | Mirrors AlgoLab UI's `_run_backtest` semantics (no Streamlit). Returns `{total_return, sharpe, max_dd, win_rate, n_trades, equity_curve, trades}`. |

### `main.py backtest` extensions
New optional flags (existing flags untouched):
| Flag | Purpose |
|------|---------|
| `--strategy NAME` | Load `generate_signals` from `algolab_strategies.db` |
| `--symbols SYM1,SYM2,…` | Override `settings.symbol_list` (only with `--strategy`) |
| `--capital AMOUNT` | Per-symbol capital (only with `--strategy`) |

When `--strategy` is provided:
1. `cmd_backtest` early-returns to a new `cmd_backtest_strategy` handler.
2. Loads the strategy callable (or fails with available-names list).
3. Resolves symbols (CLI flag → settings → fail).
4. Fetches OHLCV — Kite if `KITE_ACCESS_TOKEN` is set, else yfinance fallback.
5. Runs `simulate_strategy()` per symbol; prints a one-liner per symbol.
6. Prints a Sharpe-ranked summary table + cross-symbol mean.

When `--strategy` is NOT provided:
The original code path runs untouched — `_assert_credentials()` → heavy
`Backtester` → `PerformanceReporter`. Zero regression.

### Verified end-to-end
```
$ python main.py backtest --strategy TestEMA \
                         --symbols RELIANCE,TCS,INFY \
                         --from 2024-01-01 --to 2025-01-01 --capital 100000

✅ Loaded strategy 'TestEMA' from algolab_strategies.db
=== AlgoLab Strategy Backtest ===
Strategy : TestEMA
Period   : 2024-01-01 → 2025-01-01
Symbols  : RELIANCE, TCS, INFY
Capital  : ₹100,000 per symbol
(Kite token not set — using yfinance fallback)

  RELIANCE    : bars= 246  return=-12.01%  sharpe=-0.74  max_dd=-20.67%  trades=10  win_rate=20.0%
  TCS         : bars= 246  return= +6.25%  sharpe= 0.49  max_dd= -9.01%  trades= 3  win_rate=66.7%
  INFY        : bars= 246  return=+17.86%  sharpe= 1.16  max_dd= -9.50%  trades= 2  win_rate=50.0%

=== Ranked summary (by Sharpe) ===
  Symbol  Return %  Sharpe  Max DD %  Win %  Trades
    INFY     17.86    1.16     -9.50   50.0       2
     TCS      6.25    0.49     -9.01   66.7       3
RELIANCE    -12.01   -0.74    -20.67   20.0      10

Mean across 3 symbols: return +4.03%  ·  sharpe 0.30
```

Edge cases verified:
- Bad strategy name → `❌ Strategy 'X' not found in DB. Available: [...]`
- No `--strategy` flag → existing `_assert_credentials` path triggers (untouched)
- `--symbols` with custom list + custom capital → correct per-symbol metrics

### NOT changed (per spec)
- `ui/algolab.py`, `ui/live_runner.py`, `ui/bulk_simulator.py` — untouched
- `backtest/backtester.py` — untouched
- No `strategies/` folder, no `utils/backtest_engine.py` (would have duplicated backtest/backtester.py)
- `cmd_live` — out of scope; live mode would need risk-gate + broker integration which is a much bigger task

## Files
| File | Status | Iteration |
|------|--------|-----------|
| `ui/live_runner.py` | NEW | 2 |
| `ui/bulk_simulator.py` | NEW | 3 |
| `utils/__init__.py`, `utils/strategy_loader.py` | NEW | 4 |
| `main.py` | Updated — `--strategy/--symbols/--capital` flags, `cmd_backtest_strategy` handler; default path untouched | 4 |
| `ui/algolab.py` | Updated — wires Live Runner + Bulk Backtest + Bulk Live Scan; sandbox uses full `__builtins__` | 2-3 |
| `app.py` | Updated — autorefresh + logger + 6-tab layout | 1-2 |
| `ui/heatmap.py`, `ui/macro.py`, `ui/watchlist.py`, `ui/alert_inbox.py`, `charting/multi_tf.py` | Updated — `persist="disk"` removed | 1 |
| `requirements.txt` | `streamlit-autorefresh>=1.0.1` added | 1 |
| `algolab_strategies.db` | Now usable from CLI (schema unchanged) | 4 |

## Backlog
- P1 — Sidebar **Risk Profile** form so max ₹/trade, max positions, daily loss circuit, ATR multipliers can be tuned without editing code.
- P1 — Use Kite intraday data when token is set; fall back to yfinance otherwise (currently yfinance daily only in Live Runner).
- P1 — Per-tab `try/except` boundary so one bad tab can't blank the whole page.
- P1 — 6th risk gate: backtest Sharpe ≥ 0.5 in last 90 days before allowing live order.
- P1 — `cmd_live --strategy NAME` (CLI live trading using AlgoLab strategies + risk gates).
- P2 — Persist live orders to `live_orders` SQLite table (audit trail) + "Cancel Order" button.
- P2 — Bulk runs: emit cache hit/miss + per-symbol elapsed time to logs.
- P2 — "Send Top 5 to Live Runner" button under Bulk Live Scan results.
- P3 — APScheduler-driven optional auto-execution mode.

## Implementation dates
- Iter 1 — Stability hotfix: 2026-05-08
- Iter 2 — Live trading + tab consolidation: 2026-05-08
- Iter 3 — Bulk multi-symbol simulator: 2026-05-08
- Iter 4 — CLI ↔ AlgoLab DB bridge: 2026-05-08
