# DevBloom / Prism Quant — PRD

## Original problem statement
Streamlit-based trading terminal (`/app/app.py`, ~2000 LOC) that loaded
correctly, then went **completely blank after a few seconds**. After the
hot-fix the user wanted:
1. Live trading integration for AlgoLab strategies.
2. Tab consolidation — 23 tabs were too cluttered.
3. Bulk multi-symbol simulation for both backtest and live signals.

## Iteration 1 — Stability hotfix (2026-05-08)

### RCA
| # | Issue | Effect |
|---|-------|--------|
| 1 | Blocking `time.sleep(1)` × 30 in sidebar auto-refresh + `st.rerun()` | **Primary cause of blank screen** — WebSocket starves |
| 2 | `@st.cache_data(persist="disk", ttl=N)` on 10 functions | TTL silently ignored; stale data + disk I/O storms |
| 3 | Deprecated `components.v1.html(MONACO_HTML)` | Will break post 2026-06-01 |
| 4 | `tabs = list(tabs) + [st.empty()]*14` placeholder hack | tabs[9..22] silently dropped content |
| 5 | No structured logging | Impossible to trace freezes |

### Fixes
| Fix | Change |
|----|--------|
| F1 | Non-blocking `streamlit_autorefresh.st_autorefresh(interval=30_000)` |
| F2 | Removed `persist="disk"` from all 10 cache decorators with TTL |
| F3 | Removed deprecated Monaco editor; kept textarea fallback |
| F4 | Declared all 23 tabs properly via `st.tabs([...])` |
| F5 | Added stdlib structured logger → `/app/logs/streamlit_app.log` |

## Iteration 2 — Live trading + tab consolidation (2026-05-08)

### Choices
- Execution: **Kite live + manual confirmation** (Paper / Kite Live toggle).
- Cadence: **Manual** button-driven.
- Risk: **Defaults** (max ₹/trade, max positions, daily loss circuit, trade window).
- Tabs: **23 → 6** (Charts + Fundamentals together).

### A. AlgoLab Live Runner (`ui/live_runner.py`)
1. **Evaluate** → fetches 200 d daily OHLCV (yfinance) → executes
   `generate_signals(df)` → reads `signals.iloc[-1]`.
2. Displays Action (BUY/SELL/HOLD), LTP, ATR, evaluated timestamp.
3. Order math: `qty = capital//LTP`, `SL = LTP ∓ 2×ATR`, `target = LTP ± 4×ATR`.
4. **5 risk gates** — any ❌ disables Confirm:
   - Trading window (9:15-15:20 IST, Mon-Fri)
   - Max ₹ per trade (default ₹10,000)
   - Max concurrent open positions (default 5)
   - Daily loss circuit-breaker (default −₹5,000)
   - Strategy returned a tradable signal
5. Mode: **Paper** (writes to `paper_trading.db`) or **Kite Live**
   (`KiteClient.place_order(MARKET, tag="algolab_live")`).
6. After successful order, evaluation cache is cleared — user must re-evaluate.

### B. Tab consolidation 23 → 6 (smart-merge with zero feature loss)

| Top-level tab | Sub-tabs | Legacy indexes |
|---------------|----------|----------------|
| 🏠 Command Center | (single panel) | 0 |
| 📊 Research | 📈 Charts · 📊 Pro Charts · 📐 Multi-TF · 📊 Fundamentals · 🔬 Deep Fund · 📊 Ownership | 1, 21, 14, 2, 20, 12 |
| 🧠 Signals | 🔎 Decision · 🧠 Ensemble ML · 🌡️ Regime · ⚡ Co-Pilot · 🏦 Quant | 8, 16, 15, 3, 11 |
| 🧬 AlgoLab | (editor + backtest + Live Runner + Bulk panels) | 5 |
| ⚙️ Execute | ⚙️ Trading Cockpit · 📋 Paper Trading · ⚠️ Risk Metrics · 🔮 What-If | 4, 13, 17, 18 |
| 🔬 Tools | 🔎 Screener · 🔍 Quick Screener · 🔗 Correlation · 📓 Journal · 🌅 Pre-Market · 🌍 Global | 22, 7, 19, 6, 10, 9 |

Implementation: top-level `st.tabs([6 names])`, sub-tabs declared inside each
top via `with _top_tabs[i]:`, then a `tabs = [None]*23` legacy-mapping list
keeps all 1500 lines of existing `with tabs[N]:` body code untouched.

## Iteration 3 — Bulk multi-symbol simulator (2026-05-08)

New module: **`ui/bulk_simulator.py`** — runs the AlgoLab strategy across many
symbols in parallel, both for historical backtest and for latest-signal scan.

### Universes (selectable)
| Universe | Source | Count |
|----------|--------|-------|
| Nifty 50 | Static list | 50 |
| Nifty 100 | Nifty 50 + Nifty Next-50 | ~100 |
| Nifty 500 | First 500 symbols from `get_all_equity_symbols()` | 500 |
| Custom watchlist | Comma-separated input | user-defined |

### Parallel orchestration
- `ThreadPoolExecutor`, default 8 workers, slider 2–16.
- Per-symbol timeout slider (5–60 s, default 30 s); on timeout the symbol is
  recorded as an error and the run continues.
- `st.progress(...)` bar updates as each future completes:
  `"Backtesting 12/50 · errors 1"`.
- Errors are surfaced in an "⚠️ N symbols skipped" expander.

### OHLCV cache
`@st.cache_data(ttl=86_400, show_spinner=False)` on `_fetch_ohlcv_cached` —
yfinance is hit at most once per symbol per 24 h.

### Bulk Backtest (`render_bulk_backtest`) — mounted as expander inside AlgoLab
Output table columns: Symbol · Signal · Return % · Sharpe · Max DD % · Win % · Trades · LTP ₹.
Sorted by Sharpe desc.

Heatmap mode (radio toggle) colours cells with a green-to-red gradient:
- Return %, Sharpe, Win %  → high = green, low = red
- Max DD %                 → high (less negative) = green, low = red
- Signal column            → BUY=green, SELL=red, HOLD=amber

### Bulk Live Scan (`render_bulk_live_signals`) — mounted as expander inside AlgoLab below Live Runner
Output: top-line metric chips (BUY / SELL / HOLD / Errors counts) plus a
sortable table (Symbol · Signal · LTP ₹ · Day %) sorted with BUYs first,
then SELLs, then HOLDs (secondary by `|Day %|` desc).

Same Table / Heatmap radio toggle.

### Verification (Playwright on localhost:8501)
- ✅ Bulk Backtest on Nifty 50 — 48/50 succeeded (~90 s cold cache, ~30 s warm)
- ✅ Bulk Live Scan on Nifty 50 — BUY 34 · SELL 14 · HOLD 0 · Errors 2
- ✅ Heatmap toggle renders with gradient colors
- ✅ "Done — N succeeded, M skipped" status caption
- ✅ "⚠️ N symbols skipped" expander with error-detail table
- ✅ Lint passes on all new modules
- ✅ Live Runner end-to-end: BUY signal for RELIANCE @ ₹1,436.20 → Qty 6,
  SL ₹1,377.39, Target ₹1,553.83; Confirm correctly disabled when market
  closed, enabled inside 9:15-15:20 IST window.

## Files
| File | Status |
|------|--------|
| `ui/live_runner.py` | NEW (Iteration 2) |
| `ui/bulk_simulator.py` | NEW (Iteration 3) |
| `ui/algolab.py` | Updated — wires Live Runner + Bulk Backtest + Bulk Live Scan |
| `ui/heatmap.py`, `ui/macro.py`, `ui/watchlist.py`, `ui/alert_inbox.py`, `charting/multi_tf.py` | Iteration 1 — `persist="disk"` removed |
| `app.py` | Iteration 1 (autorefresh + logger) + Iteration 2 (6-tab layout) |
| `requirements.txt` | `streamlit-autorefresh>=1.0.1` |
| `memory/PRD.md` | This file |

## Backlog
- P1 — Sidebar **Risk Profile** form so traders can tune max ₹, max positions,
       daily loss limit, ATR multipliers without editing `live_runner.py`.
- P1 — Prefer Kite intraday data for `_fetch_recent` and `_fetch_ohlcv_cached`
       when `KITE_ACCESS_TOKEN` is set; fall back to yfinance daily otherwise.
- P1 — Per-tab `try/except` boundary so one bad tab can't blank the page.
- P1 — **6th risk gate**: backtest Sharpe ≥ 0.5 in last 90 days before live
       order is allowed.
- P2 — Persist Live Runner orders to a `live_orders` SQLite table (audit
       trail) + "Cancel Order" button for placed Kite orders.
- P2 — Bulk runs: emit cache hit/miss + per-symbol elapsed time to logs.
- P3 — APScheduler-driven optional auto-execution mode (currently always
       manual).

## Implementation dates
- Iteration 1 — Stability hotfix: 2026-05-08
- Iteration 2 — Live trading + tab consolidation: 2026-05-08
- Iteration 3 — Bulk multi-symbol simulator: 2026-05-08
