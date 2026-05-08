# DevBloom / Prism Quant — PRD

## Original problem statement
Streamlit-based trading terminal (`/app/app.py`, ~2000 LOC) that loaded
correctly, then went **completely blank after a few seconds**. After the
hot-fix the user wanted:
1. Live trading integration for AlgoLab strategies.
2. Tab consolidation — 23 tabs were too cluttered.

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
| F1 | Non-blocking `streamlit_autorefresh.st_autorefresh(interval=30_000)` — never blocks script thread |
| F2 | Removed `persist="disk"` from all 10 cache decorators with TTL |
| F3 | Removed deprecated Monaco editor; kept textarea fallback |
| F4 | Declared all 23 tabs properly via `st.tabs([...])` |
| F5 | Added stdlib structured logger → `/app/logs/streamlit_app.log` |

## Iteration 2 — Live trading + tab consolidation (2026-05-08)

### User choices
| Question | Answer |
|----------|--------|
| Live execution mode | **Kite live + manual confirmation** |
| Run cadence | **Manual** (button-driven) |
| Risk controls | **Defaults** (max ₹/trade, max open positions, daily loss circuit, trading window) |
| Tab consolidation | **Yes — Charts + Fundamentals together; rest smartly merged** |

### Implementation

#### A. AlgoLab Live Runner — new module `ui/live_runner.py`
Workflow:
1. User clicks **Evaluate** → fetches 200 d daily OHLCV via yfinance → executes
   `generate_signals(df)` → reads `signals.iloc[-1]`.
2. Displays Action (BUY/SELL/HOLD), LTP, ATR, evaluated timestamp.
3. If BUY/SELL: shows order math
   - `qty = capital // LTP`
   - `stop_loss = LTP ∓ 2×ATR`
   - `target = LTP ± 4×ATR`
4. Runs **5 risk gates** (any ❌ blocks the Confirm button):
   - Trading window (9:15-15:20 IST, Mon-Fri)
   - Max ₹ per trade (default ₹10,000)
   - Max concurrent open positions (default 5; reads `paper_trading.db`)
   - Daily loss circuit-breaker (default −₹5,000; reads today's realised P&L)
   - Strategy returned a tradable (non-HOLD) signal
5. Mode toggle: **Paper** (writes to `paper_trading.open_position`) or
   **Kite Live** (calls `KiteClient.place_order(MARKET, tag="algolab_live")`).
6. **Confirm button** is `disabled` until all gates pass; on click → places
   order, clears the cached evaluation so the user must re-evaluate before
   the next order.
7. Every step logs structured entries to `devbloom.live_runner` (file + console).

Sandbox change (algolab.py + live_runner.py):
`__builtins__` now exposes `builtins.__dict__` so user strategies can
`import pandas`, `import numpy`, etc. No security boundary claimed (user
edits and runs their own code).

#### B. Tab consolidation — 23 → 6
| New top-level tab | Sub-tabs | Legacy index map |
|-------------------|----------|------------------|
| 🏠 Command Center | (none — single panel) | 0 |
| 📊 Research | 📈 Charts · 📊 Pro Charts · 📐 Multi-TF · 📊 Fundamentals · 🔬 Deep Fund · 📊 Ownership | 1, 21, 14, 2, 20, 12 |
| 🧠 Signals | 🔎 Decision · 🧠 Ensemble ML · 🌡️ Regime · ⚡ Co-Pilot · 🏦 Quant | 8, 16, 15, 3, 11 |
| 🧬 AlgoLab | (single panel — editor + backtest + Live Runner) | 5 |
| ⚙️ Execute | ⚙️ Trading Cockpit (Order Pad / Positions / Backtest Bridge) · 📋 Paper Trading · ⚠️ Risk Metrics · 🔮 What-If | 4, 13, 17, 18 |
| 🔬 Tools | 🔎 Screener · 🔍 Quick Screener · 🔗 Correlation · 📓 Journal · 🌅 Pre-Market · 🌍 Global | 22, 7, 19, 6, 10, 9 |

**Implementation trick**: top-level `st.tabs([6 names])` declared first; inside
each `with _top[i]:` block declare the sub-tab `st.tabs([...])` so
DeltaGenerators are bound to the right parent. A `tabs = [None]*23` list then
maps each legacy index to its new sub-tab — keeps all 1500 lines of existing
`with tabs[N]:` body code untouched.

Removed duplicate "Paper Trading" sub-sub-tab inside Trading Cockpit (was a
duplicate of Execute → Paper Trading).

### Files changed (Iteration 2)
| File | What |
|------|------|
| `app.py` | Replaced flat 23-tab `st.tabs` with 6 top-level + sub-tabs + legacy mapping. Trimmed Trading Cockpit's inner sub-tabs. |
| `ui/algolab.py` | Seed `algolab_code` session-state on first render; broaden sandbox builtins; mount Live Runner. |
| `ui/live_runner.py` | **NEW** — full live-trading flow (evaluate → risk gates → confirm). |
| `memory/PRD.md` | This file. |

### Verification (Playwright on localhost:8501)
- ✅ 6 top-level tabs render; each sub-tab renders its content
- ✅ AlgoLab → Live Runner: Evaluate clicked → BUY signal for RELIANCE
- ✅ Order math correct: Qty=6, SL=LTP−2×ATR, Target=LTP+4×ATR
- ✅ All 5 risk gates render with ✅/❌ and detail text
- ✅ Confirm button correctly **disabled** when any gate fails
- ✅ Zero exception blocks across Command Center / Research / Signals / AlgoLab / Execute / Tools
- ✅ Lint pass on `ui/live_runner.py`

### Default risk-control values (in `live_runner.DEFAULT_RISK`)
| Knob | Default |
|------|---------|
| Max ₹ per trade | 10,000 |
| Max open positions | 5 |
| Daily loss limit | ₹5,000 |
| Trade window | 09:15 – 15:20 IST, Mon-Fri |
| ATR stop multiplier | 2.0× |
| ATR target multiplier | 4.0× |

User can edit these in `ui/live_runner.py`. Future enhancement: expose a
sidebar "Risk Profile" widget so non-developers can change them at runtime.

## Backlog
* P1 — Surface risk-control knobs (max ₹, max positions, daily loss, ATR mults)
       in a sidebar form so traders can tune without code changes.
* P1 — `_fetch_recent` falls back to yfinance only — when Kite token is set,
       prefer Kite intraday data so live signals reflect *intraday* moves,
       not just yesterday's close.
* P1 — Add per-tab `try/except` boundaries — one bad tab shouldn't kill the
       whole render.
* P2 — Persist Live Runner order history in a dedicated `live_orders` SQLite
       table for audit trail.
* P2 — Add a "Cancel Order" button next to recently placed Kite orders.
* P3 — Background scheduler (APScheduler — already in requirements) for
       optional auto-execution mode (currently always manual).

## Implementation dates
* Iteration 1 (stability hotfix): 2026-05-08
* Iteration 2 (live trading + tab consolidation): 2026-05-08
