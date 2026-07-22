# QuantTerm — Developer Reference

NSE India trading terminal: whole-market scanning, evidence-calibrated
signals, risk-managed execution via Zerodha, proactive Telegram delivery.
Streamlit app (`streamlit run app.py`), branch of record:
`claude/deepseek-multi-agent-system-nrO7n`.

## Architecture (data → signal → risk → execution → feedback)

```
app.py                    # Streamlit shell: nav (Today|Pulse|Stocks|Options|
                          #   Portfolio|JARVIS), startup daemons, Terminal page
├── data/                 # Market data layer
│   ├── bhavcopy_store.py # PRIMARY history: NSE official bhavcopy, ~500
│   │                     #   sessions, pickle-cached (logs/bhav/), incremental
│   ├── nse_live.py       # Intraday: NSE index-API snapshot (~750 stocks/call),
│   │                     #   overlays today's bar on the store in market hours
│   ├── live_quotes.py    # Unified quotes: Kite → NSE → Google (in that order)
│   ├── google_finance.py # Scrape fallback only (fragile — never primary)
│   ├── kite_client.py    # KiteConnect wrapper (orders, quotes, ticker)
│   ├── institutional_flows.py # 🏦 FII/DII net + bulk deals (NSE free API,
│   │                     #   3h cache) → Brain evidence stream + 🏦 card tag.
│   │                     #   Context-only (gate nahi) jab tak edge measure na ho
│   ├── instruments.py    # symbol → instrument_token map
│   └── nse_universe.py   # ~2000 clean EQ symbols; _is_valid_symbol junk filter
│
├── scan/                 # Signal layer
│   ├── auto_scan.py      # BACKGROUND BRAIN: daemon scans whole market every
│   │                     #   15 min (market hrs), orchestrates everything below,
│   │                     #   pushes Telegram, serves the shared results store
│   ├── unified_scanner.py# 16 signals: breakouts, patterns (VCP/cup/triangle/
│   │                     #   double-bottom/HTF), pre-breakout (accumulation,
│   │                     #   delivery spike, NR7, pocket pivot), momentum.
│   │                     #   Scores calibrated by backtest (_load_calibration).
│   │                     #   Close Location Value (Wyckoff): weak close (<0.5,
│   │                     #   din ki low ke paas) demotes grade A→B / rejects
│   │                     #   marginal breaks — ATR+volume alone miss bull-traps
│   │                     #   where sellers take the day back by the close.
│   │                     #   Stock-quality gates (demote-only, evidence over
│   │                     #   vibes): RSI ceiling (72 soft demote, 82 hard
│   │                     #   reject — blow-off-top, no room to run) + distance
│   │                     #   from 52-week high (>30% below = laggard zone,
│   │                     #   gentle linear conviction cut, floors at 30% —
│   │                     #   deliberately soft so a strong setup (~70+ raw
│   │                     #   conviction) still clears the BUY gate even
│   │                     #   deep in laggard territory; thins mediocre
│   │                     #   laggards, never blanket-vetoes the category)
│   ├── bulk_fetcher.py   # prefetch(): bhav store first, yfinance last resort
│   ├── signal_backtest.py# Walk-forward per-signal accuracy → JSON report →
│   │                     #   score calibration + per-card combo_edge
│   ├── live_edge.py      # LIVE feedback loop: mines signal_log's real tracked
│   │                     #   outcomes → per-signal / per-regime / per-combo
│   │                     #   expectancy (R). Blended CONSERVATIVELY into scorer
│   │                     #   calibration (≥30 = demote proven losers, never
│   │                     #   inflate); regime_calibration() demotes a signal in
│   │                     #   the CURRENT tape where it leaks. Raises expectancy.
│   ├── ev_engine.py      # 💰 NORTH STAR: Expected Value per setup from live
│   │                     #   outcomes — EV% = [P(win)×avgWin − P(loss)×avgLoss]
│   │                     #   × setup risk. ≥30 outcomes gate; ranking uses the
│   │                     #   CONSERVATIVE EV (Wilson lower-bound p_win → big
│   │                     #   samples trusted, small shrunk) + HIGH/MED/LOW
│   │                     #   confidence tier; conviction fallback for thin data
│   ├── conviction.py     # Top-40 enrichment: news buzz + earnings + verdicts
│   ├── breadth.py        # 📊 Market breadth from the FULL bulk cache (zero
│   │                     #   fetch): adv/decl, % above 50/200-DMA → HEALTHY/
│   │                     #   MIXED/NARROW. NARROW = Brain ka lean-in veto
│   ├── sector_heat.py    # Sector map (parsed from nse_universe groups),
│   │                     #   pack-boost + sector_performance()
│   └── breakout_sniper.py# Kite WebSocket ticks → instant pivot-break alerts
│
├── risk/                 # Risk layer (the gatekeepers)
│   ├── position_sizer.py # 1% rule + 10% concentration cap → exact qty
│   ├── position_manager.py# After-entry brain: R-progress, book-half/trail
│   │                     #   advice, paper auto-close on stop/target
│   ├── portfolio_risk.py # Account-level: total open risk %, sector packs,
│   │                     #   OK/CAUTION/DANGER verdict; check_new_trade()
│   ├── correlation.py    # 🧩 Positions vs asli BETS: 60-day return correlation
│   │                     #   → union-find clusters (ρ≥0.70 = same macro bet).
│   │                     #   Catches cross-sector co-movement the sector cap
│   │                     #   misses. READ-ONLY lens → Brain warn directive
│   └── watchlist_watcher.py# Buy-zone entered / setup broken / ran-away alerts
│
├── execution/
│   ├── trade_executor.py # place_trade(): validation rails → Kite entry order
│   │                     #   → GTT OCO (stop+target AT the exchange) → journal
│   │                     #   (logs/trades.db). paper=True forces paper mode.
│   └── zerodha_broker.py # (legacy engine path)
│
├── alerts/
│   ├── telegram_alerts.py# AlertEngine.send(msg, reply_markup)
│   └── telegram_actions.py# Two-way: inline buttons (paper-trade/watchlist),
│                         #   long-poll listener; taps can NEVER place live
│
├── reports/
│   ├── street_pulse.py   # Daily Pulse data (movers, buzzing, breakouts)
│   ├── verdict_dashboard.py# Equity curve from tracked signal outcomes
│   └── trade_coach.py    # Weekly behavioral review (overtrading, revenge,
│                         #   risk inconsistency) — rule-based + DeepSeek polish
│
├── ui/                   # Streamlit pages
│   ├── scanner.py        # Smart Scanner: Best Trade hero, sector pulse,
│   │                     #   freshness banner, health strip, trade ticket,
│   │                     #   positions panel, backtest panel
│   ├── command_center.py # Today page (top picks from auto_scan store)
│   └── street_pulse_page.py# Daily Pulse newsletter + Report Card + Coach
│
├── ai/jarvis_orchestrator.py # JARVIS: DeepSeek chat with FULL live context
│                         #   (setups, sectors, backtest, positions, portfolio)
├── core/
│   ├── brain.py          # 🧠 THE CONDUCTOR: composes regime × live-edge ×
│   │                     #   setups × portfolio-risk × autopilot × health into
│   │                     #   ONE read → posture (AGGRESSIVE/NORMAL/DEFENSIVE/
│   │                     #   STAND_ASIDE) + prioritised directives. READ-ONLY
│   │                     #   (survival-first, evidence-gated); surfaced as the
│   │                     #   Brain hero atop Daily Pulse. assess()/decide_posture()
│   ├── portfolio_intel.py# 💼 Portfolio Intelligence: capital as a PORTFOLIO —
│   │                     #   per-holding EV, weakest holding, opportunity-cost
│   │                     #   rotation advice (EV-gap ≥2.5pp after costs).
│   │                     #   ADVICE-ONLY (never rotates); feeds Brain directives
│   ├── decision_journal.py# 🗳️ EVERY decision logged (TAKEN + REJECTED) with
│   │                     #   its prediction (EV/p_win/conf) → outcomes 5d later.
│   │                     #   decision_report(): kaunse gates EARN vs COST;
│   │                     #   calibration_report(): "70% bola toh 70% nikla?"
│   ├── sim_lab.py        # 🧪 Monte Carlo bootstrap of OWN closed-trade Rs —
│   │                     #   compare risk settings over 500 futures BEFORE the
│   │                     #   live slider moves; scaling_advice(): capital is
│   │                     #   EARNED (PF/DD/n thresholds). Advice-only, seeded
│   ├── signal_outcome_tracker.py # every BUY logged → outcome after 5 days
│   ├── market_clock.py   # ⏰ IST-explicit clock — SAB NSE gates (market hours,
│   │                     #   entry windows, daily-limit dates) yahin se. UTC
│   │                     #   server pe bhi gates kabhi shift nahi hote
│   └── error_guard.py    # page error boundaries + logs/errors.log + config check
│                         #   (+ IST-clock check + build version)
└── tests/                # pytest; test_money_paths.py = 180+ money-critical tests
```

## Key Invariants (violate these and lose money)

1. **No fake data, ever** — a symbol with no real data is skipped, never
   simulated. Demo-data fallbacks were removed deliberately.
2. **Stale must look stale** — cards carry live/EOD tags; freshness banner
   on the scanner; never show yesterday's close as today's price.
3. **Every trade ships with an exchange-side exit** — GTT OCO placed with
   the entry; loud warning if GTT fails.
4. **Telegram taps are paper-only** — live orders require the app's ticket.
5. **1% risk per trade, 10% per name, 5% total open risk** — sizer + ticket +
   portfolio meter enforce; DANGER warns but the user keeps the final click.
6. **Evidence over vibes** — signal weights come from the walk-forward
   backtest; <30 trades = no claim; negative-edge combos auto-demote to WATCH.
7. **Every BUY is outcome-tracked** — Report Card equity curve decides
   whether the system deserves real money.

## Background daemons (started once in app.py)

- `auto_scan._worker`: scan → sector heat → conviction → edge → live overlay
  → Telegram push; plus morning pulse (8:30-10), Kite-login reminder
  (8:30-9:15), nightly backtest (off-hours), weekly coach (Sun 17+),
  position/watchlist alerts + breakout sniper (market hours).
- `telegram_actions._listener`: button-tap long-poll.

## Data source policy

Quotes: **Kite (primary when logged in) → NSE snapshot → Google Finance**.
History: **NSE bhavcopy** (official EOD + intraday overlay). yfinance is a
last-resort fallback and for intraday Terminal charts only. Yahoo/Google
must never be a primary dependency again.

## Daily Ops

1. Morning: `python main.py login` (Telegram reminds at 8:30 if forgotten)
2. `streamlit run app.py` — everything else is automatic
3. 24/7 hosting: see `docs/ALWAYS_ON.md`

## Testing / CI

- `python -m pytest tests/` — money-critical suite is network-free
- GitHub Actions runs suite + compileall on every push (credentials blanked)
- New money-path code MUST land with tests in `tests/test_money_paths.py`

## Config (.env highlights)

```
KITE_API_KEY / KITE_API_SECRET / KITE_ACCESS_TOKEN   # token daily
DEEPSEEK_API_KEY                                     # JARVIS + coach polish
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID                # alerts + buttons
TRADING_CAPITAL=100000  RISK_PER_TRADE_PCT=0.01      # position sizing
MAX_POSITION_SIZE_PCT=0.10  MAX_OPEN_POSITIONS=5
```
