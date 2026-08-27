# QuantTerm — System Overview (AI Briefing)

> **Purpose of this document.** A self-contained explanation of the QuantTerm
> trading system for an AI assistant (or a new engineer) with no prior context.
> It complements the terser `CLAUDE.md` in the repo root. Read this first to
> understand *what* the system is and *why*; read `CLAUDE.md` for the exact
> module map and coding invariants. If anything here conflicts with `CLAUDE.md`,
> `CLAUDE.md` wins.

---

## 1. What it is, in one paragraph

QuantTerm is an evidence-driven, self-learning trading terminal for the Indian
stock market (**NSE**, live via the Zerodha/Kite broker) and US equities
(**paper-only**). It scans the whole market (~2,000 clean NSE symbols) every 15
minutes, turns each stock into a graded "setup," filters those setups through
quality gates and a **measured Expected-Value** ranking (learned from the
system's own past trade outcomes, not hunches), sizes and risk-checks each
trade, executes with an exchange-side stop+target exit, and then tracks every
outcome to recalibrate itself. The canonical product UI is the **Vite/React desk**
(`bash scripts/run_quantterm_complete.sh` → `http://127.0.0.1:5173`). One command
owns the local stack (desk, terminal API, report API, autonomy, and market-ops
worker). Streamlit `app.py` is not the product path and is not started. A two-way
**Telegram** bot and an LLM analyst ("JARVIS") remain optional companions. It
runs 24/7 as a background service. Everything is narrated in plain Hinglish
(Hindi-English mix) because that's the primary user's language.

**Canonical launcher:** `bash scripts/run_quantterm_complete.sh`.
Historical research branches such as `overhaul/evidence-lab` are not the
current product path.

---

## 2. Core philosophy — the seven invariants

These are enforced in code, not just documented. Violating them is how the
system loses money.

1. **No fake data, ever.** A symbol with no real data is skipped, never
   simulated. Demo/placeholder fallbacks were deliberately removed.
2. **Stale must look stale.** Cards carry live/EOD tags; a freshness banner
   names the data path. Yesterday's close is never shown as today's price.
3. **Every trade ships with an exchange-side exit.** A GTT OCO order (stop +
   target, resting at the exchange) is placed together with the entry; a loud
   warning fires if the GTT fails.
4. **Telegram taps are paper-only.** Live orders require the app's own ticket.
   A phone button can never place real money (safety invariant).
5. **1% risk per trade, 10% max per name, 5% max total open risk.** Enforced by
   the sizer + the trade ticket + the portfolio meter. DANGER warns but the user
   keeps the final click.
6. **Evidence over vibes.** Signal weights come from a walk-forward backtest.
   Fewer than ~30 tracked trades = no statistical claim. Proven-negative
   signal combos auto-demote from BUY to WATCH.
7. **Every BUY is outcome-tracked.** A "Report Card" equity curve, built from
   real tracked outcomes, is the single artefact that decides whether the
   system deserves real capital.

**Design idioms that follow from these:**
- **Demote-only gates.** Quality filters can only *cut* a score/grade/verdict,
  never inflate one. New checks apply multiplicative penalties on a computed
  score; they never add bonuses.
- **Backward-compatible defaults.** New parameters get safe defaults (e.g.
  `rsi=0.0`, `chase_risk=False`) so old call sites behave exactly as before.
- **Fail-open, never fake.** If a data source is down, the code fails open
  (skips, returns empty) so it never *blocks* trading — but it never invents
  data to fill the gap (invariant #1).
- **Env-tunable thresholds.** Numeric thresholds are exposed via
  `os.getenv("QT_...", default)` so they can be tuned without code changes.

---

## 3. The pipeline (the spine)

Data flows in one direction, with the "Brain" reading across all stages to set
an overall posture:

```
DATA  →  SIGNAL  →  RISK  →  EXECUTION  →  FEEDBACK (learning)
                         ↑___________________________|
                 🧠 BRAIN reads all of the above → posture + directives (read-only)
```

- **Data** — official NSE bhavcopy history + live quotes.
- **Signal** — 17 setups, graded, quality-gated, ranked by measured EV.
- **Risk** — position sizing, portfolio open-risk, correlation clusters.
- **Execution** — entry + GTT exit at the exchange, journalled (paper or live).
- **Feedback** — every decision & outcome logged → recalibrates the scorer.
- **Brain** — composes everything into ONE posture (AGGRESSIVE / NORMAL /
  DEFENSIVE / STAND_ASIDE) + a prioritised to-do list. Never trades itself.

---

## 4. Layer 1 — Market Data (`data/`)

**Quote source policy (strict priority): Kite → NSE snapshot → Google Finance.**
**History policy: NSE official bhavcopy (EOD) + intraday overlay.** yfinance and
Google are last-resort fallbacks only and must never become a primary
dependency again.

- `bhavcopy_store.py` — PRIMARY price history. NSE's official end-of-day
  bhavcopy, ~500 sessions, pickle-cached, incremental. Uniquely carries
  **delivery %** (whether buyers took delivery = real ownership, vs intraday
  punting). Reads are lock-safe against the live overlay writer.
- `nse_live.py` — Intraday overlay. NSE index-API snapshot (~750 stocks/call)
  overlays today's bar onto the store during market hours, so signals form on
  today's move, not just yesterday's close.
- `live_quotes.py` — Unified quote chain (Kite→NSE→Google). Also serves index
  quotes (Nifty/BankNifty/VIX/Sensex).
- `kite_client.py` — Zerodha KiteConnect wrapper: orders, quotes, and the live
  tick WebSocket that feeds the breakout sniper.
- `institutional_flows.py` — FII/DII net flows + bulk deals (free NSE API, 3h
  cache). Feeds a Brain evidence stream and a "🏦" card tag. Context-only (not a
  trade gate) until its edge is measured.
- `instruments.py` — symbol → instrument_token map (~9,948 NSE instruments). The
  authority on what's actually fetchable/tradeable via Kite.
- `nse_universe.py` — ~2,000 clean EQ symbols. A pattern-based junk filter
  (`_is_valid_symbol`) **plus** a cross-check against the Kite instrument map:
  stale listings that look valid but would miss every fetch are dropped before
  they enter a scan.
- `google_finance.py` — fragile scrape, last resort only.

---

## 5. Layer 2 — Signal Engine (`scan/`)

### 5a. The core scanner — `scan/unified_scanner.py`

`UnifiedScanner._analyze(symbol, df)` turns a daily OHLCV DataFrame into a
`StockSignal`: a list of fired signals, plain-English reasons, a composite 0-100
score, a trade plan (entry/stop/target), and a verdict (**BUY** or **WATCH**).
It is data-source agnostic (the same engine runs on US data via `us_scanner.py`).

**The 17 signals** (`SIGNAL_META`), by category:
- **Breakout:** `BREAKOUT_52W` (52-week high), `BREAKOUT_RES` (resistance break
  on volume), `GOLDEN_CROSS` (50/200 SMA), `VOL_SQUEEZE`.
- **Chart pattern:** `VCP` (volatility contraction / tightening base),
  `FLAT_BASE`, `CUP_HANDLE`, `HIGH_TIGHT_FLAG`, `ASC_TRIANGLE`, `DOUBLE_BOTTOM`.
- **Pre-breakout evidence:** `PRE_BREAKOUT` (within 2.5% below a pivot with
  volume building), `ACCUMULATION` (volume dry-up + up-day volume dominance near
  highs), `DELIVERY_SPIKE` (rising NSE delivery %), `NR7_COIL` (narrowest range
  in 7 sessions), `POCKET_PIVOT`.
- **Trend/continuation:** `MOMENTUM`, `PULLBACK_SUPPORT` (a dip to a rising EMA
  in an uptrend — deliberately NOT at highs).

**Breakout grading** — `grade_breakout()` returns `(confirmed, grade, note)`:
grade **A** = clean clearance (≥1×ATR on ≥2× volume), grade **B** = confirmed
but less clean, unconfirmed = a marginal poke (becomes a PRE_BREAKOUT WATCH, not
a BUY). Trade plan geometry: `stop = entry − 2×ATR`, `target = entry + 4×ATR`
(≈ 2:1 reward:risk).

### 5b. The quality gates — demote-only, evidence-first

A confirmed clearance is not enough. Each gate can only cut a grade/verdict. A
*confirmed* base-breakout is exempt from the extension/falling-knife gates
(those target unconfirmed/pre-breakout momentum chases).

- **Close Location Value (CLV, Wyckoff):** `(close−low)/(high−low)`. A break that
  closes in the bottom of its day's range (sellers took the day back) is a
  bull-trap → demotes grade A→B or rejects a marginal break. ATR+volume miss
  this.
- **RSI ceiling:** RSI ≥72 soft-demotes A→B (extended); RSI ≥82 hard-rejects
  (blow-off-top, no room to run — same tier as the gap-exhaustion check).
- **Extension guard ("don't chase"), two views:** fails if EITHER the price is
  >10% above its 20-EMA with strong 5-day momentum (short-term stretched), OR
  >20% above its **50-DMA** (late-stage / far from base). The 50-DMA view
  catches a steady grinder that stays glued to its 20-EMA the whole way up (the
  ADANIENSOL case). Sets a `chase_risk` flag that demotes the verdict to WATCH
  and propagates downstream (blocks the sniper + prevents re-promotion).
- **Laggard filter:** >30% below the 52-week high = laggard zone; a gentle
  linear conviction cut that floors at a 30% cut — thins mediocre laggards,
  never blanket-vetoes the category.
- **Falling-knife / RSI-rollover guard:** a stock red today, or whose RSI has
  rolled over from a few sessions ago, is NOT breaking out — a fresh BUY demotes
  to WATCH. A `PULLBACK_SUPPORT` setup (which buys weakness by design) is the one
  exemption.
- **`breakout_conviction()`** — 0-100 "who is behind the break?" from Volume,
  Delivery %, Relative Strength, Trend stage, Base quality. Below a floor, a
  confirmed break is a WATCH, not a buy.

### 5c. Ranking, market context, and the short side

- `ev_engine.py` — **The north star.** Expected Value per setup from live
  outcomes: `EV% = [P(win)×avgWin − P(loss)×avgLoss] × setupRisk`. Gated at ≥30
  outcomes. Ranks by the *conservative* EV (Wilson score lower bound — big
  samples trusted, small ones shrunk) plus a HIGH/MED/LOW confidence tier.
- `live_edge.py` — the learning feedback loop. Mines the signal log's real
  tracked outcomes → per-signal / per-regime / per-combo expectancy (in R).
  Blended conservatively into the scorer's calibration (≥30 = demote proven
  losers, never inflate).
- `signal_backtest.py` — walk-forward per-signal backtest across the bhavcopy
  store. Simulates each historical setup with the system's **real exit
  discipline**: a breakeven trail at +2% (once a trade pops +2%, its stop moves
  to entry) so a faded pop *scratches* at ~0R instead of being logged as a FLAT
  loss. Produces per-signal accuracy/expectancy → the scorer weights + a
  per-target-geometry sweep. Nightly.
- `conviction.py` — enriches the top ~40 candidates with news buzz + earnings +
  a plain-English checklist. Respects the scanner's safety demotes (a chase-risk
  WATCH can't be promoted back to BUY by buzz/earnings).
- `breadth.py` — market internals from the full scan cache (zero extra fetch):
  advance/decline, % above 50/200-DMA → HEALTHY / MIXED / NARROW. NARROW is a
  Brain "lean-in veto."
- `sector_heat.py` — sector map, pack-boost, sector performance.
- `prime_filter.py` — the elite sieve. A momentum/breakout setup is tagged
  "💎 PRIME" only if it survives EVERY layer (verdict BUY, conviction/EV,
  liquidity ≥₹5cr/day turnover, breadth not NARROW, regime not leaky). Prime
  setups go to the top of the Telegram push.
- `breakout_sniper.py` — real-time pivot alerts. A Kite tick stream fires when a
  break clears a pivot and *holds* for 45s on confirming volume. It's a SEPARATE
  path from the scanner, so it re-applies the scanner's quality demotes itself:
  it skips chase-risk (extended) and blow-off-RSI names so it never fires a green
  "BREAKOUT CONFIRMED" (or auto-trades) a stock the scanner would demote.
- `short_scanner.py` — **the same edge pointed DOWN** (weak-market shorts). A
  full mirror of the long scanner: `grade_breakdown` (support break on volume),
  death-cross, distribution, downside-momentum, lower-highs → verdict
  **SHORT / AVOID**. Every long gate is mirrored: an RSI *floor* not ceiling
  (don't short an already-crushed stock — bounce risk), a *strong* close =
  bear-trap, a rising-rocket guard (don't short a green day), too-far-below-50DMA
  = bounce due. **PAPER-FIRST: detection only, no live orders.** (India blocks
  overnight equity shorts — cash = intraday MIS; positional needs F&O/options —
  so the edge is proven on paper first before any execution vehicle is chosen.)

---

## 6. Layer 3 — Risk (`risk/`) — the gatekeepers

- `position_sizer.py` — the 1%-risk rule + a 10% per-name concentration cap →
  an exact share count. Caps always win over conviction.
- `portfolio_risk.py` — account-level meter: total open risk %, sector packs →
  OK / CAUTION / DANGER. `check_new_trade()` vets each candidate against the
  whole book.
- `position_manager.py` — the after-entry brain: R-progress, book-half/trail
  advice, paper auto-close on stop/target. Trailing profit-lock aims for ₹1,500
  net, floors at ₹1,000, trails the peak.
- `correlation.py` — positions vs *real* bets: 60-day return correlation →
  union-find clusters (ρ≥0.70 = the same macro bet). Catches cross-sector
  co-movement the sector cap misses. Read-only warning lens.
- `watchlist_watcher.py` — buy-zone entered / setup broken / ran-away alerts,
  once per name per day.

---

## 7. Layer 4 — Execution (`execution/`)

- `trade_executor.py` — `place_trade()`: validation rails → Kite entry order →
  **GTT OCO** (stop + target sitting AT the exchange) → journal to
  `logs/trades.db`. Loud warning if the GTT fails. `paper=True` forces paper.
- `autopilot.py` — the hands-free trader. Takes the best-evidenced setups
  through ~13 gates (below), sizes by *measured* conviction (Kelly-lite: risk
  scales 0.5×–1.5× with measured edge, never with vibes), places entry + GTT,
  manages a breakeven trail + time-stop, and **compounds** realized P&L into the
  pool via a monotonic high-water mark (so P&L is never double-counted). Runs
  **paper by default**; can only be armed LIVE from the app, never from Telegram.
- `fo_executor.py` / `zerodha_broker.py` — futures/options order plumbing and
  the legacy engine path (scaffolding for a future short-side vehicle).
- `market_clock.py` (in `core/`) — every NSE gate reads an IST-explicit clock,
  so a UTC cloud host never shifts market hours / daily-limit dates by 5.5h.

**Autopilot gates (every candidate runs this gauntlet):** time window · daily
trade limit · max open positions · symbol already traded today · sector
concentration cap · score/conviction floor · proven-negative measured edge ·
market regime · breadth NARROW · Brain STAND_ASIDE · symbol memory (serial
false-breaker) · live-price anchor / chase · source self-paused · 1-share/pool
fit.

**Presets (frequency dials only — never change the safety rails):**
- Conservative: 2 trades/day, 2 open, min score 70.
- Balanced (default): 4 trades/day, 3 open, min score 60.
- Aggressive: 10 trades/day, 5 open, min score 52.

---

## 8. Layer 5 — Learning / Feedback (`core/`, `reports/`)

This is the moat — the system earns its weights from reality.

- `signal_outcome_tracker.py` — every BUY logged → outcome checked 5 sessions
  later (WIN/LOSS/OPEN). Fill-aware: an untriggered breakout is not judged.
- `decision_journal.py` — every decision (TAKEN *and* REJECTED) logged with its
  prediction (EV/p_win/confidence) → outcome 5 days later. Answers: which gates
  EARN vs COST money, and "when the system said 70%, did ~70% actually happen?"
  (calibration).
- `sim_lab.py` — Monte Carlo bootstrap (500 futures) of the user's own closed
  trade R-multiples, to compare risk settings *before* the live slider moves.
  Scaling advice is earned (profit-factor / drawdown / sample-size thresholds).
- `reports/verdict_dashboard.py` — the Report Card equity curve from tracked
  outcomes. The graduation test for real capital.
- `reports/trade_coach.py` — weekly behavioural review (overtrading, revenge,
  risk inconsistency); rule-based, optionally polished by DeepSeek.
- `reports/street_pulse.py` — the Daily Pulse newsletter data (movers, buzzing,
  breakouts).

Time-based logging in `signal_outcome_tracker` and `decision_journal` uses an
IST-explicit clock (via `market_clock`) so day-bucketing/outcome windows don't
shift on a UTC host.

---

## 9. The Brain (`core/brain.py`) + Macro radar (`core/macro_pulse.py`)

`brain.assess()` composes every subsystem — regime × live-edge × setups ×
portfolio-risk × autopilot × health × breadth × options positioning × FII flows
× **macro news** — into ONE read: a **posture** and a prioritised, plain-language
directive list. It is **read-only** (survival-first, evidence-gated) and is
surfaced as the hero atop the Daily Pulse and the `/brain` Telegram briefing.

**Postures:**
- **AGGRESSIVE** — normal-plus size, let winners run. Only when tape + edge +
  book all align, ≥30 outcomes, AND breadth healthy AND macro calm.
- **NORMAL** — standard size. The steady state, or an otherwise-strong board
  held back by NARROW breadth or a RISK_OFF news tape (demote-only).
- **DEFENSIVE** — small size, A+ setups only, tight stops. Any single caution:
  weak tape, decaying edge, negative expectancy, book CAUTION.
- **STAND_ASIDE** — no new trades. Book open-risk DANGER, or a hostile tape with
  a negative edge. Survival first (overrides everything).

**`macro_pulse.py` — the news radar the technical stack was blind to.** The
system reads the news stream (RSS + Marketaux via `news/fetcher.py`) but
previously only mined it for per-stock buzz. `macro_pulse` reads it for
market-*moving* themes — **tariffs, crude, rates, geopolitics, rupee, FII
flows, index moves** — with direction (e.g. crude UP hurts OMCs/helps ONGC; a
rate cut is bullish; a trade *deal* de-escalates). It is a keyword **context
radar, not a predictive signal**, corroboration-gated (a theme needs ≥2 fresh
articles — one headline is noise). Output: a mood (RISK_OFF / CAUTIOUS /
NEUTRAL / RISK_ON) + heat 0-100 + affected sectors + a plain-English note. It
feeds the Brain as **demote-only** caution: a RISK_OFF tape holds an
otherwise-perfect board back from AGGRESSIVE → NORMAL (like breadth NARROW) and
names the hit sectors — it never forces or blocks a trade on its own.

---

## 10. Interfaces

- **Vite/React desk (`frontend/`, `http://127.0.0.1:5173`)** — the only product
  UI. Start it with **one command**: `bash scripts/run_quantterm_complete.sh`.
  Pages: **Home**, **Market Scanner** (click starts a real `MARKET_SCAN` job),
  **Recommendations** (evidence families from the saved scan; missing scan
  queues the prerequisite job), **Market Reports** (queues a real report job
  when today's wrap is missing), **Stock Intelligence** (workspace first, then
  acquire/refresh of missing research), plus long-term, radar, paper and system
  health. Click → freshness check → durable backend job → visible progress →
  auto-refresh. Empty stays empty. No invented prices or headlines.
- **Telegram bot** — proactive pushes (prime setups, breakout-sniper alerts,
  morning pulse, Kite-login reminder) + two-way commands: `/status`, `/trade`
  (place the best setup now, paper), `/pause` & `/resume`, `/aggressive`
  `/balanced` `/conservative`, `/book 1500` (profit-lock target), `/funnel`
  (why so few trades today), `/brain` (posture on demand). Inline buttons can
  paper-trade or watchlist a setup — but **never** place a live order.
- **JARVIS (`ai/jarvis_orchestrator.py`)** — a DeepSeek chat wired to the full
  live context (setups, sectors, backtest, positions, portfolio). Needs
  `DEEPSEEK_API_KEY` with balance to answer.

---

## 11. Background daemons (started by the one-command stack)

`bash scripts/run_quantterm_complete.sh` owns the process tree. It does **not**
start Streamlit.

- `scripts/local_stack.py` / `scripts/run_quantterm.sh` — Vite desk on :5173,
  FastAPI terminal (`terminal_product_api`) on :8765, report API on :8766,
  `python -u main.py autonomy`, and the market-operations worker that runs
  `MARKET_SCAN`, `MARKET_REPORT`, news, F&O and due-diligence acquire jobs.
- `auto_scan` / market-ops lanes — whole-market scan, news refresh, report
  assembly and Investigate acquire. User clicks enqueue the same durable jobs;
  startup auto-scan is helpful but not the only way a scan can run.
- `telegram_actions._listener` — long-polls for button taps/commands; chat-id
  guarded to the single authorised user.

---

## 12. Config & running it

**.env highlights:**
```
KITE_API_KEY / KITE_API_SECRET / KITE_ACCESS_TOKEN   # Kite token is refreshed daily
DEEPSEEK_API_KEY                                      # JARVIS + coach polish
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID                # alerts + commands
TRADING_CAPITAL=100000  RISK_PER_TRADE_PCT=0.01       # position sizing
MAX_POSITION_SIZE_PCT=0.10  MAX_OPEN_POSITIONS=5
```

**Daily ops:** (1) morning `python main.py login` (Telegram nags at 8:30 if
forgotten — the Kite access token expires daily). (2) `bash scripts/run_quantterm_complete.sh`
then open `http://127.0.0.1:5173` — one command owns the desk and APIs. (3) For 24/7,
the Mac runs a sleep-proof launchd service (`deploy/setup_mac.sh`), or a
Raspberry Pi / VPS (`deploy/setup_server.sh`); see `docs/ALWAYS_ON.md`.

---

## 13. Testing & conventions

- Money-critical logic is covered by a **network-free** pytest suite:
  `tests/test_money_paths.py` (~245+ tests). It runs on every push (GitHub
  Actions) with credentials blanked.
- **New money-path code MUST land with tests** in that file. This is a hard
  rule.
- Pure/testable helpers are preferred: the trade-relevant logic (grading,
  sizing, gates, EV, simulation) is extracted into pure functions that take
  primitives and are unit-tested in isolation; the I/O shells around them stay
  thin.
- Code and user-facing copy are written in Hinglish to match the user; the
  architecture/docstrings are in English.

---

## 14. How to help work on this system (guidance for an AI)

- **Read before you change.** Confirm a filter/gate isn't already handled
  elsewhere; the system deliberately fails open in many places (that's intended,
  not a bug — unless it violates invariant #1).
- **Respect the demote-only rule.** New quality logic cuts scores; it never adds
  bonuses. New params get safe defaults.
- **Never fake data** to make a test or a demo pass. A missing quote stays
  missing.
- **Never let Telegram place a live order,** and never let the autopilot arm
  live from anywhere but the app ticket.
- **Ship tests with money-path code.** Verify against `tests/test_money_paths.py`
  and keep the suite green.
- **India market realities matter:** no overnight equity shorts (intraday MIS or
  F&O/options only); the Kite token expires daily; all NSE time gates must use
  the IST-explicit `market_clock`, never naive `datetime.now()`.
- When unsure how something ranks or gates, trace the pipeline: `data →
  unified_scanner → conviction/ev_engine/prime_filter → auto_scan store →
  brain/autopilot → execution → outcome_tracker`.
```
