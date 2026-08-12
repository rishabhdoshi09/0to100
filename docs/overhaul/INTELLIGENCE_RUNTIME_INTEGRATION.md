# Intelligence Runtime — Integration Map

Implementation map for wiring the two-brain layer (`research/intelligence/`) into the real
daily loop. Not a redesign — it names the exact seams.

## 1. Current `grow_one_day` flow (`research/auto_research/scheduler.py`)
`AutoResearchBrain.grow_one_day(date)` → search-weighted `run_once` (discovery BACKTEST +
`paper.deploy` survivors) → `_run_paper_day` (signal_fn/bars_fn → `paper.trade_day`) →
per-strategy `calibrate` → `knowledge.remember_*` → persist knowledge + paper book. The
daemon `_worker` calls `maybe_grow_today` once/day when `paper.engaged`.

## 2. Scheduler ownership
`AutoResearchBrain` (module singleton via `get_brain()`) owns the daemon thread, `paper`
(PaperAutonomyManager), `knowledge`, `regime_fn`, and `paper_state_path`. This is the ONE
authoritative loop owner — the intelligence cycle plugs in here, not a second scheduler.

## 3. Current paper deployment flow
`paper.deploy(spec, ev, …)` → `autonomous_paper_approve` (paper_autopilot) → activate →
`PAPER_EVALUATION`; entries come from `signal_fn` (currently the scanner) → `book.open_position`.

## 4. Current outcome-measurement flow
`paper.trade_day` → `book.mark(bars)` closes on stop/target/gap/max-hold → `book.stats`/
`r_stats` → `calibrate` → `knowledge`. Outcomes are numbers in the book, not canonical events.

## 5. Where canonical events should be emitted
Inside a NEW orchestrator `run_intelligence_cycle` (broker/Streamlit-free), at: cycle start,
data gate, per-strategy evaluation, signal generated/rejected, unsupported runtime, market/
execution/outcome decoded, evidence card created/updated, allocation decision, trade intent
created/blocked, paper position opened/updated/closed, allocation increased/reduced, strategy
paused/retired, cycle completed/failed-safe. Persisted in the existing `EventStore`.

## 6. Where Brain 1 runs
After outcomes for a strategy are decoded in the cycle: `evidence_brain.build_card(strategy_def,
backtest_R, forward_returns, …)` → append immutable `StrategyEvidenceCard` (history preserved;
latest = current view via `event_store.latest_cards`).

## 7. Where Brain 2 runs
After Brain 1 updates cards, per cycle: `allocation_brain.decide(cards, current_risk, clusters,
data_ok)` → append `PaperAllocationDecision`. Consumes only immutable cards.

## 8. Where allocation decisions reach paper autonomy
Brain 2 does NOT open positions. A `TradeIntent` is created from a DEPLOY/INCREASE decision +
a live signal, passes the portfolio/risk gate, then `book.open_position` executes it. REDUCE/
PAUSE/RETIRE update runtime state and (for retire) the manager lifecycle.

## 9. Persistence & recovery gaps (to fix)
Today: knowledge + paper book snapshot persist; **deployed-strategy runtime state is
re-derived**, and **completed-cycle identity is not recorded** (no idempotency guard across
restarts). Fix: a `RuntimeState` persisting per-strategy lifecycle/allocation/risk-budget/last
cycle + a set of completed `cycle_id`s, loaded on start, reconciled against the book.

## 10. Portfolio-risk integration gaps (to fix)
`PaperBook` enforces per-trade 1% / 10% per-name / 5% total / max-positions. Missing at the
loop level: **family cap, correlation-cluster cap, regime stand-down as a gate on intents**.
Brain 2 already computes family/cluster caps; the loop must enforce them via the intent gate
(reusing `risk/correlation.clusters_from_corr`) and emit a block event with a reason code.

## Seam summary
```
scheduler.grow_one_day ──▶ run_intelligence_cycle(ctx)         # NEW orchestrator
   ctx = {as_of_date, market_snapshot, strategy_registry,      #  broker/UI-free
          paper_book, runtime_state, event_store, knowledge}
run_intelligence_cycle:
   data gate → per-strategy runtime signals → decode(events)
   → manage/exit open positions → decode outcomes
   → Brain 1 cards → Brain 2 decisions → TradeIntents
   → portfolio/risk gate → book.open_position → persist state
```
Operating mode `PAPER_AUTO` runs this end-to-end; live modes stay disabled and the
`USER_APPROVED` gate is untouched.
