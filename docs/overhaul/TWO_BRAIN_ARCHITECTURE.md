# Two-Brain Intelligence Architecture

A structural split of the autonomous research system into two brains that communicate ONLY
through immutable, typed, point-in-time records held in an append-only canonical event store.

```
                    ┌─────────────────────────────────────────────┐
   raw inputs  ─────▶  DECODERS (deterministic, idempotent)        │
 (bhav, scans,      │        │                                     │
  outcomes)         │        ▼                                     │
                    │  CANONICAL EVENT STORE (append-only, 1 writer)│
                    └────────┬───────────────────────┬─────────────┘
                             │ reads                  │ reads
                    ┌────────▼─────────┐     ┌────────▼───────────────────┐
                    │ BRAIN 1          │     │ BRAIN 2                    │
                    │ Evidence Brain   │     │ Strategy & Allocation Brain│
                    │ • interprets     │     │ • selects experiments      │
                    │ • StrategyEvidence     │ • deploys to PAPER         │
                    │   Card (immutable)│───▶│ • allocates paper risk     │
                    │ • NEVER trades   │     │ • pauses / retires         │
                    └──────────────────┘     │ • NEVER edits a card       │
                                             │ • NEVER crosses live gate  │
                                             └────────────────────────────┘
```

## How it fits the existing code (reuse, don't duplicate)

| New need | Reused existing component |
|----------|---------------------------|
| statistical evidence (DSR/PSR/alpha-beta/bootstrap CI) | `research/harness.py` |
| forward-vs-backtest calibration | `research/auto_research/growth.calibrate` |
| persistent family trust / freshness | `research/auto_research/knowledge.py` |
| correlation clusters | `risk/correlation.py` |
| paper execution + frictions + journal | `research/auto_research/paper_autonomy.py`, `paper_book.py`, `costs.py` |
| strategy grammar / families | `research/strategy_studio/grammar.py`, `discovery.py` |
| lifecycle + test-enforced live boundary | `research/strategy_studio/spec.py` |
| daily loop / providers / regime | `research/auto_research/scheduler.py`, `providers.py` |

The new package `research/intelligence/` adds the **connective tissue** (schemas + event store
+ decoders) and the two **brain roles**; it orchestrates the modules above rather than
reimplementing them.

## Invariants (existing + new)

Existing (unchanged): paper-only autonomy; LIVE structurally locked; no synthetic-as-evidence;
no data ⇒ no action; no order/broker/kite/telegram import in the intelligence package
(source-scanned by tests); realistic frictions in paper.

New (this milestone):
1. Brain 1 owns evidence interpretation; Brain 2 owns experiment selection + allocation.
2. Brain 2 cannot modify evidence scores; Brain 1 cannot place/size/allocate.
3. Both brains talk only through immutable typed records.
4. Every record carries record-id, event ts, (knowledge ts), strategy id+version, rules hash,
   data snapshot id, source, schema version.
5. Decoders are deterministic + idempotent (reprocessing ⇒ no duplicate semantic event).
6. Event store is append-only, single-writer, and reconstructs both brains' state.
7. Only structured `ResearchRationale` is persisted (observation / hypothesis / supporting /
   conflicting / decision / uncertainty / next_test) — never raw chain-of-thought.
8. Graduation adds `PAPER_CONFIRMED → ELIGIBLE_FOR_HUMAN_LIVE_REVIEW → USER_APPROVED`; the
   `USER_APPROVED` transition is user-only — neither brain nor the paper autopilot can perform it.

## Conflicts found & resolutions
- **Path drift** in the spec (`research/auto_research/discovery.py` etc.) — those modules live
  in `research/strategy_studio/`. Reused at their real locations.
- **Two "live door" names** — kept the single existing user-only boundary and layered the new
  graduation states on top of it, so there is exactly ONE user-owned live gate, still enforced
  by the existing lifecycle tests.

## Phase status (honest)
- **Phase 1 (schemas · event store · decoders)** — implemented.
- **Phase 3 (Brain 1 Evidence Cards)** — implemented (reuses harness/growth/knowledge/correlation).
- **Phase 4 (Brain 2 allocation)** — implemented (transparent weighted score, risk buckets).
- **Phase 7 (graduation boundary)** — implemented (lifecycle states + user-only gate).
- **Phase 2 (per-strategy rule→signal evaluator)** — minimal chronological evaluator + one family
  adapter; remaining families explicitly UNSUPPORTED (fail loud, never fall back to scanner).
- **Phase 8 (UI)** — Brain Observatory + Live Review Candidates + Automatic Strategies as honest
  empty-state pages.
- **Phases 5 & 6 (experiment governance depth · full portfolio wiring)** — partially delivered by
  reusing `discovery` (multiple-testing burden, untouched-test isolation, attempt registry) and
  `risk/correlation` (cluster caps); deeper governance (lineage store, experiment expiry) is the
  next increment. Nothing here is claimed complete that isn't.

## Data reality
No NSE history is loaded in this environment, so decoders emit nothing, Brain 1 issues
`INSUFFICIENT_EVIDENCE`, and Brain 2 deploys nothing — by design. All logic is exercised by
deterministic injected fixtures in `tests/test_intelligence.py`, never presented as market
evidence.
