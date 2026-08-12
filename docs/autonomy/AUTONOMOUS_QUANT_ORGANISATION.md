# QuantTerm as an Autonomous Quant Organisation — design note

Phase A design note (write-before-edit). Turns QuantTerm from a scanner into an organisation that
runs the closed loop ACQUIRE → VERIFY → OBSERVE → EXPLAIN → HYPOTHESISE → CHALLENGE → TEST →
PAPER-DEPLOY → MONITOR → MEASURE → LEARN → ADAPT → REPEAT — **without duplicating any canonical
component** and **without ever increasing risk on its own**.

## Non-goals (explicit)
- No second snapshot store / registry / paper book / event ledger / allocation brain / scanner /
  retail state model.
- No LLM writing/deploying arbitrary Python. No LLM output becoming a `TradeIntent`.
- No real-money execution in this milestone. No autonomous change to constitutional limits.
- No "agent chat theatre" — dialogue is typed, auditable records tied to canonical data.

## Existing components reused (the canonical truth)
| Need | Reused component |
|------|------------------|
| Paper cycle (Brain1→Brain2→gate→PaperBook→outcomes) | `research/auto_research/scheduler.py::AutoResearchBrain` (`get_brain`, `run_intelligence_cycle_day`, `intel_book`, `enable/disable/start/stop`) |
| Immutable data | `research/intelligence/data/snapshot_store.py` |
| Genuine data activation | `research/intelligence/data/kite_activation.py::activate` + `kite_source.KiteDataSource` |
| Live feed (data-only) | `kite_live.KiteLiveOverlay` + `kite_activation.KiteTickerFeed` |
| Strategy grammar + versioning | `research/strategy_studio/spec.py::StrategySpec` (`config_hash`, `bump_version`, `LIFECYCLE`, actor-gated `_TRANSITIONS` — live door is **user-only**) |
| Failed-research memory | `research/scientific_memory.py` (`belief_id`, `is_known_dead`, `record_negative`) |
| Evidence / graduation | `research/intelligence/evidence_brain.py`, `graduation.py` (system may NOMINATE, only user APPROVES live) |
| Anti-overfitting stats | `research/harness.py` (DSR/PSR/Reality-Check/FDR) via existing gates |
| Whole-market scan | `scan/unified_scanner.py` + `product/scan_store.py` (deterministic saved scan) |
| F&O universe/funnel | `data/fno_universe.py` |
| Event/record log | `research/intelligence/event_store.py::EventStore` |
| News | existing `news/curator_service.py` (health contract only — no second loop) |
| Retail projection | `product/` (read-only) |

## Missing links this milestone adds (thin orchestration only)
1. A **durable, Streamlit-independent supervisor** with a persisted job ledger, leases and
   idempotency — replacing uncontrolled daemon threads as the scheduler of record.
2. An explicit **operational state machine** (STARTING…HALTED) not derived from "is a thread alive".
3. **Scheduled jobs** that call the existing components on a market-calendar cadence (auth health,
   data refresh, whole-market scan, paper cycle, outcome/learning) with the 09:30 opening-noise gate.
4. **Typed research dialogue records** (Observation/EvidenceGap/Hypothesis/Challenge/Experiment/
   Promotion/Allocation/Retirement/Learning/Incident) persisted append-only.
5. A **constrained hypothesis generator** over `StrategySpec` dimensions + semantic dedupe against
   `scientific_memory`, and a **deterministic adversarial council** + committee (no self-approval).
6. A **promotion ladder** mapped onto the existing lifecycle (never a conflicting model).
7. A **capability matrix** (failure policy) and a **read-only status projection** for the UI.
8. A CLI entrypoint `python main.py autonomy` and **separate supervised processes** (systemd/launchd).

## Package
```
research/autonomy/
  job_store.py       # SQLite durable job ledger: leases, idempotency keys, recovery
  supervisor_state.py# operational state machine + transition records (IST)
  schedules.py       # per-job cadence, market-calendar + opening-noise aware
  jobs.py            # handlers wiring the EXISTING components (injected for tests)
  supervisor.py      # single-instance lock, heartbeat, lease loop, incidents, status, graceful stop
  dialogue.py        # typed research/operational records
  hypotheses.py      # constrained StrategySpec successors + scientific_memory dedupe
  challenge.py       # Data Auditor / Sceptic / Reality Checker / Promotion Committee (deterministic)
  promotion.py       # ladder mapped onto existing lifecycle; retirement/decay
  health.py          # capability matrix + product status snapshot
```

## Operational state machine
`STARTING → AUTH_REQUIRED → DATA_REFRESHING → {DATA_BLOCKED | DATA_READY} → OBSERVING →
PAPER_ACTIVE → RESEARCHING`, with `DEGRADED` and `HALTED` reachable from any state. Every transition
records prev/next, reason code, plain explanation, trigger, IST timestamp, snapshot id,
`new_risk_permitted`, `positions_manageable`. State is persisted, not inferred from thread liveness.

## Durable job model
One SQLite row per operation: `job_id, job_type, scheduled_for, started_at, finished_at, status,
attempt, lease_owner, lease_expires_at, idempotency_key, input_snapshot_id, output_snapshot_id,
result_summary, error_code, error_message, next_retry_at`. Statuses: `PENDING RUNNING SUCCEEDED
BLOCKED RETRYABLE_FAILED PERMANENT_FAILED SKIPPED_IDEMPOTENT CANCELLED`. A dead process's `RUNNING`
job is recovered when its lease expires. Same `idempotency_key` → `SKIPPED_IDEMPOTENT` (no duplicate
snapshot/scan/experiment/entry/learning).

## Data flow
`auth_health → data_refresh (kite_activation) → bhavcopy/CA/universe-history → snapshot active →
index warm-up → whole-market scan (scan_store) → paper cycle (run_intelligence_cycle_day) →
outcome resolution → learning`. Each mutation job pins one active snapshot + one scan + one registry
version + one config hash + one regime + one news-context version.

## Research flow (governed dialogue)
`evidence-gap planner → constrained hypothesis (StrategySpec successor, created_before_results) →
Data Auditor + Sceptic (deterministic) → preregistration (idempotency + scientific_memory hash) →
canonical backtest → Reality Checker (existing stats gate) → Promotion Committee typed decision
(REJECT | INCONCLUSIVE | RETEST_WITH_MORE_DATA | PAPER_NOMINATED)`. Rejected rules are recorded in
`scientific_memory` so equivalent ideas are not re-tried. A successor gets a new version+hash via
`bump_version`; the frozen parent is never mutated.

## Promotion ladder → existing lifecycle mapping
`PROPOSED/PREREGISTERED→GENERATED/UNDER_REVIEW · BACKTESTING→UNDER_REVIEW · REJECTED→REJECTED ·
INCONCLUSIVE→UNDER_REVIEW(pending) · HISTORICALLY_QUALIFIED→PROMISING · PAPER_NOMINATED→
APPROVED_FOR_PAPER · PAPER_EVALUATION→PAPER_EVALUATION · PAPER_PROVEN→PAPER_CONFIRMED · DECAYED→
DECAYED · RETIRED→RETIRED`. The live door (`ELIGIBLE_FOR_HUMAN_LIVE_REVIEW → USER_APPROVED`) stays
user-only, exactly as `spec._TRANSITIONS` enforces.

## Failure matrix (capability, not one green/red flag)
Auth missing → block new entries, manage safe exits, historical research OK. Snapshot stale → block
entries, manage if trustworthy price, research OK. News down → entries OK, news-studies blocked. CA
incomplete → affected historical strategies/tests blocked. Live feed stale → symbol entries blocked,
risk-reducing exits OK, EOD research OK. Event-store failure → block new mutation, protect state,
UI read-only. Risk Governor unhealthy → block entries, risk-reduction only. Unreconciled → block
until reconciled. A non-critical failure reduces exactly the dependent capabilities, never crashes
the organisation.

## Migration plan
Additive only. The existing `AutoResearchBrain._worker` daemon remains usable, but the supervisor
becomes the durable scheduler of record; the two never run the same mutation concurrently (the paper
cycle is idempotency-keyed by `(snapshot_id, session_date)`). Retail UI reads a status snapshot file
the supervisor writes — the UI never starts the supervisor.

## Readiness progression (no live this milestone)
`RESEARCH_ONLY → PAPER_AUTO → PAPER_PROVEN → SHADOW_LIVE → LIMITED_LIVE_ELIGIBLE →
OWNER_ACTIVATED LIMITED_LIVE`. This milestone reaches PAPER_AUTO operation + the evidence package for
a later LIMITED_LIVE; `SHADOW_LIVE`/live remain **out of scope** and structurally refused.

---

## Delivered vs remaining (honest status)

**Delivered & tested (network-free, 36 acceptance tests in `tests/test_autonomy.py`):**
- Durable supervisor: single-instance lock, SQLite job ledger (`job_store.py`) with leases +
  idempotency, crash recovery via expired leases, bounded-backoff retry → permanent failure, critical
  overdue → DEGRADED + incident, explicit state machine (`supervisor_state.py`), heartbeat, append-only
  incident trail, graceful shutdown, read-only status snapshot. CLI `python main.py autonomy` runs
  **independently of Streamlit** (verified headless).
- Wired job handlers (`jobs.py`, injected deps): auth-health, data-refresh (via `kite_activation`),
  whole-market scan (via the existing scanner+scan_store), paper cycle (via
  `run_intelligence_cycle_day`), news-health. Opening-noise (<09:30) blocks new entries; a provider
  failure becomes RETRYABLE, never "no opportunity"; a stale snapshot never shows ready.
- Governed research dialogue: typed append-only records (`dialogue.py`); evidence-gap planner +
  constrained `StrategySpec` successor hypotheses with semantic hashing and `scientific_memory`
  dedupe (`hypotheses.py`); deterministic adversarial council + committee with no self-approval
  (`challenge.py`); promotion ladder mapped onto the existing lifecycle, allocation/retirement, and
  the LIMITED_LIVE readiness model that the system can never self-activate (`promotion.py`).
- Capability matrix + read-only product projection (`health.py`, `product/autonomy_status.py`,
  `ui/autonomy_page.py` — a new retail page). Two-process deployment units + corrected branch refs.

**Wired to a fake/fixture only (genuine run blocked — no Zerodha token in this environment):**
The auth/data/scan/paper handlers ran end-to-end against injected fakes and the CLI ran headless, but
NOT against a genuine Kite session or real market data (Phase E).

**Designed but NOT yet wired as scheduled jobs (thin follow-ups; slots reserved in `schedules.py`):**
official bhavcopy update, corporate-action acquisition pipeline, point-in-time universe-history
builder, index/regime warm-up job, outcome-resolution job, daily-learning job. The paper cycle's
outcome/learning already run inside the existing brain; promoting them to first-class supervised jobs
is the next increment. SHADOW_LIVE and any live path remain out of scope and structurally refused.
