# Implementation Plan — Evidence Lab overhaul (staged backlog)

Small, logically-separated commits. Each phase leaves the system runnable and behind
tests. Money- and evidence-critical items first (per `TRUTH_AUDIT.md`). Physical
folder migration is incremental; logical boundaries/interfaces come first (directive §6).

Milestone status legend: `TODO` · `IN_PROGRESS` · `DONE` · `BLOCKED(reason)`.

---

## Phase 0 — Truth & scaffolding  *(this milestone)*
- [DONE] Branch `overhaul/evidence-lab`; preserve prior branch.
- [DONE] `TRUTH_AUDIT.md`, `ADR-001`, `DATA_CLASSIFICATION.md`, this plan, `PROGRESS.md`.
- [TODO] Append a `RESEARCH_LOG.md` entry recording the architectural discovery that
  per-trade R is insufficient for portfolio-alpha claims (directive §11), without
  erasing EXP-002…005.

## Phase 1 — Safety & fail-closed (MONEY/EVIDENCE critical, smallest blast radius)
- [TODO] **C-04b** Feature-flag `QT_LIVE_ENABLED` (default false) gating all live
  arming in `execution/autopilot.py`; live arming raises if unset. Telegram live-order
  path assert-disabled. Tests: arming refused when flag off.
- [TODO] **C-06** Introduce `evidence.write()` transactional helper that RAISES on
  failure; convert `auto_scan` ledger emission and outcome writes from `except: pass`
  to fail-closed. Keep news/LLM/UI fail-open. Tests: a failed evidence write aborts.
- [TODO] `TrustClass` enum + `requires_trust` boundary stub (no behaviour change yet).

## Phase 2 — Separate services from Streamlit (RELIABILITY, C-05)
- [TODO] Extract runnable workers: `services/ingestion_worker`, `research_worker`,
  `outcome_worker`, `execution_worker` — explicit `python -m` entry points, PID-safe
  singleton, heartbeat record, structured logs, graceful shutdown, idempotent jobs.
- [TODO] `app.py` stops starting daemons; the UI reads worker state / requests work
  via a small job/heartbeat table. Duplicate-worker prevention test.

## Phase 3 — Point-in-time data platform (EVIDENCE critical, C-01/03/04/07)
- [TODO] `data_platform/raw_archive/` — immutable append-only store; every source file
  saved with source, retrieval ts, market date, **SHA-256**, parser version, trust
  class, original filename, ingestion status.
- [TODO] `data_platform/security_master/` — bitemporal identity (`security_id`, ISIN,
  symbol/series history, listing/delisting, `valid_from/to`, `knowledge_from/to`).
- [TODO] `data_platform/corporate_actions/` — explicit CA event ledger (split/bonus/
  dividend/rights/merger/demerger/spin-off/symbol-change); deterministic adjustment;
  **never infer a CA from a price gap in research grade**. Golden tests.
- [TODO] `data_platform/point_in_time/` — reconstruct investable universe at a
  timestamp (incl. then-delisted names); coverage/quality checks.
- [TODO] `data_platform/manifests/` — dataset manifest (`snapshot_id`,
  `source_file_hashes`, `transform_commit`, `schema_version`, coverage, quality,
  `trust_class`). Storage: Parquet + DuckDB.
- [TODO] Enforce `requires_trust(RESEARCH_GRADE)` at research entry (fail closed).

## Phase 4 — Chronological portfolio simulator (EVIDENCE critical, C-02)
- [TODO] `research/portfolio_engine/` — event-time loop: reveal-only-available-info →
  reconstruct universe → signals → intended orders → entry delay → fill logic →
  cash/risk enforcement → concurrent holdings → mark-to-market → costs/slippage → CA →
  suspensions/delistings → exits → **immutable daily NAV ledger**.
- [TODO] Model overlapping positions, rejected orders (insufficient capital), partial
  deployment, turnover, idle cash, concentration, sector/correlated exposure,
  liquidity, capacity, taxes/brokerage/slippage, next-tradable-price fills.
- [TODO] All CAGR/Sharpe/drawdown/alpha derive from the ledger, labelled
  `HISTORICAL_SIMULATION` / `FORWARD_PAPER` / `LIVE` / `MODELLED` / `OBSERVED`.
- [TODO] Retire per-trade `_equity_and_drawdown` modelled-CAGR as a *claim* source
  (keep for diagnostics only, relabelled).

## Phase 5 — Statistical framework upgrade (EVIDENCE, C-08/09/11)
- [TODO] Evaluate daily portfolio-return streams; add HAC/Newey-West alpha inference,
  factor-neutral attribution, CSCV + Probability of Backtest Overfitting, an untouched
  temporal holdout, complexity accounting, and a **persistent cross-experiment trial
  ledger** (every inspected variant counts).
- [TODO] Capacity / turnover / cost / universe / subperiod / sector / regime
  sensitivity sweeps as *diagnostics that also register as trials*.
- [TODO] Schema-cohort enforcement; point-in-time coverage-checked regimes.
- [TODO] Strategy lifecycle states: HYPOTHESIS → EXPLORATORY → HISTORICAL_SURVIVOR →
  FORWARD_PAPER → LIMITED_LIVE → PRODUCTION → RETIRED/REJECTED.

## Phase 6 — Evidence registry hardening (EVIDENCE)
- [TODO] Every observation/experiment links timestamp, security_id, schema version,
  snapshot_id, source-file hashes, commit, config hash, hypothesis id, outcome def,
  execution assumptions, verdict. Immutable observation + appended outcome events.

## Phase 7 — Execution isolation & graduation (MONEY)
- [TODO] `execution/` service: idempotent client order IDs, order-state machine,
  entry/exit atomicity, GTT reconciliation, stale-order detection, restart recovery,
  duplicate prevention, audit log, kill switch, explicit arming from trusted UI, no
  Telegram live path. Documented graduation criteria in `EXECUTION_SAFETY.md`.

## Phase 8 — Product simplification & CI (RELIABILITY/ARCHITECTURE)
- [TODO] Two UIs: Research (dataset health, coverage, registry, verdicts, negative
  evidence, forward-paper) and Trading (setups, positions, risk, reconciliation,
  freshness). UI renders; never recomputes research truth.
- [TODO] CI: Ruff, mypy/pyright, pytest, critical-path coverage, property-based,
  deterministic replay, point-in-time leakage, CA golden, delisting/symbol-change,
  cost-model, concurrent-position, cash-constraint, evidence-transaction, worker
  idempotency, migration, timezone, announcement-release-time, failure-injection,
  and **snapshot-id → result-hash reproducibility** tests.

## Phase 9 — Research programmes (only after the platform is trustworthy)
- [TODO] Data structures + pre-registration scaffolding for: (A) corporate-event
  underreaction; (B) ownership/capital-supply events; (C) liquidity/attention
  dislocations. No assumption of alpha — honest test harnesses only.

## Career-grade docs (produced as their phase lands)
`architecture/EVIDENCE_LAB_ARCHITECTURE.md`, `architecture/DATA_LINEAGE.md`,
`architecture/PORTFOLIO_SIMULATOR.md`, `architecture/RESEARCH_GOVERNANCE.md`,
`architecture/EXECUTION_SAFETY.md`, `research/REPRODUCIBILITY_STANDARD.md`,
`research/STRATEGY_GRADUATION.md`, `product/EVIDENCE_AUDIT_PRODUCT.md`.

## Definition of completion
The 15 criteria in directive §19 — tracked in `PROGRESS.md`.
