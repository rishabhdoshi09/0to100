# QuantTerm evidence-first roadmap
Date: 2026-11-07 | Priority order: safety -> correctness -> recovery -> performance -> research.

## Five highest-value next actions
### 1. Exact-head CI + generation-gate certification (P0/P1)
Dependency: isolated audit branch and synthetic fixtures.
Acceptance: all focused tests and all expected GitHub workflows PASS on the exact final audit commit; new entries reject mismatched/unknown snapshot identity; position management still works independently. Any failure is a blocker.

### 2. Cross-store atomicity and consumer-fail-closed tests (P1)
Dependency: disposable runtime with two synthetic snapshot generations.
Force activation between persist_public_best_setups, persist_desks_from_market_scan and save_scan. Inspect dashboard, recommendations, decision discovery, Telegram handoff and paper-entry pipeline. Acceptance: no generation-A BUY recommendation remains actionable or alertable after B becomes active; no duplicate paper entries; recovery is deterministic.

### 3. Read-only disaster-recovery proof (P1)
Dependency: available verified backup/recovery images and operator-controlled isolated machine.
Verify image/checksums, SQLite + WAL consistency, active snapshot, paper position ledger and immutable decision history. Restore only to disposable copy. Acceptance: documented recovery steps, RPO/RTO and mismatch counts; never overwrite original images or silently initialize empty runtime.

### 4. Performance and autonomous reliability baseline (P2)
Dependency: isolated runtime copy + exact benchmark SHA.
Measure p50/p95/p99 for dashboard/recommendations/decision gate, scan throughput, peak RAM, swap, file descriptors, restart duration and queue depth; trace worst bottlenecks. Acceptance: measurable improvements without changed decisions or lost evidence; singleton worker and robust mount-disconnect handling.

### 5. Scientific trading intelligence validation (P1/P2)
Dependency: point-in-time bars, universe membership, strategy versions and settled outcomes.
Reconstruct walk-forward results after real fills, fees, slippage, gaps and capital constraints. Benchmark against simple alternatives; prove calibration and paired challenger lift with protected holdout. Acceptance: public methodology, reproducible fold outcomes and uncertainty; if absent, label scores as rankings and block unsupported 'winning probability' claims.

## Later stages
Reconcile F&O instrument/Greeks/source timestamps and paper expiry risk. Validate US market calendar, DST, exits and settlement. Harden supply chain, secrets and GitHub Actions permissions. Improve accessibility and page load only after first-screen freshness/eligibility invariants pass.

## Release statuses
Engineering-ready: HOLD pending exact-head tests and real operational invariants.
Paper-trading-ready: HOLD pending isolated end-to-end + real-host acceptance and position reconciliation.
Live-trading-ready: LOCKED; separate independent broker/security/regulatory review and explicit authorization required.

## Change-control rules
All work stays on isolated branch; no production runtime mutation, no order submissions, no deployment or PR merge. Record exact test outputs and blockers. Never substitute synthetic market data for observed trading evidence.
