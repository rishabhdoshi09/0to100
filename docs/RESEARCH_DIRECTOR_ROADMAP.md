# QuantTerm Research Director Roadmap

## Mission

Build QuantTerm into a truth-first autonomous market-intelligence and paper-trading research system
that can be evaluated against institutional-grade engineering and research standards. The goal is
not to claim guaranteed market outperformance; the goal is to make every advantage measurable,
reproducible, risk-bounded, and continuously challenged.

## Non-negotiable invariants

1. No invented market values, confidence, P&L, freshness or success states.
2. HISTORICAL_REPLAY, FORWARD_PAPER and LIVE evidence never collapse into one evidence class.
3. Live-money execution remains fail-closed unless independently and explicitly authorized.
4. Every autonomous mutation has provenance: input data identity, thesis version, code/config identity,
   reason, evidence request or hypothesis, result and terminal decision.
5. Repetition without information gain is a defect.
6. A long-running job is not called stalled unless measurable progress stops.
7. Challenger research cannot silently change production selection.
8. Backtest/replay evidence never masquerades as forward validation.
9. Every batch/cycle is restart-safe and idempotent.
10. A failed subsystem is reported as failure/degradation, never as no-opportunity.

## Program map

### T0 — Autonomy bootstrap
Status: VALIDATING / PR #180

- Fresh best-trade discovery automatically grants PAPER_FORWARD/HISTORICAL_REPLAY simulation authority.
- Approval provenance distinguishes AUTONOMY from OPERATOR.
- Stale/freshness gates stay authoritative.
- Live-money interlock unchanged.

Acceptance:
- complete-stack startup requires no decision-simulation click;
- stale discovery cannot auto-authorize;
- all exact-head CI/product/browser/host gates green.

### T1 — Truthful data-refresh liveness
Status: VALIDATING / PR #182

- Separate resumable security history state from runtime telemetry.
- Publish stage, processed/total, percent, throughput, target session, last progress time.
- Stall = no measurable progress, not elapsed runtime.
- Ignore telemetry from an older worker.
- Preserve snapshot/auth/freshness fail-closed behavior.

Acceptance:
- long healthy refresh never shows false stall;
- genuine no-progress condition is detectable;
- restart resumes history without telemetry corruption;
- product status can expose factual progress.

### T2 — Research Director / Evidence Acquisition
Status: IN PROGRESS

- Replace vague RETEST_WITH_MORE_DATA outcomes with a durable EvidenceRequest.
- Every request states strategy/gap, evidence origin, allowed evidence lane, current samples,
  target samples, deficit, missing metrics, acquisition tasks and stopping conditions.
- Historical batches record the request they serve.
- Forward-only gaps cannot be satisfied by historical replay.

Acceptance:
- every retest names the missing evidence;
- every autonomous research batch has request provenance or explicitly says ordinary backlog;
- wrong evidence class cannot close a request.

### T3 — Active-learning replay curriculum
Status: TODO after T2

Priority inputs:
- uncertainty / wide confidence interval;
- signal or model disagreement;
- false positives and hard negatives;
- missed winners / costly rejection gates;
- underrepresented market regimes;
- underrepresented sectors/setups;
- decision-boundary examples;
- drifted or stale calibration buckets.

Required safeguards:
- targeted samples are TRAINING/RESEARCH evidence, not untouched validation;
- adaptive sampling history is persisted;
- separate untouched temporal holdout remains untouched.

Acceptance:
- next batch includes a quantified information objective;
- useful evidence yield is measured per replay second;
- ordinary chronological replay is fallback, not default, when a valid target exists.

### T4 — Evidence-gap progress + plateau detector
Status: TODO

Track request progress across batches:
- samples acquired;
- new regimes/sectors covered;
- CI width change;
- calibration error change;
- challenger/champion separation;
- information gain;
- batches with zero relevant yield.

Stop/replan when:
- request is satisfied;
- marginal information gain falls below threshold;
- N consecutive batches add no target evidence;
- the required lane is unavailable;
- thesis/data version changes.

Acceptance:
- RETEST cannot loop forever;
- plateau produces a new research question or closes the investigation.

### T5 — Signal Registry
Status: TODO

For every scanner signal:
- stable signal ID + version;
- enabled/disabled;
- scanner eligibility;
- calibration eligibility;
- replay eligibility;
- minimum sample requirement;
- current sample count;
- exclusion reason;
- feature/config hash;
- last calibrated data snapshot.

Acceptance:
- scanner=17/calibrated=16 can never be unexplained;
- disabled/unavailable signal has a machine-readable reason.

### T6 — Frozen CalibrationSnapshot
Status: TODO

Key calibration by:
dataset snapshot + feature version + thesis hash + signal-registry version + model version.

Acceptance:
- unchanged inputs cannot trigger recalibration;
- replay references one immutable calibration snapshot;
- recalibration cause is visible.

### T7 — High-throughput replay engine
Status: TODO

- immutable universe snapshot cache;
- cached indicator/feature matrices;
- incremental calculations;
- safe symbol-level parallelism;
- no duplicate universe reload if source hash is unchanged;
- useful_examples_per_second metric;
- CPU/I/O resource governor.

Acceptance:
- reproducible result hash for same inputs;
- speed improvement measured without changing decisions;
- foreground current-market work preempts background research.

### T8 — Counterfactual shadow portfolio
Status: TODO

Track every rejected candidate by reason:
WAIT, EXTENDED, WEAK_EVIDENCE, SECTOR_WEAK, RISK, etc.

Measure:
- avoided losers;
- missed winners;
- rejection precision;
- opportunity cost in R;
- gate-specific calibration;
- regime/sector interactions.

Acceptance:
- every meaningful rejection gate can prove whether it earns or costs expectancy;
- gate changes require preregistered evidence.

### T9 — Decision fingerprints
Status: TODO

Persist immutable decision-time fingerprint:
thesis hash, code/config hash, dataset/snapshot, calibration snapshot, feature vector,
market regime, sector rank, signal versions, entry/stop/target, confidence decomposition,
all gate outcomes and source IDs.

Acceptance:
- settled outcome can always be joined back to exact decision-time truth;
- no hindsight mutation.

### T10 — Decomposed confidence
Status: TODO

Expose separately:
- base setup evidence;
- regime adjustment;
- sector adjustment;
- extension/chase penalty;
- liquidity/risk penalty;
- historical sample/CI;
- forward-paper sample/CI;
- calibration quality;
- final calibrated score.

Acceptance:
- final confidence can be reconstructed deterministically from visible components;
- historical and forward confidence remain distinguishable.

### T11 — Champion / challenger tournament
Status: TODO

Compare on:
- out-of-sample expectancy;
- payoff ratio and hit rate;
- Brier/calibration error;
- Top-5 precision;
- max drawdown / adverse excursion;
- sector stability;
- regime stability;
- turnover/cost/capacity;
- confidence intervals;
- multiple-testing penalty.

Acceptance:
- challenger cannot promote on raw win rate alone;
- production policy changes only after predefined independent gates.

### T12 — Automatic hypothesis generation
Status: TODO

Flow:
Observation -> quantified gap -> causal hypothesis -> preregistration -> experiment ->
adversarial challenge -> decision -> memory -> next gap.

Constraints:
- grammar-bounded changes only;
- no executable-code generation by the hypothesis engine;
- semantic dedupe of known-dead ideas;
- trial ledger counts every inspected variant.

### T13 — Resource Governor
Status: TODO

Priority:
CRITICAL_DATA > CURRENT_SCAN > FORWARD_PAPER > SETTLEMENT > HISTORICAL_REPLAY > RESEARCH.

Control:
- CPU budget;
- I/O budget;
- concurrency;
- memory pressure;
- current-market preemption;
- thermal/host-safe behavior on the Intel Mac.

Acceptance:
- historical research cannot starve official refresh/current scan;
- resource decision visible in status.

### T14 — Runtime-state truth + structured logging
Status: TODO

- derive state from authoritative active work;
- deduplicate near-identical heartbeats;
- replace opaque counters with named status groups, age, owner and action requirement;
- incident object includes stage, worker, last progress, blocking dependency and recovery action.

Acceptance:
- no stale RESEARCHING/DATA_REFRESHING latch after work ends;
- one status surface explains exactly why the system is in its state.

### T15 — Research Command Center
Status: TODO

One operator surface shows:
- current beliefs;
- strongest genuine opportunities;
- unresolved evidence gaps;
- active evidence request;
- why the next batch was chosen;
- batch progress/yield;
- champion vs challengers;
- calibration snapshot;
- rejected hypotheses;
- recent learning delta;
- regime/sector coverage;
- plateau/replan decisions;
- historical vs forward evidence separation.

Acceptance:
- operator can answer "Did QuantTerm become wiser today, and why?" from one screen.

## Continuous development loop

After each tranche:

1. Merge only after exact-head tests and product gates pass.
2. Run full deterministic + browser/complete-stack validation.
3. Inspect runtime truth and generated evidence.
4. Compare observed behavior to this roadmap and institutional-grade failure modes.
5. Create the next highest-value Evidence/Engineering Gap.
6. Implement it on a small isolated branch.
7. Repeat until remaining gaps are lower-value refinements rather than structural weaknesses.

## Competitive scorecard

QuantTerm is compared to serious market systems on capabilities, not marketing labels:

- data correctness and provenance;
- research reproducibility;
- point-in-time integrity;
- statistical discipline;
- calibration;
- counterfactual learning;
- regime awareness;
- portfolio/risk intelligence;
- latency appropriate to its strategy horizon;
- resilience/recovery;
- observability;
- operator clarity;
- autonomous research quality;
- cost/capacity realism.

Market outperformance remains an empirical result to be demonstrated through historical,
forward-paper and eventually separately authorized live evidence; it is never assumed by design.
