# QuantTerm Autonomous Development Map

Status legend: `TODO` · `IN_PROGRESS` · `DONE` · `BLOCKED(reason)`

This document is the rolling control plane for QuantTerm's autonomous engineering programme.
Completing a tranche triggers a fresh audit. Any justified new work is appended here and
implemented without waiting for an operator prompt.

## Non-negotiable invariants

1. No invented market data, prices, probabilities, statuses, evidence or success states.
2. HISTORICAL_SIMULATION, FORWARD_PAPER and any future LIVE evidence remain explicitly
   separated by provenance.
3. Live-money execution stays locked/fail-closed unless separately and explicitly graduated.
4. The production thesis used to rank trades is the thesis evaluated in historical and
   forward-paper evidence. Every thesis change receives a new immutable identity.
5. No hindsight leakage: decisions persist the information set available at decision time.
6. No endless loops without measurable cursor movement, information gain or a stopping law.
7. Background work exposes measurable progress or a truthful structural wait reason.
8. UI renders persisted/derived truth; it never manufactures research truth.
9. Strategy/model promotion requires independent evidence; backtest return alone cannot promote.
10. Green unit tests are necessary but never sufficient; runtime/browser acceptance remains required.

## A. Autonomous startup authority

- [IN_PROGRESS] Fresh best-trade discovery automatically authorizes PAPER_FORWARD and
  HISTORICAL_REPLAY without a per-startup operator click.
- [IN_PROGRESS] Persist approval provenance as AUTONOMY vs OPERATOR.
- [IN_PROGRESS] Preserve freshness gates and independent live-money interlock.
- [IN_PROGRESS] Exact-head CI, Product Acceptance, Terminal UI and host migration gates.
- Tracking: PR #180.

Exit gate: startup discovers, authorizes paper/history simulation and continues autonomously
while live execution remains fail-closed.

## B. Truthful data-refresh liveness

- [DONE] Keep Kite history resume cursor separate from runtime progress telemetry.
- [DONE] Atomic runtime telemetry: stage, current/total, percent, symbol, candles fetched,
  requested session/range, error and timestamp.
- [DONE] Replace elapsed-time-only stall warnings with no-measurable-progress detection.
- [DONE] Ignore stale pre-launch telemetry until the active worker emits a new progress token.
- [DONE] Expose measured throughput and ETA when evidence permits.
- [IN_PROGRESS] Regression coverage for false-stall prevention and resume-state purity.
- [TODO] Exact-head CI + Product Acceptance.
- [TODO] Merge only after acceptance proves no regression.

Exit gate: a long but advancing refresh is never called stalled merely for being slow; an
actually motionless refresh reports how long measurable progress has been absent.

## C. Workload / Resource Governor

Priority classes:
`CRITICAL_DATA > CURRENT_SCAN > FORWARD_PAPER > SETTLEMENT > HISTORICAL_REPLAY > RESEARCH`.

- [TODO] Persist resource class and workload cost estimate on durable jobs.
- [TODO] Throttle/defer replay and research when critical data work is I/O-bound.
- [TODO] Guarantee forward settlement and risk maintenance cannot be starved by research.
- [TODO] CPU, disk-I/O and memory pressure observations with truthful degraded-mode reasons.
- [TODO] Back-pressure policy for old Mac hardware; no unbounded thread/process spawning.
- [TODO] Starvation tests and deterministic scheduling tests.

Exit gate: high-volume off-session research cannot degrade market-critical work or create
competing duplicate workers.

## D. Research Director / Evidence Acquisition Engine

Replace generic `RETEST_WITH_MORE_DATA` with a typed research request.

EvidenceGap contract:
- hypothesis/thesis/signal identity and version;
- population and decision bucket;
- current n, wins/losses, uncertainty interval;
- regime / sector / setup scarcity;
- false-positive / false-negative / missed-winner counts;
- calibration bucket weakness;
- requested additional evidence and stopping condition;
- expected information gain;
- priority and compute budget;
- source/provenance requirements.

Research flow:
`Observation -> Evidence Gap -> Experiment -> Evidence -> Decision -> Policy/Hypothesis -> next gap`.

- [TODO] EvidenceGap schema and durable ledger.
- [TODO] Research Director converts research verdicts into specific evidence requests.
- [TODO] Scheduler chooses the highest-value unresolved gap.
- [TODO] Generic RETEST becomes a compatibility label only, backed by a concrete gap.
- [TODO] Gap closes only when its explicit stopping condition is satisfied or hypothesis is retired.

Exit gate: every replay batch can answer "what unresolved question is this acquiring evidence for?"

## E. Information-driven Historical Replay Curriculum

Sampling score combines:
- uncertainty / confidence-interval width;
- disagreement among production/challenger models;
- false positives, hard negatives and missed winners;
- proximity to decision boundary;
- regime scarcity and sector scarcity;
- setup scarcity;
- drift / recent degradation;
- evidence age;
- expected information gain per estimated compute second.

- [TODO] Replay candidate pool with point-in-time-safe metadata.
- [TODO] Curriculum selector and durable selection explanation.
- [TODO] Coverage targets for bull/bear/sideways, high/low volatility and event regimes.
- [TODO] Deduplicate materially equivalent samples.
- [TODO] Never oversample simply because data are cheap/easy.
- [TODO] Track useful_training_examples / replay_second.

Exit gate: repeated replay is purposeful and coverage-aware rather than chronological/random churn.

## F. Canonical Signal Registry

Every signal has:
- stable signal_id and semantic version;
- enabled state;
- feature/schema dependencies;
- scanner eligibility;
- calibration eligibility;
- historical replay eligibility;
- forward-paper eligibility;
- minimum evidence n;
- exclusion/disabled reason;
- owner family and deprecation state.

- [TODO] Single registry consumed by scanner, calibration, replay and UI.
- [TODO] Reconcile scanner-vs-calibration counts by explicit identities.
- [TODO] Expose exact exclusions when counts differ.
- [TODO] Prevent unknown/unregistered signals from entering production evidence.

Exit gate: "17 scanned / 16 calibrated" always identifies the one excluded signal and why.

## G. Immutable CalibrationSnapshot

Key:
`data_version + feature_version + thesis_version + signal_registry_version + model_version`.

- [TODO] Frozen calibration artifact with hash and provenance.
- [TODO] Reuse unchanged snapshot; no repeated recalibration on identical inputs.
- [TODO] Historical replay references a frozen snapshot id.
- [TODO] Invalidate only when a declared dependency changes.
- [TODO] Reproducibility test: same key -> same calibration result hash.

Exit gate: calibration work is reproducible, cacheable and never silently changes mid-experiment.

## H. High-throughput Replay Engine

- [TODO] Immutable market-universe snapshot with content hash; reuse until source dataset changes.
- [TODO] Cached point-in-time feature matrices.
- [TODO] Incremental indicator computation.
- [TODO] Vectorized batch evaluation where semantics remain identical.
- [TODO] Bounded parallel symbol evaluation where safe.
- [TODO] Persistent replay checkpoint/cursor.
- [TODO] Measure useful examples/sec, not merely elapsed runtime.
- [TODO] Benchmark cold vs warm replay without sacrificing truth.

Exit gate: throughput improves materially while producing byte-for-byte equivalent decision
semantics for a fixed input snapshot/thesis/config.

## I. Decision Fingerprints / Immutable Information Sets

Persist for every accepted and rejected decision:
- decision timestamp and session;
- dataset/snapshot/manifests;
- thesis/model/signal/calibration versions;
- feature values and missingness;
- market/sector/regime context;
- entry/target/stop/risk assumptions;
- confidence components;
- gate outcomes and rejection reason;
- source hashes;
- deterministic fingerprint hash.

- [TODO] Fingerprint schema + hash.
- [TODO] Historical and forward-paper paths share the same fingerprint builder.
- [TODO] Settlement appends outcome; never mutates decision-time inputs.

Exit gate: every result can be audited without reconstructing information from future state.

## J. Counterfactual Shadow Portfolio

Track decisions rejected for:
`EXTENDED`, `LOW_EVIDENCE`, `SECTOR_WEAK`, `REGIME`, `RISK`, `WAIT`, and other gates.

- [TODO] Persist rejected-decision fingerprints.
- [TODO] Settle shadow outcomes using the same outcome definition.
- [TODO] Measure correct rejects, avoided losers and missed winners by gate/version.
- [TODO] Feed systematic rejection errors to the Research Director.
- [TODO] Never count a rejected trade as executed P&L.

Exit gate: filters learn from what QuantTerm refused as well as what it traded.

## K. Decomposed / Calibrated Confidence

Expose separately:
- base setup evidence;
- regime adjustment;
- sector adjustment;
- extension/chase penalty;
- liquidity/risk adjustment;
- historical n and interval;
- forward-paper n and calibration;
- final calibrated probability/decision score.

- [TODO] No opaque confidence with unknown provenance.
- [TODO] Historical and forward evidence contributions remain separately visible.
- [TODO] Brier score / reliability curve / calibration-error monitoring.
- [TODO] Confidence suppressed or downgraded when sample requirements fail.

Exit gate: every displayed probability can be traced to evidence and calibration artifacts.

## L. Champion / Challenger Tournament

Compare production vs challenger on:
- OOS expectancy;
- hit rate and payoff ratio;
- Brier score and calibration error;
- Top-5 precision;
- max drawdown and adverse excursion;
- turnover/cost sensitivity;
- sector and regime stability;
- sample-size confidence intervals;
- forward-paper confirmation.

- [TODO] Tournament artifact and immutable comparison ledger.
- [TODO] Promotion criteria require OOS + robustness + forward confirmation.
- [TODO] Challenger cannot promote on global win rate alone.
- [TODO] Promotion/retirement decisions identify exact evidence used.

Exit gate: promotion is an auditable evidence decision, not a performance-chasing shortcut.

## M. Automatic Hypothesis Generation

- [TODO] Detect conditional failure patterns after experiments/settlement.
- [TODO] Convert patterns into candidate hypotheses with falsifiable statements.
- [TODO] Register hypothesis before testing to limit hindsight/data-snooping.
- [TODO] Generate experiment/evidence-gap request.
- [TODO] Accept filter/weight changes only after graduation gates.
- [TODO] Rejected hypotheses remain visible negative evidence.

Exit gate: failed challengers can produce new falsifiable research directions instead of endless retests.

## N. Plateau / Stopping Laws

- [TODO] Track information gain, interval shrinkage and policy delta per added batch.
- [TODO] Stop a line of research when marginal information falls below threshold.
- [TODO] Stop repeated RETEST when decision/verdict remains unchanged after declared budget.
- [TODO] Reallocate compute to next unresolved evidence gap.
- [TODO] Explicit terminal outcomes: SUFFICIENT, REJECTED, INCONCLUSIVE_BUDGET_EXHAUSTED.

Exit gate: QuantTerm cannot replay forever simply because more history exists.

## O. Authoritative Runtime State

- [TODO] Derive high-level state from durable active work rather than "last writer wins".
- [TODO] Reconcile RUNNING/PENDING work, background workers, structural waits and failures.
- [TODO] No RESEARCHING state after research is complete and no research work remains.
- [TODO] State projection tests covering concurrent refresh/replay/research.

Exit gate: UI state always agrees with the actual durable workload.

## P. Structured Logging / Job Transparency

- [TODO] Deduplicate identical high-frequency heartbeat lines.
- [TODO] Rate-limit stable informational messages while preserving state transitions.
- [TODO] Replace opaque counters such as B/F with named categories.
- [TODO] For each nonterminal/problem category expose count, oldest age, cause, owning job,
  whether action is required and recovery policy.
- [TODO] Structured incident objects for stalls/dead workers/checkpoint recovery.

Exit gate: operations can be diagnosed without interpreting cryptic counters or log floods.

## Q. Research Command Center

One screen answers:
- What does QuantTerm currently believe?
- What evidence is weak?
- What is it researching now, and why?
- Why was the next replay batch selected?
- What new evidence arrived?
- Which hypotheses were accepted/rejected?
- Champion vs challenger state.
- Regime/sector/setup coverage.
- Recent calibration change.
- Did today's research change any future decision policy?

- [TODO] Read-only API projection from durable evidence.
- [TODO] Drilldown by gap, experiment, signal, model and decision fingerprint.
- [TODO] Never recompute evidence in the frontend.

Exit gate: an operator can determine whether QuantTerm genuinely became better-informed today.

## R. Continuous adversarial audit loop

After each merged tranche:
1. rerun deterministic, integration, browser and runtime gates;
2. inspect truth/provenance inconsistencies;
3. inspect performance bottlenecks;
4. inspect scientific/research weaknesses;
5. inspect UI/operator confusion;
6. inspect failure/restart behaviour;
7. inspect evidence leakage/data-snooping risk;
8. compare capabilities against professional terminal/research-system expectations;
9. add justified next developments to this map;
10. implement the highest-value next item.

"Competitive with giants" is treated as an engineering aspiration, not a claim. QuantTerm may only
claim capabilities that are actually implemented and validated. The programme optimizes for
evidence quality, decision usefulness, reliability, reproducibility and operator clarity rather
than feature count.
