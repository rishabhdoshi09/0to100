# Evidence Lab Overhaul — Progress Log

Running log, updated after every milestone. Honest status only: `implemented`,
`tested`, `historically simulated`, `unproven`, `blocked`, `invalid`. No completion
claim without evidence.

---

## Milestone 0 — Truth & scaffolding · 2026-07-27 · status: DONE

**Completed work**
- Created branch `overhaul/evidence-lab` off `claude/deepseek-multi-agent-system-nrO7n`
  (prior branch preserved as the historical terminal prototype).
- Inspected implementation as source of truth (not docs) across: app startup &
  daemons, bhav/index stores, universe, corporate actions, signal backtest, gauntlet
  runners, registry, momentum, auto_scan, execution/autopilot, telegram.
- Authored Phase-0 documents: `TRUTH_AUDIT.md` (12 classified contradictions),
  `ADR-001-EVIDENCE-LAB.md`, `DATA_CLASSIFICATION.md`, `IMPLEMENTATION_PLAN.md`.

**Tests run**
- None yet (documentation milestone; no code changed). Existing suites remain green
  from the prior branch (`test_money_paths`, `test_gauntlet`, `test_research`,
  `test_governance`, `test_momentum`).

**Evidence generated**
- A code-backed contradiction audit. Highest-severity confirmed defects:
  1. `MONEY_CRITICAL` — live autopilot enabled during overhaul (C-04b).
  2. `EVIDENCE_CRITICAL` — portfolio metrics synthesised from independent per-trade R,
     no NAV ledger (C-02).
  3. `EVIDENCE_CRITICAL` — research paths can reach yfinance/Google fallback (C-01).
  4. `EVIDENCE_CRITICAL` — survivorship-biased universe; point-in-time is a stub (C-03).
  5. `RELIABILITY` — Streamlit owns all background-service lifecycles (C-05).

**Unresolved risks / blockers**
- `RESEARCH_GRADE` Indian data is BLOCKED on external inputs: `ca_events.json` and
  `universe_history.json` do not exist and are not free to assemble at long history.
  The platform will run fail-closed on `OPERATIONAL_ONLY` data until supplied.

**Architectural decisions**
- ADR-001 accepted: research platform first; portfolio returns as primary evidence;
  trusted paths fail closed; no implicit research-data fallback; no new signal
  features during the overhaul.

**Next milestone**
- Phase 1 (Safety & fail-closed): `QT_LIVE_ENABLED` flag disabling live arming (C-04b);
  transactional fail-closed evidence writes (C-06); `TrustClass` boundary stub. Plus
  the `RESEARCH_LOG.md` entry recording the per-trade-vs-portfolio discovery (§11).

## Milestone 1a — C-04b live-disable (Phase 1) · 2026-07-27 · status: DONE

**Completed work**
- `execution/autopilot.py`: `_live_enabled()` (env flag `QT_LIVE_ENABLED`, fail
  closed / default off) + a hard gate at the top of the LIVE-arm path. Paper
  unaffected. LIVE now refuses regardless of the phrase until the flag is set.
- `docs/architecture/EXECUTION_SAFETY.md`: the live-trading graduation criteria the
  flag stands in front of.
- Test `TestAutopilot::test_live_disabled_during_overhaul` (LIVE refused when flag
  unset even with the correct phrase; paper still arms; flag-on falls through to the
  phrase check).

**Tests run**
- `TestAutopilot::test_live_disabled_during_overhaul` — PASS.
- `TestAutopilot::test_live_arm_needs_exact_phrase` — PASS.
- Two PRE-EXISTING failures surfaced (NOT caused by this change; both arm in PAPER):
  `test_circuit_breaker_disarms`, `test_pnl_snapshot_live_and_day` — UTC↔IST date-
  boundary flakes (run at UTC 23:58 = IST next day). Logged as **C-13** (RELIABILITY,
  money-adjacent). Deferred to the §16 timezone milestone; not fabricating green.

**Evidence generated**
- Live autopilot is now fail-closed disabled (C-04b MONEY_CRITICAL closed for arming).

**Unresolved risks**
- C-13 timezone boundary in day-P&L / circuit breaker (new).

**Next milestone**
- Phase 1 continued: transactional fail-closed evidence writes (C-06); `TrustClass`
  boundary stub (C-01/E). Then Phase 2 service extraction.

## Milestone 2 — Institutional Momentum Breakout research framework (EXP-006) · 2026-07-28 · status: implemented + unit-tested (NOT yet run on real data)

Research-only milestone. **No service extraction, no portfolio-simulator, no live/
paper/Telegram/GTT wiring, no UI redesign, no broad Phase-1 refactor.**

**Completed work**
- New package `research/momentum_breakout/`: `pit.py` (canonical point-in-time-safe
  primitives — ATR/MA/EMA/returns/drawdown/CLV/volume/rel-strength with a
  `FutureLeak` fail-closed contract), `config.py` (versioned thresholds + config
  hash), `observation.py` (`MomentumBreakoutObservation` + canonical `event_id`),
  `features.py` (six feature groups + trend-extension + weakening + transparent
  component scores), `pit_safety.py` (six-clock temporal firewall + `EventRegistry`
  dedup), `detector.py` (base detection, eligibility contract, scoring, dedup),
  `experiment.py` (EXP-006 pre-registration, PIT gap-aware trade simulator,
  ablations, wiring to the existing `research.harness` evidence gate).
- Reused existing Evidence-Lab contracts (harness, gauntlet ledger/registry/freeze,
  feature store, point-in-time universe, CA-adjusted bhav store). Did NOT reuse the
  live, non-PIT ATR/RS/breakout code (`scan/relative_strength.py`, unified_scanner) —
  documented in ADR-002.
- Docs: `ADR-002-MOMENTUM-BREAKOUT-RESEARCH.md`, `MOMENTUM_BREAKOUT_FEATURES.md`,
  `RESEARCH_LOG.md` EXP-006 pre-registration, TRUTH_AUDIT `C-15` (valuation/sector
  not PIT → surfaced + fail-closed).

**Tests run**
- `tests/test_momentum_breakout.py` — 39 passed (deterministic, synthetic,
  network-free; no wall-clock/timezone dependence): PIT primitives, prior-upmove,
  base detection (long contracting detected / deep rejected / future bars don't alter
  an earlier base / reproducible ids), breakout (confirmed vs intraday-only,
  next-bar entry, overextension), structural stop (signal-time only, deterministic
  risk, excessive-risk reject, gap-through-stop not filled at stop), sector
  (strong qualifies / weak rejected / missing membership surfaced), valuation
  (extreme flags but does NOT reject; stale flagged; future rejected; missing ≠ zero),
  deduplication (one event one obs / consecutive closes no dup / new base new event /
  equivalent detectors no double-count), reproducibility + config-hash, experiment
  plumbing, and execution isolation.
- Regression: `test_money_paths.py` + `test_research.py` + `test_gauntlet.py` +
  `test_momentum.py` — all green; PAPER autopilot, Telegram paper-only and the LIVE
  migration lock unchanged.

**Evidence generated**
- NONE yet on real data — the framework is pre-registered and unit-tested only. No
  PASS/FAIL/INCONCLUSIVE verdict is claimed; that awaits a run on `RESEARCH_GRADE`
  point-in-time NSE data (operator step, like the gauntlet).

**Unresolved risks / limitations (surfaced, not hidden)**
- Valuation has no PIT publication dates in the repo → fails closed to UNAVAILABLE.
- Sector membership not historically dated → `SECTOR_MEMBERSHIP_NOT_PIT`.
- Universe survivorship incomplete until `logs/universe_history.json` supplied.

**Next milestone**
- Unchanged: Phase 1 continued (C-06 fail-closed evidence writes; TrustClass stub),
  then Phase 2 service extraction. EXP-006 is run when point-in-time data is available.

## Milestone 3 — EXP-006 Historical Evidence Run · 2026-07-28 · status: runner implemented + tested; verdict INCONCLUSIVE (data unavailable here)

Evidence run of the FROZEN EXP-006 framework (commit 6e7968e). No redesign, no new
strategy, no execution wiring, no service extraction, no portfolio simulator, no UI.

**Primary verdict: INCONCLUSIVE — DATA_UNAVAILABLE.** No point-in-time NSE dataset
exists in this environment (empty bhav/index stores; no network; no universe/CA/
fundamental history). The data-quality gate failed CLOSED; the runner emitted
INCONCLUSIVE rather than fabricating a PASS/FAIL. This is honest process, NOT strategy
evidence about the hypothesis.

**Completed work (implementation, tested — distinct from evidence)**
- `research/momentum_breakout/dataset.py`: `DataProvider` abstraction, real
  `BhavDataProvider` (fails closed if the store can't be built), machine-readable
  `data_quality_report` (fails closed on non-positive prices / HLOC inconsistency /
  duplicate dates / absent data; records every limitation), reproducible
  `snapshot_manifest`.
- `research/momentum_breakout/runner.py`: `run_evidence()` — chronological candidate
  generation (one event per breakout), primary + two secondary exits, six frozen
  ablations, benchmark comparisons, regime/sector/valuation breakdowns, existing
  harness + BH-FDR multiple-testing, machine-readable artifacts, and the
  PASS/FAIL/INCONCLUSIVE verdict with a **research-grade downgrade** (a would-be PASS
  on survivorship-incomplete / CA-unadjusted data → INCONCLUSIVE; a FAIL is retained).
  Operator CLI: `python -m research.momentum_breakout.runner`.
- Bug fixes (implementation contradicted robustness; demonstrated by tests; hypothesis
  unchanged, no new experiment id): NaN/missing-bar fail-closed in `_detect_base` +
  simulator; `_detect_base` O(base_max²)→O(base_max) with identical output (audit C-16).

**Tests run**
- `tests/test_momentum_breakout_run.py` — 27 passed (deterministic, synthetic,
  network-free): data-quality gate + fail-closed corruption, snapshot stability,
  chronological generation + stable replay + unique event ids, no-same-bar entry,
  missing-session / IPO / delisting, benchmark alignment, cost application, no-fill +
  gap-through-stop, exit-variant separation, ablation isolation, multiple-testing,
  verdict mapping (PASS / research-grade downgrade / FAIL / UNDERPOWERED /
  DATA_UNAVAILABLE), artifact reproducibility, valuation-unavailable honesty, execution
  isolation.
- `tests/test_momentum_breakout.py` — 39 still pass (detector optimization is
  behaviour-preserving).
- Regression: money-paths / research / gauntlet suites green; PAPER autopilot,
  Telegram paper-only, LIVE migration lock unchanged; the runner imports nothing from
  execution/alerts/broker (enforced by test).

**Evidence generated**
- On real data: NONE (data unavailable → INCONCLUSIVE). On synthetic research-grade
  data the runner is verified to produce trades and a coherent verdict, and to REFUSE
  a PASS on a small sample (8 trades → UNDERPOWERED → INCONCLUSIVE).

**Artifacts (this run, fail-closed set):** `logs/experiments/EXP-006/<snapshot>/`
data_quality.json, snapshot_manifest.json, experiment_spec.json, config_snapshot.json,
limitations.json, verdict.json (+ the full observation/ledger/ablation/benchmark set
when real data is present).

**Limitations bounding any future real-data verdict:** survivorship incomplete; CA raw
unless ca_events.json present; sector membership not dated; no PIT fundamentals; no PIT
delivery. Under the research-grade gate, PASS needs ≥ survivorship + CA research-grade.

**Next milestone**
- Unchanged: Phase 1 continued (C-06; TrustClass), then Phase 2 service extraction.
  Run EXP-006 on research-grade point-in-time NSE data (operator step) for a verdict on
  the hypothesis itself.

## Milestone 1b — C-13 day-boundary money-safety · 2026-07-28 · status: DONE

Focused money-safety milestone. **No service extraction, no portfolio-simulator work.**

**Root cause**
- The NSE India money-path was already IST-*correct* (naive-IST storage +
  `today_ist()` day-filter), but the "naive-IST storage / IST-only bucketing" convention
  was **implicit and un-single-sourced**. A naive machine `datetime.now()` (= UTC on a
  VPS/CI box) could therefore be compared against an IST date and silently mis-bucket a
  trade across the UTC↔IST midnight — under which the **daily-loss circuit breaker could
  fail to fire**. The two flaky tests wrote machine-local timestamps and depended on the
  wall-clock instant pytest ran.

**Timestamp-storage contract (now single-sourced in `core/market_clock.py`)**
1. Persist trade/journal timestamps as **naive IST** wall-clock via `now_ist_naive()`
   (documented legacy convention; a tz-aware-UTC migration is deferred).
2. Convert to IST only for NSE trading-day boundaries / display.
3. Never compare a naive machine timestamp against an IST date.
4. Every "today" query resolves the IST trading day via `ist_day_of()` /
   `is_ist_today(ts, today)` — which accept naive-IST *or* tz-aware inputs, so the
   query layer already tolerates a future UTC-storage migration.
5. Tests pin the IST "today" (monkeypatch `_ist_today`) → independent of machine TZ and
   of the instant pytest runs.

**Completed work**
- `core/market_clock.py`: canonical `now_ist_naive()`, `ist_day_of()`, `is_ist_today()`
  + documented storage contract.
- `execution/trade_executor.py`: `placed_at` stamped via `now_ist_naive()`.
- `execution/autopilot.py`: day-P&L snapshot, EOD digest, and circuit-breaker
  `day_realized` all route through `is_ist_today(placed_at, today)`.
- `docs/architecture/EXECUTION_SAFETY.md`: `QT_LIVE_ENABLED` reframed as a **temporary
  migration interlock** (not graduation); future **deployment-manifest** gate described
  (strategy ID, promoted experiment ID, config hash, dataset snapshot, allowed mode,
  evidence status, broker-reconciliation status) — *not implemented this milestone*.
- `docs/overhaul/TRUTH_AUDIT.md`: C-13 marked RESOLVED; C-04b clarified (temporary
  interlock; Telegram paper ordering is intended); new **C-04c** (Telegram paper-only
  verification) and **C-14** (deferred non-NSE tz sites) added.

**Tests run**
- `tests/test_money_paths.py` — 259 passed (incl. the new boundary + Telegram suites).
- **Full network-free suite `python -m pytest tests/` — GREEN (exit 0), all edits.**
- New: `TestAutopilotDayBoundary` (market_clock contract; breaker counts a 00:01-IST
  loss when UTC is the prior day; breaker ignores a 23:59-IST prior-day loss;
  day-realised IST-only + no double-count; PAPER **and** LIVE both IST-filtered;
  per-day limit resets on the IST day) and
  `TestTelegramCommands::test_telegram_order_path_is_always_paper`.

**Evidence generated**
- The daily-loss circuit breaker now counts exactly the IST trading day at the
  UTC↔IST boundary; verified deterministically (machine-TZ- and wall-clock-independent).
- Telegram order path proven paper-only and guarded against regression.

**Evidence: paper autopilot + Telegram paper actions still work**
- `TestAutopilot` (43 tests) green — paper arming, gates, +3% target, sizing,
  compounding, report card, trailing, circuit breaker, P&L snapshot.
- `test_live_disabled_during_overhaul` green — LIVE still migration-locked; paper arms.
- `test_telegram_order_path_is_always_paper` green — Telegram tap places paper even
  when the app is armed LIVE.

**Data migration implication**
- **None.** Existing journals are already naive-IST; `ist_day_of()`/`is_ist_today()`
  read them unchanged. No backfill, no schema change. A future tz-aware-UTC storage
  migration is optional and already tolerated by the query layer.

**Unresolved risks**
- C-14 (deferred): US-paper / F&O-paper / Telegram-display timezone sites still
  machine-local — bounded, cannot affect the NSE circuit breaker or live-order path.

**Next milestone**
- Phase 1 continued: transactional fail-closed evidence writes (C-06); `TrustClass`
  boundary stub (C-01/E). Then Phase 2 service extraction.

## Milestone 3b — EXP-006 evidence run EXECUTED + artifacts committed · 2026-07-28 · status: DONE (verdict INCONCLUSIVE — DATA_UNAVAILABLE)

Executes the FROZEN EXP-006 runner (commit `a634be3`) and persists its auditable
artifact set into version control. No framework/hypothesis/detector/evidence/execution
change; no service extraction, portfolio simulator, UI or new strategy.

**What this adds over Milestone 3:** Milestone 3 built + tested the runner but its
artifacts were transient (`logs/` is git-ignored). This milestone RUNS it and COMMITS
the machine-readable artifact record under `docs/overhaul/exp006_run/` (force-added past
the repo's global `*.json` ignore, on purpose, so the evidence is auditable from the repo).

**Verdict: INCONCLUSIVE — DATA_UNAVAILABLE.** Data reality re-confirmed fresh: 0 bhav
files, `is_ready()` False, NSE HTTP 000; a bounded 45s `BhavDataProvider` build attempt
timed out (no network). The data-quality gate failed closed. Not strategy evidence.

**Committed artifacts** (`docs/overhaul/exp006_run/`): data_quality.json,
snapshot_manifest.json, experiment_spec.json, config_snapshot.json, limitations.json,
verdict.json, artifact_index.json, README.md (index + reproduce guide). Reproducibility
identities: snapshot_id `ad652107580ddae1`, config hash `4f638f99e13bf939` (== frozen
framework, no drift), code commit `a634be3`.

**Tests run**
- `tests/test_momentum_breakout_run.py` — 31 passed (27 runner + new `TestCommittedRunRecord`
  guarding the persisted verdict/manifest/gate against silent corruption).
- Full network-free suite green (regression). Execution isolation, Telegram paper-only,
  LIVE migration lock all re-asserted.

**Limitations bounding any future real-data verdict:** survivorship incomplete; CA raw;
sector membership not dated; no PIT fundamentals; no PIT delivery. Under the research-grade
gate a PASS needs ≥ survivorship + CA research-grade; a FAIL is attainable now.

## Milestone 4 — NSE data acquisition decision & minimum research dataset · 2026-07-28 · status: DONE (data BLOCKED; decision + contracts + infra delivered)

Data-availability/provenance milestone (not strategy results). No new strategy, no
EXP-006 redesign, no threshold change, no service extraction, no portfolio simulator,
no UI, no research→execution link.

**Corrected status of the prior run (commit `6a865c8`):** run attempt completed but
**BLOCKED before candidate generation** → INCONCLUSIVE — DATA_UNAVAILABLE; economic
hypothesis **UNEVALUATED**; **not a historical evidence verdict**.

**Existing-data discovery:** no usable NSE dataset exists anywhere reachable — inspected
`logs/bhav` (0 files), `logs/index`, `universe_history.json`/`ca_events.json` (absent),
`data/fundamentals_cache.db` (current only), `/mnt/user-data` (empty), `/opt/rclone` (no
remotes), env vars, CI artifacts, broker/cloud config, and a filesystem-wide search.
Machine-readable: `docs/overhaul/data_acquisition/discovery_report.json`.

**Source decision:** **NSE official archives** (existing `data/bhavcopy_store` +
`data/index_store`; includes delivery `DELIV_PER`) — official, reproducible, already the
canonical path (no competing ingestion). yfinance rejected for research (DISPLAY_ONLY;
EXP-005 survivorship lesson). Licensing: raw NSE files not redistributed (git-ignored);
only derived provenance committed.

**Delivered contracts/policies** (`docs/overhaul/data_acquisition/`): minimum dataset
contract, corporate-action & price-integrity policy (with limitation-direction
classification), universe-history/sector policy, storage & snapshot design, source
decision + licensing/reproducibility.

**Infra changes**
- Immutable run tree `docs/overhaul/experiments/EXP-006/runs/`; the no-data record moved
  to `0001-blocked/` (content unchanged) + `run_manifest.json` (artifact SHA-256s) +
  runs index. Nothing overwritten.
- `runner._decide`: FAIL-direction verdict gate (economic FAIL retained only under
  one-directional-favourable limitations; CA-raw → INCONCLUSIVE). No EXP-006 threshold/
  config-hash change.
- `tests/test_scan_core.py` → `tests/integration/` (marked `integration`);
  `tests/conftest.py` excludes it from the default run by classification. **Canonical
  network-free suite: `python -m pytest`** (no ad-hoc `--ignore`). Integration:
  `QT_INTEGRATION=1 python -m pytest tests/integration`. CI updated accordingly.

**Tests run**
- `tests/test_momentum_breakout_run.py` — 33 passed (incl. new FAIL-direction gate cases
  + committed-record guard at the new path).
- Canonical network-free suite `python -m pytest` — green (no ad-hoc ignore; integration
  excluded by classification). Integration (`test_scan_core`) passes separately (slow).

**Regression (re-verified):** C-13 timezone protections; PAPER autopilot; Telegram
paper-only; LIVE migration lock; research/execution isolation; no broker/GTT import from
EXP-006.

**EXP-006 readiness gate: NOT satisfied here** (no NSE network / no data). The economic
run was NOT executed (and must not be, against an empty environment). Precise blocker
recorded. When the gate passes on a data host, run the UNCHANGED frozen runner.

## Milestone 5 — Real-data materialisation attempt · 2026-07-28 · status: BLOCKED (host not data-capable)

Attempted to materialise the real NSE dataset for EXP-006 on this host. No new
infrastructure, no strategy, no EXP-006 change.

**Result: BLOCKED.** This environment is not data-capable — the outbound proxy denies
NSE archive hosts with `Tunnel connection failed: 403 Forbidden` (nseindia.com not in
the allowlist). Canonical ingestion `build_store` → 0 sessions / 0 files / `is_ready()`
False; readiness gate **RED — DATA_UNAVAILABLE**. Per the verdict restriction the frozen
runner was NOT executed, no economic PASS/FAIL issued, and no synthetic/display-only data
substituted. `runs/0001-blocked` untouched; no new run directory (no run occurred).

**Auditable record:** `docs/overhaul/experiments/EXP-006/acquisition_attempts/acq-0001.json`
(+ README). Unblock steps are documented there and in `docs/overhaul/data_acquisition/`.

**Tests:** no code changed; canonical network-free suite `python -m pytest` remains green
(545). Regression guarantees (C-13, PAPER autopilot, Telegram paper-only, LIVE lock,
EXP-006 isolation) unchanged.

## Milestone 6 — Historical Data Setup frontend · 2026-07-28 · status: DONE (data-management UI + ingestion)

Layman workflow to provide/validate real NSE data, save it into the canonical stores,
view readiness, and run the FROZEN EXP-006 test when the gate allows. No EXP-006 change,
no research→execution link, no order actions, no LIVE unlock.

**Completed work**
- `research/momentum_breakout/data_setup.py` (pure, tested): safe ZIP extraction
  (path-traversal / symlink / decompression-bomb / unsupported-file guards; validated by
  content, not name), dataset validation + coverage/quality, readiness (green/amber/red;
  red cannot be bypassed), deterministic content-addressed snapshot, overwrite-protected
  save (new/replace/cancel), and `run_exp006()` into a NEW immutable run dir (never
  overwrites a prior run; refuses a red gate).
- `data/bhavcopy_store.build_from_local()` + `data/index_store.build_from_local()` —
  local-load entry points feeding the SAME canonical stores (no parallel database).
- `ui/data_setup_page.py` — thin Streamlit page (simple language, Technical-details
  expanders, live-safety note "historical research cannot place broker orders"; NO order
  buttons); wired into app.py More Tools + a standalone dispatch branch.
- `docs/user-guide/HISTORICAL_DATA_SETUP.md` — plain-language guide.

**Tests run**
- `tests/test_data_setup.py` — 27 passed (deterministic, network-free, no wall-clock/
  Streamlit): valid/unsafe/unsupported/not-a-zip; malformed CSV/JSON, invalid OHLC,
  duplicate detection, insufficient history; readiness green/amber/red; stable + content-
  addressed snapshot; overwrite protection (new refuses / replace / cancel); materialise
  the canonical store from local files (no network); red-gate run prevention; new
  immutable run dir + prior-run preservation; EXP-006 config unchanged after a run;
  execution isolation (engine + UI page import no order path).
- Canonical network-free suite `python -m pytest` — green.

**Guarantees preserved (re-asserted):** PAPER autopilot, Telegram paper-only, LIVE
migration lock, C-13 timezone protections, research/execution isolation, no broker/GTT
import. EXP-006 thresholds/config-hash/detector/entry/stop/exits/ablations UNCHANGED.

**Known limitation:** Streamlit *rendering* is verified by source inspection + the pure-
engine tests (no live UI harness in CI). Real materialisation still needs real NSE files
(or network on a data-capable host).

## Milestone 7 — Autonomous discovery + Strategy Studio (human-in-the-loop) · 2026-07-28 · status: DONE (workflow built + tested; no market evidence without real data)

Research + user-approval workflow. No LIVE unlock, no research→execution link, no silent
paper deploy, no app redesign, no evidence-framework replacement, no synthetic evidence,
no self-rewriting strategies, no unrestricted RL, no service extraction/portfolio sim.

**Completed work** (`research/strategy_studio/`, all pure + tested)
- `spec.py` canonical versioned StrategySpec + user-only lifecycle; `grammar.py` approved
  PIT-safe building blocks; `discovery.py` constrained seeded/budgeted generation +
  append-only attempt registry + validity stages + overfitting controls (multiple-testing
  burden, untouched-test isolation, simpler-baseline, complexity) + data-readiness gate;
  `review.py` five SEPARATE confidences + Convince Me (defence + prosecution, labelled
  evidence/plausible/speculation, careful recommendations, non-cherry-picked trades) +
  comparison; `tweak.py` guided + NL→explicit-diff (no code exec) + impact preview +
  material-change versioning; `approval.py` user-only immutable PAPER-only record +
  separate paper activation (no live); `wizard.py` guided manual creation.
- `ui/strategy_studio_page.py` (thin, 5 sections; NO order actions) wired into app.py
  More Tools + dispatch. Guides: `docs/user-guide/STRATEGY_STUDIO.md`,
  `docs/overhaul/STRATEGY_STUDIO.md`.

**Tests**
- `tests/test_strategy_studio.py` — 42 passed (deterministic, network-free, wall-clock-
  independent): reproducible generation, budget, attempt logging, family diversity,
  leakage/impossible-entry/unsupported-PIT/cost/min-sample/concentration rejection,
  untouched-test isolation, multiple-testing burden, simpler-baseline, data-unavailable
  gate; material tweak → new version + hash, display change preserves identity, NL→diff,
  unsafe/ambiguous handling, evidence invalidation, approval not transferring; user-only
  approval, immutable PAPER-only record, red-gate + synthetic + non-survivor refusal,
  separate paper-activation, no-live; explainability (defence+prosecution, wins+losses,
  attempts disclosed, limitations, five confidences, synthetic labelled); comparison never
  auto-picks; wizard same standards; execution isolation (package + UI import no order
  path); PAPER autopilot / Telegram paper-only / LIVE lock unchanged.
- Canonical network-free suite `python -m pytest` — green.

**Limitation (honest):** without a research-grade dataset the Studio shows labelled
DEMONSTRATION fixtures only — never market evidence; a strategy cannot be approved on them.

---

## Milestone — Autonomous Research Brain (self-driving research loop)

**Ask (user):** "no proper automation… market ko khud hi data lekar uska chain of thoughts/
analysis karke ek data thread banakar uske hisab se trade improve karna chahiye. System ko
khud hi smart banna padega apne-aap se without any human intervention."

**Built (`research/auto_research/`, pure + tested):**
- `thread.py` — append-only `ResearchThread` (chain-of-thought log; OBSERVE/REASON/DECIDE/
  PROPOSE/CONCLUDE), deterministic content, JSONL persistence, torn-line-safe, wall-clock
  only as provenance.
- `loop.py` — `run_cycle()`: observe readiness → generate → reason (structural + evidence)
  → reject weak → shortlist → auto-advance lifecycle (SYSTEM hops) to the ONE gate
  `AWAITING_USER_APPROVAL` and STOP. Honest `Discovery unavailable …` when data isn't
  research-grade; synthetic evidence never becomes a proposal. `canonical_readiness()`
  fails closed to red. Report carries `acted_on_market=False` / `approved_anything=False`.
- `learning.py` — `LearningLedger`: per-family decay/improvement across cycles; proposes
  re-tested child versions (`bump_version` → new hash) for decayed families; advice only,
  never mutates an active strategy, ignores synthetic.
- `scheduler.py` — `AutoResearchBrain`: headless daemon running cycles on an interval,
  threading thread+ledger+registry across cycles, surviving errors; `run_once()` is
  synchronous/deterministic for tests + the UI button.
- `ui/auto_research_page.py` — read-only "watch it think" panel (thread, parked proposals,
  learning), think-once / start / pause; NO order actions, cannot approve. Wired into
  app.py More Tools + dispatch as "🧠 Research Brain".

**The one deliberate boundary (by the project's own safety directives):** the brain does the
entire research loop autonomously and parks its best ideas at `AWAITING_USER_APPROVAL`.
Approving for PAPER is a **person's** action in Strategy Studio; LIVE stays migration-locked.
Real-money autonomy is intentionally NOT granted.

**Tests:** `tests/test_auto_research.py` — 21 passed (deterministic, network-free): thread
append-only/kinds/reload/torn-line/stamp-not-identity; no-data honesty + canonical fallback;
autonomous reasoning/reject/propose + determinism + no-evaluator survival; safety boundary
(never acts/approves, synthetic never proposed, system-only gate walk, user-only step
raises); learning improvement→decay + synthetic-ignored + re-tested child never mutates
parent; scheduler accumulation + red-data + learning-note-to-thread; canonical readiness
fails closed. Canonical `python -m pytest` — **646 passed**.

---

## Milestone — Full PAPER autonomy (deploy → trade → learn → retire, hands-off)

**Ask (user):** "a paper money mei I want to give full autonomy. Blow up paper money, I
don't care but it needs to learn and become smarter by the day." — a deliberate override of
the earlier human-approval-before-paper gate, **for PAPER only**.

**Built (`research/auto_research/` + lifecycle):**
- `spec.py` — new `paper_autopilot` actor; multi-actor transitions. It may cross the PAPER
  gates (AWAITING→APPROVED→PAPER_EVALUATION, and DECAYED/RETIRED) but is **structurally
  barred** from `PAPER_EVALUATION → ELIGIBLE_FOR_LIVE_REVIEW` (user-only). LIVE boundary
  unchanged; studio `system`-can't-approve guarantee intact.
- `approval.py` — `autonomous_paper_approve()` (paper-autopilot, PAPER-only, refuses
  synthetic/red/non-survivor); `activate_paper` now also accepts the engaged paper autopilot.
- `paper_book.py` — self-contained simulated ledger; 1%/10%/5%/max-positions caps; stop-first
  marking; realized-R + equity curve + per-strategy stats. Imports no real-order path.
- `paper_autonomy.py` — `PaperAutonomyManager`: deploy survivors → trade day by day →
  autonomously retire proven losers (negative real paper expectancy over ≥20 trades) → keep
  winners. Per-strategy daily trade cap.
- `scheduler.py` — `engage_paper_autonomy()` / `disengage`; `run_once` now (when engaged)
  deploys `survivors_for_paper`, runs a paper day via injected signal/price providers, and
  reviews-and-adapts. New `BrainState` fields (paper_autonomy/deployed/retired).
- `ui/auto_research_page.py` — engage/disengage toggle + paper equity, deployed/active/retired
  metrics, per-strategy paper performance.

**Boundary kept (unchanged):** full paper autonomy trades ONLY simulated money and can NEVER
reach live — moving a strategy toward LIVE stays a person's action; LIVE migration-locked;
Telegram paper-only; no order/broker import in the package (test-guarded). Synthetic evidence
and a red data gate are still refused, so with no research-grade data the brain honestly
deploys nothing.

**Tests:** `tests/test_paper_autonomy.py` — 23 passed. Canonical `python -m pytest` — **669
passed**.

---

## Milestone — Growth engine: backtest → forward test → calibrate → remember (daily)

**Ask (user):** "take advantage of paper trading. Make it trade each day… Backtest →
forward test all automatic. How will it get smarter? Strategize, observe and improve. It is
an infant growing up each day."

**Answer + code (`research/auto_research/` + `research/strategy_studio/discovery.py`):**
- `growth.py` — `calibrate()`: compares a strategy's forward (paper) edge to its backtest
  edge → CONFIRMED / WEAKER_POSITIVE / DECAYED / OVERFIT / FORWARD_PENDING. Catching overfits
  (great backtest, dead forward) is the core "getting smarter" signal.
- `knowledge.py` — `Knowledge`: persistent per-family memory (backtest R, forward R, trust
  in [0,1]). Trust rises on forward-confirmation, falls on overfit; saved to
  `logs/auto_research/knowledge.json` so learning compounds across restarts. `search_weights()`
  floors every family (keeps exploring).
- `discovery.generate(family_weights=…)` + `_family_sequence` — adaptive search: draws the
  family visit sequence proportional to learned trust (never starving a family). The search
  itself concentrates on what forward-tests well.
- `providers.py` — production wiring (backtest_evaluator over bhavcopy history, daily_bars,
  signals_for from the audited scanner). Degrades honestly to empty/invalid with no data —
  never synthetic-as-real. `get_brain()` wires these by default.
- `scheduler.py` — `AutoResearchBrain.grow_one_day()` runs the full daily loop (biased
  discovery → backtest → deploy → forward-test paper day → calibrate → retire overfits →
  remember → daily thread report). `maybe_grow_today()` = once-per-day guard; the daemon
  grows daily when paper autonomy is engaged. Calibration is the retirement authority during
  growth (run_once(adapt=False)).
- UI: Control Room "What it has learned" (trust bars, backtest R vs forward R) + days-grown
  pill; Research Brain "🌱 Grow one day now" button.
- Doc: `docs/overhaul/HOW_IT_GETS_SMARTER.md` (plain-language answer).

**Boundary unchanged:** paper-only, live-locked (paper_autopilot barred from the live-review
transition), synthetic + red gate refused. With no real data it grows nothing, honestly.

**Tests:** `tests/test_growth.py` — 17 (calibration verdicts; trust rise/fall + persistence +
corrupt-safe; adaptive search weighting + never-starve + determinism; grow_one_day end-to-end
confirms a real edge and catches+retires an overfit; paper-only/live-locked mid-growth;
once-per-day idempotence; knowledge persists between brains). Canonical suite: **686 passed**.

---

## Milestone — Trustworthiness layer (frictions · noise-aware calibration · regime · memory)

**Ask (user):** take autonomy toward the apex of intelligence, coherence, connectivity,
productivity and actionability — smoothly and transparently.

**Built (non-breaking; realistic paths default-on only through the autonomy manager):**
- `costs.py` — India cash-equity round-trip cost estimate (STT/exchange/SEBI/stamp/GST) +
  `cost_in_R`.
- `paper_book.py` — entry/exit slippage, **gap-through-stop / gap-through-target** (4-tuple
  bars), NET-of-cost realized R; `r_stats()` (mean/stderr/lower R) for noise-aware judging;
  `snapshot()/restore()` + equity_curve in `as_dict()`. Frictionless by default (exact tests);
  the manager builds a realistic book (`slippage_bps=3`, India costs) by default.
- `growth.calibrate(forward_lower_R=…)` — OVERFIT judged on the conservative lower estimate,
  so luck can't fake an edge.
- `paper_autonomy.py` — realistic book default, append-only **decision journal** (deploy/retire),
  `save()/load()`; `review_and_adapt` reuses `retire` (journaled).
- `scheduler.py` — **regime gate** (`regime_fn` → stand down NEW deploys in RISK_OFF; existing
  managed), `run_once(allow_deploy=…)`, noise-aware calibration in `grow_one_day`, book/journal
  persistence via `paper_state_path`; `get_brain()` wires `providers.current_regime` +
  `logs/auto_research/paper_book.json`.
- `providers.current_regime()` — macro_pulse/breadth read, fails open to RISK_ON.
- UI: Control Room paper **equity-curve sparkline** + **recent autonomy actions** journal.
- Doc: `HOW_IT_GETS_SMARTER.md` "trustworthiness layer" section.

**Boundary unchanged:** paper-only, live-locked; synthetic + red gate refused; honest no-op
with no data.

**Tests:** `tests/test_autonomy_enhancements.py` — 14. Canonical suite: **700 passed**.

---

## Milestone — Two-Brain Intelligence architecture (Phase 1/2/3/4/7/8 slice)

**Ask (user, from a ChatGPT-drafted spec):** split the autonomous system into Brain 1
(Evidence) + Brain 2 (Strategy/Allocation), connected by immutable typed records in an
append-only event store; add a per-strategy rule→signal evaluator, evidence cards, uncertainty-
adjusted paper allocation, a graduation protocol, and honest UI — LIVE structurally locked.

**Built (`research/intelligence/`, reusing existing infra, tested):**
- **Phase 1** — `schemas.py` (immutable frozen records with deterministic ids + full PIT
  provenance: rules hash, snapshot id, version, schema version), `event_store.py` (append-only,
  single-writer, idempotent, deterministic reconstruction), `decoder_registry.py` +
  `decoders/` (signal/market/strategy/execution/outcome/explanation — deterministic + idempotent;
  `ResearchRationale` is STRUCTURED only, never raw chain-of-thought).
- **Phase 3** — `evidence_brain.py` (Brain 1): consumes canonical events → immutable
  `StrategyEvidenceCard` with rich states (INSUFFICIENT/PROMISING/FORWARD_PENDING/CONFIRMED/
  REGIME_DEPENDENT/WEAKER/DECAYING/OVERFIT/RETIRED); reuses `research.harness` (deflated Sharpe,
  alpha/beta) + `growth.calibrate`. Pure — never trades.
- **Phase 4** — `allocation_brain.py` (Brain 2): consumes ONLY cards → immutable
  `PaperAllocationDecision` via a transparent weighted score (anchored on the uncertainty-adjusted
  lower-bound edge, not recent returns); risk buckets (established/promising/exploratory);
  family + correlation-cluster caps. Cannot mutate cards; cannot cross the live gate.
- **Phase 7** — lifecycle extended with `PAPER_CONFIRMED → ELIGIBLE_FOR_HUMAN_LIVE_REVIEW →
  USER_APPROVED`; `graduation.py` nominates on explicit criteria but **only a user** can perform
  USER_APPROVED (enforced by `spec._TRANSITIONS`; no brain/autopilot can).
- **Phase 2 (minimal)** — `strategy_runtime.py`: per-strategy, bar-by-bar, point-in-time
  entries (breakout/vol-contraction adapters); unsupported families FAIL LOUD (no scanner
  fallback); fills reuse the realistic PaperBook.
- **Phase 8** — `ui/brain_observatory.py`: Automatic Strategies · Two Brains · Live Review
  Candidates (human-only queue) with honest empty states; wired into app nav.

**Conflicts flagged (see TWO_BRAIN_ARCHITECTURE.md):** spec path drift (modules live in
`strategy_studio/`, reused there); kept the single user-owned live gate rather than a parallel one.

**Not yet complete (honest):** Phases 5 & 6 depth (experiment-lineage store, expiry, full
portfolio wiring into the live loop) — partially covered by reusing discovery's multiple-testing/
untouched-test isolation + `risk/correlation`; the two brains are not yet driven by the live
daily scheduler (they're exercised by fixtures + the event store). With no NSE data, decoders emit
nothing, Brain 1 says INSUFFICIENT_EVIDENCE, Brain 2 deploys nothing — by design.

**Tests:** `tests/test_intelligence.py` — 27 (decoder determinism/idempotence; event-store
reconstruction; brain separation; card immutability; deployment gated on evidence; tiny-sample &
negative-lower-bound penalised; forward deterioration; correlation-cluster cap; regime & no-data
no-op; PIT per-strategy eval + gap-through-stop; holdout single-use; user-only live gate; no
order imports). Canonical suite: **727 passed**.

---

## Milestone — Autonomous Intelligence Runtime (end-to-end paper loop)

**Ask (user):** activate the two-brain foundation into a continuously running paper loop —
data → frozen runtime signals → decoders → event store → Brain 1 card → Brain 2 allocation →
portfolio/risk gate → paper execution → position management → exits → outcome decode → evidence
& allocation update → repeat. Headless, restartable, honest with no data, live still locked.

**Built (`research/intelligence/runtime/`, integration map in INTELLIGENCE_RUNTIME_INTEGRATION.md):**
- `autonomous_loop.run_intelligence_cycle(ctx, store, book, runtime_state)` — the ONE
  orchestrator (Streamlit-free, broker-free): data gate → per-strategy PIT runtime signals →
  canonical decode → manage/exit open positions → outcome decode → Brain 1 cards → Brain 2
  decisions → TradeIntents → portfolio/risk gate → paper open → persist. Deterministic,
  idempotent per `cycle_id`, fail-safe.
- `cycle_context.py` (deterministic cycle id over date/type/phase/snapshot/registry/config),
  `cycle_result.py` (full typed result), `events.py` (24 canonical event types + `emit`),
  `runtime_state.py` (persistent per-strategy state + completed-cycle idempotency + reconcile),
  `portfolio_gate.py` (family/cluster/duplicate/regime/position caps → block events with reason
  codes), `preflight.py` (Phase Q operational-safety checks), `controls.py` (Phase P owner
  actions — pause/resume/retire/block/close-all — each an audited canonical event), `modes.py`
  (Phase P mode ladder; PAPER_AUTO end-to-end, all live modes hard-disabled).
- Allocation bootstrap: a PROMISING card (strong backtest, 0 forward) earns a small EXPLORATORY
  allocation to start forward evidence; FORWARD_PENDING is held (thin samples never scaled).
- Schema additions: `CanonicalEvent`, `TradeIntent` (broker-independent).
- Scheduler (Phase O): `AutoResearchBrain.run_intelligence_cycle_day()` — one authoritative job,
  one lock (no overlapping mutation), its OWN intel book + event store (no conflict with the
  legacy growth path); the daemon runs it each tick in PAPER_AUTO. `get_brain()` persists to
  `logs/intelligence/{events.jsonl,runtime_state.json}` (what Brain Observatory reads).
- UI: Brain Observatory now shows the REAL loop — operating mode, last cycle summary, and recent
  canonical events — not fixtures.

**Honest scope / not done:** Phase E extra strategy adapters (only breakout + volatility-
contraction evaluate bar-by-bar; others fail loud) and a fully-populated production strategy
registry + bar provider are the next increment — they need a validated NSE dataset, which isn't
loaded, so the production loop is a safe no-op today. Multi-job scheduler (research/growth/
session/recovery as distinct locked jobs) is represented by one job + cycle types; splitting is
follow-up. No live path exists; USER_APPROVED remains user-only.

**Tests:** `tests/test_intelligence_runtime.py` — 15 (end-to-end signal→position; exit→outcome
feedback; idempotent re-run; deterministic cycle id; no-data no-op; live-mode refused; paper-
paused opens nothing; unsupported-family event; duplicate-symbol block; restart persistence +
completed-cycle no-op; unreconciled refuses new risk; owner close-all audit event; set-mode
rejects live; scheduler no-op without registry; intel book separate from legacy). Canonical
suite: **742 passed**.

---

## Milestone — Production Data Activation & Strategy Runtime Expansion (partial, honest)

**Ask (user):** unblock the loop with production NSE data ingestion, a populated strategy
registry, more genuine bar-by-bar strategy families, evidence tiers/data states, and real
forward evidence. Audit first (PRODUCTION_DATA_ACTIVATION.md), reuse existing infra.

**Delivered (tested):**
- **Phase 0** audit: `docs/overhaul/PRODUCTION_DATA_ACTIVATION.md` (reuse map + exact NSE-package
  failure reasons + implementation order).
- **Phase 12** four new genuine adapters in `research/intelligence/strategy_runtime.py`:
  `trend_following`, `pullback` (single-symbol) + `cross_sectional_momentum`, `relative_strength`/
  `sector_rotation` (cross-sectional ranking), plus PIT feature helpers (SMA/EMA/ATR/return) and a
  unified `signals(spec, as_of, universe, benchmark)` dispatch. Unsupported families still fail loud;
  no scanner fallback. Runtime supports 7 families now (was 2).
- **Phase 11** `registry.py` — production strategy registry with startup validation (unknown family,
  missing adapter, duplicate id/hash, bad params) → disable-with-reason, never crash; `deployable_specs()`.
- **Phases 16/19** `data_state.py` — data operating states (NO_DATA…FAILED, partial-degradation
  isolation) + evidence tiers (OPERATIONAL_ONLY…FORWARD_ELIGIBLE); tier stamped on Evidence Cards and
  flagged as a limitation when not forward-eligible.
- **Phase 21 (nested archives)** `data_setup.safe_extract_zip` now RECURSES into `.zip`/`.csv.zip`/
  `.gz`/`.csv.gz` members up to a depth limit, with password-protected rejection and corrupt-member
  quarantine — fixes the real `BhavCopy_*_F_0000.csv.zip` failure.
- **Loop wiring** — the cycle now calls `RT.signals` (cross-sectional-capable) and stamps the dataset
  tier on cards; a fixture end-to-end (registry → momentum signal → Brain 1 → Brain 2 → intent → gate →
  paper position) is proven.
- **Phase 18 UI** — Brain Observatory "Strategy Coverage" tab (runtime-supported vs why-unsupported).

**Not done — honest, and safe (production stays no-op until these land, by design):** full
ingestion breadth + per-file classification report + row-level quarantine severity engine (Phases 2–4);
immutable snapshot store / incremental successors / snapshot pinning (Phases 8–9) — the existing
`snapshot_manifest` is reused but the immutable versioned store is next; deeper CA-leakage/survivorship
protections beyond existing `corporate_actions` + `point_in_time_universe`; production bar provider
reading a validated snapshot + `intel_registry_fn` wiring (Phase 10 — the loop runs on a fixture ctx
today, not a live snapshot); scheduler job split (Phase 15); performance/scale (Phase 22); the full
Historical Data Setup UI overhaul (Phase 17); and the remaining Phase-21 weakness matrix. These are the
next increment and are documented so nothing over-claims.

**Tests:** `tests/test_production_data.py` — 17 (new adapters PIT/deterministic/loud-unsupported;
registry validation; data-state/tier classification + card tier flag; nested `.csv.zip`/`.gz`/
zip-of-zips ingestion; fixture end-to-end momentum→paper position). Canonical suite: **759 passed**.

---

## Milestone — Real-Data Runtime Activation (immutable snapshot → provider → pinned cycle)

**Ask (user):** complete the blocked chain — imported NSE files → validated immutable snapshot →
active pointer → snapshot-reading production provider → pinned cycle context → run_intelligence_cycle
— so automation flips from fixture-driven to real-data-driven. No broker work.

**Delivered (tested):** implementation note `docs/overhaul/REAL_DATA_RUNTIME_ACTIVATION.md`, plus
`research/intelligence/data/`:
- `snapshot_store.py` — `SnapshotStore`: **immutable, content-addressed** commit (deterministic id
  from normalized content + schema/parser versions; idempotent; successor on change), manifest +
  checksum, `verify_snapshot` (checksum + data-file hash + schema), **atomic activation**
  (`os.replace` pointer swap + audit line; crash → old OR new, never partial), `get_active`/`open`/
  `list`. A pointer to a missing snapshot resolves to None (fail-safe).
- `snapshot.py` — `Snapshot`: read-only **point-in-time** accessor (`bars(through)` never returns a
  later bar; `universe(on_date)` is contemporaneous; `benchmark`, `health`, `coverage_for`).
- `provider.py` — `SnapshotBarProvider`: pinned-snapshot-only; no internet/synthetic fallback.
- `runtime/context_builder.py` — `build_context_from_snapshot`: assembles a real `CycleContext`
  pinned to ONE snapshot id, with **data-aware per-strategy readiness** (READY / UNSUPPORTED_RUNTIME
  / MISSING_BENCHMARK / INSUFFICIENT_HISTORY) and a **forward-eligibility gate** (Part 14): a
  research-eligible-but-not-forward snapshot still runs the cycle + updates evidence but opens NO new
  entries.
- Loop: `forward_eligible` added to `CycleContext`; the loop gates new entries on it. Scheduler
  `_build_intel_ctx` now prefers an **active verified snapshot** (real data) → production context,
  else injected fn, else honest no-op. `get_brain()` wires a `SnapshotStore` + a validated registry
  so the loop flips to real-data the moment a snapshot is activated. UI shows the active snapshot +
  "no snapshot → no action" reason.

**The flip (proven):** `tests/test_snapshot_runtime.py::TestSnapshotDrivenCycle` builds a snapshot,
activates it, and runs the cycle with **no fixture ctx** → cross-sectional momentum ranks the real
snapshot universe → opens a paper position on the top name; every canonical event carries the pinned
snapshot id.

**Honest deferrals (documented):** full validation/quarantine severity engine + import
classification report persistence; incremental successor reconcile depth (Parts 5–7); the full
recovery matrix (Part 20) beyond verify/fail-safe; scalability benchmarks + columnar storage
(Parts 21–22); the full Historical Data Setup UI overhaul (Part 18); trading-calendar-accurate
freshness (currently latest≥as_of). No NSE dataset exists in this environment, so production stays a
safe no-op until a user imports + activates a real snapshot — the machinery is ready and tested.

**Tests:** `tests/test_snapshot_runtime.py` — 15 (immutable/idempotent commit + successor + tamper/
missing-file verify; atomic activate + audit + invalid-can't-activate + missing-pointer fail-safe;
PIT provider + refuses-invalid + contemporaneous universe; snapshot-driven cycle → paper position;
not-forward-eligible runs research but blocks entries; missing-benchmark blocks sector rotation;
scheduler no-op without snapshot + drives cycle with active snapshot). Canonical suite: **774 passed**.

---

## Milestone — Production Execution System + LIMITED_LIVE (simulator-certified)

**Ask (user):** build the broker-neutral Execution Management System, independent Risk Governor,
reconciliation and recovery required for genuinely automatic trading; make QuantTerm
LIMITED_LIVE-capable — simulator-certified, no real broker, live impossible without explicit
user approval + full preflight.

**Delivered (tested, `ems/`):** audit `docs/overhaul/PRODUCTION_EXECUTION_ARCHITECTURE.md` +
- `schemas.py` — broker-neutral immutable records with full provenance (idempotency key + whole
  decision chain): OperatingEnvelope (user-owned, checksummed, `approve_envelope` USER-only),
  RiskDecision, ExecutionPlan, OrderStateRecord, FillRecord, PositionRecord, ProtectionPlan/Status,
  ReconciliationReport, ExecutionIncident; operating modes + readiness states.
- `state_machine.py` — explicit order lifecycle (24 states) + legal transitions; illegal → raise.
- `broker.py` (BrokerAdapter contract) + `simulator.py` (deterministic SimBroker: ack/partial/
  reject/timeout/idempotent-by-key/protection/manual positions).
- `risk_governor.py` — INDEPENDENT governor (separate from Brain 2 + EMS): full risk hierarchy
  (per-trade/symbol/strategy/family/sector/cluster/total/positions) + capital-protection state
  machine (daily-loss/drawdown → NORMAL…HALTED). Reduces or denies; nobody overrides it.
- `ledger.py` — persistent journaled execution ledger (atomic write; survives restart).
- `ems.py` — the ONE order-lifecycle owner: intent → envelope check → Risk Governor → frozen
  plan → JOURNALED submit (persist-before-broker) → idempotent broker call (timeout ⇒ reconcile,
  never blind resubmit) → partial-fill on ACTUAL qty → BROKER-VERIFIED protection (local flag is
  not proof; failure → exit + critical incident + block) → position. `reconcile()` (broker is
  authority; manual positions never assigned to a strategy; critical mismatch blocks new risk).
  `recover()` (restart: adopt broker state, no duplicate submission, block entries until clean).
- `preflight.py` — live preflight (all hard gates) + explicit readiness states.

**Boundaries proven:** strategies/brains cannot submit (only the EMS, only in a live mode with a
user-approved envelope); PAPER_AUTO/SHADOW/HALTED cannot submit live; Brain 2 cannot override the
Risk Governor; broker adapter cannot increase risk; duplicate submission prevented; ambiguous
submission reconciled; protection broker-verified; restart cannot duplicate; daily-loss/drawdown
act automatically; the owner ceiling cannot be raised automatically; live cannot activate via UI/
env alone (only `approve_envelope(actor="user")`); the intelligence package still imports no
broker code (test-scanned). `USER_APPROVED` unchanged.

**Readiness state: `LIMITED_LIVE_SIMULATOR_CERTIFIED`** — NOT broker-connected, NOT user-activated.
No real broker adapter, credentials, or order path exists. Deferred (documented): real Kite
adapter; full health/alerts/UI; shadow certification; scaling ladder; secrets vault; calendar-
accurate freshness; and the remainder of the 75/35 test matrices (a strong subset is delivered).

**Tests:** `tests/test_ems.py` — 35 (boundaries; lifecycle + idempotency + timeout-reconcile +
partial fill; broker-verified protection; independent risk governor caps + daily-loss + drawdown +
governor-failure + brain-can't-override; envelope bounds; reconciliation incl. manual positions;
restart recovery no-duplicate; preflight + readiness; end-to-end certification with full provenance
+ governor independent stop + idempotent re-run). Canonical suite: **809 passed**.

---

## Milestone — Autonomous PAPER_AUTO operational activation & certification

**Ask (user):** certify fully autonomous PAPER_AUTO — QuantTerm independently takes, manages,
closes and learns from paper trades on snapshot-backed data; the user is an optional supervisor,
not a mandatory participant. No new architecture/strategies/dashboards.

**Verified existing chain (traced to code):** market data → `SnapshotStore` (immutable active
pointer) → `build_context_from_snapshot` (forward-eligibility + PIT) → `strategy_runtime.signals`
→ `evidence_brain.build_card` (Brain 1) → `allocation_brain.decide` (Brain 2) → `TradeIntent` →
`portfolio_gate` → `paper_book.open_position` → `_manage_positions` (stop/target/gap exits) →
`OutcomeObservation` decode → Brain 1 card update → Brain 2 allocation. Orchestrated by
`AutoResearchBrain.run_intelligence_cycle_day` (one lock, cycle-id idempotent), driven headless by
`_worker`. No Streamlit/Telegram/broker/EMS import in the loop (test-scanned).

**Operational gaps closed (no new architecture):**
- **In-sample bootstrap evidence** — `AutoResearchBrain._insample_evidence` runs a REAL in-sample
  backtest of each strategy's OWN frozen rules over the snapshot history (reusing the runtime +
  paper book), so Brain 1 can promote a fresh strategy to PROMISING for exploratory paper — never
  fabricated. This is what makes the scheduler actually trade.
- **Paper-book persistence** — `intel_book` snapshot/restore to `logs/intelligence/intel_book.json`
  (atomic); open positions + stops/targets survive restart; recovery-first (restore → reconcile
  before new entries).
- **Persistent PAPER_AUTO enable flag** — `enable_paper_auto`/`disable_paper_auto` persisted to
  `logs/intelligence/paper_config.json`; ordinary restart never reverts or re-prompts; explicit
  user disable is honoured. Paper config is NOT real-money authorization.
- Worker gates on `is_paper_auto_enabled()`; `run_intelligence_cycle_day` saves the book each cycle.

**Human intervention:** none required for routine paper trading — no per-trade entry/exit approval,
no envelope, no broker, no credentials, no Telegram ack, Streamlit need not be open, restart needs
no re-click. Optional overrides remain: `disable_paper_auto`, regime stand-down, snapshot
deactivation, manual book close.

**Tests:** `tests/test_paper_auto.py` — 14 (opens without a click; no envelope/broker/creds; no
ems/broker/streamlit import; auto risk-rejection under RISK_OFF; scheduler lock; duplicate cycle no
dup; missing snapshot blocks safely; enable flag survives restart; open positions + stops survive
restart; management resumes + auto-exit after restart; outcome→Brain 1, paper-labelled `forward`
never `live`; manual disable override; full end-to-end certification with restart, no duplication).
Canonical suite: **823 passed**.

---

## Milestone — PAPER_AUTO activation closeout (freshness · PIT · no-trade · ingestion · runbook)

**Ask (user):** final activation closeout only — real NSE session freshness, prove in-sample PIT
separation, preserve no-trade as valid, complete the genuine-data ingestion path, a headless
smoke test, and an operator runbook. No new architecture.

**Delivered (tested):**
- **Real freshness** — `research/intelligence/data/nse_calendar.py` (IST-aware NSE session
  freshness: weekends, holidays from an optional file, pre-close/post-cutoff publication window,
  publication allowance, missing/future/duplicate sessions). Wired into `context_builder`
  freshness gate for real ISO dates (synthetic fixture dates keep the simple fallback).
- **In-sample PIT proof** — audited `_insample_evidence`: evidence uses only sessions strictly
  before the decision date; the signal bar and later bars are excluded; leakage test (appending
  future profitable bars leaves the earlier card byte-for-byte identical) and same-session test
  (spiking the decision bar doesn't change its own evidence). In-sample is never FORWARD/CONFIRMED.
- **No-trade valid** — scheduler reports `eligibility: NO_ELIGIBLE_TRADE`; insufficient/negative
  evidence opens nothing; a strategy can stay inactive across cycles with no error; no
  forced/random trade.
- **Ingestion bridge** — `research/intelligence/data/from_bhav.py`: canonical bhav files (from the
  existing Historical Data Setup, incl. nested `.csv.zip`) → validated (OHLC/positive/dup/future)
  → immutable snapshot commit → verify → active pointer. No new ingestion system.
- **Headless smoke** — bounded test starts the background worker, loads the persisted flag, opens
  the active snapshot, runs a scheduled cycle unattended, survives a worker exception, and a
  restart restores state without re-approval.
- **Runbook** — `docs/overhaul/PAPER_AUTO_OPERATIONS.md` (required files, import/activate,
  enable/verify/inspect/stop/disable, restart, stale-data + corrupt-persistence responses, and the
  exact paper vs simulator vs in-sample vs live evidence distinction).

**Tests:** `tests/test_paper_auto_closeout.py` — 18 (freshness: Fri-weekend/pre-close/post-close/
allowance/delayed/holiday/missing/future/duplicate; PIT: no-future-leak/same-session/in-sample-
not-forward; no-trade: insufficient → NO_ELIGIBLE_TRADE + inactive-no-error; ingestion: nested
bhav-zip → active snapshot + quarantine/dedup; headless: worker-cycle-survives-errors + restart-
no-reapproval). Canonical suite: **841 passed**. Genuine NSE data still absent ⇒ status
`OPERATIONAL_DATA_REQUIRED` (machinery complete; awaits a user-supplied snapshot).

---

## Milestone — Zerodha Kite as the primary automatic DATA provider (PAPER_AUTO)

**Ask (user):** make Kite the automatic data source so routine PAPER_AUTO needs no manual bhavcopy
uploads (file-import kept only as offline fallback + independent verification). Kite is **DATA
ONLY** — no order/GTT/modify/cancel from the PAPER_AUTO path. Feed the EXISTING canonical provider
+ `SnapshotStore`; no new architecture. Report 5 separate states; do **not** claim
PAPER_AUTO_OPERATIONAL on mocks; do **not** build real-money execution.

**Delivered (tested):**
- **`research/intelligence/data/kite_source.py`** — `KiteDataSource` over an INJECTED, duck-typed
  data client (`profile` / `instruments("NSE")` / `historical(token, frm, to, "day")` only).
  - **Session** — `session_valid()` / `require_session()` (`KiteSessionInvalid`); an invalid/expired
    session BLOCKS refresh and leaves the last active snapshot untouched. Daily Zerodha login is the
    only human step; no credentials stored, no auth bypassed.
  - **Instrument master** — `refresh_instruments()` filters EQ/INDEX, reconciles vs the prior master
    by **canonical identity** (`canonical_id`: ISIN → `exchange:tradingsymbol`, **never the token**),
    and records added / removed / symbol_changed / token_changed. Corporate identity survives symbol
    & token rotation.
  - **Historical bootstrap** — `bootstrap_symbol()` fetches ONLY the missing date range per security,
    dedups, quarantines malformed candles, and is **resumable + idempotent** via persisted per-symbol
    history + progress (`progress_path`). Fetch is rate-limited (`RateLimiter`) with retry +
    exponential backoff + jitter (`_fetch`, injectable sleep/clock/rng ⇒ deterministic offline).
  - **Daily refresh** — `daily_refresh()` computes the required NSE session from the trading calendar,
    fetches incrementally, validates, commits an **immutable snapshot**, verifies it, and activates
    **only when FORWARD_ELIGIBLE** (`data_state.classify_tier` + freshness). Not fresh / missing
    benchmark / verify-fail ⇒ `COMMITTED_NOT_ACTIVATED`, previous active snapshot preserved.
- **`research/intelligence/data/kite_live.py`** — `KiteLiveOverlay` (injected feed): bounded-backoff
  reconnect that RESTORES subscriptions, rejects out-of-order and future-dated ticks, tracks
  per-symbol staleness (stale ⇒ block new entries, still manage exits), and **never** finalizes an
  intraday bar as daily evidence. No order methods, no snapshot commit.
- **Data-only boundary** — no `kite_client` / `KiteConnect` / order symbols in the intelligence
  package code (the `TestNoOrderImports` guard now scans code only, ignoring docstrings that merely
  name the boundary). A dedicated test asserts an order-capable client's order methods are never
  touched on the PAPER_AUTO path.
- **Fallback preserved** — the offline bhav file-import bridge (`from_bhav`) still works as an
  independent verification source; Kite refresh needs no manual upload.

**Tests:** `tests/test_kite_data.py` — 22 deterministic, network-free (FakeKite / FakeFeed):
session valid/invalid; instrument reconciliation; canonical-id≠token; incremental-only download;
rate-limiter used; retry-backs-off-then-succeeds; restart-resumes-from-progress; invalid-OHLC
quarantined; valid-refresh-activates; missing-benchmark-blocks-eligibility; failed-refresh-preserves-
active; reconnect-restores-subs; stale-ticks-block-entries; out-of-order/future-ticks-rejected;
overlay-never-finalizes-daily; PAPER_AUTO-trades-from-Kite-data; no-order-API-on-path;
data-source-uses-only-data-APIs; restart-restores-downloader-state; no-manual-upload-needed;
manual-import-still-available. Canonical suite: **863 passed**.

**Final states (honest):**
- `KITE_DATA_CONNECTED` — **NOT connected in this environment** (no live Zerodha session / outbound
  market network in the sandbox). Adapter + injected-client contract complete and proven offline.
- `HISTORICAL_BOOTSTRAP_COMPLETE` — machinery complete & tested (resumable, deduped, rate-limited);
  awaits a live session to fetch real candles.
- `DAILY_REFRESH_OPERATIONAL` — machinery complete & tested (calendar-driven incremental → verified
  immutable snapshot → tier-gated activation); awaits a live session.
- `LIVE_FEED_OPERATIONAL` — overlay logic complete & tested; awaits a live KiteTicker feed.
- `PAPER_AUTO_OPERATIONAL` — **NOT claimed on mocks.** The loop trades automatically from valid
  Kite-fed data in tests; genuine PAPER_AUTO requires a real activated Kite snapshot.
