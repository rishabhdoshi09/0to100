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
