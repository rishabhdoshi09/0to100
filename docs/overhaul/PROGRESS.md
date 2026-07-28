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
