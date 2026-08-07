# ADR-001 — QuantTerm becomes an Evidence Lab; trading is a downstream client

- **Status:** Accepted (Phase 0)
- **Date:** 2026-07-27
- **Branch:** `overhaul/evidence-lab` (the prior `claude/deepseek-multi-agent-system-nrO7n`
  is preserved as the historical terminal prototype)
- **Supersedes:** the implicit "trading terminal with a research bolt-on" architecture

## Context

QuantTerm grew as a feature-rich trading terminal (scanners, agents, Telegram,
autopilot) with research capability added later. Five pre-registered experiments
(`RESEARCH_LOG.md`, EXP-002…005) and the Phase-0 `TRUTH_AUDIT.md` show the research
layer cannot currently guarantee its own inputs or its own metrics: research paths can
reach fallback data (C-01), portfolio metrics are synthesised from independent
per-trade R (C-02), the universe is survivorship-biased (C-03), corporate actions are
un-applied (C-04), background services are owned by Streamlit (C-05), and evidence
writes fail open (C-06).

The scientific process — not any single strategy — is the asset worth protecting.

## Decision

1. **Research platform first.** The core system is a *point-in-time, reproducible
   evidence platform*. The scanner, trading UI, Telegram, JARVIS and execution engine
   are *applications* on top of it. They may read research outputs and request work;
   they must not define or recompute research truth. (Decision A.)
2. **No new signal features during the overhaul.** No indicators, patterns, agents,
   LLM verdicts, sweeps, or auto-trading enter until the trusted research pipeline is
   operational. (Decision B.)
3. **Portfolio returns are the primary unit of evidence.** A daily NAV ledger from a
   chronological simulator is the *only* source of CAGR/Sharpe/alpha/drawdown.
   Per-trade R survives for attribution/diagnostics only. (Decision C.)
4. **Trusted paths fail closed; optional paths may fail open.** Evidence capture,
   dataset generation, point-in-time construction, CA adjustment, research simulation,
   order placement, reconciliation and live risk checks must fail closed. News, LLM
   explanations and UI decoration may fail open. (Decision D.)
5. **No implicit research-data fallback.** Every dataset declares a trust class
   (`RESEARCH_GRADE` / `OPERATIONAL_ONLY` / `DISPLAY_ONLY` / `PROHIBITED`); research
   grade refuses anything lower. (Decision E, see `DATA_CLASSIFICATION.md`.)

## Consequences

- **Positive:** results become reproducible from a snapshot ID + commit; portfolio
  claims become defensible; the system can reject attractive-but-invalid results
  (as it already did for EXP-005) *by construction* rather than by vigilance; the
  codebase becomes career-grade and auditable.
- **Cost:** a large, staged refactor (see `IMPLEMENTATION_PLAN.md`). Some current
  numbers (per-trade "modelled CAGR") are retired and must be recomputed on the NAV
  ledger. Live autopilot is disabled until an explicit graduation gate.
- **Non-goals:** this ADR does not add strategies, chase alpha, or promise returns. It
  makes it *hard for QuantTerm — or its author — to lie to itself*.

## Guardrails carried from the existing system (kept, not rebuilt)

The anti-overfitting harness (DSR, PSR, White's Reality Check, BH-FDR, block bootstrap,
purged/embargoed CV, effective-N, power, alpha/beta) and the append-only Research Log
are sound and are **preserved**. The overhaul upgrades them to operate on daily
portfolio-return streams and to persist multiple-testing accounting across experiments.

## Migration boundary

Logical boundaries and interfaces come first; physical folder migration is incremental
(directive §6). No "big bang" move. Each phase lands behind tests and leaves the system
runnable.
