# QuantTerm Next Research Cycle — Final Report

> End-to-end cycle for EXP-NEXT-01/02/03. Snapshot `a7a9828ec37e09e4`. Global trust `OPERATIONAL_ONLY`. Production unchanged. Phase B not started.

## 1. Executive summary

We finished the three approved next tests on verified panel history. EXP-NEXT-01→FAIL; EXP-NEXT-02→INCONCLUSIVE; EXP-NEXT-03→FAIL. Overall recommendation: STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES. Nothing goes live.

## 2. Data / snapshot certification used

- Snapshot: `a7a9828ec37e09e4` (scoped READY_FOR_SCIENTIFIC_RERUN)
- Global trust: `OPERATIONAL_ONLY` (unchanged)
- Readiness doc: `NEXT_RESEARCH_EXECUTION_READINESS.md` — all three READY
- Partitions: discovery `2024-08-01→2025-07-31`; confirm `2025-08-01→end`

## 3. Exact experiment IDs

- `EXP-NEXT-01` / `be12db7a0d764c98` / final=`FAIL`
- `EXP-NEXT-02` / `5eb01b27fc75b885` / final=`INCONCLUSIVE`
- `EXP-NEXT-03` / `ada9d05390b78be1` / final=`FAIL`

## 4–9. Per-experiment outcomes

### EXP-NEXT-01

**What we tested:** Whether stocks that fall sharply over a few days tend to bounce back after trading costs.

**What happened:** After trading costs and the frozen checks, the effect was not reliable (or did not survive independent confirmation).

**What it means:** The idea does not currently show a proven advantage QuantTerm should use.

**What QuantTerm will do:** Nothing. The strategy / risk rule will not be used. Branch closed.


- Discovery: `FAIL` · Confirmation: `None` · Final: **FAIL**

### EXP-NEXT-02

**What we tested:** Whether quieter (lower-volatility) stocks produce better risk-adjusted results than high-volatility stocks after costs.

**What happened:** The evidence was mixed or underpowered under the frozen criteria.

**What it means:** We cannot claim the idea works or fails cleanly yet.

**What QuantTerm will do:** No tuning. No live use. Hold.


- Discovery: `INCONCLUSIVE` · Confirmation: `None` · Final: **INCONCLUSIVE**

### EXP-NEXT-03

**What we tested:** Whether unusually calm (compressed) price movement changes future downside risk — as a warning, not a buy tip.

**What happened:** After trading costs and the frozen checks, the effect was not reliable (or did not survive independent confirmation).

**What it means:** The idea does not currently show a proven advantage QuantTerm should use.

**What QuantTerm will do:** Nothing. The strategy / risk rule will not be used. Branch closed.


- Discovery: `FAIL` · Confirmation: `None` · Final: **FAIL**

## 10. Statistical significance

See per-experiment JSON in individual reports (harness DSR/PSR, FDR for EXP-NEXT-01 cell family, materiality gaps for EXP-NEXT-03).

## 11. Economic significance

- `EXP-NEXT-01`: `NET_NON_POSITIVE`
- `EXP-NEXT-02`: `NET_POSITIVE_UNCONFIRMED`
- `EXP-NEXT-03`: `NO_MATERIAL_RISK_VALUE`

## 12. Cost impact

- ALPHA tests use CNC round-trip (`research.phase_next.eval_utils.cost_pct`).
- Gross vs net reported in EXP-NEXT-01/02 packs; net≤0 cannot PASS.
- EXP-NEXT-03 is RISK diagnostic (cost reporting secondary).

## 13. Multiple-testing treatment

- EXP-NEXT-01: BH-FDR across 6 formation×hold cells; DSR n_trials=6
- EXP-NEXT-02: single primary spec; DSR n_trials=1
- EXP-NEXT-03: single frozen τ from warmup; no formula mining
- Across the three families: no post-hoc expansion; family results reported separately

## 14. Closed hypotheses (this cycle + prior)

- Prior closed: structure, network alpha, momentum, logistic, network interaction
- This cycle closures: see final verdicts FAIL / FAILED_CONFIRMATION below

## 15. Surviving hypotheses

- None confirmed for follow-up implementation.

## 16. Scientific-memory updates

- Outcomes recorded via phase_a5 preregistry + negative/watch beliefs.
- Prior A.6 interaction failure lesson retained.

## 17. Production behaviour confirmation

- production_behaviour_changed: **False**
- Brain / ranking / risk / execution / broker: **unchanged**
- phase_b_started: **False**

## 18. What NOT to build next

- Do not rescue failed reversal/low-vol/compression with ML or parameter sweeps
- Do not reopen momentum/network/structure branches
- Do not invent live risk blocks from unconfirmed RISK diagnostics

## 19. What QuantTerm should do next

**OVERALL NEXT ACTION: `STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES`**

Current certified evidence does not support these tested hypotheses. Stop model expansion. Reassess data breadth (PIT fundamentals/events) and/or wait for more independent history before new economic families. Do not reopen closed branches or add ML to rescue failures.

## 20. Plain-English summary

We finished the three approved next tests on verified panel history. EXP-NEXT-01→FAIL; EXP-NEXT-02→INCONCLUSIVE; EXP-NEXT-03→FAIL. Overall recommendation: STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES. Nothing goes live.

---

## Final decision table

| EXPERIMENT | TYPE | DISCOVERY | CONFIRMATION | ECONOMIC VALUE | FINAL VERDICT | NEXT ACTION |
|---|---|---|---|---|---|---|
| EXP-NEXT-01 | ALPHA | FAIL | None | NET_NON_POSITIVE | **FAIL** | `REJECT_CLOSE_BRANCH` |
| EXP-NEXT-02 | ALPHA | INCONCLUSIVE | None | NET_POSITIVE_UNCONFIRMED | **INCONCLUSIVE** | `HOLD_NO_TUNING` |
| EXP-NEXT-03 | RISK | FAIL | None | NO_MATERIAL_RISK_VALUE | **FAIL** | `REJECT_CLOSE_BRANCH` |

| OVERALL | — | — | — | — | — | `STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES` |

_git_sha: `654a883073b51d40773f271fe805f89b4d15c83a` · evaluated 2026-08-11T17:14:50.647763+00:00_
