# Phase A.5 — Evidence Activation Report

**Status:** RESEARCH ONLY  
**Production behaviour changed:** **NO**  
**Date:** 2026-08-11  
**Branch:** `cursor/institutional-ai-audit-80a2`  

---

## 1. Executive verdict

Phase A infrastructure works end-to-end (PitContract → horizons → challenger lab →
structure/network → registry/harness/scientific memory).

**No capability earned promotion.**

All five preregistered experiments returned formal verdict **INCONCLUSIVE** for
promotion because the only available panel is `DISPLAY_ONLY` / `LIMITED_RESEARCH`
(yfinance; no CA ledger; no survivorship-complete universe history). Per
`docs/overhaul/DATA_CLASSIFICATION.md`, this trust class **cannot** support
`PASS_ALPHA` / `PASS_RISK`.

On the exploratory panel, point estimates were mostly **negative** for
incrementality:

| Experiment | Exploratory economic/risk reading | Formal promotion verdict |
|------------|-----------------------------------|---------------------------|
| EXP-A5-01 Market structure | Discovered clusters **underperform** corr-cluster baseline on future co-movement | INCONCLUSIVE (trust-gated) |
| EXP-A6-01 Network risk | Incremental community risk **not** predictive after pairwise/sector controls | INCONCLUSIVE (trust-gated) |
| EXP-A2-01 Horizons | Momentum long/short **negative** net of costs at 5/10/22/66d | INCONCLUSIVE (trust-gated) |
| EXP-A3-01 Simple challenger | Logistic **FAIL** vs naive and rank incumbents | INCONCLUSIVE (trust-gated) |
| EXP-A5A6-01 Interactions | No FDR-significant context interactions | INCONCLUSIVE (trust-gated) |

**Bottom line:** do **not** start Phase B model/product expansion. Next value is
**research-grade data** (CA + universe ledger + NSE snapshot), then **retest**.

---

## 2. Experiment registry IDs

Isolated Phase A.5 store: `logs/phase_a5/experiments.db`  
(Results JSON: `logs/phase_a5/results.json`)

| Experiment | Hypothesis ID | Registry status |
|------------|---------------|-----------------|
| EXP-A5-01 | `81b8889792f53113` | REJECTED (failed frozen `research_grade==1` criterion) |
| EXP-A6-01 | `590571a11ee06fc2` | REJECTED |
| EXP-A2-01 | `775b4a0fce7d5b83` | REJECTED |
| EXP-A3-01 | `7842a46ee335685a` | REJECTED |
| EXP-A5A6-01 | `3734b8a0a9124a60` | REJECTED |

Pre-registration froze success criteria **before** metrics were computed.
`research_grade: {eq: 1}` was intentional — exploratory sources cannot “pass.”

---

## 3. Dataset / snapshot provenance

| Field | Value |
|-------|-------|
| Source | yfinance NSE `.NS` closes (auto-adjusted) |
| Trust class | **DISPLAY_ONLY** |
| Research tier | **LIMITED_RESEARCH** |
| Research-grade? | **False** |
| Snapshot ID | `050c77ea71b73001` (committed under `logs/phase_a5/snapshots/`) |
| Symbols | 29 liquid large-caps (sector-tagged exploratory map) |
| Sessions | 742 (2023-08-11 → 2026-08-11) |
| CA ledger | absent |
| Universe history | absent (`has_universe_history=False`) |
| Survivorship | biased (current survivors only) |
| PitContract smoke | PASS — mid-sample `as_of` returned only bars ≤ as_of |

**Limitation (blocking for promotion):** Yahoo-adjusted, survivorship-biased,
no official CA/universe ledgers. Metrics below are **provisional / exploratory**.

---

## 4. Market structure results (EXP-A5-01)

**Hypothesis:** hierarchical / k-means / PCA structure adds stable incremental
future co-movement info beyond sectors + correlation clusters.

**Null:** no incremental info and/or unstable.

### Method summary (exploratory)

| Method | Future co-movement score | Δ vs best baseline | Stability ARI | Turnover | p(Δ) |
|--------|--------------------------|--------------------|---------------|----------|------|
| sector_static (baseline) | 0.232 | — | — | — | — |
| correlation_clusters (baseline) | **0.414** | — | — | — | — |
| hierarchical | 0.149 | **−0.265** | 0.429 | 0.571 | 1.0 |
| kmeans | 0.128 | −0.286 | 0.501 | 0.499 | 1.0 |
| pca_kmeans | 0.128 | −0.286 | 0.501 | 0.499 | 1.0 |

FDR-rejected methods: **none**.

### Interpretation

- Clusters are **moderately stable** (ARI ~0.43–0.50) — structure is not pure noise.
- They do **not** beat the existing correlation-cluster baseline on future
  within-vs-between co-movement; point estimate is strongly negative.
- Complexity of discovered structure is **not justified** on this panel.

**Formal verdict:** INCONCLUSIVE (trust gate).  
**Provisional scientific reading:** fails incrementality vs incumbent corr clusters.

---

## 5. Network risk results (EXP-A6-01)

**Hypothesis:** network community / centrality / incremental community risk
explains correlated losses beyond pairwise clusters + sector HHI.

**Null:** no conditioned improvement.

| Metric | Value |
|--------|-------|
| Rows | 100 candidate–date observations |
| Partial corr(incr_risk, cand_loss \| pairwise, sector) | **−0.019** |
| Partial p | 0.85 |
| FDR rejects | none |
| auto_block | False (honoured) |

**Formal verdict:** INCONCLUSIVE (trust gate).  
**Provisional reading:** no evidence network adds value after existing controls —
**reject added complexity** until research-grade retest says otherwise.

---

## 6. Horizon results (EXP-A2-01)

**Preregistered family (not blind search):** `5d, 10d, 22d, 66d`  
**Strategy:** 60d cross-sectional momentum rank, long top 20% / short bottom 20%.  
**Costs:** CNC round-trip via `core.costs` (~0.32% assumed one-way turnover drag).  
**Multiple testing:** `n_trials=4` into DSR + BH-FDR across horizons.

| Horizon | N OOS | Mean net | Sharpe | DSR | Harness |
|---------|-------|----------|--------|-----|---------|
| 5d | 218 | −0.0032 | −0.10 | 0.067 | REJECT |
| 10d | 213 | −0.0059 | −0.14 | 0.017 | REJECT |
| 22d | 201 | −0.0152 | −0.29 | 0.000 | REJECT |
| 66d | 157 | −0.0172 | −0.20 | 0.005 | REJECT |

FDR-rejected horizons: **none**.  
Effect region: **no_effect** on exploratory panel (all net-negative).

**Formal verdict:** INCONCLUSIVE (trust gate).  
**Provisional reading:** this momentum specification is not failing “because the
horizon is wrong” on this panel — it fails across the preregistered family.

---

## 7. Simple challenger results (EXP-A3-01)

**Challenger:** logistic regression  
**Features:** mom_5/10/20/60 + vol_20  
**Target:** 10d classification (±1%)  
**Incumbents:** naive majority; momentum-rank sign  

| Bake-off | Verdict | Econ Δ (challenger − incumbent) | Pred corr |
|----------|---------|----------------------------------|-----------|
| logistic vs naive | FAIL | −0.014 | — |
| logistic vs rank | FAIL | **−0.037** | −0.06 |

Harness did **not** promote challenger.  
**Formal verdict:** INCONCLUSIVE (trust gate).  
**Provisional reading:** simple statistical learning did **not** extract incremental
value — **do not escalate to RF/GBM/DL/ensembles**.

---

## 8. Interaction results (EXP-A5A6-01)

Preregistered interactions only (no open mining):

| Interaction | Δcorr (high−low context) | p | FDR |
|-------------|--------------------------|---|-----|
| signal × cluster stability | −0.006 | 0.96 | no |
| signal × network concentration | +0.019 | 0.90 | no |
| signal × incremental community risk | +0.066 | 0.79 | no |

N rows: 319. FDR rejects: **none**.

**Formal verdict:** INCONCLUSIVE (trust gate).  
**Provisional reading:** no support that structure/network are useful *context*
modulators for momentum on this panel.

---

## 9. Statistical significance

- Harness `evaluate` applied to horizon R streams (with `n_trials=4` deflation).
- BH-FDR applied within each experiment’s preregistered family.
- Challenger lab used registry `should_promote` + autonomy `promotion_committee`
  (via bake-off) without live champion persistence.
- **No family cleared FDR** on exploratory metrics.

---

## 10. Economic significance

- Horizon portfolio: **negative** net expectancy after costs at all preregistered
  horizons.
- Challenger: **negative** economic-value delta vs incumbents.
- Structure/network: no positive incremental risk/co-movement value vs existing
  correlation/sector controls.

Economic significance for promotion: **absent** on this panel.

---

## 11. Regime robustness

Not separately powered: exploratory sample is ~3y, single NSE large-cap panel,
no RESEARCH_GRADE regime labels attached. Horizon/structure walks used OOS
tail (~30%) only. **Regime robustness = untested at research-grade.**

---

## 12. Multiple-testing treatment

| Experiment | Treatment |
|------------|-----------|
| EXP-A5-01 | BH-FDR across 3 discovery methods |
| EXP-A6-01 | BH-FDR across network features + partial test |
| EXP-A2-01 | Fixed horizon family (4); DSR `n_trials=4`; BH-FDR |
| EXP-A3-01 | Single primary challenger algorithm; two incumbent baselines |
| EXP-A5A6-01 | BH-FDR across 3 preregistered interactions |

No post-hoc horizon or cluster-parameter optimisation against future returns.

---

## 13. Failed hypotheses (provisional + registry)

Registry marked all REJECTED due to failed `research_grade==1` criterion.

Provisional empirical failures (exploratory):

1. Discovered clusters beat corr-cluster co-movement baseline — **failed** (negative Δ).
2. Network incremental risk predicts losses after controls — **failed** (≈0 / wrong sign).
3. Momentum has a winning preregistered horizon after costs — **failed** (all negative).
4. Logistic challenger beats incumbents — **failed** (both bake-offs FAIL).
5. Structure/network modulate momentum (FDR) — **failed** (no rejects).

Negative evidence preserved in Phase A.5 scientific memory (`logs/phase_a5/scientific_memory.db`) as WATCH notes.

---

## 14. Inconclusive hypotheses (formal)

All five remain **formally INCONCLUSIVE for promotion** until RESEARCH_GRADE
inputs exist. This is correct fail-closed behaviour, not a loophole to advance.

---

## 15. What should be promoted to further research

1. **Research-grade data activation** (CA ledger + universe history + NSE bhav
   snapshot) — blocking prerequisite.
2. **Retest EXP-A5-01 / EXP-A6-01** on RESEARCH_GRADE data only (structure/network
   as *risk lenses*, not alpha factories).
3. Keep Phase A codepaths (PitContract, horizons, challenger lab, seams) as the
   evaluation substrate — they proved operable.

---

## 16. What should be abandoned (for now)

1. Escalating to RF / GBM / SVM / deep learning / ensembles after logistic FAIL.
2. Treating discovered clusters as alpha features.
3. Automatic trade blocking from network metrics.
4. Meta-ensemble / HRP production cutover.
5. Blind expansion of horizon grid beyond preregistered families.

---

## 17. What Phase B should contain (evidence-based)

Phase B should **not** mean “build more AI.” If anything follows:

1. **Data grade promotion path** to RESEARCH_GRADE (operator CA + universe archive).
2. **Retest harness** reusing Phase A.5 experiment IDs/protocols unchanged.
3. Only if retests PASS_RISK: consider advisory Brain *warnings* (not gates) for
   network concentration — still behind evidence levels.

---

## 18. What Phase B should NOT contain

- Meta-ensemble  
- Financial RAG expansion as a trading input  
- Deep learning / RL  
- gs-quant dependency  
- Production HRP replacement  
- New model families “because infrastructure exists”  
- Wiring A4 seams into Brain decisions without PASS evidence  

---

## 19. Production behaviour confirmation

| Check | Result |
|-------|--------|
| Scanner / unified_scanner scoring changed? | No |
| Brain posture logic changed? | No |
| portfolio_gate / check_new_trade blocking on network? | No (`auto_block=False`) |
| Broker / place_trade path changed? | No |
| Challenger `persist_champion` for live roles? | False |
| CycleContext seams consumed by execution? | No |
| `production_behaviour_changed` flag in run | **False** |

---

## Decision matrix

| CAPABILITY | EVIDENCE | VALUE TYPE | COMPLEXITY | VERDICT | NEXT ACTION |
|------------|----------|------------|------------|---------|-------------|
| PitContract (A1) | Operable; PIT smoke passed | Infrastructure | Low | **ADVANCE** (as access layer) | Keep; use for all future research |
| Horizons framework (A2) | Operable; momentum family net-negative exploratorily | Infra + provisional no-effect | Low | **HOLD** infra / **RETEST** economics | Retest strategies on RESEARCH_GRADE |
| Challenger lab (A3) | Operable; logistic FAIL vs incumbents | Infra + negative ML signal | Low | **HOLD** infra / **REJECT** complexity escalation | No RF/GBM/DL; retest simple models on RG data |
| CycleContext seams (A4) | Present, unused by Brain | Plumbing | Low | **HOLD** | Do not wire to decisions yet |
| Market structure (A5) | Stable-ish; **no** incremental co-movement vs corr clusters | Risk structure (unproven) | Medium | **RETEST** | RG data only; abandon as alpha |
| Portfolio network (A6) | No conditioned loss predictability | Risk advisory (unproven) | Medium | **RETEST** / lean **REJECT** complexity | No auto-block; RG retest before any Brain warn |
| Horizon term structure claim | All preregistered horizons REJECT | Alpha timing | Low | **RETEST** | Don’t expand grid; fix data grade first |
| Structure×network context | No FDR interactions | Context | Medium | **REJECT** (for now) | Revisit only after A5/A6 PASS_RISK |

### Verdict legend used above

- **ADVANCE** — keep/use in research workflow  
- **RETEST** — same protocol on RESEARCH_GRADE data  
- **HOLD** — freeze; no expansion  
- **REJECT** — stop investing complexity here until evidence overturns  

---

## Appendix — how to reproduce

```bash
# exploratory panel must exist under logs/phase_a5/ (gitignored)
python -m research.phase_a5.run_all
# writes logs/phase_a5/results.json
```

Network-free unit smoke: `pytest tests/test_phase_a5.py`.
