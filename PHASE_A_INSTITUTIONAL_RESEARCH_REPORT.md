# Phase A — Institutional Research Report

**Branch:** `cursor/institutional-ai-audit-80a2`  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`  
**Date:** 2026-08-11  

**Production behaviour change:** **NONE** (by design).

---

## Architecture before / after

### Before

```text
Snapshot / data_state / ledgers
        │
        ▼
gauntlet + harness + registry + challenge committee
        │
        ▼
CycleContext → Brain / allocation → portfolio_risk / correlation → execution
        │
        ▼
outcomes → scientific memory
```

Gaps: no unified PIT read facade; horizons hard-coded in `ml/multi_horizon`;
`ml/` not wired through bake-offs; CycleContext had no research seams;
no market-structure discovery; network risk beyond pairwise ρ missing.

### After (Phase A)

```text
PitContract (A1) ──► Snapshot / ledgers / data_state   [no new store]
        │
research/horizons (A2) ──► harness.purged_kfold_indices
        │
research/challenger_lab (A3) ──► registry + harness + committee + scientific_memory
        │                         (persist_champion=False by default)
CycleContext seams (A4) ◄── optional typed overlays (None by default)
        │
research/market_structure (A5) ── RESEARCH_ONLY
research/portfolio_network (A6) ── advisory; complements risk.correlation
        │
(existing spine unchanged)
```

---

## Exact reuse points

| Milestone | Reused | Not duplicated |
|-----------|--------|----------------|
| A1 | `Snapshot`, `SnapshotStore`, `data_state`, `point_in_time_universe`, `corporate_actions`, `pit_valuations` | No FeatureStore / SnapshotStore |
| A2 | `harness.purged_kfold_indices`, `ml.multi_horizon._HORIZONS` thresholds | No second harness |
| A3 | `registry.register_hypothesis` / `should_promote`, `challenge.promotion_committee`, `harness.evaluate`, `scientific_memory`, `core.costs` | No second registry; no live `evaluate_challenger` by default |
| A4 | `CycleContext` | No DecisionResolver |
| A5 | PitContract inputs, `MarketStructureView` seam, sklearn | Does not overwrite `features/market_structure.py` |
| A6 | `risk.correlation` corr dict / cluster threshold intent, `NetworkRiskView` | Does not replace pairwise guards |

---

## Files changed / added

### Documentation
- `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md` (pre-Phase-A audit)
- `docs/overhaul/A1_PIT_CONTRACT_NOTE.md`
- `docs/overhaul/PIT_DATA_CONTRACT.md`
- `docs/overhaul/A2_HORIZONS_NOTE.md`
- `docs/overhaul/A3_CHALLENGER_LAB_NOTE.md`
- `docs/overhaul/A4_CYCLECONTEXT_SEAMS_NOTE.md`
- `docs/overhaul/A5_MARKET_STRUCTURE_NOTE.md`
- `docs/overhaul/A6_PORTFOLIO_NETWORK_NOTE.md`
- `PHASE_A_INSTITUTIONAL_RESEARCH_REPORT.md` (this file)

### Code
| Path | Role |
|------|------|
| `research/intelligence/data/pit_contract.py` | A1 facade |
| `research/intelligence/data_state.py` | `INCOMPLETE` / `NOT_PIT_SAFE` / `BLOCKED` + `PIT_READ_STATES` |
| `research/intelligence/data/__init__.py` | exports |
| `research/horizons/*` | A2 TargetSpec / labels / splits / catalog |
| `ml/multi_horizon.py` | docstring pointer only |
| `research/challenger_lab/*` | A3 bake-off wiring |
| `research/intelligence/runtime/cycle_context.py` | A4 optional seams |
| `research/intelligence/runtime/research_seams.py` | typed seam views |
| `research/market_structure/*` | A5 discovery |
| `research/portfolio_network/*` | A6 network complement |

### Tests
- `tests/test_pit_contract.py`
- `tests/test_horizons.py`
- `tests/test_challenger_lab.py`
- `tests/test_cyclecontext_seams.py`
- `tests/test_market_structure.py`
- `tests/test_portfolio_network.py`

---

## New contracts

1. **PitContract** — `history` / `latest` / `as_of` / `coverage` with explicit statuses  
2. **HorizonSpec / TargetSpec** — horizon, entry/exit, costs, purge/embargo, overlap policy  
3. **BakeOffResult** — incumbent vs challenger comparison payload + verdicts  
4. **CycleContext seams** — `market_structure` / `network_risk` / `horizon_view` / `challenger_evidence`  
5. **MarketStructureResult** / **NetworkRiskResult** — research outputs with `evidence_status=RESEARCH_ONLY`

---

## Tests added / run

Phase A focused suite (plus snapshot/intelligence regressions):

```text
tests/test_pit_contract.py
tests/test_horizons.py
tests/test_challenger_lab.py
tests/test_cyclecontext_seams.py
tests/test_market_structure.py
tests/test_portfolio_network.py
tests/test_snapshot_runtime.py
tests/test_intelligence_runtime.py
→ 87 passed
```

---

## Backward compatibility

- `Snapshot` / `SnapshotBarProvider` APIs unchanged  
- Automation `DATA_STATES` / `allows_new_entries` unchanged (new PIT statuses do not unlock entries)  
- `CycleContext.cycle_id()` identity fields unchanged (seams excluded)  
- `ml.multi_horizon` live inference path unchanged  
- `risk.correlation` / `portfolio_risk` unchanged  

---

## Evidence / research boundaries

| Component | Production authority |
|-----------|----------------------|
| PitContract | Access only |
| Horizons | Labels/splits only |
| Challenger lab | Research nomination; `live_behaviour_changed=False` by default |
| CycleContext seams | Optional; Brain does not read them yet |
| Market structure | `RESEARCH_ONLY` / `production_authority=False` |
| Portfolio network | `advisory_only=True` / `auto_block=False` |

Authority hierarchy preserved:

`RISK LIMITS > EVIDENCE GATE > PORTFOLIO CONSTRAINTS > MODEL > LLM/AGENT`

---

## Known limitations

1. Fundamentals / sectors remain `NOT_PIT_SAFE` until dated ledgers exist.  
2. Universe ledger path refuses biased “today’s survivors” fallback (stricter than raw `point_in_time_universe`).  
3. Challenger economic mapping uses a simple directional R proxy — not a full portfolio backtest.  
4. Market-structure communities use transparent sklearn methods only; stability is half-window ARI.  
5. Network communities are connected components on ρ-threshold graphs (not Louvain); sufficient for Phase A complement.  
6. Seams are not populated by the autonomous loop yet (intentional).  

---

## Explicitly deferred (still out of scope)

- HRP production / allocator cutover  
- Meta-ensemble  
- Financial RAG / `available_at` document corpus  
- Controlled research agent rewrite  
- Deep learning / RL / gs-quant / quantum  
- Automatic trade blocking from network metrics  
- Production promotion of market-structure clusters  

---

## Candidate experiments enabled

1. **Horizon bake-off:** 5d vs 10d vs 22d vs 66d absolute-return targets via `TargetSpec` + challenger lab.  
2. **Model bake-off:** Naive vs logistic vs existing rule-engine score streams under identical PIT matrices.  
3. **Structure vs sectors:** ARI of hierarchical clusters vs NSE sector map; does stability survive regimes?  
4. **Network incremental risk:** Does blocking high `incremental_cluster_risk` candidates improve realised cluster drawdowns vs pairwise-only? (paper shadow only)  
5. **Seam attachment experiment:** Attach A5/A6 views onto CycleContext in paper cycles without execution branching; measure directive usefulness later.  

---

## Did production behaviour change?

**No.** No scanner scoring, Brain posture, portfolio gate, ticket, or broker path was altered to consume Phase A research outputs. Optional champion persistence is off by default and still does not touch live trading.

---

## Milestone commits (this branch)

1. Audit document  
2. A1 PIT facade  
3. A2 horizons  
4. A3 challenger lab  
5. A4 CycleContext seams  
6. A5 market structure  
7. A6 portfolio network  
8. This report  
