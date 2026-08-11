# QuantTerm Institutional AI Audit

**Status:** READ-ONLY AUDIT — no implementation in this deliverable  
**Date:** 2026-08-11  
**Branch of record inspected:** `cursor/suggest-filters-evidence-5e9e` (+ working tree)  
**Authority:** Repository source of truth (not the prompt’s assumed architecture)

---

## Executive verdict

QuantTerm already has an institutional **evidence spine**:

`DATA → FEATURE SNAPSHOT → HARNESS/GAUNTLET → EVIDENCE LEVELS → BRAIN/ALLOCATION → RISK → PAPER/LIVE EXECUTION → OUTCOME → SCIENTIFIC MEMORY`

What it does **not** yet have (and must not invent as parallel truth) are: a discovered **market-structure** engine, a **portfolio network graph**, **HRP as a research challenger**, a **generic multi-horizon label framework**, a **unified ML challenger lab** that binds `ml/` to the evidence gate, a **meta-ensemble**, production **financial RAG with `available_at`**, or **RL / gs-quant / quantum**.

**Mandate preserved:** no model, LLM, agent, ensemble, or RL system may bypass `core/evidence_levels.promote` / gauntlet / champion–challenger. A complicated model has zero production entitlement until OOS evidence earns it.

---

## Mandatory Step 0 — Capability trace (22 systems)

| # | System | Classification | Primary files / symbols |
|---|--------|----------------|-------------------------|
| 1 | Market-data providers | **EXISTS** | `data/bhavcopy_store.get_ohlcv`, `data/nse_live.fetch_live_snapshot`, `data/live_quotes.get_live_quotes` (Kite→NSE→Google), `data/kite_client.KiteClient`, `data/index_store`, `data_platform/provider_registry.py`, `fundamentals/fetcher.py`, `news/fetcher.py` |
| 2 | Snapshot / PIT infrastructure | **EXISTS** | `research/intelligence/data/snapshot.Snapshot` (`bars`, `universe`, `benchmark`, `coverage_for`), `SnapshotStore.commit_snapshot` / `activate_snapshot`, `data/nse_universe.point_in_time_universe`, `data/universe_history.py`, `data/pit_valuations.get_valuation`, `research/momentum_breakout/pit_safety.py` |
| 3 | Feature store / computation | **PARTIAL** | `research/feature_schema.py` + `research/feature_store.snapshot` / `load_matrix` (immutable observation store); `features/indicators.IndicatorEngine`; `research/momentum_breakout/features.py`. **No `FeatureStore` class.** Dual stacks. |
| 4 | Research harness | **EXISTS** | `research/harness.py`: `deflated_sharpe_ratio`, `whites_reality_check`, `benjamini_hochberg`, `purged_kfold_indices`, `block_bootstrap_mean_ci`, `alpha_beta`, `evaluate` → `PROMOTE\|REJECT\|UNDERPOWERED\|INCONCLUSIVE` |
| 5 | Experiment registry | **EXISTS** (multi) | `research/registry.py` (pre-registration + champion/challenger SQLite), `gauntlet/registry.register`, `research/intelligence/registry.StrategyRegistry`, `research/strategy_studio/discovery.AttemptRegistry` |
| 6 | Evidence gates | **EXISTS** | `core/evidence_levels.py` (`E0`–`E6`, `promote`, `demote`); `research/autonomy/challenge.promotion_committee`; `research/autonomy/promotion.py` |
| 7 | Backtesting engines | **EXISTS** | `scan/signal_backtest.run_backtest`, `backtest/backtester.Backtester`, `backtest/walk_forward.WalkForwardValidator`, `gauntlet/runner.run_gauntlet`, `research/momentum_breakout/experiment.py` |
| 8 | Strategy definitions | **EXISTS** | `scan/unified_scanner.SIGNAL_META` / `StockSignal`, `playbooks.Playbook` / `REGISTRY`, `research/strategy_studio/spec.StrategySpec`, `research/momentum_breakout/config.MomentumBreakoutConfig` |
| 9 | Bayesian / confidence | **PARTIAL** | Beta-style: `core/adaptive_engine.AdaptiveEngine`, `expectancy/expectancy_engine.py`; Wilson LB: `scan/ev_engine.wilson_lb`. **No full Bayesian model lab.** |
| 10 | Calibration | **EXISTS** | `scan/live_edge.regime_calibration` / `live_calibration`, `research/calibration.py`, `analytics/calibration.py`, `core/decision_journal.calibration_report` |
| 11 | Drift detection | **EXISTS** | `research/drift.assess_drift` / `page_hinkley`, `research/drift_attribution.py`, `core/regime_drift.detect_drift`, `monitoring/decay_monitor.ModelDecayMonitor` |
| 12 | Counterfactual system | **EXISTS** | `research/counterfactual.gate_attribution` (gate ATE / FDR — association, not CFML); fed by `core/decision_journal` |
| 13 | Correlation engine | **EXISTS** | `risk/correlation.pairwise_corr`, `clusters_from_corr` (ρ≥0.70 union-find), `book_correlation_report`; `risk/correlation_guard.py` |
| 14 | Portfolio risk | **EXISTS** | `risk/portfolio_risk.portfolio_risk_report` / `check_new_trade`, `risk/position_sizer.size_position`, `risk/governor.evaluate`, `core/portfolio_intel.rotation_advice`, `portfolio/state.PortfolioState` |
| 15 | EV / ranking | **EXISTS** | `scan/ev_engine.estimate_ev` / `tag_ev` / `ev_rank_key` (MIN_N=30, Wilson LB), `scan/ranking_engine.RankingEngine.rank` |
| 16 | Regime detection | **EXISTS** | `core/regime_engine.compute_regime`, `scan/regime_filter.RegimeFilter`, `analysis/hmm_regime.HMMRegimeDetector` (2-state), `core/regime_monitor.py`, `core/regime_analog.py` |
| 17 | Research / Scientific Memory | **EXISTS** | `research/scientific_memory.py` (`record_belief`, `beliefs_as_of`, `promote_from_experiment`); `research/autonomy/hypotheses.ResearchMemory`; `research/market_memory.py`; `research/evidence_graph.py`. `/workspace/memory/` is docs-only PRD, not code. |
| 18 | JARVIS / Assistant | **EXISTS** | `ai/jarvis_orchestrator.JarvisOrchestrator`, `ai/jarvis_agents.*`, `llm/context_builder.ContextBuilder`, `llm/devil_advocate.challenge_signal`, `ui/jarvis.py` |
| 19 | Execution engine | **EXISTS** | `execution/trade_executor.place_trade` (paper-default + GTT), `execution/autopilot.py`, `ems/ems.EMS` + `ems/simulator.SimBroker`, OMS under `execution/oms/` |
| 20 | Broker boundary | **EXISTS** | `data/kite_client.KiteClient`, `execution/zerodha_broker.ZerodhaBroker`, `ems/broker.BrokerAdapter` (abstract). Live legacy gated by `QT_ENABLE_UNSAFE_LEGACY_LIVE`. |
| 21 | Model / version provenance | **EXISTS** | `gauntlet/registry` (`git_commit`, `dataset_hash`, `config_hash`, `seed`), `gauntlet/freeze.verify_unchanged`, `research/registry` (`code_hash`, `seed`), `StrategySpec.config_hash`, `research/evidence_graph` |
| 22 | Outcome tracking | **EXISTS** | `core/signal_outcome_tracker.log_signal` / `update_outcomes`, `core/decision_journal.log_decision`, `ai/thesis_recorder.py`, `ai/signal_memory.py` |

---

## Capability matrix (prompt targets A–N)

Legend for **Current maturity:** E0–E6 = `core/evidence_levels`; qualitative notes where not yet gated.

| Capability | Existing implementation | Files / classes / functions | Current maturity | Gap | Reuse opportunity | Research value | Engineering complexity | Data requirement | PIT risk | Recommended action | Priority |
|------------|------------------------|-----------------------------|------------------|-----|-------------------|----------------|------------------------|------------------|----------|--------------------|----------|
| **A. Market Structure Engine** | **MISSING** as discovery engine. Closest: swing-point TA (`features/market_structure.py`), pairwise corr clusters (`risk/correlation.py`), rule regime (`core/regime_engine.py`), 2-state HMM (`analysis/hmm_regime.HMMRegimeDetector`), static NSE sectors (`scan/sector_heat.py`). | `features/market_structure.find_swing_points`; `risk/correlation.clusters_from_corr`; `core/regime_engine.compute_regime`; `analysis/hmm_regime.HMMRegimeDetector`; `scan/sector_heat.sector_of` | Operational regime ≈ production UI; **structure discovery = E0** | No rolling hierarchical / spectral / DBSCAN / PCA-ICA factor discovery; no `cluster_stability`, `structural_shift_score`, sector-comparison research reports | Reuse PIT returns from `Snapshot.bars` / bhav store; compare vs `sector_heat`; gate via `harness.evaluate` + `registry` | Medium — may explain co-movement; **must not assume alpha** | Medium (sklearn already in `requirements.txt`) | Daily returns panel, PIT universe, ≥2–3y sessions preferred | High if fit on full panel without rolling `as_of`; medium if rolling-only | Build **research-only** module under `research/market_structure/`; no scanner demotion until OOS | **P0** |
| **B. Portfolio Network Engine** | **MISSING** graph engine. Pairwise corr + union-find clusters + sector packs + cluster risk caps in intelligence runtime. | `risk/correlation.py`; `risk/correlation_guard.py`; `risk/portfolio_risk.check_new_trade`; `research/intelligence/runtime/portfolio_gate.check`; `target_portfolio._existing_group_risk`; CycleContext.`clusters` | Correlation lens ≈ live advisory; network science = **E0** | No edges beyond ρ; no community detection, centrality, contagion paths, `incremental_cluster_risk` beyond current cluster caps | **Complement** `book_correlation_report` + `check_new_trade` + portfolio_gate; do **not** replace | High for risk questions: “what incremental risk does this candidate add?” | Medium–High (graph libs may be new dep; can start with numpy/scipy) | Same return panel + current open book | Medium (rolling edges only) | Research challenger that **feeds** Brain directives / gate reasons; keep union-find as incumbent | **P0** |
| **C. Hierarchical Risk Parity** | **MISSING**. Mentioned as future in `docs/MOAT_RESEARCH_DIRECTIVE.md`. Allocation is evidence-card / risk-budget based, not HRP. | `research/intelligence/allocation_brain.decide`; `research/intelligence/runtime/position_sizing.size_long_cash`; `risk/position_sizer.size_position` | Allocator = production rails; HRP = **E0** | No inverse-vol / risk-parity / HRP bake-off vs current sizer | Reuse gauntlet + harness metrics; champion/challenger in `research/registry.evaluate_challenger` | Medium — often improves diversification vs naive weights; **not** alpha | Medium | Historical returns of candidate universes + cost model (`core/costs.py` if used) | Low–Medium if weights estimated rolling | Implement as **research challenger only**; promote only via evidence gate | **P0/P1** |
| **D. Multi-Horizon Research Framework** | **PARTIAL**. Hard-coded 1d/5d/10d LGBM consensus; backtest `horizon` param; signal tracker multi-horizon columns. | `ml/multi_horizon.MultiHorizonSignalGenerator`; `scan/signal_backtest.run_backtest(..., horizon=10)`; `research/market_memory.build_corpus(..., horizon=...)`; `screener/signal_tracker.py` | ML multi-horizon ≈ **E0–E1** (not gauntlet-bound); scanner horizon configurable | No generic 5…252d framework; no purged/embargoed label construction shared across models; no horizon dispersion / best-supported horizon for rule strategies; overlapping-label leakage not systematically prevented in `ml/` | Reuse `harness.purged_kfold_indices`; outcome trackers; `feature_store` | High — horizon misspecification is a common false edge | Medium | PIT bars + explicit label windows | **High** if labels overlap without purge/embargo | Build `research/horizons/` label+eval contract first; **do not** expand `ml/multi_horizon` horizons until labels are leakage-safe | **P0** |
| **E. Model Challenger Lab** | **PARTIAL**. Champion/challenger + adversarial committee exist; `ml/` XGB/LGBM/Ensemble do **not** call evidence/promote. | `research/registry.should_promote` / `evaluate_challenger`; `research/autonomy/challenge.{data_auditor,sceptic,reality_checker,portfolio_examiner,promotion_committee}`; `ml/xgboost_signal.XGBoostSignalGenerator`; `ml/lgbm_signal.py`; `ml/ensemble_signal.EnsembleSignalGenerator`; `research/harness.evaluate` | Registry mechanics = mature; ML bake-off = **E0** | No identical data/features/target/cost/portfolio harness binding rule engines + ML + naive benchmarks to one verdict schema | **Extend** registry + challenge + gauntlet; wrap existing scanners as incumbents | Very high — prevents fashionable models from skipping the gate | Medium (orchestration); Low (new algos if sklearn) | Shared feature matrix from `feature_store.load_matrix` or Snapshot | High if ML uses non-PIT fundamentals | Wire `ml/` through challenger + gauntlet; add naive/linear baselines; verdicts only | **P0** |
| **F. Meta-Ensemble** | **MISSING** (evidence-weighted). Fixed 0.5/0.5 vote exists. | `ml/ensemble_signal.EnsembleSignalGenerator`; weight concepts already in `scan/live_edge`, `scan/ev_engine`, `research/calibration`, `core/adaptive_engine` | Ensemble vote = **E0**; reliability systems higher | No regime×calibration×evidence weighting; disagreement not first-class | **REUSE** live_edge / EV / calibration / adaptive_engine — do not invent a second weighting OS | Medium only **after** challenger lab proves ≥2 models with incremental OOS value | Medium | Per-model OOS histories + regime tags | Medium | Defer until Phase A challengers produce comparable prediction streams | **P1** |
| **G. Financial RAG / Doc Intelligence** | **PARTIAL**. News semantic index + setup analogs; filing embeddings stubbed. | `news/semantic_index.SemanticNewsIndex`; `news/curator.py`; `research/market_memory.find_analogs`; `research/similar_history.similar`; `fintel/.../embeddings.py` (TODO); `fintel/.../ai_extraction.py` (TODO) | News retrieval = operational; PIT document RAG = **E0** | No `available_at` contract; no unified document schema (source, document_id, symbol, published_at, available_at, hash); LLM not forced to cite retrieval for filings/CA | Reuse news curator models, Scientific Memory, Snapshot; JARVIS `ContextBuilder` | Medium for research narrative; low for trade alpha unless PIT | High (ingest + PIT) | Filings/announcements with honest availability timestamps | **Very high** without `available_at` | Document schema + PIT retrieval first; generation second; no fake facts | **P1** |
| **H. Controlled Research Agent** | **EXISTS** (research-scoped). Auto-research stops at user approval; committee is deterministic. | `research/auto_research/loop.run_cycle`; `research/autonomy/research_loop.run_research_cycle`; `research/strategy_studio/discovery.py` + `approval.approve_for_paper`; `ai/jarvis_agents.ResearchAgent`; `research/autonomy/challenge.promotion_committee` | Autonomy paper path mature; agent must not gain live/broker/promote powers | Permissions matrix not fully enumerated as code policy object; JARVIS still conversationally broad | Extend existing loops + Scientific Memory search; keep LLM commentary non-authoritative (already in challenge.py) | High for research throughput **if** bounded | Medium | Scientific Memory + registries + Snapshot | Medium | Harden allowlists; do **not** grant promote/order/credential powers | **P1/P2** |
| **I. Decision Resolution Architecture** | **EXISTS** (two related paths). Intelligence CycleContext + live Brain/journal. | `research/intelligence/runtime/cycle_context.CycleContext`; `context_builder.build_context_from_snapshot`; `autonomous_loop.run_intelligence_cycle`; `evidence_brain.build_card`; `allocation_brain.decide`; `core/brain.assess` / `decide_posture`; `core/decision_journal.log_decision` | Paper intelligence cycle strong; retail ticket path parallel | No single named DETECT→RESOLVE→EVALUATE object across retail+intelligence; risk of a **third** context object | **Reuse CycleContext** for research/paper; map retail ticket fields onto journal + risk checks — do not invent `DecisionResolver` duplicate | High for reproducibility | Low–Medium (glue + schema alignment) | Existing snapshots + book + registry | Low if reusing Snapshot ids | Audit mapping doc + thin adapters; **no second SOFT** | **P0** |
| **J. Unified PIT Data Contract** | **PARTIAL**. Snapshot + data_state + coverage patterns; no single facade; `NOT_PIT_SAFE` absent. | `Snapshot.bars/universe/benchmark/coverage_for`; `research/intelligence/data_state` (`READY`/`DEGRADED`/`STALE`/tiers); `data_platform/coverage.audit_symbol`; `data/corporate_actions.adjust_frame`; `point_in_time_universe`; fundamentals **not** PIT (`fundamentals_point_in_time: False` in long-term path) | Ops data strong; research-grade depends on `logs/ca_events.json` + universe ledger | No unified `history/latest/as_of/coverage` across market/fundamentals/CA/filings/sectors/news; status vocab fragmented; planned `data_platform/point_in_time/` **does not exist** | Extend Snapshot + data_state + provider_registry; do not build a second snapshot store | Foundational — required by A/D/E/G | Medium | Operator ledgers for CA + universe; PIT valuations | **Critical** if research silently uses live universe/fundamentals | Introduce thin facade + `NOT_PIT_SAFE` / `INCOMPLETE` / `BLOCKED` states; fail closed | **P0** |
| **K. Model Explainability + Governance** | **PARTIAL**. Gate explainability + gauntlet provenance; no SHAP / full ML card. | `research/explainability.explain_reason`; `gauntlet/registry.register`; `gauntlet/freeze.py`; `research/evidence_graph.py`; `models/*.pkl` artifacts without governance schema | Provenance for strategies/gauntlet mature; ML model cards thin | Missing model_id/version/training windows/feature_schema_version/kill_criteria for `ml/`; SHAP not present (and must not substitute for evidence) | Attach governance schema to registry + evidence_graph | Medium for trust/audit | Medium | Training manifests | Low | Model card schema on promote path; optional SHAP for trees later | **P1** |
| **L. RL execution sandbox** | **MISSING** | Closest non-RL: `core/sim_lab.simulate`, `ems/simulator.SimBroker`, `execution/tca/` | N/A | No gym env / policy / shadow execution RL | Reuse EMS simulator + TCA metrics if ever started | Low for alpha; medium for execution cost only | High | Tick/LOB or realistic fill model — **not available as research-grade** | High if fake fills | **Do not implement** in Phase A–B | **P2/P3 — NOT NOW** |
| **M. gs-quant** | **NOT REQUIRED** | No imports/deps; architectural ideas already mirrored (resolve-before-evaluate ≈ CycleContext; portfolio risk exists) | N/A | Marquee credentials not confirmed | Borrow patterns only | None operationally | Isolation cost high if added | N/A | N/A | **Do not add dependency** | **NOT REQUIRED** |
| **N. Quantum computing** | **NOT REQUIRED** | None | N/A | N/A | N/A | Watchlist only | N/A | N/A | N/A | **No engineering time** | **NOT REQUIRED** |

---

## 1. Current architecture diagram

```text
                    ┌─────────────────────────────────────────────┐
                    │  Providers                                  │
                    │  Kite → NSE live → Google (quotes)          │
                    │  NSE bhavcopy (history, CA adjust on READ)  │
                    │  fundamentals/news (mostly as-of-now)       │
                    └───────────────┬─────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────────────┐
              ▼                     ▼                             ▼
     SnapshotStore / Snapshot   bhavcopy_store              live scanner path
     (research PIT bars)        + universe_history          unified_scanner
              │                 + pit_valuations              + auto_scan
              ▼                     │                             │
     feature_store (schema)         │                             ▼
     momentum_breakout feats        │                      ev_engine / live_edge
              │                     │                      conviction / breadth
              ▼                     ▼                             │
     gauntlet + harness ◄── scientific_memory / registry          │
     evidence_levels.promote                                      │
              │                                                   │
              ▼                                                   ▼
     CycleContext → evidence_brain → allocation_brain      core/brain.assess
              │              │                             decide_posture
              ▼              ▼                                    │
     portfolio_gate / target_portfolio                    portfolio_risk
     position_sizing                                      correlation clusters
              │                                                   │
              ▼                                                   ▼
     EMS / paper book  OR  trade_executor.place_trade (+ GTT)
              │                                                   │
              └────────────────────┬──────────────────────────────┘
                                   ▼
                     decision_journal / signal_outcome_tracker
                                   ▼
                     drift / calibration / counterfactual / memory
```

**Parallel truth risk today:** retail `ml/*` signal generators and live `unified_scanner` are not the same evidence path as `gauntlet` / `CycleContext`. That is acceptable only while ML remains non-authoritative.

---

## 2. Proposed minimal architecture (Phase A only)

Add **research modules that plug into existing sockets** — no second Brain, no second registry, no second snapshot store, no gs-quant.

```text
Snapshot / data_state / point_in_time_*
        │
        ├─► research/horizons/          # label builders + purge/embargo + eval windows
        ├─► research/market_structure/  # rolling clusters/factors → research outputs only
        ├─► research/portfolio_network/ # graph metrics → incremental risk lens
        └─► research/challenger_lab/    # binds naive|linear|trees|rule-engines
                 │                      # identical splits/costs → harness + registry
                 ▼
        research/registry + autonomy/challenge + evidence_levels
                 │
                 ▼
        CycleContext / Brain directives / portfolio_gate   (consume, don't replace)
```

**Explicit non-goals for Phase A:** meta-ensemble weights, RAG generation, RL, HRP production cutover, deep learning, quantum, gs-quant.

---

## 3. Components that should NOT be built

| Temptation | Why not |
|------------|---------|
| Second `FeatureStore` class / parallel feature DB | `research/feature_store.py` + schema already exist |
| Second experiment registry | Use `research/registry.py` + gauntlet E5 |
| Second snapshot store | Use `research/intelligence/data/SnapshotStore` |
| Second portfolio risk engine that ignores `check_new_trade` / `portfolio_gate` | Extend incremental risk **into** them |
| Production HRP allocator | Challenger until evidence |
| Meta-ensemble before ≥2 validated models | No substance to weight |
| Deep learning | Dataset/use-case not justified; sklearn+boosting already present |
| gs-quant runtime dependency | No Marquee credentials; isolation cost |
| Quantum anything | Prompt forbids |
| RL with broker authority | Forbidden until separate validation; fill data insufficient |
| `NOT_PIT_SAFE` silent ignore / fake fundamentals fallback | Violates invariants |
| LLM self-promotion of strategies | `challenge.promotion_committee` already forbids self-approval |

---

## 4. Duplicate-functionality risks

| Risk | Incumbent | Proposed | Rule |
|------|-----------|----------|------|
| Correlation clusters vs network communities | `risk/correlation.clusters_from_corr` | Portfolio Network Engine | Network **augments**; union-find remains default until bake-off |
| Regime engine vs market structure clusters | `core/regime_engine`, `HMMRegimeDetector` | Structure engine | Structure = cross-sectional; regime = tape state — different questions |
| `features/market_structure.py` name collision | Swing points | New market-structure research package | **Do not overwrite** TA module; use `research/market_structure/` |
| `ml/ensemble_signal` vs meta-ensemble | Fixed vote | Evidence-weighted meta | Keep vote as naive challenger baseline |
| `ml/multi_horizon` vs horizons framework | 1/5/10d LGBM | Generic horizons | Framework owns labels; ML becomes one consumer |
| CycleContext vs new DecisionResolver | CycleContext + decision_journal | Decision lifecycle mapping | **Map**, don’t mint a third canonical object |
| Scientific Memory vs mem0 / memory vault | `scientific_memory.py` vs `ai/mem0_store` / `ui/memory_vault` | RAG over research | Keep trader chat memory separate from scientific beliefs |
| data_state READY/STALE vs setup freshness STALE | Multiple meanings | PIT contract | Namespace statuses (`data.*` vs `setup.*`) |

---

## 5. Dependency implications

**Already present (`requirements.txt`):** `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `hmmlearn`, `qdrant-client`, `sqlalchemy`, `kiteconnect`.

| Phase A need | Dependency | Notes |
|--------------|------------|-------|
| Clustering / PCA / spectral | sklearn | Prefer existing; no new dep |
| Graph centrality / communities | optional `networkx` | Only if numpy implementation too costly; isolate |
| SHAP (Phase B) | `shap` | Optional; never required for promote |
| gs-quant | **do not add** | Credentials unconfirmed |
| RL libs | **do not add** | Phase D only |
| Deep learning (torch/tf) | **do not add** | Not justified |

---

## 6. Migration risks

1. **Survivorship / CA incomplete:** Without research-grade `logs/universe_history.json` + `logs/ca_events.json`, gauntlet/`classify_tier` correctly stays limited — new engines must inherit that fail-closed behavior, not invent data.
2. **ML path currently ungated:** Expanding `ml/` features without routing through challenger lab would **worsen** governance.
3. **Dual brains:** Live `core/brain.py` vs intelligence `evidence_brain`/`allocation_brain` — new risk lenses must declare which path they feed.
4. **Status vocabulary collision:** Introducing `NOT_PIT_SAFE` without mapping to `data_state` / `QualityStatus` will confuse UI and automation.
5. **Sector membership not historically dated:** `BhavDataProvider.sector_ctx` returns `None` historically — structure-vs-sector comparisons must treat static sectors as **labels of convenience**, not PIT truth.
6. **Fundamentals / news look-ahead:** Any feature using `fundamentals/` or news without `available_at` is research-contaminated — block from FORWARD_ELIGIBLE claims.
7. **Name collision:** Implementing “market structure” inside `features/market_structure.py` would destroy swing-breakout helpers.

---

## 7. Proposed test strategy

Follow existing money-critical patterns (`tests/test_money_paths.py`, `tests/test_research.py`, `tests/test_gauntlet.py`, `tests/test_intelligence_runtime.py`): **network-free, synthetic fixtures**.

| Area | Must assert |
|------|-------------|
| PIT / facade | `as_of` never returns future bars; missing ledger → `NOT_PIT_SAFE`/`LIMITED_RESEARCH`, never silent OK |
| Horizons | Overlapping labels + purge/embargo; identical calendar splits across challengers |
| Market structure | Deterministic seeds; rolling window exclusivity; stability metric finite; no production scanner mutation |
| Portfolio network | Incremental risk ≥ 0 when adding perfectly correlated twin; complements `clusters_from_corr` |
| Challenger lab | Naive baseline included; identical costs; verdict ∈ {PROMOTE, KEEP INCUMBENT, INCONCLUSIVE, REJECT}; ML cannot promote without committee |
| Decision mapping | CycleContext still sole intelligence cycle identity (`cycle_id` stable) |
| Authority | LLM/agent fixtures cannot call `place_trade(live)` or `evidence_levels.promote` |

---

## 8. Exact P0 implementation plan

**STOP after each gate. No Phase B until Phase A items justify continuation.**

### P0-1 — Unified PIT data contract (thin facade)

1. Add a **facade module** (suggested: `data_platform/pit_contract.py` or `research/intelligence/data/contract.py`) exposing `history / latest / as_of / coverage` that **delegates** to `Snapshot`, `bhavcopy_store`, `point_in_time_universe`, `pit_valuations`, `corporate_actions`, `data_state.classify_tier`.
2. Add explicit states: reuse `READY`/`DEGRADED`/`STALE`; add `INCOMPLETE`, `NOT_PIT_SAFE`, `BLOCKED` (do not invent fake fills).
3. Tests: future leak blocked; missing CA/universe → not FORWARD_ELIGIBLE.
4. **Do not** implement `data_platform/point_in_time/` mega-package yet.

### P0-2 — Multi-horizon research framework

1. Audit actual horizons in use (`signal_backtest` default 10; outcome tracker ~5d; ML 1/5/10; long-term paths).
2. Create `research/horizons/` with label constructors + purge/embargo using `harness.purged_kfold_indices`.
3. Outputs: per-horizon metrics + agreement/dispersion — **research reports**, not live BUY.
4. Leave `ml/multi_horizon.py` unchanged until it consumes the new labels.

### P0-3 — Model challenger lab (wire existing pieces)

1. Thin orchestrator `research/challenger_lab/` (or extend `research/registry` + `autonomy/challenge`) that runs: naive → linear/logistic → optional RF/GBM → incumbent rule engine on **identical** PIT matrix.
2. Score with `harness.evaluate` + economic metrics (costs/turnover via existing cost helpers where available).
3. Persist via `register_hypothesis` / `evaluate_challenger`; committee verdicts only.
4. **Bind** a smoke path so `ml/` models are challengers, not authorities.

### P0-4 — Decision resolution mapping (no new SOFT)

1. Document DETECT→RESOLVE→EVALUATE→RISK→DECIDE→EXECUTE→TRACK mapping onto: scanner/candidate → `CycleContext` / ticket fields → `evidence_brain`/`ev_engine` → `portfolio_gate`/`check_new_trade` → `allocation_brain`/`decision_journal` → `place_trade`/EMS → outcome trackers.
2. Fill only **missing RESOLVE fields** (model versions, snapshot_id, evidence tier) where cheap — prefer extending `log_decision` payload / CycleContext metadata.
3. Forbid a new `DecisionResolver` class unless audit proves CycleContext cannot carry retail path (unlikely).

### P0-5 — Market Structure Engine (research-only)

1. New package `research/market_structure/` (avoid `features/market_structure.py`).
2. Rolling returns → baseline k-means + hierarchical; optional PCA factors; stability across adjacent windows.
3. Compare cluster labels to `sector_heat` (agreement stats only).
4. Register as hypothesis; gauntlet/harness before any Brain demote/boost.

### P0-6 — Portfolio Network Engine (complement correlation)

1. New package `research/portfolio_network/` building graph from rolling correlations (statistically thresholded).
2. Metrics: community_id, concentration, centrality, diversification_score, **incremental risk vs current book**.
3. Surface as Brain **warn** directive / gate reason candidate — default path remains `risk/correlation` + `portfolio_risk`.
4. Challenger bake-off: does network incremental risk reduce realized cluster drawdowns OOS vs incumbent?

### Phase A exit criteria

- PIT facade used by new research modules.
- At least one horizon-safe evaluation report.
- At least one challenger bake-off recorded in `logs/experiments.db` with KEEP INCUMBENT or INCONCLUSIVE acceptable.
- Market structure + network produce **research artifacts only** (no live order path).
- No new production allocator, ensemble, RAG generator, RL, or gs-quant.

### Explicit deferrals

| Item | Defer until |
|------|-------------|
| HRP challenger (P1) | Network + challenger lab harness can evaluate allocators fairly |
| Meta-ensemble (P1) | ≥2 models with incremental OOS evidence |
| Explainability/SHAP (P1) | Model cards on promote path |
| Financial RAG (P1) | Document `available_at` ledger exists |
| Controlled agent expansion (P1/P2) | Scientific Memory query UX + permission allowlist |
| RL (P2/P3) | Realistic execution simulator evidence |
| gs-quant / quantum | Never / watchlist |

---

## Appendix A — Prompt capability scoreboard

| ID | Capability | Classification |
|----|------------|----------------|
| A | Market Structure Engine | **MISSING** |
| B | Portfolio Network Engine | **MISSING** |
| C | HRP Challenger | **MISSING** |
| D | Multi-Horizon Research Framework | **PARTIAL** |
| E | Model Challenger Lab | **PARTIAL** |
| F | Meta-Ensemble | **MISSING** |
| G | Financial RAG | **PARTIAL** |
| H | Controlled Research Agent | **EXISTS** (bounded) |
| I | Decision Resolution Architecture | **EXISTS** / map, don’t rebuild |
| J | Unified PIT Data Contract | **PARTIAL** |
| K | Explainability + Governance | **PARTIAL** |
| L | RL Sandbox | **MISSING** — not now |
| M | gs-quant | **NOT REQUIRED** |
| N | Quantum | **NOT REQUIRED** |

## Appendix B — Authority hierarchy (unchanged)

```text
RISK LIMITS
  > EVIDENCE GATE (evidence_levels / gauntlet / challenge committee)
    > PORTFOLIO CONSTRAINTS (portfolio_risk / portfolio_gate / correlation)
      > MODEL (scanner, ML challengers)
        > LLM / AGENT (commentary only)
```

---

**Audit complete. No implementation performed beyond this document.**  
Await explicit instruction before Phase A coding.
