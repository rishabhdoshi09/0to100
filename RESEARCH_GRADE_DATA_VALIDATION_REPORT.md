# RESEARCH_GRADE Data Validation Report

**Date:** 2026-08-11  
**Branch:** `cursor/institutional-ai-audit-80a2`  
**Gate result:** **NOT EARNED** — `trust_class=OPERATIONAL_ONLY`  
**Phase A.5 RESEARCH_GRADE rerun:** **BLOCKED** (no `PHASE_A5_RESEARCH_GRADE_RERUN.md`)  
**Production trading behaviour:** **unchanged**

---

## In plain English

**Conclusion:** Research quality — **Not ready**.

**Why:** QuantTerm loaded a large official NSE price history and several official
ledgers, but two scientific checks still fail:

1. Price adjustments for splits/bonuses are **not yet trustworthy enough**
   (too many unexplained price jumps remain after applying the corporate-action
   ledger we could parse from NSE).
2. Symbol identity history is **still incomplete** (not every rename/ISIN link
   is proven).

**What it means:** This data is good enough for charts and operational paper
views. It is **not** good enough to prove or disprove a strategy scientifically.

**What to do:** Do **not** promote Phase A.5 ideas. Do **not** start Phase B.
Supply missing corporate-action factors from official filings for remaining
gaps (never invent them), and close identity lineage gaps with evidence only.
Then re-run the earned RESEARCH_GRADE gate.

**Data quality today:** Good enough for today's trading view, not fully
reconstructible research history (`OPERATIONAL_ONLY`).

<details>
<summary>Technical details</summary>

```
earned=false
trust_class=OPERATIONAL_ONLY
failed=[symbol_lineage_complete, adjustment_verified, gauntlet_validate]
```

Plain-language helper: `product.plain_language.explain_trust_class("OPERATIONAL_ONLY")`.

</details>

---

## What was materialized (reuse-only architecture)

| Asset | Status | Location / source |
|-------|--------|-------------------|
| RAW NSE equity bhav | **YES** | `logs/bhav/` — 764 sessions, 2853 EQ symbols via existing `data.bhavcopy_store.build_store` |
| Index / Nifty / VIX | **YES** | `logs/indices/` — ~360 sessions via existing `data.index_store` |
| Security identity | **PARTIAL** | `logs/security_identity.json` — EQUITY_L + symbolchange + delisted (`data/security_identity.py`) |
| PIT universe ledger | **PARTIAL→improved** | `logs/universe_history.json` — EQUITY_L + delisted (`data/nse_universe_ingest.py`); `survivorship_complete=True` stamped; **research_grade=False** until gate |
| Corporate-action ledger | **PARTIAL** | `logs/ca_events.json` — 183 share-count events parsed from NSE corporates API (`data/nse_ca_ingest.py`); 5408 dividends stored as provenance only |
| Adjustment policy | **YES** | `ca_sharecount_v1` — dividends not auto-applied |
| SnapshotStore / PitContract | **reused** | no second store created |
| RESEARCH_GRADE gate | **YES (fail-closed)** | `research/intelligence/data/research_grade_gate.py` |

---

## Gate scoreboard

| Check | Result | Notes |
|-------|--------|-------|
| security_identity present | PASS | 2449 rows |
| official_delistings | PASS | NSE `delisted.csv` ingested |
| symbol_lineage_complete | **FAIL** | Completeness flag remains False — unknown transitions not invented |
| universe_source | PASS | `nse_equity_l+nse_delisted` (not `bhav_inferred`) |
| universe_not_survivor_only | PASS | Delist archive applied |
| survivorship_complete | PASS | Completeness stamp True for window-local criteria |
| corporate_actions present | PASS | 183 adjusting events |
| adjustment_verified | **FAIL** | `gap_rate≈0.16` (need ≈0); verify did not PASS |
| bhav_coverage | PASS | 764 sessions / 2853 symbols |
| benchmark + VIX | PASS | after index cache load |
| gauntlet_validate | **FAIL** | `no_phantom_gaps`, `no_symbol_mismatch` |

**Earned RESEARCH_GRADE?** No.

Writers refuse to stamp `trust_class=RESEARCH_GRADE` without this gate
(`stamp_manifest_if_earned`).

---

## D1–D11 milestone status

| ID | Milestone | Outcome |
|----|-----------|---------|
| D1 | Security identity | Implemented + materialized from NSE; lineage **incomplete** |
| D2 | PIT universe ledger | Materialized from EQUITY_L + delisted; survivor-only reconstruction refused |
| D3 | Corporate-action ledger | Materialized parseable bonus/split events only; dividends provenance-only |
| D4 | Raw + adjusted view | Raw bhav immutable; on-read `adjust_frame`; policy version `ca_sharecount_v1` |
| D5 | NSE research panel path | Bhav+index on disk; Snapshot/PitContract reuse ready |
| D6 | Validation | Gauntlet + CA verify + gate checks run |
| D7 | RESEARCH_GRADE gate | **Not earned** |
| D8 | PitContract enforcement | Existing contract reused; promotion path still blocked by gate |
| D9 | Immutable RESEARCH_GRADE snapshot | **Not created** (refused — gate failed) |
| D10 | Freeze Phase A.5 protocols | Confirmed unchanged in `PHASE_A5_FROZEN_PROTOCOLS.md` |
| D11 | RESEARCH_GRADE rerun | **BLOCKED** — would violate scientific standard |

---

## Frozen Phase A.5 protocols (unchanged)

| Hypothesis ID | Experiment |
|---------------|------------|
| `81b8889792f53113` | EXP-A5-01 |
| `590571a11ee06fc2` | EXP-A6-01 |
| `775b4a0fce7d5b83` | EXP-A2-01 |
| `7842a46ee335685a` | EXP-A3-01 |
| `3734b8a0a9124a60` | EXP-A5A6-01 |

Hypotheses, horizons, costs, benchmarks, success criteria, and FDR treatment
were **not** altered based on DISPLAY_ONLY results.

---

## Remaining blockers (exact)

1. **Corporate-action completeness**  
   NSE API subject strings yielded 183 unambiguous share-count events for
   2023–2026. `verify_ca_adjustment` still sees ~16% unexplained discontinuities
   on sampled symbols.  
   **Required:** operator/official factor rows for remaining gaps
   (`logs/ca_events.todo.csv` workflow). **Must not invent factors from prices.**

2. **Symbol lineage closure**  
   Identity ledger has ISINs for current EQ + 1045 symbol-change rows + 328
   delistings, but `symbol_lineage_complete=False` by design until unresolved
   transitions are evidenced.  
   **Required:** evidence-backed closure (or explicit per-panel exclusion list)
   without guessing.

3. **Optional:** longer index history (currently ~360 sessions vs 764 equity
   sessions) — not the binding blocker once cache is loaded, but coverage is
   thinner than equity.

---

## What was deliberately NOT done

- No yfinance substitution for scientific PASS  
- No fabricated CA factors from gaps  
- No fabricated listing dates for undated delistings  
- No second SnapshotStore / FeatureStore / PitContract  
- No Phase B models / ensembles / RL / HRP / RAG  
- No Brain / risk / execution / live-signal changes  
- No Phase A.5 protocol edits  

---

## Code added / changed (activation spine)

- `data/security_identity.py`  
- `data/nse_ca_ingest.py`  
- `data/nse_universe_ingest.py`  
- `research/intelligence/data/research_grade_gate.py`  
- `research/intelligence/data/research_grade_activation.py`  
- `data/universe_history.py` (fail-closed research_grade)  
- `data/corporate_actions.py` (research_grade requires verify PASS)  
- Tests: `tests/test_security_identity.py`, `tests/test_nse_ca_ingest.py`,
  `tests/test_research_grade_gate.py`

---

## Next action after blockers clear

1. Complete CA ledger until `verify_ca_adjustment` PASS (`gap_rate ≤ 0.002`).  
2. Close or honestly scope identity lineage; set completeness only with evidence.  
3. Re-run `evaluate_research_grade` — earn stamp.  
4. Commit immutable Snapshot via existing `SnapshotStore`.  
5. Rerun **frozen** Phase A.5 protocols → write `PHASE_A5_RESEARCH_GRADE_RERUN.md`.

---

## STOP

**RESEARCH_GRADE was not earned. Phase A.5 scientific rerun did not run.
Phase B must not begin.**
