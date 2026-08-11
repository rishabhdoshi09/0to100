# RESEARCH_GRADE Data Audit

**Status:** AUDIT ONLY — no implementation in this deliverable  
**Date:** 2026-08-11  
**Branch:** `cursor/institutional-ai-audit-80a2`  
**Authority:** Phase A / A.5 complete; production behaviour must remain unchanged  

**Scientific blocker:** data quality / missing ledgers — **not** architecture or model complexity.

---

## In plain English

**Conclusion:** QuantTerm already has the tools to build trustworthy historical
tests. It does **not** yet have the trustworthy historical files on disk.

**Why:** Official day-by-day NSE prices, corporate-action records, and
“who was listed when” membership history are missing in this environment.

**What it means:** Research results so far are exploration-only. They must not
steer real-money strategy promotion.

**What to do:** After approval, load the missing ledgers through the **existing**
data path, run the quality checks, earn a research-quality stamp, then rerun the
same frozen Phase A.5 tests. Do not invent missing history. Do not change live
trading behaviour.

**Data quality today:** Good enough for charts and exploration, not for proving
a strategy (`DISPLAY_ONLY` exploratory panel only).

**Research status:** UNPROVEN until a research-quality snapshot is earned.

<details>
<summary>Technical details</summary>

Missing on disk: `logs/bhav/`, `logs/index/`, `logs/ca_events.json`,
`logs/universe_history.json`, `logs/snapshots/`. Reuse existing
`bhavcopy_store`, `SnapshotStore`, `PitContract`, `corporate_actions`,
`universe_history`, validators — do **not** create parallel stores.
See scoreboard below. Presentation contract: `COMMON_MAN_READABILITY.md`.

</details>

---

## Required end-state (target pipeline)

```text
RAW NSE HISTORY
+ POINT-IN-TIME UNIVERSE LEDGER
+ CORPORATE-ACTION LEDGER
+ SECURITY IDENTITY HISTORY
        ↓
PIT-CONFORMANT RESEARCH VIEW  (PitContract / Snapshot.bars through=as_of)
        ↓
DATA VALIDATION               (gauntlet/validator, data_integrity, classify_tier)
        ↓
RESEARCH_GRADE GATE           (earned, never manually asserted)
        ↓
IMMUTABLE HASHED SNAPSHOT     (existing SnapshotStore)
        ↓
FROZEN PHASE A.5 RERUN        (protocols below — unchanged)
```

---

## Phase A.5 protocol freeze (confirmed)

The following hypothesis IDs are **FROZEN**. Do not alter hypotheses, horizons,
benchmarks, transaction-cost assumptions, success criteria, or multiple-testing
treatment based on DISPLAY_ONLY exploratory results.

| Hypothesis ID | Experiment | Status at freeze |
|---------------|------------|------------------|
| `81b8889792f53113` | EXP-A5-01 Market structure | REJECTED (failed `research_grade==1`) |
| `590571a11ee06fc2` | EXP-A6-01 Network risk | REJECTED |
| `775b4a0fce7d5b83` | EXP-A2-01 Horizons (`5d/10d/22d/66d`) | REJECTED |
| `7842a46ee335685a` | EXP-A3-01 Logistic challenger | REJECTED |
| `3734b8a0a9124a60` | EXP-A5A6-01 Interactions | REJECTED |

Committed freeze artifact (full protocol dump; `*.json` is gitignored):

`PHASE_A5_FROZEN_PROTOCOLS.md`

Local working copy also at `docs/overhaul/PHASE_A5_FROZEN_PROTOCOLS.json` (gitignored).  
Source DB (gitignored): `logs/phase_a5/experiments.db`  
Seed: `42` · code_hash: `phase_a5` for all five.

**Rerun rule:** replace only *data provenance* (snapshot id, trust class, research_grade
earned by validation). Keep success criteria / protocol / hypotheses identical.

---

## Capability scoreboard (12 areas)

| # | Capability | Classification | Primary files / symbols | Notes |
|---|------------|----------------|-------------------------|-------|
| 1 | Historical NSE/bhav ingestion | **EXISTS** (code) / data **MISSING** | `data/bhavcopy_store.build_store`, `build_from_local`, `get_ohlcv`, `iter_raw_frames`; `data/bhavcopy_runtime`; `research/intelligence/data/from_bhav.py`; CLI via `main.py` | Default `logs/bhav/`, pickle `store_cache.pkl`, `DEFAULT_DAYS=500`. Store stays **raw**; CA applied on read. **No `logs/bhav` in this environment.** |
| 2 | SnapshotStore + manifests | **EXISTS** (+ **DUPLICATE** non-research snapshot types) | `research/intelligence/data/snapshot_store.SnapshotStore`, `snapshot.Snapshot` | Content-addressed; default `logs/snapshots/` (**absent**). Phase A.5 uses same class under `logs/phase_a5/snapshots/` (DISPLAY_ONLY). Duplicates in *other domains*: `execution/reconciliation/snapshot_store.BrokerSnapshotStore`, `options/eod_snapshot.py` — do **not** reuse for market research. |
| 3 | PitContract | **EXISTS** | `research/intelligence/data/pit_contract.PitContract` | `history/latest/as_of/coverage`; refuses network; fundamentals/sectors → `NOT_PIT_SAFE`; missing universe ledger → `NOT_PIT_SAFE` (no survivor READY). |
| 4 | universe_history / PIT universe | **PARTIAL** | `data/universe_history.py`; `data/nse_universe.point_in_time_universe` | Schema + CLI exist. `bhav_inferred` → `research_grade=False`. Without file: **today’s survivors** + `survivorship_complete=False` (**UNSAFE** if treated as READY). File **absent**. |
| 5 | Stable security identity / symbol changes | **PARTIAL** / lineage **MISSING** | Kite `canonical_id` (ISIN) in `research/intelligence/data/kite_source.py`; `data/dead_symbols.py`; `data_platform/security_master.py` | Bhav keyed by **current trading symbol**, not ISIN. No bitemporal rename/merger master. Product readiness flag `symbol_lineage_complete` expected → currently blocker. |
| 6 | Listing / delisting handling | **PARTIAL** | universe ledger rows `{symbol,listed,delisted?}`; momentum_breakout IPO/terminal gap counts | Code path EXISTS; official archive data **MISSING**. Bar-inferred listing ≠ research-grade membership. |
| 7 | Corporate-action ingestion / `ca_events.json` | **EXISTS** (code) / data **MISSING** | `data/corporate_actions.load_events`, `write_events`, `ingest_from_path`, `export_gap_todo`; CLI `ca-ingest` | Default `logs/ca_events.json`. Types: `split\|bonus\|consolidation` (dividends rejected). Never invents from gaps. File **absent**. |
| 8 | `adjust_frame` + adjustment policy | **EXISTS** | `data/corporate_actions.adjust_frame`; `docs/overhaul/data_acquisition/CORPORATE_ACTION_POLICY.md`; `core.data_integrity.verify_ca_adjustment` | On-read only; raw immutable. PASS verify needs events loaded **and** gap_rate ≤ 0.002. Without ledger → RAW (honest). |
| 9 | Sector membership history | **MISSING** / **UNSAFE** for PIT | `scan/sector_heat.py` (static NIFTY groups); PitContract `sectors` → `NOT_PIT_SAFE`; `BhavDataProvider.sector_ctx` → `None` | Phase A.5 used static `sector_map.json`. Dated sector history not in-repo. EXP-006 already treats as optional/`SECTOR_MEMBERSHIP_NOT_PIT`. |
| 10 | Data-quality validation | **EXISTS** | `gauntlet/validator.validate`; `core/data_integrity.*`; `research/intelligence/data_state.classify_tier`; `research/momentum_breakout/dataset.data_quality_report`; `product/institutional_readiness.py` | E4 abort-on-fail pattern for gauntlet. Tier gate needs CA coverage ≥0.9 + universe history + adjustment_consistent + benchmark. |
| 11 | RESEARCH_GRADE vs DISPLAY_ONLY gating | **PARTIAL** | `docs/overhaul/DATA_CLASSIFICATION.md`; `research/phase_a5/metrics.gate_research_grade`; parallel tiers in `data_state` | Docs define trust classes. **No Python `TrustClass` enum / `@requires_trust` yet.** Phase A.5 forces `research_grade=False` for yfinance. Risk: universe `research_grade` can be **label-stamped** by source string — must tighten to validation-earned for activation milestone. |
| 12 | Historical path used by research harness / A.5 / gauntlet | **EXISTS** (paths differ) | See § below | Harness itself is stats-only (no I/O). |

---

## Exact historical data paths by consumer

| Consumer | Data path today |
|----------|-----------------|
| `research/harness.py` | **No market I/O** — consumes R arrays only |
| Phase A.5 (`research/phase_a5/`) | `logs/phase_a5/exploratory_closes.csv` (yfinance) → optional Snapshot under `logs/phase_a5/snapshots/`; **DISPLAY_ONLY** |
| Gauntlet | `data.bhavcopy_store` + `data.index_store` (when present); momentum CLI may use `--source yf` → DISPLAY_ONLY |
| EXP-006 / momentum_breakout | `BhavDataProvider` → bhav `get_ohlcv` + index `^NSEI` + `point_in_time_universe` + `load_events` |
| PitContract research reads | Bound `Snapshot` (+ optional ledgers for universe/CA/valuations) |
| Live scanner / production | Separate path (Kite/NSE live) — **out of scope; must not change** |

---

## On-disk inventory (this environment)

| Path | Expected role | Present? |
|------|---------------|----------|
| `logs/bhav/*.csv` + `store_cache.pkl` | RAW NSE equity EOD | **NO** |
| `logs/index/` | RAW NSE index EOD | **NO** |
| `logs/ca_events.json` | CA ledger | **NO** |
| `logs/universe_history.json` | Membership ledger | **NO** |
| `logs/snapshots/` | Default research SnapshotStore | **NO** |
| `logs/dead_symbols.json` | Operational skip list | **NO** |
| `logs/phase_a5/*` | Exploratory DISPLAY_ONLY artifacts | **YES** (gitignored) |
| `PHASE_A5_FROZEN_PROTOCOLS.md` | Frozen protocol export (tracked) | **YES** (this audit) |
| `docs/overhaul/PHASE_A5_FROZEN_PROTOCOLS.json` | Same dump (gitignored `*.json`) | **YES** (local) |

Prior discovery (`docs/overhaul/data_acquisition/discovery_report.json`): **NO** usable PIT NSE research dataset; NSE archive fetch previously unreachable in that environment.

---

## What can already be reused (do not rebuild)

1. `data/bhavcopy_store` — sole equity EOD ingestion  
2. `data/index_store` — Nifty / VIX / sector index OHLC  
3. `data/corporate_actions` + `adjust_frame` — CA ledger + on-read adjustment  
4. `data/universe_history` + `point_in_time_universe` — membership ledger API  
5. `SnapshotStore` / `Snapshot` / `SnapshotBarProvider` — immutable research snapshots  
6. `PitContract` — unified PIT read facade  
7. `from_bhav` — bhav → Snapshot bridge  
8. `gauntlet/validator` + `data_integrity` + `data_state.classify_tier` — validation / tiering  
9. Phase A.5 runners — swap data provenance only; protocols frozen  
10. `gate_research_grade` — promotion hard gate  

**Explicit non-builds:** second SnapshotStore, FeatureStore, PitContract, parallel bhav ingest, yfinance scientific path.

---

## What must be sourced externally

| Asset | Source | Cannot fabricate |
|-------|--------|------------------|
| Daily NSE equity bhav (`sec_bhavdata_full`) | NSE archives (`nsearchives.nseindia.com`) or operator ZIP | No |
| Index EOD (`ind_close_all` / Nifty) | NSE archives / existing `index_store` fetch | No |
| Corporate actions (split/bonus/consolidation with factors + ex-dates) | NSE filings / operator-curated ledger (gap TODO from prices is **hints only**) | **Never invent factors from gaps** |
| Listing / delisting membership | NSE listing/delisting archive or operator research-grade ledger | **Never from today’s survivors** |
| Symbol change / ISIN lineage (renames, mergers) | Official master / operator table | Unknown stays unknown |
| Dated sector membership | **Not available in-repo** — remain `NOT_PIT_SAFE` / optional limitation | No fake history |

yfinance / Google: **DISPLAY_ONLY only** — forbidden for scientific PASS.

---

## Minimum dataset for RESEARCH_GRADE (Phase A.5 rerun)

Aligned with `MINIMUM_DATASET_CONTRACT.md` + Phase A.5 needs:

1. **≥ ~500 trading sessions** of RAW NSE EQ OHLCV in `logs/bhav/` (matches `DEFAULT_DAYS`; prefer longer if available).  
2. **Nifty 50 (`^NSEI`) benchmark** history covering the same window (`logs/index/`).  
3. **`logs/ca_events.json`** with research-usable share-count events; `verify_ca_adjustment` must PASS (gap_rate ≤ 0.002 on sample).  
4. **`logs/universe_history.json`** with **non-`bhav_inferred`** source label **and** validation that membership is not survivor-only; `survivorship_complete=True`, `research_grade=True` only after deterministic checks.  
5. **Security identity policy documented for the snapshot window** — at minimum: no silent rename joins; missing lineage → symbols excluded or run limited with explicit `symbol_lineage_complete=False` blocker if renames affect the universe.  
6. **Immutable Snapshot** committed via existing `SnapshotStore` from bhav (+ index), with manifest fields earned by validation:
   - `trust_class="RESEARCH_GRADE"`
   - `research_grade=true` (only after gate)
   - `has_universe_history=true`
   - `adjustment_consistent=true`
   - `corporate_action_coverage≥0.9`
   - content hashes / `snapshot_id`
7. **PitContract coverage** on that snapshot returns research-eligible tier (not `NOT_PIT_SAFE` / `OPERATIONAL_ONLY`).  
8. **Sector history:** remain unavailable → Phase A.5 structure baselines that use static sectors must keep `SECTOR_MEMBERSHIP_NOT_PIT` limitation (already true); do not pretend PIT sectors.

Optional (mark unavailable): dated fundamentals, delivery if missing columns — never fabricate.

---

## How RESEARCH_GRADE must be earned (not manually set)

Deterministic gate (proposed activation sequence — **not implemented yet**):

1. Raw bhav + index present and hashable  
2. CA ledger loaded; `verify_ca_adjustment` PASS  
3. Universe ledger present; source ∉ `{bhav_inferred, bhav_*}`; row schema valid; survivor-fallback **not** used  
4. `classify_tier(health)` ≥ `RESEARCH_ELIGIBLE`  
5. `gate_research_grade(manifest)` true only if `trust_class=="RESEARCH_GRADE"` **and** validation report PASS  
6. Snapshot `extra_manifest` stamped from validation report — **writers must refuse** stamping RESEARCH_GRADE if any check fails  

Today’s gap: universe `research_grade` can flip true from a non-`bhav_*` source **label** alone (`is_research_grade_source`). Activation work must require the validation bundle, not the label alone.

---

## Expected disk / storage (order-of-magnitude)

| Component | Estimate |
|-----------|----------|
| ~500 session equity CSVs (`logs/bhav/`) | ~0.5–2 GB uncompressed CSVs (full EQ universe/day); pickle cache typically hundreds of MB |
| Index store | small (tens of MB) |
| `ca_events.json` | small (KB–MB) |
| `universe_history.json` | small (MB) |
| Research Snapshot (subset used for A.5, ~30–200 names × 500–1000 days) | tens of MB |
| Full-universe multi-year Snapshot | potentially 1–5+ GB depending on width/depth |

Exact size depends on universe width (all EQ vs liquid subset). Phase A.5 frozen protocols used ~29 names; RESEARCH_GRADE rerun may keep a **certified liquid subset** if full-universe CA/membership incomplete — but subset membership itself must be PIT-honest.

---

## Exact implementation sequence (for approval — do not execute yet)

1. **Freeze check** — confirm `PHASE_A5_FROZEN_PROTOCOLS.md` matches DB ids (done in this audit).  
2. **Acquire RAW bhav + index** — network NSE fetch via existing `build_store` / `build_index_store`, **or** operator ZIP through existing `data_setup.safe_extract_zip` / `build_from_local`.  
3. **Acquire / ingest CA ledger** — operator CSV/JSON → `ca-ingest`; run `--verify`; never invent factors.  
4. **Acquire / ingest universe ledger** — official/operator listing–delisting → `universe-history`; **forbid** promoting `bhav_inferred` to RESEARCH_GRADE.  
5. **Identity policy** — document rename handling; exclude unmapped renames; do not invent lineage.  
6. **Build PIT research view** — `from_bhav` → existing `SnapshotStore` (default `logs/snapshots/` or dedicated research root — **same class**, not a new store type).  
7. **Validate** — `gauntlet.validator.validate` + `data_integrity` + `classify_tier` + PitContract `coverage`.  
8. **Earn RESEARCH_GRADE stamp** — only if all checks PASS; write validation report beside snapshot.  
9. **Wire Phase A.5 loader** to refuse DISPLAY_ONLY for promotion path; load RESEARCH_GRADE snapshot closes/panel via PitContract.  
10. **Rerun frozen protocols** — same criteria; new data_window provenance; record new results without editing frozen hypotheses.  
11. **Update evidence report** — append research-grade rerun section; production still unchanged.

---

## Blockers

| Blocker | Severity |
|---------|----------|
| No `logs/bhav` / index history in environment | **P0** |
| No `logs/ca_events.json` | **P0** |
| No research-grade `logs/universe_history.json` | **P0** |
| NSE archive network may be restricted (prior discovery: unreachable) | **P0** — needs allowlist or operator USB/ZIP |
| No ISIN/rename lineage for bhav symbols | **P1** — limits clean multi-year joins |
| No dated sector membership | **P1** — keep NOT_PIT_SAFE; limitation flag |
| TrustClass enum / fail-closed `@requires_trust` not implemented | **P1** — Phase A.5 gate helps but is not global |
| Universe `research_grade` label can be set without full validation bundle | **P1** — must fix during activation |
| Raw per-file SHA planned but not fully enforced in SnapshotStore extra fields | **P2** — content-addressed snapshot id exists; extend provenance |

---

## Unsafe patterns to refuse during activation

1. Reconstructing historical universe from today’s survivors and calling it RESEARCH_GRADE.  
2. Inventing CA factors from price gaps.  
3. Stamping `trust_class=RESEARCH_GRADE` without validation PASS.  
4. Using yfinance / Google for scientific verdicts.  
5. Mutating raw bhav CSVs in place.  
6. Changing frozen Phase A.5 success criteria after seeing DISPLAY_ONLY results.  
7. Building a second SnapshotStore / PitContract / FeatureStore.

---

## Production behaviour

**No production behaviour needs to change** for RESEARCH_GRADE data activation.

- Live quote waterfall, scanner, Brain, portfolio gates, broker/`place_trade` remain untouched.  
- Activation is research-path only: ledgers + SnapshotStore + PitContract + Phase A.5 rerun.  
- Advisory structure/network modules stay non-authoritative until a RESEARCH_GRADE rerun yields PASS_*.

---

## Audit conclusion

QuantTerm already has the **code spine** for research-grade data. It does **not** have the **ledgers and raw history** on disk. The scientific next step is acquisition + deterministic certification through existing validators, then a frozen Phase A.5 rerun.

**STOP. Awaiting approval before implementation.**
