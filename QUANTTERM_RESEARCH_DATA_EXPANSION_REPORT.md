# QuantTerm Research Data Expansion Report

> End-to-end **data breadth/depth** cycle only. No Phase B. No AI/ML. No new
> strategies. No reopening of rejected hypotheses. No EXP-NEXT-02 retest executed.
> Global historical trust remains **`OPERATIONAL_ONLY`**. Production Brain / risk /
> execution unchanged.

---

## Plain English — What QuantTerm can now research honestly

QuantTerm used to have a carefully checked group of **29** large stocks over about
**three years**. That was enough for some tests, but too small for others
(especially low-volatility, which stayed inconclusive).

This cycle did **not** invent missing history. It:

1. Pulled **more official NSE daily files** going back to **2020**
2. Checked which stocks have trustworthy identity, listing history, and
   corporate-action handling
3. Built a **larger certified research group** of **870** liquid stocks
4. Saved that group as an **immutable snapshot** so future tests can repeat exactly

**What this means for a normal user:** QuantTerm can now run serious
price-history research on a much broader, longer stock set — without pretending
the entire 3,000+ name database is fully certified, and without guessing splits
or ticker changes.

**What it does *not* mean:** The live trading system did not change. Old failed
ideas were not revived. Company fundamentals and earnings still cannot be used
for honest historical tests until QuantTerm records **when** those facts first
became public (`AVAILABLE_AT`).

---

## 1. Starting data state

| Item | Before this cycle | Notes |
|------|-------------------|--------|
| Global trust | `OPERATIONAL_ONLY` | Unchanged after expansion |
| Prior scoped panel | 29 names | Snapshot `a7a9828ec37e09e4` |
| Prior date span | ~2023-08-23 → 2026-08-11 | ~764 sessions in store |
| Store symbols | ~2,853 EQ | Operational bhav cache |
| CA ledger | 378 events / 317 symbols | Official-only; never invents |
| Identity ledger | ~2,449 securities | NSE EQUITY_L + symbolchange + delisted |
| Universe ledger | 2,121 rows | Listings/delistings; research_grade=False |
| PIT sector history | Missing | Static map only |
| Fundamentals PIT | Missing | As-of-now caches |

**Verified prior snapshot (required before reliance):**

- Path: `logs/phase_a5_scoped/snapshots/a7a9828ec37e09e4/`
- `SnapshotStore.verify_snapshot` → **True**
- Manifest: 29 instruments, 22,156 equity bars, dates `2023-08-23`→`2026-08-11`
- `scoped_certification=READY_FOR_SCIENTIFIC_RERUN`, `trust_class=OPERATIONAL_ONLY`

---

## 2. Existing 29-name certification

Unchanged and still valid for its **original** frozen window:

| Field | Value |
|-------|--------|
| Snapshot | `a7a9828ec37e09e4` |
| Certification | `READY_FOR_SCIENTIFIC_RERUN` (scoped) |
| Global trust | `OPERATIONAL_ONLY` |
| Use | Prior Phase A.5 / A.6 / next-cycle experiments |

On the **expanded 2020–2026** window, **27/29** remain in the new CERTIFIABLE set.
**TATASTEEL** and **BAJAJFINSV** become `BLOCKED_CA` because each has one
consecutive-session ~−89.5% jump (2022-07-28 and 2022-09-13) with **no official
parseable adjustment factor** in the CA ledger. Those names are **not removed
from the global database**; they are simply excluded from the expanded
research-eligible scope until official CA evidence is ingested.

After **2022-09-15**, all 29 are again CA-clean on consecutive-session audit
(~1,056 sessions / ~49 monthly rebalances) — relevant for an identical-protocol
low-vol retest path (not executed here).

---

## 3. Expanded certified universe

| Metric | Value |
|--------|--------|
| Classification window | 2020-01-01 → 2026-08-11 |
| Store after expansion | **3,130** symbols · **1,900** sessions |
| **CERTIFIABLE** | **870** |
| PARTIAL | 608 |
| Immutable snapshot | **`2f683be0c73eaa33`** |
| Scoped status | `SCOPED_RESEARCH_READY` |
| Global trust | `OPERATIONAL_ONLY` (not upgraded) |
| Median names / session | 816.5 |
| Security-sessions | 1,497,079 |
| Parent snapshot | `a7a9828ec37e09e4` |
| Index context in snapshot | Nifty 50 + India VIX · 2020-01-01→2026-08-11 (1,642 bars each) |

**CERTIFIABLE gates (evidence-only):**

- ISIN + `security_id` in identity ledger; mid-window `resolve_as_of` OK
- Universe membership with known listing date
- ≥1,000 sessions; still trading into 2026
- Median volume ≥ 50,000
- **Zero** unresolved consecutive-session discontinuities (≤3 calendar days)
- CA ratios **never** inferred from price jumps

Code: `research/data_expansion/` (`classify.py`, `snapshot.py`, `assess.py`, `run_expansion.py`).

Committed machine summary: `docs/overhaul/RESEARCH_DATA_EXPANSION_SUMMARY.json`.

---

## 4. Securities excluded and why

| Class | Count | Meaning (user) | Technical |
|-------|------:|----------------|-----------|
| BLOCKED_IDENTITY | 1,036 | Older/other records cannot be linked confidently | Mostly missing identity ledger rows (survivors not in EQUITY_L master / undated) |
| INSUFFICIENT_HISTORY | 362 | Too little price history for serious tests | `<400` sessions |
| BLOCKED_CA | 116 | Large price breaks without official adjustment proof | Unresolved consecutive jumps |
| OTHER | 137 | Clean enough on paper but too illiquid / thin | CA-clean but below liquidity gates |
| BLOCKED_UNIVERSE | 1 | Listing membership cannot be defended | Missing/invalid universe row |
| PARTIAL | 608 | Usable with caution; thinner than CERTIFIABLE | ≥400 sessions, lower liquidity, or shorter depth |

**Explicit preferable exclusions** (examples): TATASTEEL, BAJAJFINSV on full 2020+ window until official split/demerger factors are filed into `logs/ca_events.json`.

Hard names were **not** deleted from the global bhav store to decorate statistics.

---

## 5. Historical range

| Source | Trustworthy range established | Sessions |
|--------|-------------------------------|----------|
| NSE equity bhavcopy (official archives) | **2020-01-01 → 2026-08-11** | **1,900** |
| NSE index OHLC (Nifty 50 / India VIX) | **2020-01-01 → 2026-08-11** | **1,642** |

History was extended by downloading missing official `sec_bhavdata_full_*.csv` /
`ind_close_all_*.csv` files and rebuilding local stores. One corrupt equity day
(`2022-08-08`) failed parse and was skipped (not fabricated).

**Do not assume older prices are valid merely because files exist** — certification
still requires identity + CA + universe gates per security.

**Independent calendar years covered:** 2020–2026 (COVID crash/recovery, 2021
strength, 2022 rate-hike risk-off, 2023 consolidation, 2024 election/mid-cycle,
2025–2026 recent tape).

---

## 6. CA coverage

| Item | Status |
|------|--------|
| Ledger events | 378 events / 317 symbols (pre-existing official ingest) |
| Policy | `ADJUSTMENT_POLICY_VERSION` on-read; raw bhav immutable |
| Invention | **Forbidden** — gaps stay unresolved |
| Expanded panel CA gate | 0 unresolved consecutive among 870 CERTIFIABLE |
| Global CA | Still **not** research-grade for all 3,130 names |
| Notable blockers | TATASTEEL (2022-07-28), BAJAJFINSV (2022-09-13) ~−89.5% without ledger factor |
| NSE live CA API for 2022 ranges | Returned empty in this environment — cannot auto-heal |

**Next CA investment:** official filings for unresolved consecutive names (todo /
ingest path), especially the two prior-29 blockers.

---

## 7. Identity coverage

| Item | Status |
|------|--------|
| Ledger | ~2,449 securities; source `nse_equity_l+nse_symbolchange+nse_delisted` |
| Lineage flag | `symbol_lineage_complete=True` at ledger level |
| Dual-ISIN guessing | Not done |
| CERTIFIABLE requirement | ISIN + security_id + mid-window resolve |
| Blocked | 1,036 store symbols lack defendable identity rows |

User: *"Some older stock records cannot yet be linked confidently, so QuantTerm
will leave them out of serious historical tests."*  
Technical: `BLOCKED_IDENTITY` / `PARTIAL_IDENTITY_COVERAGE` for the global store.

---

## 8. Universe coverage

| Item | Status |
|------|--------|
| Ledger | 2,121 rows; survivorship_complete=True at ledger metadata |
| research_grade | False (global) |
| Delisted omitted (no listed date) | 327 (honest omission) |
| CERTIFIABLE requirement | Membership row + known listed date |
| Mode | Explicit **research-eligible scope**, not “whole NSE is certified” |

---

## 9. Sector-history readiness

**`NOT_RESEARCH_READY`**

- Today's static sector map ≠ historical truth
- No dated PIT sector membership ledger in-repo
- Does **not** block OHLCV-only research

User: *"QuantTerm knows today's sector labels, but not a trustworthy year-by-year
sector history."*

---

## 10. Fundamental-data readiness

| Dataset | Status | AVAILABLE_AT |
|---------|--------|--------------|
| Financial statements | OPERATIONAL_ONLY | MISSING |
| Profitability | OPERATIONAL_ONLY | MISSING |
| Valuation | OPERATIONAL_ONLY | MISSING |
| Shareholding / ownership | OPERATIONAL_ONLY | MISSING |

PitContract continues to refuse current fundamentals for research reads.

---

## 11. Earnings / event-data readiness

| Dataset | Status | AVAILABLE_AT |
|---------|--------|--------------|
| Earnings / results | MISSING | MISSING |
| Earnings dates | MISSING | MISSING |
| Corporate announcements | PARTIAL | PARTIAL |
| Reported-vs-available timestamps | MISSING | MISSING |

Key requirement for future event/fundamental factors: **`AVAILABLE_AT`**, not
merely reporting period.

---

## 12. Immutable snapshot IDs

| Snapshot | Role |
|----------|------|
| `a7a9828ec37e09e4` | Prior 29-name scoped cert (verified; retained) |
| `0e51fa13587372c9` | Intermediate expanded equity snapshot (shallow index attach) |
| **`2f683be0c73eaa33`** | **Canonical expanded CERTIFIABLE snapshot** (full Nifty/VIX) |

Store root: `logs/research_expansion/snapshots/` via existing
`research.intelligence.data.snapshot_store.SnapshotStore` (no duplicate store).

Manifest preserves: snapshot id, security set, date range, source/CA/identity/
universe hashes, adjustment policy, validator version, git provenance,
`scoped_certification`, `global_trust_class=OPERATIONAL_ONLY`.

---

## 13. Validation results

| Check | Result |
|-------|--------|
| Prior snapshot `a7a9828ec37e09e4` verify | PASS |
| Expanded snapshot `2f683be0c73eaa33` verify | PASS |
| Unresolved consecutive CA in CERTIFIABLE set | 0 (by construction) |
| Global RESEARCH_GRADE upgrade | **Not claimed** |
| Unit tests `tests/test_research_data_expansion.py` | PASS |
| Production modules modified | **None** (Brain/signals/risk/execution untouched) |

---

## 14. Research-power improvement

| Metric | Prior 29 panel | Expanded CERTIFIABLE | Gain |
|--------|----------------|----------------------|------|
| Securities | 29 | **870** | ~30× |
| Sessions | ~764 | **1,900** | ~2.5× |
| Security-years (approx) | ~88 | **~5,941** | **~68×** |
| CS breadth | 29 | ~817 median / 870 max | large |
| Calendar years | ~3 | **7** | more regimes |

This is **not** a profitable backtest — it is sample-size / regime coverage for
future honest tests.

---

## 15. Low-vol retest readiness

**`LOW_VOL_RETEST_READY`**

EXP-NEXT-02 remains **`INCONCLUSIVE` / `HOLD_NO_TUNING`**. **Not rerun** in this task.

Frozen items preserved: hypothesis, vol definition (20d), hold/rebalance (21),
costs (CNC), metrics/thresholds/success criteria.

| Path | Ready? | Notes |
|------|--------|-------|
| A — identical 29 after 2022-09-15 | Yes | ~49 rebalances vs prior ~13; CA-clean |
| B — expanded 870 CERTIFIABLE | Yes | ~89 rebalances; **new experiment id** if universe changes |

Full 2020+ window for exact 29 still blocked by TATASTEEL / BAJAJFINSV CA gaps.

---

## 16. Newly testable hypothesis families

| Family | Class |
|--------|-------|
| Cross-sectional **price-only** factors on expanded OHLCV (incl. low-vol retest) | **READY_TO_TEST** |

---

## 17. Still-blocked hypothesis families

| Family | Class |
|--------|-------|
| Momentum / short-horizon reversal / dynamic structure / network alpha / network×concentration / logistic challenger / vol-compression | **CLOSED_REJECTED** (do not reopen) |
| Value / quality / profitability / earnings growth | **DATA_MISSING** (`AVAILABLE_AT`) |
| Post-earnings drift / event reactions | **DATA_MISSING** / PARTIAL |
| Sector-neutral designs needing PIT sectors | **PIT_UNSAFE** |
| Ownership / shareholding effects | **DATA_MISSING** |

---

## 18. Recommended data investments (priority)

1. Official CA closure for unresolved consecutive names (start: TATASTEEL, BAJAJFINSV)
2. Identity coverage for store symbols missing EQUITY_L / ISIN rows (without guessing lineage)
3. PIT **AVAILABLE_AT** fundamentals + earnings/announcement timestamps
4. PIT sector membership history (only if sector-neutral tests are prioritized)
5. Keep extending official bhav/index archives as NSE continues to publish

---

## 19. What NOT to build

- New indicators / strategies / Phase B AI
- Random Forest, XGBoost, GBM, neural nets, ensembles, RL
- New clustering / network / interaction mining to “rescue” failures
- Production HRP / live ranking / Brain / risk / execution changes
- Fabricated CA ratios or guessed symbol lineage
- Global `RESEARCH_GRADE` claims for the full store

---

## 20. Production behaviour confirmation

| Surface | Status |
|---------|--------|
| Brain / posture | Unchanged |
| Live ranking / signals | Unchanged |
| Risk limits / position sizing | Unchanged |
| Execution / broker / autopilot | Unchanged |
| Research-only additions | `research/data_expansion/*`, report, tests, docs summary |
| Local data artifacts | Extended `logs/bhav/`, `logs/indices/`, expansion snapshots (gitignored `logs/`) |

---

## Status card

| Field | Value |
|-------|--------|
| **CURRENT CERTIFIED UNIVERSE** | 870 liquid NSE names (`SCOPED_RESEARCH_READY`); prior 29 snapshot retained |
| **CURRENT DATE RANGE** | 2020-01-01 → 2026-08-11 (1,900 equity sessions) |
| **SECURITY-YEARS** | ~5,941 (vs ~88 prior) |
| **DATA QUALITY** | Global `OPERATIONAL_ONLY`; expanded scope certified; sector/fundamentals not PIT-ready |
| **LOW-VOL RETEST READY?** | **YES** (`LOW_VOL_RETEST_READY`) — not executed |
| **NEXT DATA PRIORITY** | Official CA factors for unresolved consecutive names + `AVAILABLE_AT` fundamentals/events |
| **NEXT SCIENTIFIC ACTION** | Preregister and run **frozen** EXP-NEXT-02 retest on Path A (29 post-2022-09-15) or a **new-id** expanded-panel low-vol protocol — no tuning, no ML |

---

## Reproducibility

```bash
# Rebuild classification + snapshot (uses local official bhav/index; network only if archives missing)
.venv/bin/python -m research.data_expansion.run_expansion

# Unit tests (network-free)
.venv/bin/python -m pytest tests/test_research_data_expansion.py -q
```

Canonical expanded snapshot id: **`2f683be0c73eaa33`**  
Prior certified snapshot id: **`a7a9828ec37e09e4`**
