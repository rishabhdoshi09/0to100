# Data foundation synthesis

**Date:** 2026-08-22  
**Stop:** STOP A — data foundation materially complete *as architecture*. Ledgers that require official offline ingest remain empty (STOP B adjacent, not fatal).  
**Not started:** the recommended next phase.

Production firewall held. FEATURE-002 specification unchanged. No EDGE-007.

---

## Answers

### 1. Why was the FEATURE-002 live ledger initially empty?

Protocol activation is `2026-08-22T00:00:00+05:30`. This host had (and has) no `logs/scan_store.json` and a `shadow.db` with **zero** rows. Saturday, NSE closed, no post-activation market-hours scan. Friday 21 Aug, if scanned, would be before protocol IST and could only be `implementation_test` or refused. **Expected, not a rank-spec bug.**

### 2. Is FEATURE-002 now collecting valid future rows?

**Not yet — and that is still expected today.** Logging path is verified end-to-end (production `_scan_once_locked` → candidate set → immutable row → persist → reload → resolver; snapshot frozen; no orders). Watchdog classifies empty primary as `no_post_activation_production_scan` (not a bug). First valid primary rows require a weekday market-hours production scan after protocol IST.

A session-date bug (hist as-of used as `event_id`) was fixed so those future rows will not collide with Friday’s bar date.

### 3. What percentage of NSE names have research-grade PIT fundamentals?

**0%.** `logs/pit_fundamentals.json` is absent. Schema and as-of/restatement rules exist. Live Yahoo/Screener cache is UNUSABLE historically and was not treated as coverage.

### 4. How many years / quarters are covered?

**0 years / 0 quarters** on disk. Ingest was not run (network stage is separate).

### 5. Are filing timestamps trustworthy?

**The contract is.** A row without `available_at` (broadcast/filing) is rejected; `fetched_at` cannot be mapped in. Tests: a 3 Nov filing is invisible on 2 Nov; a later restatement cannot rewrite the earlier known value. **No archive is loaded**, so there are no timestamps to trust in production data yet.

### 6. Can historical earnings events now be studied causally?

**Not yet.** Timeline schema + session class (before/during/after market) + “no surprise without consensus” are in place. `logs/pit_events.json` is empty. Post-result *infrastructure* exists; no EDGE experiment was launched.

### 7. What is sector coverage?

**408** mapped names (NIFTY500 comments + documented large-cap overlay). Unmapped stays UNKNOWN. Official sector **index** history is on disk from 2024-04-08 (research context only).

### 8. Is sector history PIT or static?

**STATIC_BACKFILL.** Not `PIT_SECTOR_STRONG`. Frozen snapshots are immutable; a later rebuild writes a new file.

### 9. Is CA research acceptability materially improved?

**Bounded, not completed.** 607 official share-count events remain `CA_RESEARCH_ACCEPTABLE`. Rights / demergers / mergers / symbol changes stay quarantined or segmented. The global verifier is unchanged. `CA_COMPLETE` is still false. No gap-inferred factors.

### 10. Is the universe still PIT_DEGRADED?

**Yes.** 2751 bhav-inferred rows. `point_in_time_universe_v2` was **not** created. First-seen ≠ official listing.

### 11. Which official benchmarks are now available?

Local NSE `ind_close_all` CSVs (586 sessions, **2024-04-08 → 2026-08-21**), offline:

- **Price-return:** Nifty 50, Nifty 500, Nifty Total Market, Nifty 100, sector indices (Bank/IT/Pharma/FMCG/Auto/Metal/Energy/Realty), India VIX, Smallcap 100.
- **Total-return names in the same files:** mostly futures/leverage TR, **not** a cash Nifty 50 TRI.

Missing official return is UNKNOWN, never 0. Do not compare a dividend-excluding book to TRI without saying so.

### 12. Can a historical experiment be fully replayed from an immutable evidence snapshot?

**Yes, for what is on disk.** `EvidenceSnapshot(as_of=…)` reads versioned local ledgers under a network guard. The manifest hashes price store, CA, universe, fundamentals, events, sector map, benchmarks, code SHA, and experiment config. Changing the config hash changes `snapshot_hash`. Empty fundamentals/events replay as empty — honestly.

### 13. Which future research families are now newly feasible?

- FEATURE-002 **forward** rank validation (once weekday scans run).
- Short-window (2024-04+) official **Nifty 500 / sector-index context** (descriptive or as a labelled PR benchmark).
- Post-result studies **after** NSE results/XBRL ingest (growth / result-strength, not surprise).
- PIT fundamental screens **after** the same ingest, fail-closed on missing filings.

### 14. Which remain blocked?

- Any fundamental or earnings-surprise book on this host today (empty ledgers; no consensus).
- Survivorship-complete 2019–2024 membership (`PIT_DEGRADED`).
- Long official Nifty / TRI from 2019.
- PIT sector rotation (static map only).
- Further CS Top-20 mining on the same bhav store (prior STOP A / STOP F still apply).

### 15. Is the next bottleneck still data, or can strategy research responsibly resume?

**Still data — specifically unfilled official archives — plus forward FEATURE-002 observation.** Strategy search on the existing bhav tape already exhausted its budget and found no robust tradable edge. The system can now *run better research* once ledgers are ingested; it cannot honestly resume hypothesis mining on the same empty fundamentals/earnings/universe gaps.

---

## Recommendation (exactly one)

**CONTINUE DATA FOUNDATION**

Do not resume hypothesis research. Do not start EDGE-007. Do not retune FEATURE-002.

Highest-value remaining work (future mandate, not this one):

1. Offline NSE financial-results + XBRL ingest into versioned PIT ledgers.
2. Official listing/delisting archive if licensable (only then consider universe v2).
3. Let weekday production scans fill FEATURE-002; keep the watchdog on.

This mandate improved **what the system can know, when it knew it, and the evidence trail.** It did not manufacture a backtest result.
