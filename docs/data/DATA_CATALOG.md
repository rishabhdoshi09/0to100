# QuantTerm data catalog

Authoritative inventory. Status uses exactly one of:  
`RESEARCH_READY` · `RESEARCH_READY_WITH_LIMITATIONS` · `DESCRIPTIVE_ONLY` · `UNUSABLE`.

---

## NSE equity bhav (OHLCV)

- **Purpose:** Official EOD prices; primary historical tape.
- **Source:** NSE bhavcopy → `data/bhavcopy_store.py` (`logs/bhav/`).
- **Coverage:** ~3156 symbols, ~1787 sessions, 2019-08-23 → 2026-08-21 (this host).
- **Refresh:** Incremental official daily file. New sessions append; do not rewrite old bars.
- **PIT quality:** Strong for unadjusted prints. CA-adjusted reads depend on the CA ledger.
- **Versioning:** Store pickle / session files; snapshot manifest hashes the pickle.
- **Known defects:** Unadjusted splits/bonuses until CA apply-on-read; live overlay mutates *today* only.
- **Consumers:** Scanner, FEATURE-001/002, EDGE studies, EvidenceSnapshot.prices.
- **Status:** RESEARCH_READY_WITH_LIMITATIONS
- **Cadence:** Daily after EOD; never mid-backtest fetch.

## Corporate actions

- **Purpose:** Share-count continuity (split / bonus / consolidation).
- **Source:** `logs/ca_events.json` (nse_corporates_api), 607 events / 462 symbols.
- **Coverage:** bonus 306, split 300, consolidation 1. No rights/demerger/merger factors.
- **Refresh:** `ca-ingest` → new ledger version. Verifier unchanged.
- **PIT quality:** CA_RESEARCH_ACCEPTABLE for listed share-count types; **not** CA_COMPLETE.
- **Versioning:** File hash in evidence manifest.
- **Known defects:** Unresolved rights/demergers/mergers/symbol changes must be quarantined or segmented. Never infer factors from gaps.
- **Consumers:** `get_ohlcv` adjust-on-read, gauntlet validator, ca_research.
- **Status:** RESEARCH_READY_WITH_LIMITATIONS
- **Cadence:** After official CA file updates; keep prior hash for old experiments.

## Universe / listing history

- **Purpose:** Point-in-time membership.
- **Source:** `logs/universe_history.json` source=`bhav_inferred` (2751 rows, 264 inferred exits).
- **Coverage:** First-seen dates from local bhav (listed range starts 2024-12-24 in this file — not 2019 listing truth).
- **Refresh:** Re-infer from bhav or ingest an official archive (new file).
- **PIT quality:** PIT_DEGRADED. `point_in_time_universe_v2` **not** created.
- **Versioning:** File hash.
- **Known defects:** First appearance ≠ listing date; missing suspensions/symbol changes.
- **Consumers:** `point_in_time_universe`, EvidenceSnapshot.universe, EDGE studies.
- **Status:** DESCRIPTIVE_ONLY
- **Cadence:** When bhav coverage changes or an official listing file arrives.

## PIT fundamentals

- **Purpose:** Statement metrics knowable as of T.
- **Source:** NSE results API + XBRL (`data/nse_results_ingest.py`) → `logs/pit_fundamentals.json`.
- **Coverage:** **0 rows on disk.**
- **Refresh:** Ingest stage only; freeze snapshot; never fetch inside a backtest.
- **PIT quality:** Architecture PIT_STRONG for filing-dated XBRL fields; ledger empty.
- **Versioning:** `content_hash()`; restatements are new rows.
- **Known defects:** No cash-flow/EBITDA map; live Screener/Yahoo cache is a different, UNUSABLE dataset.
- **Consumers:** EvidenceSnapshot.fundamentals, pit_ratios, future (not current) experiments.
- **Status:** DESCRIPTIVE_ONLY (empty) / schema RESEARCH_READY_WITH_LIMITATIONS
- **Cadence:** After each official results season; new version file.

## Earnings / result events

- **Purpose:** Announcement timeline (not surprise).
- **Source:** Same NSE results broadcast times → `logs/pit_events.json`.
- **Coverage:** **0 rows on disk.**
- **Refresh:** Ingest then freeze.
- **PIT quality:** Broadcast timestamp can be PIT_STRONG; none loaded.
- **Versioning:** Event id + content hash.
- **Known defects:** No historical consensus series ⇒ no earnings surprise.
- **Consumers:** EvidenceSnapshot.earnings_events; post-result *infrastructure* only.
- **Status:** DESCRIPTIVE_ONLY
- **Cadence:** With fundamentals ingest.

## Sector / industry map

- **Purpose:** Descriptive sector labels + sector-index context.
- **Source:** NIFTY500 comment parser + large-cap overlay (`data/sector_map.py` v2). 408 mapped names.
- **Coverage:** Static modern map. Official `ind_close_all` sector indices 2024-04-08 → 2026-08-21.
- **Refresh:** New frozen snapshot file (`logs/sector_maps/…`); never rewrite an old freeze.
- **PIT quality:** STATIC_BACKFILL. Not PIT_SECTOR_STRONG.
- **Versioning:** `content_hash` + filename.
- **Known defects:** No dated industry archive; unmapped → UNKNOWN.
- **Consumers:** SEPA-003, FEATURE-002 sector field, EvidenceSnapshot.sector (research only).
- **Status:** RESEARCH_READY_WITH_LIMITATIONS
- **Cadence:** When the comment map is deliberately revised (new version).

## Official index / benchmarks

- **Purpose:** Comparable market and sector benchmarks.
- **Source:** Local NSE `logs/indices/ind_close_all_*.csv` (586 files, 2024-04-08 → 2026-08-21).
- **Coverage:** Nifty 50, Nifty 500, Nifty Total Market, sector indices present. Price-return unless name contains TR.
- **Refresh:** Download missing official daily files in the *ingest* stage only (`data.index_store` build is production; research uses `data.benchmarks` offline).
- **PIT quality:** Official PR from 2024-04-08. No deep 2019 official Nifty. No cash TRI.
- **Versioning:** `files_hash` of local CSV names+sizes.
- **Known defects:** Short history; TR names in the file are mostly futures/leverage, not Nifty 50 TRI.
- **Consumers:** EvidenceSnapshot.benchmark, EDGE Nifty comparisons (label PR vs TR).
- **Status:** RESEARCH_READY_WITH_LIMITATIONS
- **Cadence:** Daily official file; do not treat missing return as 0.

## FEATURE-002 shadow ledger

- **Purpose:** Future-only rank-feature observations.
- **Source:** Production auto_scan hook → `logs/feature002/shadow.db`.
- **Coverage:** 0 primary live_scan rows (expected 2026-08-22 Saturday).
- **Refresh:** Each post-activation market-hours scan (first write wins per symbol/session).
- **PIT quality:** Future protocol; implementation_test excluded from primary stats.
- **Versioning:** `feature-002.v1` / protocol hash.
- **Known defects:** Daemon thread can lose an in-flight cycle on process exit.
- **Consumers:** FEATURE-002 evaluator only. Not production rank.
- **Status:** RESEARCH_READY (logging path) / experiment QUIET
- **Cadence:** Every production scan after `_save_state`; watchdog after hook.

## Live Screener / Yahoo fundamentals

- **Purpose:** UI / conviction colour, not history.
- **Source:** `fundamentals/fetcher.py`, `scan/conviction.py`.
- **Coverage:** Current website.
- **Refresh:** Live cache.
- **PIT quality:** UNUSABLE historically.
- **Versioning:** none for research.
- **Known defects:** Restated current values, no filing date.
- **Consumers:** Production UI only.
- **Status:** UNUSABLE (research)
- **Cadence:** n/a for historical experiments.

## Intelligence Snapshot (CSV bars)

- **Purpose:** Older committed bar snapshots for the intelligence runtime.
- **Source:** `research/intelligence/data/snapshot.py`.
- **Coverage:** Whatever was committed into a snapshot directory.
- **Refresh:** New snapshot id; never mutate old CSVs.
- **PIT quality:** As good as the committed bars.
- **Consumers:** Intelligence runtime. Complementary to EvidenceSnapshot.
- **Status:** RESEARCH_READY_WITH_LIMITATIONS
