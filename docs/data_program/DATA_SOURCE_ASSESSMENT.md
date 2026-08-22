# Data source assessment (Phase II)

**Date:** 2026-08-22  
**Method:** Repository inventory first, then live probes of official NSE endpoints (egress unrestricted).  
**Rule:** Prefer official exchange/regulator files over aggregator websites.

| Source | Dataset | Historical coverage (probed) | Publication timestamp | Identifiers | Form | Access | License (visible) | PIT suitability | Ingest feasibility | Confidence | Chosen? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NSE `corporates-financial-results` API | Result events + XBRL links | At least 2015–2026; Jan-2025 window = 1212 quarterly rows | `broadCastDate` / `exchdisstime` (IST wall clock) | symbol, ISIN, seqNumber, xbrl URL | JSON list | Public; cookie warmup; ~1–3s/window | NSE website terms; research reuse of published filings | **PIT_STRONG** for announcement time | High — already wrapped in `data/nse_results_ingest.py` | High | **Yes — events + XBRL index** |
| NSE XBRL instance on `nsearchives.nseindia.com` | Statement metrics | Same as results list; sample parse OK | Filing time from parent results row, not fetch time | xbrl URL, ISIN | XML | Public HTTP; cacheable | NSE published XBRL | **PIT_STRONG** for mapped Ind-AS tags when parent `available_at` kept | High — `parse_xbrl_metrics` exists | High for revenue/PBT/PAT/EPS; degraded for ratios | **Yes — fundamentals** |
| NSE `EQUITY_L.csv` | Current EQ listing master | Current members only (2291 EQ + BE/BZ); official `DATE OF LISTING` + ISIN | Listing date (date only) | SYMBOL, ISIN | CSV | Public archives | NSE | **PIT_STRONG** for current names’ listing date; survivors-only without delist file | High — `data/security_identity.py` | High | **Yes — universe** |
| NSE `delisted.csv` | Official delistings | 328 rows; dates from 2002+ | Delisted date (date only) | Symbol | CSV | Public archives | NSE | **PIT_STRONG** delist date | High | High | **Yes — universe** |
| NSE `symbolchange.csv` | Symbol transitions | Multi-year; mixed MF + equity | Effective date | old/new symbol | CSV | Public archives | NSE | **PIT_STRONG** when parseable | High | Medium (format noise) | **Yes — identity** |
| NSE `ind_close_all_DDMMYYYY.csv` | Official index PR OHLC | Files exist at least 2015, 2017–2026 (2016-01-02 404 = holiday). Local store only 2024-04+ | Session date | Index Name | CSV | Public archives | NSE | **PIT_STRONG** price-return | High — same files as `index_store` | High | **Yes — extend benchmarks** |
| NSE corporate-announcements API | Text announcements | Recent windows | `an_dt` / `exchdisstime` | symbol | JSON | Public | NSE | PIT_STRONG time; unstructured type | Medium | Medium | Secondary for event type, not numbers |
| NSE corporates-corporateActions API | CA subjects | Already ingested 607 share-count | ex date | symbol | JSON | Public | NSE | PIT for dated events; factor only if subject unambiguous | Already used | High for bonus/split | Keep; no gap-inferred extras |
| BSE XBRL / filings | Cross-check metrics | Not probed this run | Filing time | scrip code | XML | Public | BSE | Could be PIT_STRONG | Extra identity map needed | Medium | **Sample validation only** |
| Screener.in / Yahoo `.info` | Live fundamentals | Current restated website | Fetch time only | ticker | HTML/JSON | Fragile / ToS | Not official | **UNUSABLE** historically | Already in `fundamentals/` | High that it is wrong for PIT | **No** |
| NIFTY500 comment + overlay sector map | Sector labels | Static 408 names | None | symbol | Code | In-repo | Internal | STATIC_BACKFILL | Done | High | Keep; do not upgrade PIT |
| Local bhav first/last | Membership proxy | 2019–2026 store | First print | symbol | pickle | Local | NSE bhav | PIT_DEGRADED | Done | High | Supplement only |

**Chosen primary sources:** NSE results API + XBRL; EQUITY_L + delisted + symbolchange; official `ind_close_all` archives.

**Rejected as historical truth:** Yahoo/Screener live cache; inferring CA factors from price gaps; projecting today’s sector backward as PIT.
