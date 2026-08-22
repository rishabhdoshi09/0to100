# QuantTerm data program — state

**As of:** 2026-08-22 (Phase II end)  
**Branch:** `cursor/data-acquire-942f`  
**Mandate:** DATA ACQUISITION & PIT ENRICHMENT — Phase II  
**Firewall:** FEATURE-002 spec frozen; no EDGE-007; no strategy metrics during ingest.

## Ending state

| Dataset | Status | Coverage | PIT class | Blocking issues | Next action |
|---|---|---|---|---|---|
| FEATURE-002 live_scan ledger | RESEARCH_READY (logging); experiment QUIET | 0 primary rows | Future-only after 2026-08-22 IST | No post-activation market-hours scan | Call `evaluate_first_real_scan()` after a genuine weekday scan |
| PIT fundamentals | **RESEARCH_READY_WITH_LIMITATIONS** | 19151 rows / 2057 symbols; median 9 quarters; 2019-04-10 → 2025-06-17 | FUNDAMENTAL_PIT_STRONG when OneD XBRL + broadcast date | Thin 2020–21; 32 names with ≥12 quarters; no CFO/op. profit; 34k XBRL not downloaded | Optional deeper XBRL years |
| Earnings / result events | **RESEARCH_READY_WITH_LIMITATIONS** | 137212 events / 2489 symbols; 2016-01-08 → 2026-08-06 | EVENT_TIMESTAMP_STRONG (100% this dump) | No consensus; some names absent (TATAMOTORS) | DATE_ONLY → NEXT_SESSION if time ever missing |
| Sector / industry map | RESEARCH_READY_WITH_LIMITATIONS | **845** STATIC_BACKFILL (752 official current industry) | STATIC_BACKFILL | No dated reclass archive; UNKNOWN still dominant vs all bhav | Do not upgrade PIT class |
| Sector index context | RESEARCH_READY_WITH_LIMITATIONS | Official CSVs **2015-11-09 → 2026-08-21** (Nifty 50/500) | Price-return official | No cash TRI | Keep PR/TRI separate |
| Corporate actions | RESEARCH_READY_WITH_LIMITATIONS | 607 share-count / 462 symbols | CA_RESEARCH_ACCEPTABLE; not CA_COMPLETE | Rights/demerger/merger unresolved | Segment; no inferred factors |
| Universe / listing history | DESCRIPTIVE_ONLY (default bhav) + limited official v2 | 3156 bhav-inferred default; 2293 official v2; 324 omitted delists | PIT_DEGRADED | Official listing dates for current EQ only | Keep v2 opt-in; do not manufacture complete survivorship |
| Official benchmarks | RESEARCH_READY_WITH_LIMITATIONS | Nifty 50/500: 2660 sessions from 2015-11-09; TM from 2016-07-07 | Price-return official; TRI absent | No TRI | Research pickle bound in snapshot |
| EvidenceSnapshot | RESEARCH_READY | Accessors + provenance + network ban + freshness overlay | As good as constituent ledgers | — | Bind hashes per experiment |
| Live Screener/Yahoo fundamentals | UNUSABLE (historical) | Live UI cache | Not PIT | Restated current website | Never backtest |

Phase II replaced empty DESCRIPTIVE_ONLY fund/event ledgers with official coverage. Empty schema is no longer the bottleneck. Remaining holes are **depth** (older XBRL, dead-name listing dates, TRI, CA completeness), not missing APIs.
