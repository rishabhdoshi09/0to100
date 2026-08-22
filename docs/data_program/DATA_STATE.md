# QuantTerm data program — state

**As of:** 2026-08-22 (Phase II start)  
**Branch:** `cursor/data-acquire-942f` off `cursor/data-foundation-942f` @ `6c6fe45`  
**Mandate:** DATA ACQUISITION & PIT ENRICHMENT — Phase II  
**Firewall:** FEATURE-002 spec frozen; no EDGE-007; no strategy metrics during ingest.

## Starting state (Phase I synthesis, unchanged until ingest lands)

| Dataset | Status | Coverage | PIT class | Blocking issues | Next action |
|---|---|---|---|---|---|
| FEATURE-002 live_scan ledger | RESEARCH_READY (logging); experiment QUIET | 0 primary rows; E2E path verified | Future-only after 2026-08-22 IST | No post-activation market-hours scan | Acceptance evaluator; wait for genuine scan |
| PIT fundamentals | DESCRIPTIVE_ONLY (empty); schema RESEARCH_READY_WITH_LIMITATIONS | 0 symbols / 0 quarters | Filing-dated XBRL can be PIT_STRONG | No `pit_fundamentals.json` | **Ingest NSE results + XBRL** |
| Earnings / result events | DESCRIPTIVE_ONLY (empty) | 0 events; no consensus series | Broadcast `available_at` PIT_STRONG when ingested | Empty `pit_events.json` | **Ingest NSE results list** |
| Sector / industry map | RESEARCH_READY_WITH_LIMITATIONS | 408 STATIC_BACKFILL names | STATIC_BACKFILL | No dated official industry archive | Broaden coverage; do not upgrade PIT class |
| Sector index context | RESEARCH_READY_WITH_LIMITATIONS | Official CSVs 2024-04-08 → 2026-08-21 | Price-return official | Short history | Extend official index files backward |
| Corporate actions | RESEARCH_READY_WITH_LIMITATIONS | 607 share-count events / 462 symbols | CA_RESEARCH_ACCEPTABLE; not CA_COMPLETE | Rights/demerger/merger/symbol change unresolved | Classify remaining; no gap-inferred factors |
| Universe / listing history | DESCRIPTIVE_ONLY | 2751 bhav-inferred; 264 inferred exits | PIT_DEGRADED — no v2 | First-seen ≠ listing date | EQUITY_L + delisted + symbolchange |
| Official benchmarks | RESEARCH_READY_WITH_LIMITATIONS | Nifty 50/500/Total Market + sectors; 586 sessions from 2024-04-08 | Price-return official; TRI absent | Too short for 2019–2024 official bench | Older official index files if obtainable |
| EvidenceSnapshot | RESEARCH_READY | Versioned offline API + manifest | As good as constituent ledgers | Empty fund/event ledgers | Bind hashes after ingest |
| Live Screener/Yahoo fundamentals | UNUSABLE (historical) | Live UI cache | Not PIT | Restated current website | Never backtest |

Phase II goal: replace empty DESCRIPTIVE_ONLY rows with real official coverage where the network and licenses allow. Empty schema is no longer success.
