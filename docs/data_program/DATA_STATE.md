# QuantTerm data program — state

**As of:** 2026-08-22  
**Mandate:** DATA & EVIDENCE FOUNDATION — synthesis complete  
**Firewall:** FEATURE-002 spec frozen; no EDGE-007; production ranking/orders untouched.

| Dataset | Status | Coverage | PIT class | Blocking issues | Next action |
|---|---|---|---|---|---|
| FEATURE-002 live_scan ledger | RESEARCH_READY (logging); experiment QUIET | 0 primary rows; E2E path verified; no `scan_store.json` | Future-only after 2026-08-22 IST | No post-activation market-hours scan (Saturday) | Watchdog; wait for weekday scans |
| PIT fundamentals | DESCRIPTIVE_ONLY (empty); schema RESEARCH_READY_WITH_LIMITATIONS | 0 symbols / 0 quarters | Filing-dated XBRL can be PIT_STRONG; Yahoo live UNUSABLE | No `pit_fundamentals.json` | Offline NSE XBRL ingest (future) |
| Earnings / result events | DESCRIPTIVE_ONLY (empty) | 0 events; no consensus series | Broadcast `available_at` PIT_STRONG when ingested | Empty `pit_events.json` | Same ingest; never call it surprise |
| Sector / industry map | RESEARCH_READY_WITH_LIMITATIONS | 408 STATIC_BACKFILL names | STATIC_BACKFILL | No dated official industry archive | Descriptive use only |
| Sector index context | RESEARCH_READY_WITH_LIMITATIONS | Official CSVs 2024-04-08 → 2026-08-21 | Price-return official | Short history; not in production | Research context only |
| Corporate actions | RESEARCH_READY_WITH_LIMITATIONS | 607 share-count events / 462 symbols | CA_RESEARCH_ACCEPTABLE; not CA_COMPLETE | Rights/demerger/merger/symbol change unresolved | Keep verifier; quarantine |
| Universe / listing history | DESCRIPTIVE_ONLY | 2751 bhav-inferred; 264 inferred exits | PIT_DEGRADED — no v2 | First-seen ≠ listing date | Official archive or stay degraded |
| Official benchmarks | RESEARCH_READY_WITH_LIMITATIONS | Nifty 50/500/Total Market + sectors; 586 sessions from 2024-04-08 | Price-return official; TRI absent | Too short for 2019–2024 official bench | Label PR vs TR; no 0-fill |
| EvidenceSnapshot | RESEARCH_READY | Versioned offline API + manifest | As good as constituent ledgers | Empty fund/event ledgers replay empty | Use for all future experiments |
| Live Screener/Yahoo fundamentals | UNUSABLE (historical) | Live UI cache | Not PIT | Restated current website | Never backtest |

Status vocabulary (exactly one per dataset): `RESEARCH_READY` · `RESEARCH_READY_WITH_LIMITATIONS` · `DESCRIPTIVE_ONLY` · `UNUSABLE`.
