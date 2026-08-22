# QuantTerm data program — state

**As of:** 2026-08-22  
**Mandate:** DATA & EVIDENCE FOUNDATION  
**Firewall:** FEATURE-002 spec frozen; no EDGE-007; production ranking/orders untouched.

| Dataset | Status | Coverage | PIT class | Blocking issues | Next action |
|---|---|---|---|---|---|
| FEATURE-002 live_scan ledger | RESEARCH_READY (logging); experiment QUIET | 0 primary rows; empty `shadow.db`; no `scan_store.json` | Future-only after 2026-08-22 IST | No post-activation market-hours scan (Saturday) | Watchdog + E2E; wait for weekday scans |
| PIT fundamentals | RESEARCH_READY_WITH_LIMITATIONS (schema); DESCRIPTIVE_ONLY until ingest | Schema + as-of/restatement rules; no research-grade XBRL ledger on disk | Field-level: filing-dated rows can be PIT_STRONG; Yahoo/Screener live cache is UNUSABLE historically | No official XBRL archive in `logs/pit_fundamentals.json` | Offline ingest only; never Yahoo `.info` |
| Earnings / result events | RESEARCH_READY_WITH_LIMITATIONS (schema); DESCRIPTIVE_ONLY until ingest | Event timeline schema; no surprise/consensus series | Announcement `available_at` PIT_STRONG when from NSE broadcast; reconstructed calendars UNUSABLE | Empty `logs/pit_events.json` | Ingest NSE results offline; no EDGE test |
| Sector / industry map | RESEARCH_READY_WITH_LIMITATIONS | Static NIFTY500 comment + overlay map | STATIC_BACKFILL (not PIT_SECTOR_STRONG) | No dated official industry archive | Use as descriptive context only |
| Sector index context | RESEARCH_READY_WITH_LIMITATIONS | Official `ind_close_all` CSVs 2024-04-08 → 2026-08-21 | Price-return official; short history | No long TRI; not in production | Research context only |
| Corporate actions | RESEARCH_READY_WITH_LIMITATIONS | 607 share-count events (bonus/split/consol) from NSE corporates API | CA_RESEARCH_ACCEPTABLE for those types; not CA_COMPLETE | Rights/demerger/merger/symbol change unresolved; no gap-inferred factors | Keep verifier; quarantine unresolved |
| Universe / listing history | DESCRIPTIVE_ONLY | 2751 bhav-inferred rows; 264 inferred exits | PIT_DEGRADED — not `point_in_time_universe_v2` | First-seen ≠ official listing; listed dates start 2024-12-24 | Do not invent membership |
| Official benchmarks | RESEARCH_READY_WITH_LIMITATIONS | Local NSE index CSVs: Nifty 50, Nifty 500, Total Market, sectors; 586 sessions from 2024-04-08 | Price-return official; TRI mostly absent | Too short for full-sample 2019–2024 official bench | Label PR vs TR; no silent 0 fill |
| EvidenceSnapshot | RESEARCH_READY | Versioned offline API + manifest | As good as constituent ledgers | Historical research must not fetch | Use for all future experiments |
| Live Screener/Yahoo fundamentals | UNUSABLE (historical) | Live UI cache only | Not PIT | Restated current website | Never backtest |

Status vocabulary (exactly one per dataset): `RESEARCH_READY` · `RESEARCH_READY_WITH_LIMITATIONS` · `DESCRIPTIVE_ONLY` · `UNUSABLE`.
