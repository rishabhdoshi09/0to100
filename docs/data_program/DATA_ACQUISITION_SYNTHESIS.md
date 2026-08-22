# Phase II acquisition synthesis

**Date:** 2026-08-22  
**Branch:** `cursor/data-acquire-942f`  
**Stop:** Phase II populated the empty official ledgers. Remaining holes are depth, not missing APIs.  
**Not started:** EDGE-007, any strategy book, FEATURE-002 spec change, Phase III ingest.

Production firewall held.

---

## What changed versus Phase I

| Dataset | Phase I | Phase II |
|---|---|---|
| PIT fundamentals | 0 rows | 19,151 / 2,057 symbols |
| Result events | 0 | 137,212 / 2,489 symbols |
| Official index PR | from 2024-04 | Nifty 50/500 from 2015-11-09 |
| Sector map | 408 STATIC_BACKFILL | 845 STATIC_BACKFILL |
| Universe | bhav-inferred only | official v2 + bhav default + identity graph |
| FEATURE-002 | 0 primary; watchdog | same + acceptance evaluator / ops states |

## Dataset classes (honest)

- Fundamentals: **RESEARCH_READY_WITH_LIMITATIONS** (official chronology; uneven years; bounded XBRL).
- Events: **RESEARCH_READY_WITH_LIMITATIONS** (broadcast times; no surprise).
- Universe: **PIT_DEGRADED** default; official v2 **RESEARCH_READY_WITH_LIMITATIONS** (324 omitted delists).
- Sector: **STATIC_BACKFILL** (coverage ≠ PIT).
- CA: **CA_RESEARCH_ACCEPTABLE**, not complete.
- Benchmarks: **RESEARCH_READY_WITH_LIMITATIONS** (PR yes, TRI no).
- FEATURE-002 ops: **NO_POST_ACTIVATION_SCAN** + insufficient new data.

## Event policy

Unknown time → `EVENT_DATE_ONLY` → causal **NEXT_SESSION**. This dump is timestamp-strong.

## FEATURE-002

Unchanged. First genuine weekday production scan is still the operational checkpoint. `evaluate_first_real_scan()` will not invent a primary row.

## Newly feasible (backlog only)

See `docs/research_program/FUTURE_DATA_ENABLED_HYPOTHESES.md`. Not tested here.

## Still blocked

- Earnings surprise (no consensus).
- Complete 2019–2024 dead-name survivorship.
- PIT sector rotation.
- Cash TRI vs PR.
- Rights/demerger-adjusted continuous prices.
- FEATURE-002 graduation (no post-activation scan).

---

## Recommendation (exactly one)

**PHASE II COMPLETE — do not start strategy research.**

The empty-ledger bottleneck is gone. The next useful data mandate (not started) would deepen XBRL years and official dead-name listing dates. Do not start EDGE-007. Do not retune FEATURE-002.
