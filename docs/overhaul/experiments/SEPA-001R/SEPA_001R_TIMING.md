# SEPA-001R VCP timing diagnostics

Old detector = frozen SEPA-001 (pattern-high pivot + 92% near-pivot VCP fail).  
New detector = causal last-contraction pivot; distance-to-pivot is **entry state**, not a VCP fail.  
NSE rows use official bhavcopy with share-count CA applied on read. Every date is an as-of slice.

| Symbol | Setup Start | First Knowable VCP | Pivot Knowable | Breakout | Old Detection | Old Dist. to Pivot | New Detection | New Dist. |
|---|---|---|---|---|---|---|---|---|
| PLANTED_TIGHT | 2020-12-30 | 2021-02-12 | 2021-02-02 | 2021-02-16 | 2021-02-12 | -4.902 | 2021-02-12 | -3.897 |
| PLANTED_TWO | 2020-12-30 | 2021-02-12 | 2021-02-02 | 2021-02-18 | 2021-02-12 | -3.724 | 2021-02-12 | -3.195 |
| PLANTED_EXTENDED | 2020-12-30 | 2021-02-12 | 2021-02-02 | 2021-02-16 | 2021-02-12 | -4.902 | 2021-02-12 | -3.897 |
| GRIND_NO_VCP |  |  |  |  |  |  |  |  |
| CHENNPETRO | 2025-07-07 | 2025-09-01 | 2025-08-22 | 2025-10-20 | 2025-12-15 | 15.556 | 2025-12-15 | 36.298 |
| LAURUSLABS | 2025-06-16 | 2025-09-03 | 2025-09-01 | 2025-10-20 | 2025-12-02 | 10.591 | 2025-12-02 | 16.395 |
| MOTHERSON | 2025-05-27 | 2025-07-09 | 2025-07-08 | 2025-10-20 | 2025-11-06 | -3.926 | 2025-11-06 | -0.205 |
| SBIN | 2025-04-30 | 2025-09-10 | 2025-08-22 | 2025-10-20 | 2025-10-20 | 7.715 | 2025-10-20 | 8.294 |
| RELIANCE | 2025-07-09 | 2025-11-12 | 2025-11-04 | 2025-10-20 | 2025-12-02 | -0.303 | 2025-12-02 | 2.519 |
| TCS | 2025-05-29 | 2025-08-25 | 2025-08-22 | 2025-11-12 |  |  | 2025-11-14 | -0.767 |

## What this shows

- Planted coils become knowable **before** the breakout date. Pivot knowable date ≥ pivot extreme date (confirmation, not back-dating).
- **MOTHERSON**: last-contraction pivot puts first detection at **−0.21%** (inside the 1.5% band). The old pattern-high pivot was **−3.9%** (still below the zone).
- **TCS**: new detector finds a coil the legacy 92% rule missed; still below the buy-zone (`ENTRY_BELOW_PIVOT`), not a chase.
- **CHENNPETRO / LAURUSLABS**: first *structural* print in the walk is already extended. The setup **lifecycle** still found a later `ENTRY_READY` date (CHENNPETRO 2026-03-20, LAURUSLABS 2026-03-10) without widening the zone.
- Across 944 unique NSE setups, median distance at first snapshot is still **+10.7%** (75% already >1.5%). That is now labelled `EXTENDED`, not filled.

Primary evidence: `tests/test_sepa_001r.py` (causality) + `setups.jsonl` (944 unique bases).
