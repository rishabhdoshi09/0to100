# Next Research Execution Readiness

> Pre-execution audit against certified snapshot `a7a9828ec37e09e4`. Global trust remains `OPERATIONAL_ONLY`.

## Snapshot

- snapshot_id: `a7a9828ec37e09e4`
- scoped_certification: `READY_FOR_SCIENTIFIC_RERUN`
- trust_class: `OPERATIONAL_ONLY`
- equity_sha256: `9fd4550df76a23fdecd199058f49c1d17eda8d716d64bbef93b78e5f500ffc20`
- n_symbols: 29
- n_sessions: 764
- date_range: 2023-08-23→2026-08-11

## Temporal partitions (frozen before outcomes)

- Warmup / τ-fit: ≤ `2024-07-31`
- Discovery OOS: `2024-08-01` → `2025-07-31`
- Confirmation OOS: `2025-08-01` → panel end (untouched until discovery known)

## Per-experiment matrix

| EXPERIMENT | FIELDS | UNIVERSE | DATES | IDENTITY | CA | INDEX | VOLUME | PIT | COSTS | STATUS | BLOCKERS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-NEXT-01 | adjusted close | FIXED_PREREGISTERED_29 | 2023-08-23→2026-08-11 | VERIFIED (scoped cert) | VERIFIED for panel consecutive jumps | NOT_REQUIRED (panel-relative CS) | NOT_REQUIRED | PitContract on scoped snapshot | CNC round_trip_cost_pct | **READY** | — |
| EXP-NEXT-02 | adjusted close → realized vol | FIXED_PREREGISTERED_29 | 2023-08-23→2026-08-11 | VERIFIED (scoped cert) | VERIFIED for panel consecutive jumps | NOT_REQUIRED | NOT_REQUIRED | PitContract on scoped snapshot | CNC round_trip_cost_pct | **READY** | — |
| EXP-NEXT-03 | adjusted close → vol10/vol60 | FIXED_PREREGISTERED_29 | 2023-08-23→2026-08-11 | VERIFIED (scoped cert) | VERIFIED for panel consecutive jumps | NOT_REQUIRED | NOT_REQUIRED | PitContract on scoped snapshot | CNC round_trip_cost_pct | **READY** | — |

## Classification

- EXP-NEXT-01: **READY**
- EXP-NEXT-02: **READY**
- EXP-NEXT-03: **READY**

No experiment blocked. Proceeding with all three.

_Written at 2026-08-11T17:14:47.779634+00:00_
