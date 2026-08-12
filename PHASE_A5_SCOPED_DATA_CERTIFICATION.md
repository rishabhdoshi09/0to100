# Phase A.5 Scoped Data Certification

> Scope-specific fitness for the frozen Phase A.5 protocol. **Not** a global RESEARCH_GRADE stamp.

## Common-man explanation

QuantTerm's full historical database is not yet certified for scientific research. However, we separately checked the exact historical data needed for this specific frozen test. The specific data used by this test passed the required historical checks, so the test can now be rerun scientifically.

Global trust stays OPERATIONAL_ONLY. Await approval before Phase A.5 rerun.

---

## 1. Global trust state

- **Global trust:** `OPERATIONAL_ONLY`
- This remains unchanged. Unresolved global names (ABFRL demerger, ETF unit splits, unmatched CA factors) are **not** excluded from the main dataset to make global quality look better.
- **Scoped certification:** `READY_FOR_SCIENTIFIC_RERUN`
- Phase A.5 rerun executed: `False`
- Phase B started: `False`

## 2. Exact frozen panel

- Date range: `2023-08-11` → `2026-08-11`
- N securities: **29**
- Hypothesis IDs: `81b8889792f53113, 590571a11ee06fc2, 775b4a0fce7d5b83, 7842a46ee335685a, 3734b8a0a9124a60`
- Protocol version: `PHASE_A5_FROZEN_PROTOCOLS@2026-08-11`
- Validator version: `phase_a5_scoped_certification.v1`
- Adjustment policy: `ca_sharecount_v1`
- Git SHA: `f5c2715d80bda409dfd065f2f7d6d7e7d2d54e17`

| # | Symbol | security_id |
|--:|--------|-------------|
| 1 | RELIANCE | `isin:INE002A01018` |
| 2 | ONGC | `isin:INE213A01029` |
| 3 | BPCL | `isin:INE029A01011` |
| 4 | TCS | `isin:INE467B01029` |
| 5 | INFY | `isin:INE009A01021` |
| 6 | WIPRO | `isin:INE075A01022` |
| 7 | HCLTECH | `isin:INE860A01027` |
| 8 | HDFCBANK | `isin:INE040A01034` |
| 9 | ICICIBANK | `isin:INE090A01021` |
| 10 | SBIN | `isin:INE062A01020` |
| 11 | KOTAKBANK | `isin:INE237A01036` |
| 12 | AXISBANK | `isin:INE238A01034` |
| 13 | ITC | `isin:INE154A01025` |
| 14 | HINDUNILVR | `isin:INE030A01027` |
| 15 | NESTLEIND | `isin:INE239A01024` |
| 16 | SUNPHARMA | `isin:INE044A01036` |
| 17 | DRREDDY | `isin:INE089A01031` |
| 18 | CIPLA | `isin:INE059A01026` |
| 19 | M&M | `isin:INE101A01026` |
| 20 | MARUTI | `isin:INE585B01010` |
| 21 | TATASTEEL | `isin:INE081A01020` |
| 22 | JSWSTEEL | `isin:INE019A01038` |
| 23 | HINDALCO | `isin:INE038A01020` |
| 24 | NTPC | `isin:INE733E01010` |
| 25 | POWERGRID | `isin:INE752E01010` |
| 26 | LT | `isin:INE018A01030` |
| 27 | ADANIENT | `isin:INE423A01024` |
| 28 | BAJFINANCE | `isin:INE296A01032` |
| 29 | BAJAJFINSV | `isin:INE918I01026` |

## 3. Exact protocol dependencies (S1)

| EXPERIMENT | REQUIRED DATA ASSET | REQUIRED DATE RANGE | REQUIRED SECURITIES | STATUS | BLOCKER |
|---|---|---|---|---|---|
| EXP-A5-01 | adjusted_close_panel | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | corporate_actions | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | security_identity | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | listing_delisting_window | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | universe_membership_history | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A5-01 | sector | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | index_vix | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A5-01 | symbol_lineage | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | features | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | benchmark | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5-01 | transaction_costs | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | adjusted_close_panel | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | corporate_actions | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | security_identity | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | listing_delisting_window | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | universe_membership_history | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A6-01 | sector | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | index_vix | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A6-01 | symbol_lineage | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | features | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | benchmark | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A6-01 | transaction_costs | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | adjusted_close_panel | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | corporate_actions | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | security_identity | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | listing_delisting_window | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | universe_membership_history | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A2-01 | sector | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A2-01 | index_vix | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A2-01 | symbol_lineage | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | features | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | benchmark | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A2-01 | transaction_costs | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | adjusted_close_panel | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | corporate_actions | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | security_identity | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | listing_delisting_window | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | universe_membership_history | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A3-01 | sector | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A3-01 | index_vix | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A3-01 | symbol_lineage | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | features | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | benchmark | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A3-01 | transaction_costs | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | adjusted_close_panel | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | corporate_actions | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | security_identity | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | listing_delisting_window | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | universe_membership_history | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A5A6-01 | sector | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | index_vix | 2023-08-11→2026-08-11 | FROZEN_29 | NOT_REQUIRED | — |
| EXP-A5A6-01 | symbol_lineage | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | features | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | benchmark | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |
| EXP-A5A6-01 | transaction_costs | 2023-08-11→2026-08-11 | FROZEN_29 | VERIFIED | — |

## 4. Security identity verification (S2)

- Panel identity OK: `True`
- Blockers: `[]`

| Symbol | security_id | ISIN | listing | delisting | identity | lineage |
|---|---|---|---|---|---|---|
| RELIANCE | `isin:INE002A01018` | `INE002A01018` | 1995-11-29 | — | VERIFIED | NOT_APPLICABLE |
| ONGC | `isin:INE213A01029` | `INE213A01029` | 1995-07-19 | — | VERIFIED | NOT_APPLICABLE |
| BPCL | `isin:INE029A01011` | `INE029A01011` | 1995-09-13 | — | VERIFIED | NOT_APPLICABLE |
| TCS | `isin:INE467B01029` | `INE467B01029` | 2004-08-25 | — | VERIFIED | NOT_APPLICABLE |
| INFY | `isin:INE009A01021` | `INE009A01021` | 1995-02-08 | — | VERIFIED | VERIFIED |
| WIPRO | `isin:INE075A01022` | `INE075A01022` | 1995-11-08 | — | VERIFIED | NOT_APPLICABLE |
| HCLTECH | `isin:INE860A01027` | `INE860A01027` | 2000-01-06 | — | VERIFIED | NOT_APPLICABLE |
| HDFCBANK | `isin:INE040A01034` | `INE040A01034` | 1995-11-08 | — | VERIFIED | NOT_APPLICABLE |
| ICICIBANK | `isin:INE090A01021` | `INE090A01021` | 1997-09-17 | — | VERIFIED | NOT_APPLICABLE |
| SBIN | `isin:INE062A01020` | `INE062A01020` | 1995-03-01 | — | VERIFIED | NOT_APPLICABLE |
| KOTAKBANK | `isin:INE237A01036` | `INE237A01036` | 1995-12-20 | — | VERIFIED | VERIFIED |
| AXISBANK | `isin:INE238A01034` | `INE238A01034` | 1998-11-16 | — | VERIFIED | VERIFIED |
| ITC | `isin:INE154A01025` | `INE154A01025` | 1995-08-23 | — | VERIFIED | NOT_APPLICABLE |
| HINDUNILVR | `isin:INE030A01027` | `INE030A01027` | 1995-07-06 | — | VERIFIED | VERIFIED |
| NESTLEIND | `isin:INE239A01024` | `INE239A01024` | 2023-08-01 | — | VERIFIED | NOT_APPLICABLE |
| SUNPHARMA | `isin:INE044A01036` | `INE044A01036` | 1995-02-08 | — | VERIFIED | NOT_APPLICABLE |
| DRREDDY | `isin:INE089A01031` | `INE089A01031` | 2003-05-30 | — | VERIFIED | NOT_APPLICABLE |
| CIPLA | `isin:INE059A01026` | `INE059A01026` | 1995-02-08 | — | VERIFIED | NOT_APPLICABLE |
| M&M | `isin:INE101A01026` | `INE101A01026` | 1996-01-03 | — | VERIFIED | NOT_APPLICABLE |
| MARUTI | `isin:INE585B01010` | `INE585B01010` | 2003-07-09 | — | VERIFIED | NOT_APPLICABLE |
| TATASTEEL | `isin:INE081A01020` | `INE081A01020` | 1998-11-18 | — | VERIFIED | VERIFIED |
| JSWSTEEL | `isin:INE019A01038` | `INE019A01038` | 2005-03-23 | — | VERIFIED | VERIFIED |
| HINDALCO | `isin:INE038A01020` | `INE038A01020` | 1997-01-08 | — | VERIFIED | VERIFIED |
| NTPC | `isin:INE733E01010` | `INE733E01010` | 2004-11-05 | — | VERIFIED | NOT_APPLICABLE |
| POWERGRID | `isin:INE752E01010` | `INE752E01010` | 2007-10-05 | — | VERIFIED | NOT_APPLICABLE |
| LT | `isin:INE018A01030` | `INE018A01030` | 2004-06-23 | — | VERIFIED | NOT_APPLICABLE |
| ADANIENT | `isin:INE423A01024` | `INE423A01024` | 1997-06-04 | — | VERIFIED | VERIFIED |
| BAJFINANCE | `isin:INE296A01032` | `INE296A01032` | 2003-04-01 | — | VERIFIED | VERIFIED |
| BAJAJFINSV | `isin:INE918I01026` | `INE918I01026` | 2008-05-26 | — | VERIFIED | NOT_APPLICABLE |

## 5. Corporate-action verification (S3)

- Panel CA OK: `True`
- Consecutive large-move events in window: `9`
- Verified CA transitions: `9`
- Unresolved consecutive: `0`
- Adjustment policy: `ca_sharecount_v1`

| security_id | symbol | type | source | ex-date | factor | verification | source_hash |
|---|---|---|---|---|---|---|---|
| `isin:INE002A01018` | RELIANCE | bonus | nse_ca | 2024-10-28 | 2.0 | VERIFIED | `3d52e222910ea2f9` |
| `isin:INE029A01011` | BPCL | bonus | nse_ca | 2024-06-21 | 2.0 | VERIFIED | `34a1709abe5868ce` |
| `isin:INE075A01022` | WIPRO | bonus | nse_ca | 2024-12-03 | 2.0 | VERIFIED | `d1550b181cbc0062` |
| `isin:INE040A01034` | HDFCBANK | bonus | nse_ca | 2025-08-26 | 2.0 | VERIFIED | `b172f9371bef1992` |
| `isin:INE237A01036` | KOTAKBANK | split | nse_ca | 2026-01-14 | 5.0 | VERIFIED | `9e65cc241296f2f9` |
| `isin:INE239A01024` | NESTLEIND | split | nse_ca | 2024-01-05 | 10.0 | VERIFIED | `f3837bc82beaf2a1` |
| `isin:INE239A01024` | NESTLEIND | bonus | nse_ca | 2025-08-08 | 2.0 | VERIFIED | `905dd623bc0f8a44` |
| `isin:INE089A01031` | DRREDDY | split | nse_ca | 2024-10-28 | 5.0 | VERIFIED | `1f3e30fe364eb13f` |
| `isin:INE752E01010` | POWERGRID | bonus | nse_ca | 2023-09-12 | 1.3333333333333333 | NO_LARGE_JUMP | `6b13cf15ff45dcf3` |
| `isin:INE296A01032` | BAJFINANCE | bonus | nse_ca | 2025-06-16 | 5.0 | VERIFIED | `6611b1d38d609597` |
| `isin:INE296A01032` | BAJFINANCE | split | nse_ca | 2025-06-16 | 2.0 | VERIFIED | `8c63ec300625a03b` |

Global unresolved CA outside this panel do **not** fail scoped certification.

## 6. Universe-history requirement and result (S4)

- Mode: **FIXED_PREREGISTERED_29** (protocol mode `A`)
- Dynamic PIT membership required: `False`
- OK: `True`
- Ledger source: `nse_equity_l+nse_delisted`

Frozen registrations list the exact 29 symbols in registered_data_window.universe. Cross-sectional selection occurs only within this panel (e.g. top momentum quintile), not via a dynamic historical NSE membership scan. The panel was preregistered before DISPLAY_ONLY results; not converted post-hoc to pass a gate.

## 7. Sector-history requirement and result (S5)

- Requirement: `STATIC_MAP_ONLY`
- PIT sector history required: `False`
- OK: `True`

EXP-A5-01 / A6-01 / A5A6-01 freeze known_limitation 'no PIT sector history' and use the static sector_map.json for the 29 names. EXP-A2/A3 do not use sectors. Global sector-history incompleteness is not a scoped blocker.

## 8. Discontinuity / price continuity (S6)

- Metric: `unresolved_consecutive_session_symbol_rate`
- Threshold: `≤ 0.002`
- Total consecutive-session transitions: **21953**
- Verified CA transitions (large-move class): **9**
- Genuine large market moves: **0**
- Unresolved discontinuities: **0**
- Unresolved rate (symbol): **0.0**
- Unresolved event rate vs all transitions: **0.0**
- Sparse/suspension events (not counted as CA failure): **0**
- Thin history symbols: `[]`

Long suspensions (>3 calendar days between bars) are classified SUSPENSION_OR_RELISTING and do not count as unresolved CA failures.

## 9. PIT safety

- OK: `True`
- Mode: `FIXED_PANEL_ASOF_BARS`

Experiments read only closes ≤ evaluation date via walk-forward locations. No dynamic full-NSE PIT membership required by frozen protocol. CA applied on-read; raw bhav immutable.

## 10. Snapshot reproducibility (S8)

- snapshot_id: `a7a9828ec37e09e4`
- root: `/workspace/logs/phase_a5_scoped/snapshots`
- verify_ok: `True`
- bhav panel sha256: `9fd4550df76a23fdecd199058f49c1d17eda8d716d64bbef93b78e5f500ffc20`
- panel EW index sha256: `8e8bd9d5b09757e2a23b72ad0d41c294a1b7d5d965c502514ace2a8c6dbc863b`

### Provenance hashes

| Asset | sha256 |
|---|---|
| ca_events | `840efb14b9196b0b7b7d1a99e69477b4831d3dee648c5224337820ffb80eacb1` |
| security_identity | `b47474aa4c7e95fd1d7c9d6f21634fb94c83d2626238fe6615027b39192d59a0` |
| universe_history | `1422fbb3f4322dd6b35f5e2d408a31e9d1a9a317aa360cabeed70bd916ef8f78` |
| phase_a5_sector_map | `755236655f51f91a8f4dd6e9f09fb3a4af9e8b8750fba9d311e6564e55b25986` |
| frozen_protocols_md | `710af91d5592766bd29a77cdd7a573d94fb141abb29439b5c808180a95cbd4c9` |
| scoped_bhav_panel | `9fd4550df76a23fdecd199058f49c1d17eda8d716d64bbef93b78e5f500ffc20` |
| scoped_index_panel_ew | `8e8bd9d5b09757e2a23b72ad0d41c294a1b7d5d965c502514ace2a8c6dbc863b` |

## 11. Per-experiment blockers / certification matrix

| EXPERIMENT | IDENTITY | CA | UNIVERSE | SECTOR | PRICE | PIT | SNAPSHOT | CERTIFICATION |
|---|---|---|---|---|---|---|---|---|
| EXP-A5-01 | VERIFIED | VERIFIED | VERIFIED | VERIFIED_STATIC | VERIFIED | VERIFIED | COMMITTED | **READY_FOR_SCIENTIFIC_RERUN** |
| EXP-A6-01 | VERIFIED | VERIFIED | VERIFIED | VERIFIED_STATIC | VERIFIED | VERIFIED | COMMITTED | **READY_FOR_SCIENTIFIC_RERUN** |
| EXP-A2-01 | VERIFIED | VERIFIED | VERIFIED | NOT_REQUIRED | VERIFIED | VERIFIED | COMMITTED | **READY_FOR_SCIENTIFIC_RERUN** |
| EXP-A3-01 | VERIFIED | VERIFIED | VERIFIED | NOT_REQUIRED | VERIFIED | VERIFIED | COMMITTED | **READY_FOR_SCIENTIFIC_RERUN** |
| EXP-A5A6-01 | VERIFIED | VERIFIED | VERIFIED | VERIFIED_STATIC | VERIFIED | VERIFIED | COMMITTED | **READY_FOR_SCIENTIFIC_RERUN** |

## 12. Final scoped certification

```
GLOBAL TRUST:              OPERATIONAL_ONLY
PHASE A.5 FROZEN SCOPE:    READY_FOR_SCIENTIFIC_RERUN
```

Possible values for scoped certification: `READY_FOR_SCIENTIFIC_RERUN` | `BLOCKED`.

Do **not** interpret this as strategy PASS/FAIL. No Phase A.5 scientific rerun was executed in this milestone. Do **not** begin Phase B.

---

_Evaluated at: 2026-08-11T16:49:39.893035+00:00_
