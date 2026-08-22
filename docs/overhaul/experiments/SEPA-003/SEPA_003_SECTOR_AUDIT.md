# SEPA-003 sector audit

R2.1 UNKNOWN (3,131 / 4,208) came from `scan.sector_heat.sector_of`
parsing only NIFTY500 comment groups. NIFTY50 names (RELIANCE, INFY, …)
were unmapped — a coverage hole, not “no industry”.

## Map `sector_map_v1`

| Source | n |
|---|---|
| NIFTY500 comment groups | 337 |
| Documented large-cap overlay | 71 |
| **Total mapped** | **408** |

- `sector_identity_pit = false` (current classification applied historically)
- Unmapped stays **UNKNOWN**
- No sector inferred from price
- Sector *returns* / ranks at as_of use only members with bars ≤ as_of

## Coverage on reconstructed F fills

| Year | F fills | Mapped | % mapped |
|---|---|---|---|
| 2020 | 137 | (see overall) | — |
| Winning era | 2,020 | 537 | 26.6% |
| Weak era | 1,412 | 348 | 24.7% |
| **All** | **3,432** | **885** | **25.8%** |

## H8

**INSUFFICIENT_PIT_SECTOR_DATA**

Mapped coverage is 25.8% and identity is not a PIT archive. Leading vs
weak group contrast on the mapped slice is not a leadership claim
(thin, wrong-way point estimate, FDR reject is not interpretable).

No sector gate.
