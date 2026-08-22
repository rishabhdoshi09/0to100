# SEPA-003 results

**Not VALIDATED_EDGE. Not untouched confirmation. Not paper-eligible.**

Protocol committed first. Frozen R2.1 detectors. Official NSE bhav
2019-08-23 → 2026-08-21. Index: `NIFTY50_EQUALWEIGHT_PROXY_BHAV`
(`regime_pit_v1`). Sector: `sector_map_v1` (current map, UNKNOWN if
unmapped).

## Reconstruction vs frozen R2.1 F

| Book | n | E[R] |
|---|---|---|
| R2.1 official F (deduped, embargoed) | 4,208 | +0.123 |
| R2.1 F confirmation (consumed) | 975 | −0.122 |
| SEPA-003 reconstructed fills | 3,432 | +0.172 |

Ledger F setups: 5,797. Skipped/refused: MISSED 1,731, GAP_THROUGH 628,
CA-censored 2, NO_FILL 4. Reconstruction is the **feature sample**. It is
not a second official expectancy. Thresholds were not changed to match
4,208.

## Reconstructed F by year

| Year | n | E[R] | CI |
|---|---|---|---|
| 2020 | 137 | +0.171 | [−0.338, +0.735] |
| 2021 | 686 | +0.538 | [+0.299, +0.823] |
| 2022 | 472 | −0.027 | [−0.314, +0.249] |
| 2023 | 725 | +0.404 | [+0.117, +0.694] |
| 2024 | 743 | −0.066 | [−0.265, +0.142] |
| 2025 | 384 | −0.106 | [−0.307, +0.120] |
| 2026 | 285 | +0.026 | [−0.219, +0.355] |

Same shape as R2.1: 2021/2023 carry the pooled plus; 2022 is already
negative before the weak era.

## Decay

**Verdict: UNSTABLE_EDGE**

Winning era n=2,020 vs weak era n=1,412. STRONG_BULL share falls
318/2020 → 41/1412, but year-level sign flips inside both eras, so
regime mix alone does not explain the collapse.

## Primary FDR (q=0.10, 8 hypotheses, 5 with p)

| ID | p | q | Rejected | Stability class |
|---|---|---|---|---|
| H4 final contraction | ~0 | ~0 | yes | UNSTABLE (year flips) |
| H5 dry-up | 0.28 | 0.28 | no | UNSTABLE |
| H6 pivot vs MAE | 0.015 | 0.037 | yes | UNSTABLE (year flips) |
| H7 regime | 0.26 | 0.28 | no | CONTEXT (descriptive) |
| H8 sector | 0.032 | 0.054 | yes | INSUFFICIENT_PIT_SECTOR_DATA |

Pooled FDR is not a licence to pick a new threshold.

## Decision

**A — RETIRE CORE SEPA; RETAIN SELECT FEATURES**
