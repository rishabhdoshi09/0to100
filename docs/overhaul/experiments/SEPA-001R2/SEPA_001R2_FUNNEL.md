# SEPA-001R2 funnel

Two funnels. Snapshot rows are symbol×date. Unique rows are setups / opportunities. Do not add them together or compare 1,500 daily A rows to 50 unique F setups as if `n` meant the same thing.

## Snapshot funnel (symbol × date)

| Stage | Count | % of investable snapshots |
|---|---|---|
| candidates | 3474133 | 164.6040% |
| investable | 2110601 | 100.0000% |
| stage2 | 627636 | 29.7373% |
| rs_pass | 399311 | 18.9193% |
| vcp_detected | 215487 | 10.2097% |
| pivot_defined | 627591 | 29.7352% |
| entry_ready | 59346 | 2.8118% |

## Unique-opportunity funnel

| Stage | Unique count |
|---|---|
| vcp_detected | 22441 |
| valid_pivot | 26247 |
| entry_ready | 16943 |
| valid_fill | 6734 |
| gap_through | 3308 |
| observed_extended | 7192 |
| left_censored | 10801 |
| ca_censored | 215 |
| stop_too_wide | 1225 |
| expired_failed | 1385 |
| pivot_retest | 325 |

Unique setups (ledger): 19336
Left-censored unique: 10801
CA-censored outcomes (path crossings): 269

Core F fills only `valid_fill` on the unique funnel, after next-open classification (gap-through / extended / stop-too-wide / left-censored / CA-censored are refusals or exclusions, not fabricated trades).
