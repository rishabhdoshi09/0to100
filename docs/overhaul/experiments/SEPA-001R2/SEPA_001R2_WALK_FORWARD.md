# SEPA-001R2 walk-forward

Split was **predeclared** in `SEPA_001R2_VALIDATION_PROTOCOL.md` before final performance was calculated. Blocks are assigned by signal `as_of`.

- Development: first eligible date → `2023-12-31`
- Validation: `['2024-01-01', '2024-12-31']`
- Confirmation: `['2025-01-01', '2026-08-21']`

| Variant | Block | n | E[R] or 20d mean % | Verdict |
|---|---|---|---|---|
| A | development | 49924 | -0.0329 | REJECT |
| A | validation | 18554 | -0.1751 | REJECT |
| A | confirmation | 23071 | -0.3028 | REJECT |
| B | development | 26237 | 0.0366 | STATISTICAL_SIGNAL |
| B | validation | 10658 | -0.1325 | REJECT |
| B | confirmation | 5381 | -0.0685 | REJECT |
| C | development | 14738 | 0.0895 | STATISTICAL_SIGNAL |
| C | validation | 5753 | -0.0687 | REJECT |
| C | confirmation | 4939 | -0.0542 | REJECT |
| D | development | 7896 | 0.0435 | INCONCLUSIVE |
| D | validation | 3085 | -0.1198 | REJECT |
| D | confirmation | 2579 | -0.0796 | REJECT |
| E | development | 1820 | 0.2152 | STATISTICAL_SIGNAL |
| E | validation | 715 | 0.0324 | UNDERPOWERED |
| E | confirmation | 857 | -0.1096 | REJECT |
| F | development | 2305 | 0.2644 | STATISTICAL_SIGNAL |
| F | validation | 928 | 0.0281 | UNDERPOWERED |
| F | confirmation | 975 | -0.1219 | REJECT |
| G | development | 12001 | 3.3583 |  |
| G | validation | 4735 | 0.9398 |  |
| G | confirmation | 4077 | -0.0585 |  |

## Confirmation block (deployment evidence)

Core F confirmation n = `975`  
`has_unseen_block` for F uses this block, not a hardcoded False.  
Deployment label: `NOT_DEPLOYMENT_ELIGIBLE`  
Reasons: ['confirmation_block_statistical=REJECT', 'confirmation_expectancy_r=-0.1219', 'confirmation_ci_includes_nonpositive', 'pooled_STATISTICAL_SIGNAL_is_not_confirmation_evidence', 'pit_class=PIT_DEGRADED', 'ca_complete=false (global verifier unchanged)']

