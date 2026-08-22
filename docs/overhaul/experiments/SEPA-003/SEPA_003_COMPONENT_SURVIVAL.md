# SEPA-003 component survival

R2.1 ladder (cited, not re-run):

| Variant | Deduped n | E[R] | Verdict |
|---|---|---|---|
| A scanner | 91,549 | −0.130 | REJECT |
| B + Stage-2 | 42,276 | −0.019 | REJECT |
| C + RS≥70 | 25,430 | +0.026 | INCONCLUSIVE |
| D + VCP | 13,560 | −0.017 | REJECT |
| E + entry | 3,392 | +0.095 | INCONCLUSIVE |
| F core SEPA | 4,208 | +0.123 | SIGNAL pooled / confirmation REJECT |
| G Stage-2+RS % | 20,813 | +2.14% 20d | NOT_SEPA_R |

## Reconstructed F continuous classes

| Feature | n | Monotone? | Class |
|---|---|---|---|
| Final contraction depth | 3,432 | no | UNSTABLE |
| Dry-up ratio | 3,432 | no | UNSTABLE |
| Pivot distance | 3,432 | no | UNSTABLE |
| Stop width | 3,432 | no | UNSTABLE |
| Breakout gap | 3,432 | no | UNSTABLE |
| Tightness | 3,432 | no | UNSTABLE |

Prespecified RS buckets only. No threshold was taken from the best bin.

## Survival summary

| Concept | Role after SEPA-003 |
|---|---|
| Stage-2 (7 structural rules) | Retain as quality feature |
| RS percentile | Retain as ranking feature |
| VCP binary | Retire as hard gate |
| VCP continuous | UNSTABLE — not a durable carrier |
| Entry geometry | UNSTABLE — do not retune buy-zone |
| Regime | Descriptive; no gate |
| Sector leadership | INSUFFICIENT_PIT_SECTOR_DATA |
