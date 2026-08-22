# FEATURE-001 — Ranking study

Research only. Production ranking is unchanged.

Days with ≥8 simultaneous fires: **302**.

| Rank key | Top−bottom E[R] |
|---|---|
| score | 0.064 [-0.003, 0.115] |
| trend | 0.141 [0.100, 0.187] |
| rs | 0.395 [0.319, 0.477] |
| score_plus_rs | 0.095 [0.035, 0.150] |

Precision (share of top-quintile names with net_R > 0):

- `score`: 20.1%
- `rs`: 24.0%
- `trend`: 19.6%

Within-day ranks among simultaneous fires. Research only.

`rs` dominates `score`. A naive `score + RS/10` blend (0.095) is worse than RS alone (0.395). Do not ship that blend. Precision remains low (~20–24% of top-quintile names have net_R > 0), so ranking is a sort, not a licence.
