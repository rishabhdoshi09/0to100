# FEATURE-002 — Decision

**FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA**

This is the correct result. FEATURE-001 nominated Trend and RS. Only new
live-scan observations recorded on or after `2026-08-22T00:00:00+05:30`, with
`session_date >= 2026-07-24`, can promote them.

## Classification (not yet allowed)

| Feature | Label |
|---|---|
| Trend (`trend_features_v1`) | *deferred — insufficient new data* |
| RS (`rs_features_v1`) | *deferred — insufficient new data* |

Allowed labels later (exactly one each): `GRADUATE_RANK_FEATURE` |
`EXTEND_FORWARD_VALIDATION` | `KEEP_RESEARCH_ONLY` | `RETIRE`.

`GRADUATE_RANK_FEATURE` would still **not** change production. FEATURE-003
is a separate milestone and is not started.

## The thirteen questions

All answers are: **not estimable on the primary live-scan sample yet.**

1. Does RS beat production rank? — insufficient new data
2. Does Trend beat production rank? — insufficient new data
3. Top-1? — insufficient new data
4. Top-3? — insufficient new data
5. Top-5? — insufficient new data
6. Tail loss? — insufficient new data
7. Families that benefit? — insufficient new data
8. Families that do not? — insufficient new data
9. Monthly stability? — insufficient new data
10. Regime stability? — insufficient new data
11. RS beyond production score? — insufficient new data
12. Trend beyond production score? — insufficient new data
13. Combined R3 vs RS alone? — insufficient new data (R3 remains exploratory)

Ledger: `{'observations': 0, 'primary': 0, 'resolved_primary_5d': 0, 'candidate_sets': 0}`.
