# A6 — Portfolio Network Risk Complement

**Milestone:** Phase A / A6  
**Status:** Advisory / research outputs only — **no automatic trade blocking**  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Upgrade portfolio intelligence beyond pairwise correlation **without replacing**
`risk.correlation` / `portfolio_risk` / sector caps.

Answer (research-grade):

> What incremental network/community risk does adding this candidate create
> relative to the CURRENT portfolio?

## Graph

- Nodes = tradable assets
- Edges = PIT rolling correlation above a statistical threshold
- No fundamental / text / news edges in Phase A

## Metrics

community_id, centrality (degree / eigenvector / betweenness),
portfolio_network_concentration, incremental_cluster_risk, contagion_score

## Reuse

| Need | Source |
|------|--------|
| Pairwise ρ incumbent | `risk.correlation.pairwise_corr` / `clusters_from_corr` |
| CycleContext seam | `NetworkRiskView` |
| PIT returns | caller / PitContract closes |

## Non-goals

- No replacement of pairwise correlation guards
- No automatic blocking in ticket / portfolio_gate
- No news/fundamental edges
