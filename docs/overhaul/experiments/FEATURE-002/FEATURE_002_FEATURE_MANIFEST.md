# FEATURE-002 — Feature Manifest

| Field | Value |
|---|---|
| Experiment | FEATURE-002 |
| `feature_set_version` | `feature-002.v1` |
| Trend | `trend_features_v1` (unchanged from FEATURE-001) |
| RS | `rs_features_v1` wrapping `rs_cs_v1` (formula unchanged) |
| Production rank | `auto_scan.final_order.v1` |
| Shadow ranks | `feature-002.ranks.v1` |
| FEATURE-001 freeze SHA | `aa2dc3b3ef5ff611b2cdd25faeabff93f80dae58` |
| Last FEATURE-001 sample | `2026-07-23` |
| Forward start date | `2026-07-24` |
| Protocol activation (IST) | `2026-08-22T00:00:00+05:30` |
| Primary ranks | R0 production, R1 `rs_percentile`, R2 `n_structure_passed` |
| Exploratory | R3 `0.67 * PctlRank(RS) + 0.33 * PctlRank(Trend)` |
| Forbidden | `score + RS/10`; RS≥70 gate; Stage-2 AND RS trade rule |
| Ledger | `logs/feature002/shadow.db` |

A version string change is a new experiment. Old rows stay with their version and drop out of primary stats.
