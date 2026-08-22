# FEATURE-002 — Shadow schema

SQLite `logs/feature002/shadow.db` (WAL). Features are write-once. Outcomes are a separate table.

## `candidate_sets`

`candidate_set_id`, `scan_cycle_id`, `session_date`, `recorded_at`, `n_candidates`, `family_composition` (JSON), `source`, `feature_set_version`, `protocol_hash`

Membership is the production serialized list at that cycle. A later cycle with the same `scan_cycle_id` is ignored (`INSERT OR IGNORE`).

## `observations`

Identity: `event_id` = sha256(`feature-002.v1|session_date|SYMBOL`)[:20]  
Also: `candidate_set_id`, `scan_cycle_id`, `symbol`, `exchange=NSE`, `session_date`, `recorded_at`, `source`, `families[]`, `primary_family`

Production snapshot: `production_score`, `production_rank`, `production_verdict`, `production_signals`, `production_decision`, `would_trade`, `ready_status`, `entry`, `stop`, `target`, `chase_risk`

Shadow: `n_structure_passed`, `structure_pass`, `rs_percentile`, `rs_score`, `rs_rank`, `trend_rank`, `combined_shadow_rank`, `r3_score`

Context (not used in rank): `regime_label`, `sector`, `sector_map_version`

Frozen blob: `feature_snapshot` JSON (`trend` + `rs` vectors). Never updated by the resolver.

`eligible_primary` is computed at insert from source + dates + version.

## `outcomes`

`event_id` PK. `resolved_at`, `next_open`, `ret_1d/5d/10d/20d`, `mae`, `mfe`, `hit_1r`, `hit_2r`, `production_traded`, `production_outcome`, `unresolved_reason`

Missing horizon → `NULL` + reason. Never store `0` to mean “not yet”.
