# FEATURE-002 — Test report

Suite: `tests/test_feature002.py` (plus FEATURE-001 headline/PIT tests still green).

| # | Requirement | Test |
|---|---|---|
| 1 | Shadow ON/OFF → identical production BUY cards | `test_shadow_off_and_on_same_decision_objects` |
| 2 | Identical order tickets (entry/stop/target) | `test_ticket_fields_unchanged` |
| 3 | Autopilot queue unchanged / hook after `on_setups` | `test_autopilot_queue_helper_is_prefix_of_production_order`, `test_auto_scan_calls_shadow_after_autopilot` |
| 4 | Future prices cannot alter as-of Trend features | `test_future_bar_does_not_change_frozen_features` |
| 5 | Outcome writer cannot mutate feature snapshot | `test_resolver_cannot_mutate_feature_snapshot` |
| 6 | Candidate-set membership is the cycle list | `test_candidate_set_membership_is_the_input_list` |
| 7–8 | RS/Trend ranks use only that set’s values | `test_same_cycle_ranks_only_use_that_set` |
| 9 | Re-running the same event is idempotent | `test_first_write_wins_idempotent` |
| 10 | Duplicate scans do not overwrite frozen features | same + `test_same_event_id_for_duplicate_scan` |
| 11 | Unresolved stays NULL, not zero | `test_unresolved_is_null_not_zero` |
| 12 | Version bump is a new experiment | `test_version_change_is_a_new_experiment` |
| 13 | Pre-FEATURE-002 dates cannot enter primary / live insert | `test_pre_freeze_cannot_be_primary`, `test_live_scan_refuses_pre_freeze_date` |

`unified_scanner`, `trade_executor`, and `autopilot` do not import `feature002`.

Current live ledger: 0 primary observations. Status remains **FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA**.
