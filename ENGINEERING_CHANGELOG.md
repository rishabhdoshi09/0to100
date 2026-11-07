# QuantTerm audit branch engineering changelog
Branch: audit/snapshot-pointer-failclosed-20261107
Base: PR #289 head e5ed0d01ddcab5f3f5edce11ea69ac0d317517f7.
Date: 2026-11-07.

## Confirmed source changes
1. product/scan_store.py: scan_source_status now refuses an invalid existing ACTIVE pointer, errors when snapshot authority cannot be probed, and Kite-labelled artifacts without an active Kite snapshot. Existing offline NSE scans remain admissible only when no active pointer exists and source policy permits.
2. tests/test_scan_kite_snapshot_freshness.py: added synthetic regressions for corrupted pointer, probe exception, missing Kite authority and legitimate offline NSE.
3. product/recommendations_liveness.py: mismatched saved recommendation generation is no longer returned as visible cards during a rebuild. The response reports REFRESHING/FAILED with zero cards, and the rebuild remains background-only.
4. tests/test_recommendations_liveness.py: checks the stale-card suppression and updates the prior expectation that old cards would be displayed.

## Verification ledger
- Baseline PR #289 at e5ed0d01: GitHub CI, Product Acceptance, Host Migration Gate and Terminal UI all showed SUCCESS.
- Audit branch focused tests: ADDED but results must be checked on the final audit SHA; do not mark PASSED from code review alone.
- Recovered Mac host: NOT RUN.
- Production broker: NOT TOUCHED.
- Destructive storage tests: NOT RUN.
- Before/after runtime latency: NOT MEASURED.
- Before/after investment performance: NOT MEASURED.

## Compatibility and rollback
Changes affect read-time authorization and recommendation presentation, not persisted schema or paper positions. Genuine offline NSE scans remain supported. If regression evidence disproves compatibility, revert audit commits on this isolated branch; never force-restart or switch the installed production checkout. This branch is not merged or deployed.

## Why changes were made
The source accepted a stale/unverifiable scan in one no-ID error case, and a visible UI could render old BUY cards while the backend marked the projection stale. Both conflict with the no-stale-entry/clear-operator-truth invariants.
