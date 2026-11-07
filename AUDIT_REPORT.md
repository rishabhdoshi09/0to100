# QuantTerm master audit — first verified engineering pass
Date: 2026-11-07
Source reviewed: PR #289 at e5ed0d01ddcab5f3f5edce11ea69ac0d317517f7.
Audit branch: audit/snapshot-pointer-failclosed-20261107.
Status: IN PROGRESS / PARTIAL. This is not a comprehensive production certification.

## Executive verdict
The reviewed code has meaningful fail-closed improvements, but neither paper-trading reliability nor trading edge is yet independently proved on the recovered host. No live-money authorization is implied.

## Repository and actual architecture
GitHub recursive tree at the review head has 1,533 tracked files, about 16.1 MB of tracked file contents (distinct from any mounted runtime or sparsebundle). Main directories include tests (413 files), product (254), research (147), frontend (103), scan, risk, execution and data.
The documented product path is the Vite/React desk. The complete launcher is scripts/run_quantterm_complete.sh; installed macOS host runs through launchd. Durable authority uses core/runtime_paths.py, research/intelligence/data/snapshot_store.py, scan/market_scan_service.py, product/scan_store.py, recommendations/discovery projections and research/autonomy/jobs.py. Actual installed process and runtime state were NOT inspected.

## Verified existing controls
- PR #289 (draft, not merged) adds active Kite snapshot binding, source/date checks, stale scan publication rejection, delayed-bhavcopy compatibility, paper-entry guards and focused regression tests.
- At PR head e5ed0d01, GitHub CI, Product Acceptance, Terminal UI, and Host Migration Gate all completed successfully. CI includes hermetic shards and an Intel macOS smoke lane, but not the recovered host.
- product/evolution/promotion.py explicitly sets AUTO_PROMOTION_ENABLED=False.
- core/runtime_paths.py has a strict existing-runtime mode; this should still be tested during actual storage faults.
- execution/trade_executor.py says its legacy live path is locked by default; enforcement at the final broker mutation boundary requires separate review.

## Findings
### QT-AUD-001 — P1, CONFIRMED in source: active pointer/probe fail-open
Original product/scan_store.py scan_source_status(), approximately lines 28-65 at e5ed0d01, returned success on a snapshot verification exception whenever the scan had no source_snapshot_id. research/intelligence/data/snapshot_store.py get_active_snapshot() also converts a malformed or missing-target ACTIVE pointer to None. Consequence: a broken pointer can be misread as genuinely absent broker authority, allowing an unbound offline-source scan to pass this validator. Fixed on the isolated audit branch by rejecting pointer-present-but-invalid, probe errors, and Kite-labelled scans without active identity. Four synthetic regression tests added. GitHub CI on the audit head must be checked before marking VERIFIED FIX.

### QT-AUD-002 — P1, CONFIRMED in source/UI: stale recommendation cards still visible
product/recommendations_liveness.py build_fast_response(), approximately lines 295-355, returned prior-generation recommendation categories and only relabeled the response REFRESHING when _matches() was false. frontend/src/recommendationsViews.tsx, near line 586, renders category.cards without a records_status eligibility check. A stale BUY card could remain visible despite a textual warning. Audit branch now emits an empty truthful category projection while background rebuild continues. Dedicated regression added. CI/host UI acceptance still required.

### QT-AUD-003 — P1 risk hypothesis: sidecar publication can precede final scan generation check
scan/market_scan_service.py, approximately lines 495-559, runs persist_public_best_setups(), long-term overlay and persist_desks_from_market_scan() before the final active-snapshot check and canonical save_scan(). product/desk_scan_overlays.py persists recommendations and discovery independently. If the active snapshot rotates during those writes, a failed scan publication can still leave stale auxiliary files. Consumer gating mitigates some impact; an injected pointer-rotation test across all sidecar readers is NOT RUN. Do not claim an actual stale entry occurred from this path.

### QT-AUD-004 — P2 gap: manifest-only scan liveness
product/scan_store.py checks active snapshot manifest fields but does not invoke SnapshotStore.verify_snapshot() for data-file hashes. Data readers in scan/bulk_fetcher.py do verify on adopting Kite frames. Assess checksum-verification cadence and cache invalidation without introducing expensive full-file hashes into every HTTP request.

## Unverified areas
No live runtime/backup/SQLite inspection, Kite authentication, historical paper reconciliation, F&O option-chain validation, US paper exit validation, complete trading-path safety proof, backtest result reproduction, real latency p50/p95/p99 or external security penetration test was performed. Do not interpret absence of findings as passing.

## Acceptance status
Engineering-ready: NOT ESTABLISHED.
Paper-trading-ready: NOT ESTABLISHED.
Live-trading-ready: NOT AUTHORIZED / NOT ESTABLISHED.

## Next actions
1. Check exact audit-head CI; resolve any failures without merging.
2. Inject snapshot rotation during sidecar writes and assert all consumers suppress stale actions.
3. On a copy of runtime only, reconcile scan, recommendations, discovery, Telegram and paper positions by snapshot IDs.
4. Verify a read-only recovery mount and restore drill to a disposable location.
5. Run point-in-time replay with calibrated out-of-sample outcome and cost evidence.
