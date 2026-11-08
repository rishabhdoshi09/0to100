# QuantTerm production-readiness audit — 8 November 2026

Status: IN PROGRESS. This document is the durable checkpoint for the master
production-readiness directive. It records only what was verified against exact
commit SHAs in this session; absence of a finding is not a pass.

Production readiness verdict: **NOT READY**. Paper-only, live execution locked.
No production deployment, live order, or transition to live trading is
authorized by this audit.

## 1. Verified repository state (as of 2026-11-08 02:30 UTC)

| Ref | SHA | Verified CI at that SHA |
| --- | --- | --- |
| `claude/build-ai-trading-system-miHHd` (canonical) | `eda3d95ffa23841f9edffb76a39979d912b2e78c` | CI, Product Acceptance, Terminal UI, Host Migration Gate: success |
| PR #289 `fix/scan-kite-snapshot-freshness-20261106` | `e5ed0d01` | **No completed run.** Every workflow for this head was cancelled by the `cancel-in-progress` concurrency group when the audit branches were pushed from it. Its code is covered only transitively by the #296/#299 runs below. |
| PR #290 `audit/snapshot-pointer-failclosed-20261107` | `1e00746e` | CI **failure**: `tests/test_recommendations_rebuild_claim.py::test_failed_current_generation_is_explicit_not_refreshing_forever` (its `recommendations_liveness.py` rewrite conflicts with the claim contract). Other three workflows green. |
| PRs #291–#295 | `c2e70848`, `c9863d2e`, `b5f0e935`, `cb211531`, `e23f886e` | Byte-identical, file for file, to their counterparts inside #296 (verified with `git diff <pr> <296> -- <files>` → empty). |
| PR #296 `audit/integrated-quantterm-safety-20261108` | `2b1686d5` | All four workflows success (01:23 UTC push + PR runs). #297 and #298 are merged into it. |
| PR #299 `audit/paper-config-corruption-fail-closed-20261108` | `12ec8cfc2430d82a743e5e48ab9c8f0477cfdb16` | All four workflows success, re-confirmed by the 02:26 UTC runs. |

Dependency chain (each row is 0 commits behind its parent):
`canonical ← #289 (27 commits) ← #296 (13) ← #299 (2) ← this branch (3)`.

### Integration recommendation

1. Merge in order #299 → #296 → #289 → canonical. The merge commit onto
   canonical gets its own CI run; that SHA is the release-candidate SHA.
2. Close #291–#295 as superseded by #296 (no unique content).
3. Close #290 as superseded: its `scan_store.py` half is ported onto this
   branch (finding QT-PRA-002 below); its `recommendations_liveness.py` half is
   superseded by #292's version, which passes the rebuild-claim test #290 broke.
   Its root-level `AUDIT_REPORT.md` / `ROADMAP.md` etc. should not be merged;
   the content that was confirmed is carried here.
4. Do not merge #289 on its own CI evidence; it has none.

## 2. Findings this session

Classification: P0 financial loss / unsafe execution / silent position loss;
P1 incorrect decisions, stale data, major reliability; P2 performance,
inconsistency, observability; P3 cosmetic.

| ID | Pri | Finding (evidence) | Status |
| --- | --- | --- | --- |
| QT-PRA-001 | **P0** | `research/auto_research/scheduler.py::get_brain()` wrapped durable-path resolution in `try/except: pass`. Installed hosts run with `QT_RUNTIME_ROOT_REQUIRE_EXISTING=1`, so a detached external runtime makes `logs_dir()` raise; the singleton was then built path-less: empty in-memory paper book, `paper_auto_enabled=True` (first-run default because `paper_config_path=None`), `_save_intel_book()` returning `True` with no file, cached for the process lifetime. The PAPER_CYCLE job (`research/autonomy/jobs.py::Deps.run_paper_cycle`) hands exactly that book and flag to `run_reco_paper_cycle`. Reproduced by `tests/test_auto_research.py::TestGetBrainRuntimePathsUnavailable` (3 of 4 tests failed before the fix: `is_paper_auto_enabled()` was `True`, blocked brain was cached). | **FIXED** on this branch: brain is latched `durable_storage_unavailable`, entries blocked (`BLOCKED_RECOVERY`, `paper_enabled=False`), durability callback reports `False`, instance not cached so recovery rewires real paths. The 15 s `RuntimeStorageGuard` remains the outer defence. |
| QT-PRA-002 | P1 | `product/scan_store.py::scan_source_status()` treated a present-but-unreadable `ACTIVE` pointer as "no snapshot is active" and a failed snapshot probe as a pass for unbound scans (QT-AUD-001 in #290). | **FIXED** by porting #290's `scan_store.py` change and its four regression tests. |
| QT-PRA-003 | P1 | `scan/market_scan_service.py::run_whole_market_scan` persisted the coverage audit, public best setups, long-term overlay and desk recommendations/discovery before the final pre-publication snapshot identity check (QT-AUD-003 in #290, previously "NOT RUN"). Reproduced: with the identity rotating between the post-scan check and publication, all four sidecars were written while the canonical scan was withheld (`persisted == ['audit','public_best_setups','long_term','desks']`). Consumers (`decision_simulation_gate`, `recommendations_liveness`) do detect the mismatch, so the impact is churn and a permanently "refreshing" board, not a stale entry. | **FIXED**: identity re-verified immediately before the first sidecar write; regression `test_snapshot_switch_before_publication_persists_no_sidecars`. Residual window: a rotation *during* sidecar writes still leaves sidecars without a canonical scan. Closing it fully needs staged sidecar commit (P2 follow-up). |
| QT-PRA-009 | P1 | `research/autonomy/operational_guards.py::_open_position_count()` read `get_brain().intel_book.open` without checking `intel_book_recovery_error`. A brain whose book failed to restore (or whose runtime root is gone, QT-PRA-001) holds an empty in-memory book, so the PAPER_CYCLE fast path reported `NO_POSITIONS_TO_MANAGE` / `SUCCEEDED` instead of letting the canonical handler return `BLOCKED_RECOVERY`. Reproduced by `tests/test_autonomy_operational_guards.py::test_unrestored_brain_book_is_unknown_not_empty` (failed before fix). | **FIXED**: unrestored brain → position count unknown → canonical handler runs. |
| QT-PRA-004 | P1 | PR #289's head has no completed CI run (all cancelled). The whole stack rests on it. | OPEN. Mitigated by #296/#299 green runs on supersets; must be re-run on the final merge SHA. |
| QT-PRA-005 | P2 | Whole-market scan can be started by two schedulers (`operations/market_ops.py` MARKET_SCAN via terminal controls / host supervisor startup kick, and `research/autonomy/supervisor.py` `startup_discovery_scan:` / `snapshot_scan:` jobs). Coordination is best-effort: the autonomy supervisor reads `logs/market_ops/runtime.json` for an in-flight MARKET_SCAN (`_market_scan_in_flight`) but there is no cross-process single-flight lock in `run_whole_market_scan`. On an 8 GB Intel Mac two concurrent full scans are a memory/latency risk. | OPEN. Needs host measurement before changing; a file-lock single-flight would require the autonomy job to handle a `SCAN_IN_PROGRESS` result. |
| QT-PRA-006 | P2 | `PaperBook.open_position` sizes risk against starting `capital`, not current equity (`capital + realized_pnl`). Bounded by the 10 % name cap × 5 positions, so buying power cannot be exceeded, but risk-per-trade drifts from "1 % of equity" after P&L. | OPEN (design decision; document or change deliberately). |
| QT-PRA-007 | P2 | `_load_paper_config()` returns `True` when `paper_config.json` is missing even when the intel book has history. A lost opt-out file re-enables paper autonomy. The supervisor's persisted `owner_state.new_entries_paused` is the primary operator pause and is unaffected. | OPEN. |
| QT-PRA-008 | P2 | Public read-only posture is opt-in via `QT_PUBLIC_READ_ONLY`; an exposed instance without the flag accepts mutations without a bearer token (`product/public_access.py`). CORS is restricted to localhost origins. `docs/PUBLIC_LAUNCH.md` + `scripts/verify_public_launch.py` are the preflight. | OPEN. Acceptable for private operation; must be fail-closed before any public deployment (consider defaulting to read-only when bound to a non-loopback address). |

### Controls verified present (code + tests, not host)

- Live broker mutations are hard-locked at the SDK boundary: `product/live_execution_interlock.py` returns a constant `LOCKED/unauthorized` state and `data/kite_client.py`'s guarded proxy calls `assert_live_execution_allowed` on every order/GTT mutation; `KiteConnect(` is constructed only there. `execution/trade_executor.py` additionally refuses legacy live unless an explicit unsafe override is set and governance approves.
- Installed-host runtime root is strict (`core/runtime_paths.py` `QT_RUNTIME_ROOT_REQUIRE_EXISTING`), with a 15 s identity/write-through watchdog (`product/runtime_storage_guard.py`).
- Equity paper book: strict restore, checksum, cross-file commit marker, duplicate identity rejection, durability-failure latch (#294, #296); event store fail-closed on corrupt/partial records (#297, #298); PAPER_AUTO persisted preference fails closed (#299).
- F&O paper store: corrupt/mismatched open rows raise instead of being skipped; `commit_cycle` is one SQLite transaction (#295).
- Scan freshness: recommendations bound to the active Kite snapshot, stale cards withheld (#289, #292); index-bar checksum in snapshot verification (#291).

## 3. Tests executed this session (Ubuntu, Python 3.12, hermetic, credentials blank)

- CI-equivalent canonical sweep on `12ec8cfc` working tree: 409 test files, one interpreter per file as in `.github/workflows/ci.yml`. 408 files rc=0; the single rc≠0 was `tests/test_auto_research.py`, executed while the new QT-PRA-001 regression tests existed and the fix did not yet (expected red).
- After fixes: `tests/test_auto_research.py` (36 passed), `tests/test_scan_kite_snapshot_freshness.py` (17 passed), 54 files touching `scan_store`/`desk_pipeline`/`decision_simulation_gate`/`startup_check` (656 passed), 17 files touching `market_scan_service` (199 passed), neighbouring get_brain consumers (`test_autonomy*`, `test_evolution_*`, `test_paper_cycle_liveness_dedupe`, `test_paper_position_sector_persistence`, `test_parallel_release_followup`) all passed.
- `python -m compileall` over the CI module list: OK.
- Full canonical sweep on the committed branch head (`3ac04390` + docs; QT-PRA-009's files re-run afterwards at `1ad9be51`): 409/409 files rc=0, 4092 tests passed, 0 failed, 0 skipped, 0 errors. `tests/integration` (network-controlled, `continue-on-error` in CI) was not run here.

## 4. Release gates

| Gate | State | Evidence / blocker |
| --- | --- | --- |
| P0 defects resolved and validated | PARTIAL | QT-PRA-001 fixed with regression tests. No other P0 found this session; audit coverage is not complete. |
| No open P1 without blocking decision | FAIL | QT-PRA-004 (no CI on #289 head) is open until the merge SHA runs. |
| CI + regression suites at release SHA | NOT TESTED | No release SHA exists yet; stack must be merged first. |
| Data/portfolio invariants under fault injection | PARTIAL | Hermetic fault tests exist for corrupt JSON/JSONL, torn writes, SQLite row corruption, pointer corruption, detached runtime root (this session). No real detach of the user's external drive was exercised. |
| Backup restoration demonstrated | NOT TESTED | Requires the owner's Mac and recovery assets. Recovery backups must not be modified. |
| Hardware / external storage validated | BLOCKED | Physical disconnections reported; hardware is untrusted until tested on the host. |
| Real-host performance and soak (p50/p95/p99, memory, swap) | NOT TESTED | Needs the Intel MacBook. No latency numbers are claimed here. |
| Frontend / API / Telegram integration | PARTIAL | Hermetic Product Acceptance + Terminal UI workflows green at `12ec8cfc`; no real-browser session against live data this session. |
| Security controls tested | PARTIAL | Public mutation boundary tests exist; QT-PRA-008 default posture open. No dependency-vulnerability scan this session. |
| Rollback verified | NOT TESTED | |
| Operator instructions complete | PARTIAL | `docs/MACBOOK_RUNBOOK.md`, `docs/ALWAYS_ON.md`, `docs/PUBLIC_LAUNCH.md` exist; recovery-acknowledgement procedure for `BLOCKED_RECOVERY` is not written up. |
| Independent release review | NOT TESTED | |

## 5. Requires the real Mac / owner action

- Detach/reattach the external runtime volume with the stack running and confirm: children stop within the guard interval, no file is created on the internal disk, `get_brain()` reports `RUNTIME_PATHS_UNAVAILABLE` until the exact runtime returns, and the persisted book is restored unchanged afterwards.
- Restore drill from a recovery backup into a disposable location (read-only source).
- Cold/warm latency, memory and swap measurement for `/api/health` and `/api/dashboard`.
- Kite daily login and a live-session scan to confirm snapshot identity flows end to end.

## 6. Next engineering actions (priority order)

1. Merge the verified chain (#299 → #296 → #289 → canonical) and re-run all four workflows on the merge SHA.
2. QT-PRA-003 residual: stage sidecar outputs and commit them with `save_scan`, or make consumers treat a sidecar without a matching canonical scan as absent.
3. QT-PRA-005: measure concurrent-scan incidence from `logs/market_ops/runtime.json` and autonomy job ledger on the host before adding a single-flight lock.
4. QT-PRA-007/008 decisions.
5. Add a security dependency scan (`pip-audit`) to CI as a non-blocking lane first.
