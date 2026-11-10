# QuantTerm production-readiness audit — 8 November 2026

Status: IN PROGRESS. This document is the durable checkpoint for the master
production-readiness directive. It records only what was verified against exact
commit SHAs in this session; absence of a finding is not a pass.

Production readiness verdict: **NOT READY — no merged release-candidate SHA;
host, storage, restore, soak and rollback gates are BLOCKED / NOT TESTED.**
Paper-only, live execution locked. No production deployment, live order, or
transition to live trading is authorized by this audit.

Section 7 (session 2) supersedes the status column of QT-PRA-005…008 in
section 2 and the gate table in section 4. Section 9 (session 4) holds the
current gate table and verdict.

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

## 7. Session 2 — production engineering directive (2026-11-08, from 04:00 UTC)

### 7.1 Verified stack state (exact head SHAs, actual workflow conclusions)

`gh pr checks` lists superseded/cancelled runs as `fail`; the states below are
the conclusions of the runs whose `headSha` equals the PR head, for both the
`push` and `pull_request` events of CI, Product Acceptance, Terminal UI, Host
Migration Gate and Dependency Vulnerability Advisory.

| PR | Head | Result at that exact head |
| --- | --- | --- |
| #302 paper-auto missing config | `1c826f98` | all 5 workflows success (push + PR) |
| #303 paper-book equity risk | `398a2bf7` | all 5 success |
| #304 read-only host probe | `0144d9dd` | all 5 success |
| #305 public-mode malformed flag | `6e7a5890` | all 5 success |
| #306 paper mark atomicity | `124141b6` | all 5 success |
| #307 paper-cycle upstream fail-closed | `44f4f4cf` | **CI failure** (push + PR); other 4 success. 5 tests, see QT-PRA-013 |
| #310 outcomes-returned fail-closed (on #307) | `1f1db28f` | **CI failure**: exactly the same 5 inherited tests |
| #308 (this session, on #302) | `5f6979e6` | all 5 success |
| #311 (this session, on #307) | `7bc1e07a` | all 5 success |
| #309 / #312 / #313 / #314 (this session, on #311) | `5e7a93d6` / `c74774ae` / `1b6d355c` / `06ed58e5` | all 5 success each |

Local integration sweep: `7bc1e07a` with #309, #312, #313 and #314 merged
(local merge `c6a8da07`, never pushed). 413/413 test files rc=0, one
interpreter per file as in `ci.yml`. 4148 passed, 0 failed, 1 deselected.
`tests/integration` was not run.

### 7.2 Evidence register

Status vocabulary: IMPLEMENTED → TESTED (local regression) → INTEGRATED (green
at exact head on its dependency branch) → VERIFIED ON HOST → RELEASE CERTIFIED.
Nothing here is VERIFIED ON HOST or RELEASE CERTIFIED.

**QT-PRA-010 (P1): #302 disables PAPER_AUTO on the first restart of a fresh install.**
- Root cause: the first-run default ON was granted but never recorded. After the first durable book save, history exists without a preference file, so the next load looks like a deleted opt-out.
- Evidence: CI at `968fa2da` (`test_paper_auto_closeout::TestHeadlessSmoke::test_restart_restores_without_reapproval`). `1c826f98` changed that test to persist an explicit opt-in, and production behaviour was unchanged. 3 regression tests fail at `1c826f98`.
- Fix SHA: `5f6979e6` (#308). The default is persisted (temp file, fsync, atomic replace) before it is granted. If it can't be recorded, it is not granted.
- Regression: `tests/test_auto_research.py` first-run / EROFS / deleted-preference tests.
- Integration: on #302 head. CI: all 5 success at `5f6979e6`.
- Host verification: NOT TESTED.
- Residual risk: a read-only runtime volume on first boot starts with PAPER_AUTO off.
- Review: self-reviewed, plus the #302 author's alternative (test-only change) compared. Release impact: blocks a clean first-install experience; not a money path.

**QT-PRA-011 (P1): Telegram delivers recommendations from an unpublished or invalid scan generation.**
- Root cause: `drain_desk_alerts()` sent `load_recommendations()` without binding it to `load_scan()`. The sidecars are persisted before canonical publication, which can then be withheld.
- Evidence: 3 regression tests were red (the alert was sent).
- Fix SHA: `5e7a93d6` (#309). The alert requires `scan_scanned_at == load_scan().scanned_at`, and fails closed on store errors.
- Integration: on #311. CI: all 5 success.
- Host verification: NOT TESTED (no bot token in this environment).
- Residual risk: none on the execution path.
- Release impact: decision-truth consistency between Telegram and the paper executor.

**QT-PRA-003 (update): sidecar-publication race.**
- Every acting reader is now generation-bound:
  - paper entries (`STALE_SCAN_SNAPSHOT`);
  - Telegram (QT-PRA-011);
  - discovery (`decision_simulation_gate`);
  - Home cards (#292 liveness);
  - public best setups (`load_persisted_best_setups(scanned_at)`).
- Residual (P2, display-only): `long_term_scan.json` carries no market-scan binding, so a withheld publication can leave a long-term shortlist derived from an unpublished scan. No execution path consumes it.
- Staged commit of four independent stores was set aside as too invasive for that residual.

**QT-PRA-005 (closed, measured): duplicate whole-market scans.**
- Measured in-process: after `run_supervisor()`'s `install_parallel_runtime()`, `JOBS.HANDLERS[MARKET_SCAN]` is `parallel_runtime._delegated_market_scan`. Autonomy only enqueues or observes a market_ops operation.
- market_ops holds a process-exclusive `flock` (`logs/market_ops/worker.lock`) and runs scans on the single `market_scan` lane.
- A bare `import research.autonomy` still shows `jobs.run_market_scan`. Only non-canonical entry points (`ui/retail_home_momentum.py`, archived Streamlit) call `run_whole_market_scan` directly.
- No code change. Host incidence: NOT TESTED.

**QT-PRA-006 (superseded by #303, follow-up QT-PRA-012).**
- #303 sizes the book on realized equity (`capital + realized_pnl`) and adds a free-cash check.
- Contract: every paper risk fraction (1 % per trade, 10 % per name, 5 % total open) applies to realized equity. Unrealized P&L is excluded, gains compound and drawdowns shrink exposure.
- The combined-account gate `risk.portfolio_risk.account_exposure_gate` still uses `settings.trading_capital`. It is shared with the legacy engine and the live-facing display, and remains the looser outer bound.

**QT-PRA-012 (P2): with #303, the autopilot pre-size and the book disagree, so drawdown entries are refused.**
- Root cause: `product/paper_autopilot.py` pre-sized on `book.capital`, and the book refuses quantities above its realized-equity limit.
- Evidence: on `398a2bf7`, realized −30k, a risk-bound entry gave `taken=[]` with `BOOK_REFUSED: target portfolio quantity exceeds house limits`.
- Fix SHA: `c74774ae` (#312). `PaperBook.risk_capital()` is the single sizing base for both.
- Regression: `tests/test_money_paths.py::TestPaperAutopilotSizesOnBookRiskCapital` (3 tests). All 63 PaperBook-related files pass per file (1074 tests).
- CI: all 5 success.
- Release impact: without it, the modern engine silently stops entering in a drawdown.

**QT-PRA-013 (P1 release blocker on #307 / #310): CI red at #307 and #310 heads.**
- Root cause: #307's new gate (`b00a5a81`) requires status `OK`, `NO_ACTION` or `ALREADY_DONE`. 5 test fakes returned no `status`.
- The production `IntelligenceCycleResult.status` always exists and defaults to `OK`, so this is not a production defect.
- Bisected: green at `124141b6`, red at `b00a5a81`.
- Fix SHA: `7bc1e07a` (#311). The fakes match the real contract, and my #300 test now asserts the stronger "executor never invoked" invariant.
- CI: all 5 success. #310 must rebase onto #311 (or carry an equivalent fix) before it can go green.
- Review note: `SKIPPED_LOCKED` (intel-lock contention) is also mapped to `PERMANENT_FAILED` by the #307 gate, so a contended slot is retired instead of retried. This fails closed. It needs a deliberate owner decision.

**QT-PRA-014 (P1): any web page can drive operator controls on the loopback API.**
- Root cause: CORS hides responses but doesn't stop delivery of simple cross-site POSTs or DNS-rebound requests.
- Evidence: a cross-site `text/plain` POST of `RESUME_NEW_PAPER_ENTRIES` returned 200 and was accepted, on a disposable runtime root.
- Fix SHA: `06ed58e5` (#314). Browser-asserted provenance (`Origin` exact, else `Sec-Fetch-Site: cross-site`) other than the desk is refused. Headers can only refuse; a valid bearer token still authorizes; non-browser clients are unchanged.
- Real-browser check: headless Chrome 148 against real uvicorn. The attacker form got 403; a desk-origin cross-port fetch reached the handler.
- CI: all 5 success. Host verification: NOT TESTED.
- Residual risk: a desk served from another origin needs `QT_DESK_ORIGINS`.

**QT-PRA-008 (update): public posture.**
- All APIs are hard-bound to `127.0.0.1` (`product/host_supervisor.py`, `scripts/run_quantterm_complete.sh`).
- #305 makes a malformed `QT_PUBLIC_READ_ONLY` fail closed, and QT-PRA-014 covers browser-originated mutations in private mode.
- Remaining: a tunnel or reverse proxy without `QT_PUBLIC_READ_ONLY` is still unauthenticated for non-browser clients. Public deployment (Stage B) is not authorized and must set the flag. OPEN (P2).

**QT-PRA-015 (P2, accepted with gate): `autobahn==19.11.2` exact-pinned by `kiteconnect` 5.2.2.**
- Advisories: PYSEC-2020-25 (CVE-2020-35678) and PYSEC-2026-4027 (CVE-2026-77528).
- Exposure: client-only use (KiteTicker → `wss://ws.kite.trade`). The client never offers permessage-deflate, which CVE-2026-77528 requires.
- Evidence: `tests/test_kite_ticker_wire_compat.py` (`1b6d355c`, #313) passes on 19.11.2. It also passes on 26.11.1 in a disposable venv, where `pip check` reports the pin conflict. A mutation check proved the deflate assertion can fail.
- Decision: keep the pin. Don't force-override a declared exact dependency.
- CI: all 5 success.

### 7.3 Release gates (supersedes section 4)

| Gate | State | Evidence / blocker |
| --- | --- | --- |
| P0 defects resolved and validated | PARTIAL | QT-PRA-001 integrated (#300). No new P0 found in session 2. Audit coverage is not exhaustive. |
| No open P1 without blocking decision | FAIL | QT-PRA-013 until #311 lands under #307 and #310 rebases. QT-PRA-004 (#289 head never CI-complete) until the merge SHA runs. |
| CI at release SHA | NOT TESTED | No merged release-candidate SHA exists. Every PR in 7.1 is verified only at its own head. |
| Data / portfolio invariants under fault injection | PARTIAL | Hermetic only. No physical detach. |
| Backup restoration | NOT TESTED | Owner's Mac required. Backups must not be modified. |
| Hardware / external storage | BLOCKED | Untrusted until tested on the host. |
| Real-host performance and soak | NOT TESTED | No latency, memory or swap numbers are claimed. #304 adds a read-only probe, not yet run on the host. |
| Frontend / API / Telegram | PARTIAL | Product Acceptance + Terminal UI green at every head in 7.1. One real-browser security check (QT-PRA-014). No live-data browser session. No real Telegram delivery. |
| Security controls | PARTIAL | QT-PRA-014 fixed. #305 merged into the stack. pip-audit lane (#301) green. QT-PRA-008 tunnel residual and QT-PRA-015 pin open. |
| Dependency vulnerabilities | PARTIAL | QT-PRA-015 triaged and gated. Not remediated (upstream pin). |
| Rollback verified | NOT TESTED | |
| Independent release review | NOT TESTED | Session-2 PRs are drafts awaiting review. |

### 7.4 Integration order (proposed)

`#302 ← #308`, then `#303 → #307 ← #311 ← {#309, #312, #313, #314}`, with
#310 rebased onto #311. The merge onto canonical needs its own CI run. That SHA,
not any PR head above, is the first possible release candidate.

## 8. Session 3 — integrated release candidate (2026-11-08, from 17:00 UTC)

### 8.1 Release-candidate branch

`cursor/release-candidate-f43f` (PR #319) is built on #310's head `1f1db28f`,
which already contains the heads of #300–#307. On top of it, #311, #308, #309,
#312, #313, #314 and #315 were merged in that order, and every merge was clean.
#289 and #296–#315 are ancestors. The content of #291–#295 is present,
and #290 is superseded. The session-3 fixes below are separate commits on top
of merge head `ea3e61dd`.

Not integrated:
- #285 (perf bundle; CI-green at its own head `853c9cf7`): see QT-PRA-028.
- #286: only its cross-book risk idea was ported (QT-PRA-016).
- #287, #316 and #317 were excluded as features or duplicates.

### 8.2 Evidence register (session 3)

Every item below was reproduced by a test that failed before the fix and
passes after it. Status: TESTED. It becomes INTEGRATED once CI is green at
the RC SHA. Nothing here is VERIFIED ON HOST.

**QT-PRA-017 (P0): EOD paper settlement crashed in production.**
- Root cause: `providers.current_regime()` wraps `intel_book` in `InstitutionalPaperBookAdapter`. `settle_paper_session` and the intelligence loop call `mark(..., allow_entry_session=False)`, and the adapter's `mark()` did not accept that keyword. The result was a `TypeError`, `OUTCOME_RESOLUTION` retried forever, and no stop, target or time exit was ever applied.
- Fix: `45b5a1b7`. The adapter forwards `allow_entry_session`.
- Regression: two tests in `tests/test_outcome_liveness.py` exercise settlement through the adapter, including keeping the entry-session bar out of exits.

**QT-PRA-016 (P1): the 5% total open-risk ceiling was per book, not per account.**
- Root cause: legacy equity (`trades.db`), modern equity (`intel_book.json`) and F&O (`fo_paper.sqlite3`) each enforced their own 5%. A book that existed but couldn't be read counted as zero exposure.
- Fix: `12190d5e`.
  - `account_exposure_gate` reads the durable F&O ledger and returns `RISK_BOOK_UNREADABLE` when any existing book can't be read.
  - `FoPaperBook` subtracts equity open risk and is capped at the account capital.
  - The F&O cycle reads equity risk and commits its entries under `account_mutation_lock`.
  - The modern engine saves its book before releasing the lock.
  - The trade plan shows heat `UNKNOWN` instead of `OK` when risk can't be read.
- Regression: `tests/test_money_paths.py::TestAccountOpenRiskSpansEveryPaperBook` (13 tests) and `tests/test_trade_plan.py::test_unreadable_account_book_is_unknown_not_ok`.

**QT-PRA-018 (P1): outcome look-ahead from the decision-day bar.**
- Root cause: `core.outcome_resolver.first_touch_path` included the decision date's own OHLC. An EOD signal could be settled as a stop or target from price action that happened before the signal existed. Affected: `signal_log` → `live_edge` / `ev_engine`, Case memory and Evolution grading. The walk-forward backtest starts at the next session.
- Fix: `a775b773`.
- Regression: `tests/test_outcome_resolver.py`, covering the decision-day bar and the horizon counted after the decision.

**QT-PRA-019 (P1): US paper lane outside the 1% / 10% / 5% rails.**
- Root cause: the lane is armed by default and fed by the supervisor's US scans. Conviction stretched risk to 1.5%, the per-name cap defaulted to 20%, and there was no total cap. Its USD rows also counted as rupee risk and used NSE slots in the account gate.
- Fix: `241b0e5c`.
  - Ceilings are clamped at sizing time, which also covers persisted config.
  - Entries past 5% of the USD pool are refused.
  - An unreadable US book refuses entries.
  - `US_AUTOPILOT` rows are excluded from the NSE gate.
- Regression: `TestUsPaperLaneRiskRails` (5 tests).

**QT-PRA-020 (P1): a paper trade was reported placed when its journal write failed.**
- Fix: `7f7b4fd2`. `place_trade(paper=True)` returns `ok=False` unless the row was written.
- Regression: `TestPaperTradeIsOnlyOkWhenRecorded`.

**QT-PRA-021 (P1): US paper positions never closed; target wins booked above target.**
- Root cause: the `trades` DDL has no `exit_price` column; only the dormant legacy NSE autopilot adds it. On a journal written only by the US lane, every close `UPDATE` failed at debug log level. Separately, target exits booked the later polled price.
- Fix: `c2baff61`. `trade_executor.ensure_schema()` adds the column idempotently, close failures log as warnings, and target exits book the target price.
- Regression: `TestUsPaperTargetExitPrice` (2 tests, including an upgrade of a journal created with the old schema).
- Host check needed: whether the owner's `trades.db` already has the column.

**QT-PRA-022 (P1): US scan reported SUCCEEDED when its paper step failed.**
- Fix: `f8bf2271`. The operation finishes `BLOCKED/US_PAPER_FEED_FAILED` and keeps the scan summary. The supervisor does not rescan the same slot.
- Regression: 3 tests in `tests/test_us_parity_autonomy.py`.

**QT-PRA-023 (P1): OUTCOME_RESOLUTION reported SUCCEEDED when learning ingest failed.**
- Fix: `7f6886ad`. A `paper_learning` or `forward_soak` error gives `RETRYABLE_FAILED/PAPER_LEARNING_INGEST_FAILED`. Both ingests are keyed, so a retry is idempotent.
- Regression: 3 tests in `tests/test_outcome_integrity.py`.

Registered, not changed:
- **QT-PRA-024 (P2, dormant):** the legacy NSE autopilot sizes up to 2× of 1% risk and caps a name at 20%, or 40% in fast-book mode. It is disarmed by default, and its entry points (`breakout_sniper`, `telegram_commands`) are not started by the canonical stack. The account 5% gate still applies. It must be fixed before anyone re-enables it.
- **QT-PRA-025 (P2, dormant):** `PaperAutonomyManager.trade_day` marks the entry day's bar. Its signal source, `providers.signals_for`, reads legacy `auto_scan`, which is empty in the canonical stack. Existing tests encode the same-day behaviour.
- **QT-PRA-026 (P2, disclosed model):** modern equity paper fills at the card's intended entry (cards must be `ready`, not chase-risk, and under 36 hours old). Execution-reality runs as a shadow and the cycle response says so.
- **QT-PRA-027 (P2):** US live prices (yfinance `fast_info`) carry no timestamp, so their staleness can't be measured. They are used only for paper anchoring and exits.
- **QT-PRA-028 (perf, not integrated):** #285 bundles 5,473 inserted lines across 52 files from an older base. A trial merge needed 11 hunks in `data/bhavcopy_store.py` to reconcile two parallel bhavcopy-integrity designs. After resolution, 11 tests failed on conflicting semantics: #285 accepts a legacy archive without `DATE1`, which the RC rejects; and its fixture CSVs fail the RC's archive validation. Merge aborted. Its latency evidence (dashboard GET 347 s) is host-measured and was not reproduced here. A dedicated port is needed.

### 8.3 Release gates (supersedes 7.3)

| Gate | State | Evidence / blocker |
| --- | --- | --- |
| P0 defects resolved and validated | PARTIAL | QT-PRA-017 fixed and tested. The audit is targeted, not exhaustive. |
| No open P1 without a blocking decision | PASS (code) | QT-PRA-016 and 018–023 fixed. QT-PRA-013 resolved by #311 in the RC. |
| CI at release SHA | PENDING | All 5 workflows must be green on PR #319's exact head. See the PR. |
| Data / portfolio invariants under fault injection | PARTIAL | Hermetic: corrupt and unreadable books fail closed (QT-PRA-016). No physical detach. |
| Backup restoration | BLOCKED | Owner's Mac required. Backups must not be modified. |
| Hardware / external storage | BLOCKED | No host access. |
| Real-host performance and 72 h soak | BLOCKED | No host access. #285 is not integrated (QT-PRA-028). |
| Rollback verified | BLOCKED | No host access. |
| Frontend / API / Telegram | PARTIAL | vitest 25 files / 127 tests and production build pass. No live-data browser session. No real Telegram delivery. |
| Security controls | PARTIAL | Loopback binding, cross-site mutation guard, live interlock (`QT_ENABLE_UNSAFE_LEGACY_LIVE` off, `live_money_unlocked=False`): 24 test files pass. QT-PRA-008 tunnel residual and QT-PRA-015 pin remain open. |
| Independent release review | NOT TESTED | PR #319 awaits review. |

**Verdict: NOT READY.** The code-level P0 and P1 items found so far are fixed
and tested on one branch. Host, storage, restore, rollback, performance and
the 72 h soak gates are BLOCKED without access to the owner's Mac. CI at the
RC SHA must be confirmed. Live execution stays locked; everything is paper.

## 9. Session 4 — RC finalization branch (2026-11-09, from 02:40 UTC)

### 9.1 Branch and integration

`claude/quantterm-rc-finalization-20261109`, created from PR #319's head
`db42e6ccb17100b75b4844b892089593e29639e6` (`cursor/release-candidate-f43f`).
Canonical `claude/build-ai-trading-system-miHHd` is still `eda3d95f`; nothing
was pushed to it.

Integrated in this order, each as a real merge commit so the PR history stays
attributable:

| PR | Head verified | Merge | Notes |
| --- | --- | --- | --- |
| #321 exact-SHA host probe | `9585e1f8` | `7727b59` | Clean. The probe's checked fields all exist in `product.host_install.host_status()` (`supervisor.production_sha` is written at `product/host_supervisor.py`). Read-only: `service_action("status")` only. 26 probe tests pass. |
| #322 US settlement idempotency | `f54308db` | `d978322` | Clean. |
| #320 US state fail-closed | `6a4b6110` | `a5bb8c7` | Conflicted with #322 in `execution/us_autopilot.py`; resolved into one behavior (QT-PRA-029). |

Not integrated: #285 (see QT-PRA-036/037), #316/#317 (features, deferred as
instructed), #287 (research).

### 9.2 Evidence register (session 4)

Every fix below was reproduced first, either by a regression test that failed
before the fix or, where a failing test could only show a missing key, by a
probe against the prior code showing the wrong behavior. Status: TESTED. It is
INTEGRATED once the exact-head CI in 9.3 is green. Nothing is VERIFIED ON HOST.

**QT-PRA-029 (P1): #320 and #322 were two incompatible fixes for one defect.**
- #320 used a module-level `_state_load_error` latch and the label `STATE_RECOVERY_REQUIRED`; #322 used an in-state `state_recovery_required` flag and `US_PAPER_STATE_RECOVERY_REQUIRED`. Merging both verbatim gives two authorities for "is the US book trustworthy".
- Resolution `a5bb8c7`: one authority in the state dict (`state_recovery_required`, `state_recovery_error`), runtime-only (never read from disk, never written). Kept from #322: exact-ID settlement, legacy watermark migration flagged for reconciliation, fsync'd save under the lock, checkpoint before alerts, `review_cycle` propagates failures. Kept from #320: saved-state schema validation, read-only journal probe, blocked `set_config`.
- Gap neither PR closed: a migrated state that lost `accounted_close_ids` re-booked every closed trade. Probe on #322's head: realized P&L 97.50 → 196.67 after one settlement. The validator now type-checks the settlement ledger and requires the IDs once `accounting_ids_migrated` is true.
- #320's tests were adapted to the unified label and to deferred settlement raising instead of silently returning; their safety assertions are unchanged.
- Regression: `tests/test_us_paper_state_integration.py` (21 tests, all fail on both PR heads), plus both PRs' own files (18 tests).

**QT-PRA-030 (P1): detached runtime volume looked like a fresh US install.**
- With `QT_RUNTIME_ROOT_REQUIRE_EXISTING=1` and the volume gone, both `us_autopilot.json` and `trades.db` are missing. On #322's head the lane armed with the default $100k pool and `_save()` recreated the vanished volume path on the internal disk. The window is bounded by the 15 s `RuntimeStorageGuard`, but within it a US cycle could run.
- Fix (`a5bb8c7`): `_load` asks `core.runtime_paths.runtime_root()` before treating a missing file as a fresh install, and `_save` asks again before `mkdir`. Strict mode raises → recovery latched; no path is created.
- Regression: `test_detached_installed_runtime_is_not_a_fresh_install`, `test_save_refuses_to_recreate_a_volume_that_vanished_after_load`.

**QT-PRA-031 (P1): US lane could put 20% of its pool in one name.**
- `traded_symbols` resets each US session and was the only duplicate guard. A name still held from an earlier session could be bought again (two 10%-capped positions). NSE has `DUPLICATE_POSITION` in `portfolio_selection_authority`; F&O has `UNDERLYING_ALREADY_OPEN`.
- Fix `882721d`: `consider()` reads the open book once, strictly, before the live quote; a held name is refused, and the same read feeds the 5% open-risk check.
- Regression: `TestUsPaperLaneRiskRails::test_name_held_from_an_earlier_session_is_not_bought_again` (failed before: the second AAPL position opened).

**QT-PRA-032 (P1): expired F&O paper options were carried forever.**
- Supervision skips any open option without a quote, and Kite stops quoting a contract once it expires. `bars_held` advances only on quoted sessions, so missed sessions (e.g. no Kite login) can carry a position past the expiry the selector cleared. The position then kept its slot and premium risk in the 5% account budget, its outcome was never booked, and new entries continued. Probe on the RC: a 2026-09-29 contract was still open on 2026-10-07, and a new TCS position opened on 2026-10-01 beside it.
- Fix `2375892`: `product.fo_paper.expired_unsettled_positions()` is the one rule, used by the paper cycle and the desk projection. The cycle refuses new F&O entries (`EXPIRED_POSITION_UNSETTLED`), keeps the position durable and counted, and reports `expired_unsettled_positions`, `unquoted_open_positions`, `blockers` and `operator_action_required`; the Home F&O panel renders the blocker. No settlement price is invented.
- Regression: `TestFoExpiredPositionIsNeverSilentlyCarried` (4 tests).
- Owner action defined: reconcile the position against the exchange's expiry settlement. A tooled reconciliation path does not exist yet (see 9.5).

**QT-PRA-033 (P2): US daily-loss breaker attributed losses to the entry day.**
- Closed trades were summed by entry session. Probe on the RC: a $3,230.50 loss realized today on a position entered yesterday, against a $3,000 (3%) limit, left the lane armed.
- Fix `bf6549e`: additive nullable `trades.closed_at` (DDL + existing `ensure_schema` upgrade); US closes stamp it in US Eastern time; the breaker attributes by close session; legacy rows without it keep entry attribution.
- Regression: 2 tests in `TestUsPaperLaneRiskRails`.
- Host check needed: `ensure_schema` must add `closed_at` to the owner's existing `trades.db` on first US close (it is additive and idempotent).

**QT-PRA-034 (P2): one "network-free" test made a real market-data download.**
- `test_us_scan_feed_exception_is_recorded` stubbed `get_us_daily`, but `scan_us` uses `get_us_daily_batch`. It downloaded "AAA" and passed only where that request succeeded; it failed identically on the untouched RC base in this sandbox. Fixed in `b9c5fc4`.

**QT-PRA-035 (P3):** US Telegram BUY and CLOSED alerts printed a literal `\n`. Fixed in `a5bb8c7`; regression `test_closed_trade_alert_uses_a_real_line_break`.

**Display (`b101a55`):** an unrestored US paper state rendered as a routine "paper auto OFF". The Home card now shows BLOCKED with the reason, the disarm reason when present, and a reconciliation note for a legacy-watermark pool. `product/us_market_status.py` reports an unreadable US status as a blocker, not OFF. vitest: `frontend/src/usPaperTruth.test.ts`.

Registered, not changed:
- **QT-PRA-036 (P2):** `product/autopilot_journal.record_cycle` reads leniently, so a corrupt cycle journal is replaced by an empty one, and two writers can lose a row (no lock across the read-modify-write). It is a cycle blotter (latest 120 cycles), not financial state. #285 contains a locked, strict version that could be ported on its own.
- **QT-PRA-037 (perf, measured, not ported):** #285's `OperationStore` schema/connection cache. Measured on the RC (Linux, Python 3.11): construct plus `latest_summary` p50 0.12 ms, p95 0.33 ms; a construct-and-read while another connection held a 1.5 s write transaction was not blocked (0 ms). The claimed writer-lock contention is not reproduced, so the change is not justified by evidence here.
- **QT-PRA-038 (P3):** 12 private `_atomic_json` copies across `product/` and `research/`; no shared durable-write helper.
- **QT-PRA-039 (P2):** the US breaker's unrealized term is P&L since entry for every open position, not the day's change. That over-counts, which is the conservative direction for a breaker.

### 9.3 Tests executed this session

All hermetic: Linux container, Python 3.11.x venv from `requirements.txt`, Node 22 for the frontend, credentials blank. None of this ran on the owner's Mac.

- Integration-touching files run one by one at each step: US (`test_us_*`, `test_money_paths`, `test_home_os`, `test_momentum_breakout`, `test_runtime_root_isolation`, `test_terminal_audit_regressions`, `test_us_parity_autonomy`), F&O (`test_fo_*`, `test_fno_*`, `test_terminal_api`), the shared trades journal (18 files listed in the `bf6549e` run), and the release probe. All passed at the commit they were run against.
- Frontend at `b101a55`: `npm test` 26 files / 131 tests passed; `npm run build` succeeded.
- GitHub at `b9c5fc4` (push): CI, Product Acceptance, Terminal UI, Host Migration Gate, Dependency Vulnerability Advisory: all success.
- Later heads: see the PR. The exact-head results for the final SHA are recorded there, not assumed here.

Hermetic API latency (in-process `TestClient`, empty runtime root, 4 vCPU; **not** host evidence): `/api/health` cold 151 ms, warm p50 5 ms, max 6 ms; `/api/dashboard` cold 296 ms, warm p50 10 ms, max 12 ms. These numbers say nothing about a runtime with ~500 sessions of history and full-universe scans on an 8 GB Intel Mac.

### 9.4 Release gates (supersedes 8.3)

| Gate | State | Evidence / blocker |
| --- | --- | --- |
| P0 defects resolved and validated | PASS (code, hermetic) | No new P0 found in session 4. QT-PRA-001 and QT-PRA-017 remain fixed. The audit is targeted, not exhaustive. |
| No open P1 without a blocking decision | PASS (code) | QT-PRA-029…032 fixed with regressions. |
| CI at release SHA | PENDING | Must be all five green on this branch's final head (push and PR events). |
| Data / portfolio invariants under fault injection | PARTIAL | Hermetic: corrupt/missing US state, damaged settlement ledger, detached strict runtime, unreadable journal, expired F&O contract. No physical detach. |
| Backup restoration | BLOCKED | No host access. Recovery backups were not touched. |
| Hardware / external storage | BLOCKED | No host access. |
| Real-host performance and 72 h soak | BLOCKED | No host access. Only the hermetic numbers above exist. |
| Rollback verified | BLOCKED | No host access. |
| Kite data / Telegram delivery / real browser | BLOCKED | No broker session, bot token or host browser here. |
| Frontend / API integration | PARTIAL | vitest + build + hermetic Product Acceptance. |
| Security controls | PARTIAL | Unchanged from 8.3. Live interlock untouched; every change here is paper-only. QT-PRA-008 tunnel residual and QT-PRA-015 pin remain open. |
| Independent release review | NOT TESTED | |

**Verdict: NOT READY.** The code-level P0/P1 items found so far are fixed and
regression-tested on one branch. Production approval still needs green CI at
the final SHA, an independent review, and every host gate (storage, restore,
rollback, Kite, Telegram, browser, performance, 72 h soak), which are BLOCKED
without the owner's Mac. Live execution stays locked; everything is paper.

### 9.5 Next actions (priority order)

1. Review and merge this branch into `cursor/release-candidate-f43f`, then #319 into canonical; re-run all five workflows on the merge SHA.
2. On the Mac, with the exact SHA installed: run `scripts/quantterm_release_readonly_probe.py --expected-sha <sha>` (#321), the detach/reattach drill from section 5, and the restore drill into a disposable location.
3. F&O expiry reconciliation tooling (QT-PRA-032): a reviewed operator command that closes an expired paper position at the exchange's published settlement for that expiry, recorded as non-evidence. Until then the blocker holds new F&O entries.
4. Port #285's strict, locked `autopilot_journal` write (QT-PRA-036) on its own.
5. Profile `/api/dashboard` on the host before porting #285's dashboard snapshot cache (QT-PRA-037).

## 10. Session 4b — India-first directive (2026-11-09)

Owner directive: NSE equities and NSE F&O only; pause new US paper entries
without deleting or corrupting US state; live money stays locked. Work
continues on `claude/quantterm-rc-finalization-20261109` (PR #323). Base
`db42e6cc` and canonical `eda3d95f` are unchanged.

### 10.1 Evidence register

Each item was reproduced first, either by a failing test or by running the
prior code and showing the wrong result. Status: TESTED. Nothing is VERIFIED
ON HOST.

**QT-PRA-040 (directive): India-only mode.** `85f977b`
- `core/market_scope.py` is the one authority. `QT_MARKET_SCOPE=INDIA_ONLY` is the default; `INDIA_AND_US` opts back in; any other value fails closed to INDIA_ONLY.
- The existing `QT_US_MARKET_ENABLED=0` was not usable for this. It stops US scheduling entirely, and US exits and settlement only run inside `scan_us → review_cycle`, so open US positions would freeze.
- In INDIA_ONLY:
  - us_autopilot refuses entries at the execution owner (`_passes_gates`, `arm`, `on_setups`). `on_setups` evaluates nothing, so it records no counterfactual REJECT evidence.
  - The supervisor schedules `US_MARKET_SCAN` with `mode=SUPERVISE_ONLY`; market_ops runs `review_cycle` only (stops, targets, trailing, time-stops, exact-once settlement, breaker). No universe scan runs. Scope is re-checked at execution time.
  - A failed pass is `BLOCKED/US_PAPER_SUPERVISION_FAILED` and retries next slot, not every tick.
  - The Home projection and parity report `PAUSED_INDIA_ONLY`.
- Arm consent, positions, journal rows and learning records are untouched.
- Regression: `tests/test_market_scope_india_only.py` (16). Tests of the US lane's own gates now opt in explicitly.

**QT-PRA-041 (P2, intelligence): F&O volume confirmation compared a partial day with full days.** `9c31503`
- During the session the live scan's "today" bar holds Kite's cumulative volume so far, while RVOL divided by the 20 completed-day average. At 10:00, a stock trading at about three times normal read RVOL ≈ 0.7: the prefilter rejected it (< 1.0) and the false-breakout test (< 1.2) flagged it. The 20-day turnover also included today's partial volume.
- This failed conservative: fewer trades, never an unsafe one.
- Ported from #286 (head `c6b47995`, RVOL pieces only): `product/fno_intraday_volume.same_clock_rvol`.
  - Today is compared with the median of ≥5 prior sessions' observed Kite minute volume from 09:15 to the same minute; sessions with < 95% minute coverage are skipped. Nothing is extrapolated.
  - At most 16 minute-history calls per scan, spent on the strongest price breaks first. The threshold is ≥ 1.2.
  - Unavailable volume evidence gives `BLOCKED/FNO_INTRADAY_VOLUME_COVERAGE_UNAVAILABLE`, not "no setup".
  - Turnover now uses completed sessions only.
- The paper cycle still marks and exits positions when the directional scan is BLOCKED.
- Regression: `tests/test_fno_intraday_volume.py` (7).

**QT-PRA-042 (P1): option liquidity floors were in the wrong unit.** `34ff008`
- `min_volume=100` and `min_oi=500` had no unit and were compared with Kite quote volume/OI. A user report on Kite's developer forum (not official documentation) says Kite gives those in contract units (shares). On that basis 500 units is two lots of a 250-share contract: the floor barely bound, and the liquidity score saturated at 500 units.
- Floors are now 10 lots of volume and 50 lots of OI, measured with each contract's `lot_size`. A missing lot size is `LOT_SIZE_UNAVAILABLE`. Selected contracts record `volume_lots` and `oi_lots`.
- If Kite in fact reports lots, this only makes the filter stricter (fail closed).
- Regression: 2 tests in `tests/test_directional_option_selector.py`.

**QT-PRA-043 (P1): the "52-week" high and low were two-year extremes.** `1df8afd`
- `hi52 = max(high[:-1])` (and the short side's `lo52`) spanned the whole frame, and the bhavcopy store holds ~500 sessions. A new 52-week high below an older peak was labelled `BREAKOUT_RES`, and the laggard conviction cut (> 30% below the 52-week high) penalised a stock that was at its 52-week high. The short side mirrored this.
- Both now use the prior `YEAR_SESSIONS` (the scanner's existing 250-session falling-knife lookback).
- Regression: 2 tests in `TestUnifiedScanner` (both failed: `BREAKOUT_RES` / `BREAKDOWN_SUP`).

**Home best-five (no defect found).** The chain is:
1. `home_os._canonical_best_trade_rows` → `decision_simulation_gate.status`. It requires `scan_fresh` and `discovery_ready`; `discovery_ready` itself requires a non-empty scan id.
2. Each row goes through `recommendation_truth.project_candidate_truth` and must be canonical BUY and ENTER_NOW. At most 5 rows are kept.
3. A win probability is shown only when forward-paper effective N ≥ `CLAIM_MIN_EFFECTIVE_N`, labelled by source (`trading_thesis.decision_quality`). Historical-only evidence cannot produce a claimed probability.

**QT-PRA-047 (P1, caught by CI): the acceptance contract encoded the old US capability.**
- Product Acceptance failed at `eb94a5d` (PR event): `internal FAILED feature: US forward-paper parity truth`, the only failing row. The check hard-required `FORWARD_PAPER_PARITY`, `autonomous_scan` and `paper_auto_execution`, which India-only mode correctly turns off. The intermediate pushes' Product Acceptance runs had been cancelled by later pushes, so it surfaced only here.
- The projection also still claimed `forward_counterfactuals: true` while paused, though a paused lane records none.
- Fix: `scripts/_product_acceptance_core.grade_us_market_truth()` grades by reported scope: INDIA_ONLY requires the pause to be stated truthfully; INDIA_AND_US requires full parity; a missing scope fails. The projection now reports `forward_counterfactuals` from scope.
- Regression: 4 tests in `tests/test_market_scope_india_only.py` grade the real projection for both scopes and reject stale capability claims.

Registered, not changed:
- **QT-PRA-044 (P2):** the equity scanner's intraday `vratio` has the same partial-day bias as QT-PRA-041. A same-clock fix across ~2,000 symbols needs a bounded minute-history design (price breaks only, as in F&O). It is conservative today.
- **QT-PRA-045 (owner decision):** F&O "1% risk" is stop-based. The worst case for a long option is the premium, capped at 10% of F&O equity per trade and 10% per day, and an overnight gap can pass through a paper stop.
- **QT-PRA-046 (host check):** confirm on the Mac that Kite option `oi`/`volume` equal NSE option-chain lots × `lot_size`.

### 10.2 Release gates (supersedes 9.4)

| Gate | State | Evidence / blocker |
| --- | --- | --- |
| P0 defects resolved and validated | PASS (code, hermetic) | No P0 found in 4b. Targeted, not exhaustive. |
| No open P1 without a blocking decision | PASS (code) | QT-PRA-042/043 fixed. QT-PRA-045 needs an owner decision (registered). |
| CI at release SHA | PENDING | All five workflows, push and PR, at this branch's final head (see PR #323). |
| Data / portfolio invariants under fault injection | PARTIAL | Hermetic only. |
| Backup restoration, hardware/storage, rollback | BLOCKED | No host access. |
| Kite data, Telegram delivery, real browser | BLOCKED | No broker session or bot token; QT-PRA-046 needs the host. |
| Real-host performance and 72 h soak | BLOCKED | No host access. |
| Security controls | PARTIAL | Unchanged; live interlock untouched. |
| Independent release review | NOT TESTED | |

**Verdict: NOT READY.** Same reasons as 9.4: host gates are BLOCKED and no
independent review has happened. India-only mode is active by default; all
trading is paper.

## 11. P0 incident: a completed scan never reached Opportunities (2026-11-09)

Branch `claude/opportunities-p0-recovery-20261109`, based on
`integration/quantterm-india-universal-20261109` @ `980b922`. Paper only; the
live interlock is untouched.

**Host access: none.** No folder on the owner's Mac is shared with this cloud
session (`list_computer_folders` returned none). Nothing here was run on the
Mac: launchd, the sparsebundle runtime, the installed SHA, backups, Chrome on
the host, deployment and host measurements are all BLOCKED.

### 11.1 Reproduction (real app, real browser; sandbox, not the host)

- **Fixture.** `tests/fixtures/opportunities` holds the output of the real `run_whole_market_scan(save=True)`: real scanner, coverage ledger, long-term overlay and recommendation publication. Only the OHLCV input is synthetic, because NSE archives are blocked by this environment's proxy. It has 100 records and 22 BUY verdicts. A larger runtime of the same kind (827 records) was used for browser and latency runs.
- **Routes.** In `api.app:app`, which `run_quantterm_complete.sh → run_quantterm.sh` serves, there were two `GET /api/recommendations-workspace` routes, `product.observer_api` (served) and `api.runtime` (shadowed). Under `terminal_product_api_parallel:app`, which the launchd host supervisor serves, a startup hook replaced them with the persisted-first route. The handler therefore depended on the launch module.
- **Browser.** Chromium (Playwright) against uvicorn plus the Vite desk:
  - Opening Opportunities fired `POST /api/controls/RUN_SCAN_NOW` in the first second, because the dashboard read cache's WARMING placeholder had an empty scan.
  - "Desk snapshot refreshing · saved data only" flapped on through the minute (25, 40, 45, 55 and 60 s), and recommendations were hidden while it showed.
  - `/api/access` was requested once with a 4 s timeout; a failure would have set PUBLIC READ ONLY for the session.
  - PREPARING DATA came from the dashboard cache flag, not official history.

### 11.2 Defects fixed (each failed first, then passes)

| ID | Pri | Defect | Fix | Regression | Residual risk |
| --- | --- | --- | --- | --- | --- |
| QT-PRA-048 | P0 | Duplicate synchronous Opportunities GET; persisted-first route only under one launch module; the request rebuilt the workspace ("read waited 5.0 s") | `7f63791` | `tests/test_opportunities_p0.py` | `?refresh=true` no longer recomputes on GET; refresh goes through a scan |
| QT-PRA-049 | P1 | Observer GET served yesterday's saved cards when the scan was missing | `7f63791` | `test_missing_scan_withholds_saved_cards` | Opportunities stays empty during a scan outage (intended) |
| QT-PRA-050 | P1 | The browser judged currency from the 15 s dashboard cache | `7f63791` (`display_authority`), `66c8829` | vitest `recommendationAuthority.test.ts` | Older backends fall back to the previous rule |
| QT-PRA-051 | P0 | Opening Opportunities started a whole-market rescan | `66c8829` | vitest `opportunitiesReadOnly.test.ts`; browser: no `RUN_SCAN_NOW` | First-run scans now come from the scheduler / explicit click only |
| QT-PRA-052 | P1 | Dashboard cache rebuilt only after expiry, so every other 10 s poll was REFRESHING | `6a11363` | `test_snapshot_refreshes_ahead_so_steady_polls_stay_current` (was 6/12 REFRESHING) | One background build per ~7.5 s while polled |
| QT-PRA-053 | P1 | PREPARING DATA shown from cache staleness while official history was current | `66c8829` | vitest `deskStartupState.test.ts` | — |
| QT-PRA-054 | P1 | One failed `/api/access` → PUBLIC READ ONLY for the session | `66c8829` | vitest `accessTruth.test.ts` | Relies on `/api/health` access projection (canonical) |
| QT-PRA-055 | P2 | A failed recommendation publication was silent | `ccf37de` | `test_failed_publication_is_explained_not_silent` | — |
| QT-PRA-056 | P2 | `scan_is_fresh()` ignored `QT_SCAN_PATH`; a Kite-freshness test had become vacuous | `7f63791` | `test_scan_freshness_reads_the_same_scan_artifact...` | — |
| QT-PRA-057 | P1 | Telegram "sent" = any HTTP 2xx; no message id | `2c51fcd` | `tests/test_telegram_delivery_truth.py` (13) | A chat id written differently from what Telegram returns now reads as unconfirmed; check on host |
| QT-PRA-058 | P2 | Scan JSON parsed twice per Opportunities read | `f45e2c0` | existing suites | `scan_is_fresh()` still re-reads the scan (shared authority) |
| QT-PRA-059 | P0 | Old-day BUY cards served as current: a scan saved with `data_current=true` stayed "current" after the NSE session moved on | PR #331 (`693b4c3`, `f41689b`, `923e30d`), reconciled in `7601c52` | `test_canonical_get_withholds_old_day_buy...`, `test_stale_scan_provenance_withholds_saved_buy_without_rebuild`, `test_stale_official_scan_is_not_presented_as_current`, `test_invalidated_scan_provenance_withholds_cards_with_authority` | Opportunities is empty whenever `scan_is_fresh()` cannot verify today's session (intended fail-closed); confirm it reads true on the host after the scheduled scan |
| QT-PRA-060 | P1 | PR #331 on its own kept the startup hook, so under `terminal_product_api_parallel:app` (launchd) its gated route would be swapped for an ungated one | `7601c52` (hook stays removed; one route, registered at import) | `test_every_launch_module_serves_one_canonical_recommendations_route` | — |
| QT-PRA-061 | P2 | Concurrent `desk_pipeline.json` writers shared one `.tmp` path (ENOENT → HTTP 500) | PR #331 `3360e35` | `test_concurrent_pipeline_publish_uses_unique_atomic_temp_files` | — |
| QT-PRA-062 | P2 | A withheld (STALE) workspace still said `rebuilding: true` and "rebuilding in the background" | `7601c52` | `test_stale_official_scan_is_not_presented_as_current`; vitest STALE message | — |
| QT-PRA-063 | P2 | The fixture JSON was excluded by the repo-wide `*.json` ignore, so the P0 tests errored in a clean checkout | `5c25f7c` | per-file suite in a clean worktree | — |

**PR #331 overlap.** PR #331 (`fix/rc330-recommendations-fast-route-20261109`, another author) fixed the same route from a different angle and added the freshness gate. It was merged into this branch with its commits kept, and resolved to one implementation: the gate and withholding from #331, the import-time single route and `display_authority` from this branch. #331 can be closed in favour of this PR.

**`owner_paused`: no defect.** It comes from `owner_state.json new_entries_paused` (the owner's PAUSE or HALT control, or a disabled PAPER_AUTO preference on first start). It blocks only new paper entries; exits, scans, research and alerts continue. It was not changed.

### 11.3 After the fix (same browser harness, sandbox)

- **As-is runtime** (no official history in the sandbox):
  - no `RUN_SCAN_NOW`;
  - the refreshing banner appears only during the first 10 s warm-up;
  - the header shows HISTORY STALE, which is true here;
  - Opportunities shows "Saved scan is not current for this session";
  - no console errors.
- **Scan freshness forced true** (display verification only): the published workspace renders. It shows the decision gate and the Super Trends category: "Checked 78 scored names… Good setups: 48; watch: 30".
- **Latency** (4-vCPU Linux container, 827-record runtime, not the Mac):

  | Route | Sequential p50 / p95 / p99 | 8-way concurrent p50 / p95 / p99 |
  | --- | --- | --- |
  | `/api/health` | 10 / 13 / 15 ms | 125 / 253 / 295 ms |
  | `/api/recommendations-workspace` (before f45e2c0) | 69 / 165 / 174 ms | 1192 / 1377 / 1412 ms |

  After `f45e2c0`, the Opportunities read took 47 ms (was 92 ms) in-process. API RSS was 255 → 323 MB and fds 10 → 90 under the load test.

### 11.4 Release gates (supersedes 10.2 for this branch)

| Gate | State |
| --- | --- |
| Incident root cause reproduced and fixed (code, sandbox browser) | PASS |
| One canonical Opportunities route, reading never rebuilds or rescans | PASS |
| Python suite / frontend vitest + build at the final SHA | See PR (exact-SHA evidence) |
| All GitHub workflows at the final SHA | See PR |
| Installed host SHA, launchd, sparsebundle, verified backup | BLOCKED (no host access) |
| Deployment to the Mac and local-browser verification on the host | BLOCKED |
| Real Telegram delivery with message ids | BLOCKED (no bot token here) |
| Kite session and live data | BLOCKED |
| Mac p50/p95/p99, memory, swap, fds | BLOCKED (sandbox figures only) |
| 72 h soak, rollback | BLOCKED |

**Verdict: NOT READY.** The P0 path is fixed and verified in a real browser
against the real API in the sandbox. It has not been deployed or verified on
the owner's Mac, and the host, soak and rollback gates are BLOCKED.

## 12. Owner mandate on PR #332: 4/10 toward 10/10 (2026-11-09/10)

The standing specification is the owner directive on PR #332
(comment 6748303757). Work continued on the same branch
(`claude/opportunities-p0-recovery-20261109`), starting at `b2d850a`. Paper
only; the live interlock and owner pauses are untouched.

**Access.**
- No folder on the owner's Mac is shared with this cloud session (`list_computer_folders` is empty).
- NSE, Kite and Telegram hosts are unreachable from the container.
- Everything below was measured in a 4-vCPU Linux sandbox. The 827-record runtime was produced by the real scan pipeline from synthetic OHLCV.
- Every host, broker, Telegram, soak and rollback gate is BLOCKED, not PASS.

### 12.1 Defects found and fixed (each reproduced first, then a regression test)

| ID | Pri | Defect | Reproduction | Fix | Regression |
| --- | --- | --- | --- | --- | --- |
| QT-PRA-064 | P1 | Read-time TOCTOU: freshness was judged on one read of the scan file and the payload served from another, so a scan replaced in between inherited the old verdict | An older-session scan with the same `scanned_at`, swapped in after the freshness check, was served as CURRENT | `8b8ca6d`: `scan_payload_is_fresh(payload)`; the route parses once and judges and serves that parse; `display_authority` carries `scan_id` and `scan_market_session`; a missing scan reads SCAN_UNAVAILABLE | `test_freshness_is_judged_on_the_same_scan_artifact_that_is_served`, `test_absent_scan_is_reported_as_no_scan_not_as_stale` |
| QT-PRA-065 | P1 | 65 durable writers used a fixed `<name>.tmp`, so concurrent writers shared one inode. Real collisions: recommendation publication (pipeline vs API rebuild), `save_scan` and `instruments_cache.csv` (market_ops vs autonomy) | 6 concurrent publications: 5 raised `FileNotFoundError` | `a72a1c0`: `core.atomic_io` (unique mkstemp file, fsync, keep permissions, atomic replace, clean up on failure) at all 65 sites; guard test against the idiom | `tests/test_atomic_writers.py` |
| QT-PRA-066 | P1 | A background rebuild for generation A could overwrite generation B's recommendations, published by the pipeline while A was running | Deterministic interleaving test: final file held A | `ef03e92`: cross-process `publication_lock`, held by the pipeline from its first sidecar write through `save_scan` and by the rebuild around its generation check and save | `test_background_rebuild_never_overwrites_a_newer_pipeline_publication`, `test_generation_publication_holds_the_cross_process_publication_lock` |
| QT-PRA-067 | P2 | A rebuild that failed deterministically was retried every 30 s, each retry a full projection build, for as long as the page was open | 5 failures, 5 retries at 30 s | `ef03e92`: per-generation backoff 30 s → 15 min cap; the attempt count survives the rebuild claim | `test_a_deterministically_failing_rebuild_backs_off_...` |
| QT-PRA-068 | P1 | On a fresh runtime (autonomy not started yet), the first `GET /api/autonomy-desk` created an empty `logs/autonomy/jobs.db`; every later read was HTTP 500 "no such table: jobs" | Sweep of all 57 GET routes on a fresh runtime | `6eec070`: `connect_readonly` (mode=ro, None if absent); a missing store or table means no jobs | `tests/test_read_paths_no_side_effects.py`; sweep ×2: 0 errors |
| QT-PRA-069 | P2 | One Home request parsed 11.3 MB of JSON (recommendations file ×3, scan ×9) | `json.loads` instrumentation | `94c19de`, `3ece956`: `core.json_cache.read_json_shared` (keyed by inode/mtime/size; never caches a version replaced mid-parse) for read-only callers; Telegram delivery status reads the resolved scan path | cache tests in `test_atomic_writers.py` |
| QT-PRA-070 | P2 | Every Opportunities read re-hashed constant method rules about 780 times | cProfile | `4676cef6` (method identity memoised; each caller gets its own dict) | `test_method_identity_is_memoised_but_each_caller_gets_its_own_dict` |
| QT-PRA-071 | P2 | `telegram_delivery.delivery_status` read the import-time default scan path, ignoring `QT_SCAN_PATH` | Code inspection | `3ece956` | existing Telegram suites |

**Checked and not changed (evidence, no defect):**
- **Paper risk caps.** 1% per trade, 10% per name and 5% total open risk are enforced by `PaperBook`. The combined legacy, modern and F&O account gate (`account_exposure_gate` under the cross-process `account_mutation_lock`) covers duplicates, position count, sector and 5% risk. F&O premium worst case counts against the same account. Covered by `test_cross_engine_account_exposure.py`, `test_fo_premium_worst_case_risk.py` and `test_money_paths.py`.
- **Sandbox paper journey.** The 78 published cards produced NO_TRADE: 48 `HISTORICAL_EVIDENCE_PENDING`, because the history-first gate waits for the autonomous-evolution bootstrap, which needs official NSE history the sandbox lacks; 30 `LOW_QUALITY_SETUP`. This is the designed fail-closed result. Opening it on the host is a host gate.
- **Historical bootstrap.** `run_bootstrap` already holds a cross-process lock, so no duplicate replays. Residual (P2): whichever process first evaluates policies hosts the replay thread. That includes `GET /api/decisions`, which no desk view currently calls.
- **Regime warm-up.** A single-flight background thread; only Kite and Yahoo are network calls, after the official index store. Bounded, not on the request path.
- **Hung rebuild.** The projection build is network-free (quotes are `None`, overlays are local files) and its lock wait is bounded (300 s, then FAILED with backoff). No watchdog added.
- **Reverted experiment.** Pre-rendering the Opportunities JSON to skip FastAPI's encoder was 59 ms vs 49 ms p50 on FastAPI 0.162, whose pydantic-core serializer is faster. The 71 ms "encoder" cost seen in a profile was not on the request path.

### 12.2 Verification at the code head `6eec070` (sandbox)

- **Python suite:** 431 files and 4,357 collected tests, one interpreter per file in a clean worktree, credentials blanked. Every file exits 0; no failures, errors or skips.
- **Advisory integration gate:** `tests/integration`, `continue-on-error` in CI with a 120 s limit, did not finish within 300 s here. Recorded as not run, not as a pass.
- **Frontend:** vitest 33 files and 167 tests pass, `tsc` clean.
- **GET sweep:** all 57 GET routes, twice, on a fresh runtime: zero 5xx and no job store created.
- **Browser E2E:** Chromium against uvicorn plus the Vite desk, 60 s per scenario.
  - Under `api.app:app` and under `terminal_product_api_parallel:app` (launchd), the stale sandbox scan shows "Saved scan is not current", with cards withheld by the server.
  - With freshness forced and current provenance (display check only), the published workspace renders.
  - All three: no `RUN_SCAN_NOW`, no console errors.

**Latency.** 30 sequential plus 40 at 8-way concurrency per route, same 827-record runtime and machine, starting commit `b2d850a` vs head (ms):

| Route | seq p50 b2d850a → head | seq p95 | 8x p50 | 8x p99 |
| --- | --- | --- | --- | --- |
| `/api/radar-home` (Home) | 130 → **71** | 216 → 178 | 1413 → **856** | 2049 → 1397 |
| `/api/recommendations-workspace` (920 kB) | 46 → **35** | 133 → 103 | 780 → 743 | 954 → 959 |
| `/api/research-director` | 37 → 19 | 132 → 30 | 666 → 427 | 878 → 566 |
| `/api/learning-dashboard` | 28 → 13 | 121 → 43 | 398 → 175 | 678 → 335 |
| `/api/research-status` | 15 → 8 | 21 → 8 | 169 → 88 | 328 → 128 |
| `/api/command-center-workspace` | 20 → 13 | 88 → 16 | 265 → 188 | 484 → 286 |
| `/api/health` | 9 → 8 | 66 → 10 | 92 → 90 | 208 → 175 |
| `/api/dashboard` | 5 → 6 | 36 → 17 | 24 → 39 | 58 → 50 |

API RSS after the same load: 337 MB → 316 MB.

These are container figures. The 8 GB Intel Mac is expected to be slower and has not been measured.

### 12.3 Evidence scorecard (1–10)

**Rubric, per area:**
- 1–3: known P0/P1 defects remain, or the area is unverified.
- 4–5: defects fixed with tests and verified only in a sandbox; real-world gates unproven.
- 6–7: no known P0/P1, verified end-to-end in a realistic environment, real-host evidence partial.
- 8–9: verified on the real host under sustained operation, with budgets met.
- 10: every relevant gate proven on the host, including soak, restore/rollback and real broker/Telegram receipts.

A 10 is not awarded anywhere.

| Area | Start (`b2d850a`) | Now | Evidence and what limits it |
| --- | --- | --- | --- |
| (a) User-visible workflow reliability | 4 | 5 | 57-route sweep clean ×2; Opportunities and Home verified in a sandbox browser under both launch modules. No browser check on the Mac. |
| (b) Data provenance/freshness & decision correctness | 5 | 6 | One-parse freshness (TOCTOU closed); generation-safe publication; withheld old-day cards; history-first gate fails closed. Real NSE/Kite data path unverified. |
| (c) Paper-risk & crash safety | 5 | 6 | Atomic, fsynced durable writes at all 65 sites plus a guard test; cross-engine 1%/10%/5% caps tested. Restart/recovery of the real paper state on the host unverified. |
| (d) UI/API performance on real hardware | 3 | 3 | Sandbox: Home −45% p50, no memory increase. No measurement on the 8 GB Intel Mac, so not raised. |
| (e) Unattended operations/alerts | 3 | 3 | Rebuild backoff, no-500 desk on a fresh runtime. No real Telegram receipt, no soak. |
| (f) Recovery/backup/deployment | 2 | 2 | No host access: backups, restore, rollback and deployment unverified. |
| (g) Test coverage & exact-SHA delivery | 7 | 8 | +15 regression tests (races, concurrency, side-effect-free reads); full suite and all workflows at the exact SHA (see PR). |
| **Mean** | **4.1** | **4.7** | Limited by the BLOCKED host gates, not by known code defects. |

### 12.4 Release gates

| Gate | State |
| --- | --- |
| No known open P0/P1 in code reachable from the sandbox | PASS |
| Python suite, vitest and typecheck at the code head | PASS |
| All GitHub workflows at the final SHA | See PR #332 |
| Real-browser desk verification (sandbox) | PASS |
| Installed host SHA, launchd owners, sparsebundle identity/durability, backups | BLOCKED (no host access) |
| Controlled paper-only deployment and browser check on the Mac | BLOCKED |
| Real Kite session/quotes; real Telegram receipt with message id | BLOCKED (no network or credentials here) |
| History-first gate opening on real official history (paper trades actually flow) | BLOCKED |
| Mac p50/p95/p99, memory, swap, fds, CPU | BLOCKED |
| 72 h supervised soak; restore and rollback drill | BLOCKED |

**Verdict: NOT READY.** No known P0/P1 remains in code reachable here, but
every host, broker, Telegram, soak and rollback gate is unproved.

### 12.5 Host handoff (the one owner action needed)

**Give a session access to the Mac.** Either share the QuantTerm folder via
Remote Control (Claude desktop app → Settings → Claude Code → Remote Control),
or run a Claude Code session on the Mac in that folder. Credentials stay in
`.env`; none are needed in chat.

**With access, the remaining gates run in this order.** Everything stays
read-only until step 4, and live-money stays locked.
1. Record the installed SHA vs `origin`, launchd owners, sparsebundle identity and mount, free space and the latest backup.
2. Take a fresh verified backup and write a rollback note (previous SHA plus restore command).
3. Run the suite on the Mac.
4. Deploy the PR head through `scripts/run_quantterm_complete.sh` (no duplicate launcher), paper only.
5. Real-browser check of Home, Opportunities and the autonomy desk under the persisted scan generation.
6. One real Telegram message: confirm the receipt and that its message id is stored.
7. Kite login and quote check.
8. Confirm the history-first bootstrap reaches `analysis_complete` and that a paper cycle can enter on its own.
9. Measure p50/p95/p99, RSS, swap and fds.
10. Start the 72 h soak; rehearse rollback.

**Needs the owner's explicit approval first:** a disk-detach test, a destructive restore, enabling live trading.
