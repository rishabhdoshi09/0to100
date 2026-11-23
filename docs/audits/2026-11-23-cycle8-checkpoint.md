# Cycle 8 checkpoint: Stage 2 read-only checks, F1 remediation, checkpoint barriers (23 Nov 2026)

This journal records facts only and contains no secrets.

**Status:** development only.

- **Live trading:** HARD-LOCKED.
- **Production:** `125880a6`, not deployed, restarted or repaired.
- **Data:** no ledger mutated. Production was only read: the runbook §0 commands, hashes, and byte copies taken to the SSD.

## 1. Stage 2: read-only checks (23 Nov, 08:44–08:46 IST, before the open)

**§0 preflight, as written: every line passed.**

- AC power; only the WORKING image attached.
- Free space: runtime 103 GB, Expansion 412 GB, SSD 11 GB.
- Checkout clean at `125880a6`, and `8f9dc5e0` is a fast-forward of it.
- Health READY; 0 RUNNING jobs; CI 5/5 on both branches.
- `host_install status`: live LOCKED, HEALTHY.
- launchd holds only the desk and the relay.

**Identity facts.**

- Store pickle `76133eaf…`, last written 20 Nov 23:38.
- Newest archive: 20 Nov.
- The Stage 1 manifest is intact.
- Paper-state hashes snapshotted to `$M/paper_state_stage2_morning.sha256`.

**The store re-plan (the 11 gates) is deferred to after 21:00 IST tonight.** Two reasons:

- §0 itself requires running outside 09:15–15:30 IST.
- Today's end-of-day archive will change the pickle, and a plan made before it is stale by the runbook's own rule.

A wake-up is scheduled for 21:15 IST.

## 2. Finding F1 and its root cause

See `docs/audits/2026-11-23-f1-freeze-ledger-index.md`.

**Both data losses came seconds before a lid-close sleep** with the USB runtime attached:

- 16 Nov 10:21, 84 s after the lost write;
- 19 Nov 11:06, about 30 s after it.

**Why the order went wrong.** `decision_freeze.freeze()` opens and closes a connection per decision, so every freeze checkpoints. With no `F_FULLFSYNC` barrier, the checkpoint's database writes and the WAL deletion reached the disk in any order.

## 3. Fixes

| Commit | What |
|---|---|
| `7d078c45` | `core.atomic_io.make_sqlite_checkpoint_safe` on every WAL store, plus a repository guard test |
| `e06701c1` | `product/sqlite_audit.py`: the full `integrity_check`, since `quick_check` reported F1's ledger as `ok` |
| `dfe062e3` | `product/sqlite_index_repair.py` v1, runbook stage 3b, the F1 investigation |
| (this commit series) | The review's findings below: the tool redesigned as v2, in place under SQLite's exclusive lock |

**Correction to `7d078c45`'s message.** It says the barrier "costs nothing per commit". That holds only for persistent connections. A store that opens a connection per write checkpoints on every close, so each such write pays one full flush. On 18 Nov the decision-freeze ledger took 3,026 freezes.

**The real cost is to be measured on the USB stack after the close.** The docstring and the F1 report now say this.

## 4. Independent GPT review (`8f9dc5e0..dfe062e3`, one pass)

**Method.** Each sketch was saved verbatim to `tests/test_cycle8_review.py`, and all 14 failed on `dfe062e3`. The F1-07 sketch failed, but for a different reason (see its row). Adaptations are marked `ADAPTED` in the file.

| Finding | Status | Fix |
|---|---|---|
| F1-01 (P0): a writer can commit between the last check and the rename | **Confirmed** | v2 never renames. The repair is REINDEX in place, under `locking_mode=EXCLUSIVE`, held through the checks, the backup, the repair and the verification. A writer in another process is blocked throughout (tested) |
| F1-02 (P0): an interrupted rollback could erase newer writes | **Confirmed** | In every state, rollback refuses content that is neither the repaired nor the original |
| F1-03 (P0): an edited manifest could install an unrelated database | **Confirmed** | v2 installs no candidate. Apply verifies the live content is the planned original and re-checks every required gate, not just `ready` |
| F1-04 (P0): `--out` could overwrite the target; symlinked targets | **Confirmed** | Outputs may not be in the database's directory, and existing non-manifest files are refused. Symlinked targets are refused |
| F1-05 (P1): a WAL appearing during the copy; `check` passing a base file | **Confirmed** | Sidecars are checked before and after, together with the descriptor's and the path's identity. `check` returns inconclusive (exit 3) on a hot WAL |
| F1-06 (P1): the `LIKE 'sqlite_%'` filter, a shadowed rowid, the header and FK proofs | **Confirmed** | Tables come from `PRAGMA table_list`, through an unshadowed rowid alias. A covering scan is required. Header bytes 0-23 and 40-91 must be unchanged, with the schema cookie split out (§6). The full FK list is compared |
| F1-07 (P1): the audit turned a live-view integrity failure into a pass | **Confirmed as a classification defect; the sketch's premise does not hold** | A live view that opens and reports diagnostics is `LIVE_VIEW_INTEGRITY_FAILURE` and escalates. Measured: on SQLite 3.45 a read-only `integrity_check` does not verify CHECK constraints, so the sketch's live view reads `ok` |
| F1-08 (P2): the guard misses spellings; `host_install` checkpoints without the barrier | **Confirmed** | `host_install._checkpoint_sqlite_sources` and `sqlite_runtime.connect(wal=False)` on a WAL-mode file now get the barrier. The guard is case-insensitive and covers `wal_checkpoint` |
| F1-09 (P2): unique lookups ignored the index collation | **Confirmed** | Each lookup uses its index's collation |
| F1-10 (P2): the stage-3 F1 copy was never compared; the rollback command was not runnable | **Confirmed** | The copy fails with `F1_BACKUP_MISMATCH` against the plan. There are two complete rollback commands. Stage 2 branches on `nothing_to_repair`. Tests run the blocks as written |

## 4b. GPT verification pass (`dfe062e3..6ebb95bf`, the one the protocol allows)

**Status of the earlier findings:**

| Status | Findings |
|---|---|
| Closed | F1-05, F1-06, F1-09 |
| Closed for the original case | F1-02 |
| Partial | F1-01, F1-03, F1-04, F1-07, F1-08, F1-10 (all now fixed below) |

**New findings.** 8, **all confirmed**: each sketch, saved to `tests/test_cycle8_verification.py`, failed on the code at its claim. Two sketches are adapted, each marked `ADAPTED`: V2-03 to the new `verify=` argument, and V2-08 to the new block.

| Finding | What it was | Fix |
|---|---|---|
| V2-01 (P0) | Rollback hashed the backup, then reopened it by path: a file swapped in between, or an edited `backup_sha256`, was restored (200 rows became 1) | It restores only from a PRIVATE snapshot, proven by sha256 AND by content to be the pre-repair ledger before the live one is touched |
| V2-02 (P0) | The "already applied" proof dropped the integrity result: an out-of-order index passed | The state proof carries the integrity result, stale, missing and lookup totals, and unsupported objects |
| V2-03 (P0) | A recorded backup that is a hard link to the target was hashed under the lock, and that close dropped SQLite's lock; it was also accepted as the backup | A recorded backup is checked before the lock and never reused when it is the live inode. Nothing caller-named is opened under the lock |
| V2-04 (P0) | The shared manifest writer opened a predictable `<name>.tmp-<pid>` with `"w"`: a symlink there **overwrote the source ledger with JSON**. Staging symlinks, and the `/Volumes` root itself, escaped the guards | The tool's own writer uses an exclusive, no-follow temp file. Output confinement compares filesystem identity up the resolved path. `_on_live_volume` includes `/Volumes` |
| V2-05 (P1) | A malformed schema committed only to the WAL raised during the check and was classified as a sidecar constraint (exit 0) | Errors are classified by SQLite result code: CORRUPT and NOTADB escalate at any stage |
| V2-06 (P2) | `apply` ignored the conditional `expected_stale` gate, and an edited `expected_content` was found only after COMMIT, leaving an unrecoverable state | Every recorded gate must pass. The trial repair is proven against the plan INSIDE the transaction, BEFORE COMMIT; a mismatch rolls back with nothing committed. `applied_unverified` (a header-only difference after a verified commit) is accepted by rollback |
| V2-07 (P2) | Writable readers of existing WAL stores (`reco_methods`, `operator_health`) could checkpoint on close without the barrier | Both apply the barrier |
| V2-08 (P2) | In an interactive shell, a failed `cp` left an older matching SSD copy accepted | One chain: remove, copy to a temp file, verify, rename; any failure prints `F1_BACKUP_MISMATCH` and leaves no usable copy |

**Mutants.** 9 of 11 new guards, reverted on their own, fail a test.

- The content proof's integrity field first survived, because the lookup field also catches the reviewer's case. A freelist-damage test now pins it on its own.
- The lookup field is a defensive layer: `integrity_check` sees every lookup failure that could be constructed.

There is no further GPT loop this cycle.

## 5. Found while fixing (by this cycle's own tests)

- **A leaked candidate file.** v1's `apply` left its verified 25 MB copy beside the target when a crash hit the manifest write. Fixed before v1 was committed.
- **v2's first draft dropped its own lock.** It read the live file with ordinary `open()`/`close()` while SQLite held its exclusive lock, and on POSIX that close releases every lock the process holds. Every byte read now goes through one descriptor, opened before SQLite and closed after it. A mutant that copies with plain file I/O lets another process write (killed by test).
- **A weak test.** A symlink test matched the word "symlink" in the pytest directory name. It now matches the refusal's own words.
- **The runbook test hit the real desk.** An in-process apply in the "run as written" test found the real desk running on the Mac and refused, which is the production guard working. The test now uses copy mode for that step.

## 6. Measured SQLite facts (3.45.1)

- **`quick_check`** reports `ok` on F1's ledger on the Mac's 3.45.1, the production host; CI's Linux build flags the entry count. `integrity_check` catches it on both.
- **A read-only connection's `integrity_check`** sees committed WAL rows but does not verify CHECK constraints; a read-write connection's does.
- **What `REINDEX` changes in the header:** only the change counter, the version-valid-for number and, on the real file, the freelist. Root pages and the schema cookie are unchanged.
- **The backup API** bumps the destination's schema cookie (3 to 4), so a restored ledger has identical content but different bytes.
- **`locking_mode=EXCLUSIVE`** keeps both readers and writers out after `COMMIT`, through checkpoints and backup-API restores, until the connection closes.

## 6b. Other defects found this cycle

- **A time-bomb test fired on 23 Nov.** `test_research_coverage_is_not_quality_score` fails on the base commit too: it judged a Jun 2026 quarter against the wall clock, so it went stale once the Sep 2026 results deadline passed. Pinned to the fixture's fetch date; 24 other calls in that file still use the wall clock.
- **Platform-dependent audit tests.** A read-only open of a WAL file without `-shm` fails on CI's Linux SQLite and works on the Mac's.
- **The power watch** polled once a minute and only alerted. It now polls every 5 s and flushes the runtime on battery (`83cbc862`).

## 6c. The barrier's measured cost (23 Nov, 20:33 IST, after the close)

**Setup.** A throwaway APFS sparsebundle (`QTBench`) on the Expansion USB disk: the production storage stack, never the WORKING or backup image. It was deleted afterwards. The desk was running its end-of-day catch-up at the time, so these are contended numbers. Script and results: `$M/bench_barrier.py`, `$M/bench_barrier.json`.

| Operation | Without barrier | With barrier |
|---|---|---|
| Bare `F_FULLFSYNC` | n/a | median 43 ms (p90 52, max 108) |
| Connection per write, as in `decision_freeze.freeze()` | median 2.2 ms | median 309 ms, p90 1,020 ms, max 1,596 ms (mean about 0.5 s) |
| Persistent connection | median 0.1 ms | median 0.1 ms (max 358 ms when a checkpoint lands) |

**Two earlier claims were wrong.**

- The "~0.4 s per flush" in `make_sqlite_durable`'s docstring overstates the flush itself, which measures 43 ms.
- The real cost is the connection-per-write pattern: every close checkpoints.

**Volume.**

- The counterfactual freeze store took 2,001 freezes on 11 Nov, 1,800 on 12 Nov, 1,297 on 18 Nov, and about 430–450 on ordinary days.
- `decision_freeze.db` took 3,026 on 18 Nov.
- So a heavy day reaches more than 4,000 freezes.

At about 0.5 s each, that is roughly 4 minutes on a light day and 35 minutes on a heavy one. All of it falls in background scan and paper jobs, never the UI.

**Superseded on 24 Nov (below).** The 23 Nov decision called the barrier's cost acceptable. A larger measurement found a 110 s stall, so the freeze stores now keep one idle connection per process.

**Re-measured 24 Nov, 01:31–01:37 IST.**

- **Setup:** a throwaway `QTBench2` sparsebundle on the Expansion disk, deleted afterwards. The desk was down and the full test suite was running on the SSD, so these are loaded numbers.
- **Each run:** n = 300 per variant, plus 60 s sustained runs.
- **Files:** script `bench2/bench.py`, results `$M/bench_barrier2.json`.

| Variant (one freeze write) | p50 | p95 | p99 | max | ops/s | 3,026 a day |
|---|---|---|---|---|---|---|
| Production today: connection per write, no barrier | 3.4 ms | 10 ms | 32 ms | 342 ms | 179 | ≈17 s |
| **Candidate `2787a690`: per write + barrier** | 280 ms | 922 ms | 964 ms | 1.04 s | 2.6 | ≈19.5 min |
| Candidate, 60 s sustained | 244 ms | 954 ms | **109.7 s** | **109.7 s** | **0.1** | (one stall) |
| Per write + **one idle keeper** + barrier, `synchronous=NORMAL` | **0.95 ms** | 2.2 ms | 5.1 ms | 227 ms | 433 | ≈7 s |
| Per write + keeper + barrier, `synchronous=FULL` | 85 ms | 551 ms | 760 ms | 858 ms | 7.8 | ≈6.4 min |
| Persistent connection + barrier | 0.07 ms | 0.17 ms | 0.28 ms | 733 ms | 487 | — |
| Bare `F_FULLFSYNC` | 58 ms | 352 ms | 614 ms | 623 ms | 11.9 | — |

**Decision, final after GPT C8.6-01: row 5, a keeper plus `synchronous=FULL`.**

- **What it is:** `sqlite_runtime.keep_open` holds one idle keeper connection per process and freeze store. It is pid-keyed, capped at 8 per process, reset in a forked child, and closed at exit.
- **Why the keeper:** a per-write close is then not the last close, so it no longer pays a barriered checkpoint. That checkpoint caused the 280 ms p50 and the 110 s stall.
- **Why `FULL`:** each freeze commit is one `F_FULLFSYNC` of the WAL, so a freeze is **durable when `freeze()` returns**.
- **Why that matters:** the counterfactual JSONL references the freeze's id at once, and a reference must never outlive its freeze in a power cut. Row 4 (`NORMAL`, 0.95 ms) fails that test: its commits sit in unsynced WAL. `2cf44fc0` had chosen it, and GPT caught it.
- **Cost:** p50 85 ms, p99 760 ms and max 0.86 s per freeze. That is about 6.4 minutes on a 3,026-freeze day, in background paper and scan jobs, with no 110 s stall in its sustained run.
- **The barrier still holds:** every checkpoint is still barriered.
- **The next step is group commit:** one barrier per scan's batch of freezes. It belongs with the O(n²) ledger deduplication fix (Open), on the same code path.
- **The WAL now persists while the desk runs.** A clean stop checkpoints it, while a killed desk leaves it for the next opener. The repair tool's `plan` refuses a hot WAL, so stage 3b still needs the desk cleanly stopped.

## 6d. Host events after the close (23 Nov)

| When (IST) | Event |
|---|---|
| 10:47 | The USB disk unmounted on battery; the desk failed closed until the lid was opened at 14:31 |
| 15:10 | The battery reached 2%: Low Power Sleep, then AutoPowerOff until 20:23 |

**The desk missed the close and the evening jobs.** After the wake the WORKING image was attached through a stale disk-image process (Errno 5). A plain `hdiutil detach` was done, as on 22 Nov. The supervisor re-attached the image within about 60 s, the desk returned at 20:28, and the end-of-day archive landed at 20:28.

**Two more paper cycles were orphaned** (10:47 and 15:02). They were retired correctly as `ORPHANED_PAPER_CYCLE_RESTART`.

**Later that evening (evidence: `pmset -g log`, `/Library/Logs/DiagnosticReports/Kernel-2026-11-23-212503.panic`):**

| When (IST) | Event |
|---|---|
| ~21:04 | The charger came out. At 21:05:11 the Mac was **awake on battery at 100%**, and the Seagate was absent from the USB bus. It sits directly on the bus, with no hub |
| 21:05:18 | **Kernel panic** `wire_count == 0 @vm_pageout.c:7661`, with `com.apple.driver.DiskImages` in the backtrace. A desk process was inside a WAL commit's `F_FULLFSYNC` at that moment |
| 21:24–21:25 | Reboot. Battery health flipped to "Poor / Check Battery" (capacity 65%). launchd restarted the desk by itself, and storage re-attached |
| 22:14–22:33 | The stage 2 store re-plan ran and passed (section 7). The 21:00 run had been cut by the drop (EIO on `13012025.csv`) |
| 22:40:10 | Clamshell sleep on battery (45%). The WORKING image was left attached but unmounted, and the desk was down from then on |

**Findings.**

- **A USB drop is now a whole-machine crash.** On this macOS 12.7.6 host, the drive leaves the bus on the AC→battery switch, even with the lid open and the Mac awake. A degraded battery that cannot carry the bus-powered HDD's transient is a plausible cause, not yet proven. The disk-image driver then panicked the kernel.
- **No software fix exists on this host.** The remedies are physical: keep the charger in, use a powered hub or self-powered disk, service the battery, or move the runtime to internal storage.
- **The pickle survived.** `store_cache.pkl` read `e5accf5a…ffe0dbfa` before and after both events.

## 6e. Two independent reviews of `6ebb95bf..62750347` (GPT C8-01..07 and a Claude subagent)

Every finding was reproduced by a failing test first, and all are fixed in `2787a690`. The tests are in `tests/test_cycle8_part4.py`, which maps each finding to its test. Targeted mutants: **15 of 15 killed.**

| Finding | Severity | Fix |
|---|---|---|
| **C8-01** An edited manifest makes rollback install another database | P0 | Rows, schema, settings and foreign keys are compared database to database, never through the manifest |
| **C8-02** A verified file was re-opened by name: the restore source, and a lock release through a hard link | P0 | The live proof is diagnosed in memory, from one read through the pinned descriptor. The backup is written `O_EXCL\|O_NOFOLLOW` and verified through its own descriptor. Rollback restores from the in-memory connection it proved |
| **C8-03** A manifest directory or the target is swapped after the checks | P0 | The manifest is resolved once and must be `.json` outside the database's directory. The pin opens `O_NOFOLLOW` on the checked inode |
| **C8-04** A hard-linked target hides committed WAL frames | P0 | `st_nlink != 1` is refused |
| **C8-05** A header mismatch plus a crash after COMMIT strands recovery | P2 | The trial proof is recorded before COMMIT |
| **C8-06** A crash during the backup write leaves an untracked file | P2 | The backup is written at its recorded name (refined in 6f) |
| **C8-07** A stalled flush blocks the battery warning | P1 | The flush runs in its own thread, one at a time |
| **S-2** A rollback-journal target cannot recover after a hard crash | P2 | Only WAL databases are planned |
| **S-4** Re-running the stage-3 block deletes the verified SSD copy | P2 | A fresh copy every run; an existing copy is never deleted |
| **S-5** `..` is collapsed as text before symlinks resolve | P2 | `..` is refused |
| **S-6** `/System/Volumes/Data/Volumes/...` defeats every `/Volumes` guard | P2 | `core/live_volume.py` answers by identity; 9 guards route through it, and a test forbids new text-only checks |

## 6f. GPT review of the 6e fixes (`62750347..2787a690`, C8.5-01..10)

All 10 are reproduced and fixed in `2cf44fc0`, with tests in `tests/test_cycle8_part5.py`. Mutants: **16 of 17 killed, plus 3 of 3 re-run after strengthening**. The one survivor is an equivalent mutant: a redundant belt-and-braces read.

| Finding | Severity | Fix |
|---|---|---|
| **C8.5-01** Partial-backup cleanup deletes whatever an edited manifest names | P0 | Nothing is ever deleted. Earlier attempts' files are recorded in `earlier_backups` |
| **C8.5-02** `WITHOUT ROWID` rows escape the rollback's row proof | P0 | Rollback refuses unsupported objects on either side |
| **C8.5-03** `_fsync_dir` opens a name that a swap can point at the database | P0 | `O_DIRECTORY\|O_NOFOLLOW` |
| **C8.5-04** `on_live_volume` normalized `..` before resolving | P0 | The identity walk follows the spelling as given, in kernel order. Guards check before resolving |
| **C8.5-05** The trial flag let rollback ignore any later header write | P1 | The flag is bound to the attempt. Rollback accepts only the header apply observed after its own proven commit, and WAL is checked through the pin before SQLite opens the file |
| **C8.5-06** A matching pending backup was reused without being synced | P1 | Never reused: every attempt writes and syncs its own |
| **C8.5-07** The manifest was re-resolved after the guards | P1 | Resolved once; `..` is refused everywhere |
| **C8.5-08** The stage-3 block certified a symlinked copy and trusted an inherited `T` | P1 | `T` starts empty. OK only for a separate regular file (`-ef`), verified after publication |
| **C8.5-09** A slow alert send delayed the flush | P2 | The flush starts first |
| **C8.5-10** The audit test's window opened late | P2 | The window opens at the maintenance connection, and opens are classified by their flags |

## 6g. GPT review of the 6f fixes (`2787a690..2cf44fc0`, C8.6-01..09)

All 9 are reproduced and fixed, with tests in `tests/test_cycle8_part6.py`. Mutants: **10 of 10 killed.**

| Finding | Severity | Fix |
|---|---|---|
| **C8.6-01** The keeper left a returned freeze in unsynced WAL | P0 | `synchronous=FULL` with `F_FULLFSYNC`: durable at return (6c) |
| **C8.6-02** An anchor behind an alias of an alias escaped the live-volume walk | P0 | A component-wise resolver: every component met through every symlink is compared with the anchor, and a loop (over 40 links) is refused |
| **C8.6-03** A header first seen after a restart could authorize rollback over a later write | P2 | Apply records `applied_unverified` with no acceptable header, so an automatic rollback refuses and the repaired database stands |
| **C8.6-04** v2 manifests carry other recovery semantics | P2 | Manifest version 3; older manifests are refused, so re-plan. **The 23 Nov stage-2 F1 manifest is v2: stage 2 must be re-run with the final candidate** |
| **C8.6-05** A fork with the keeper registry locked hangs the child | P2 | An at-fork reset: fresh lock and registry; inherited connections are parked, never used or closed |
| **C8.6-06** `_full_fsync` downgraded an `EIO` to a plain fsync | P2 | It falls back only for unsupported (`EINVAL`/`ENOTSUP`/`ENOTTY`); everything else is raised |
| **C8.6-07** The stage-3 block certified a copy that rollback refuses (a `-journal` or non-empty `-wal` beside it) | P2 | Sidecars are checked as rollback checks them; the copy and its directory are synced before OK |
| **C8.6-08** A failed flush-thread start suppressed the battery alert | P2 | Reported; the alert and the state update proceed |
| **C8.6-09** The audit window ignored a second SQLite open of the target | P2 | The arming connect is consumed once |

**Also found and fixed in this round: the broker lane trusted any passed auth check, however old.** That is invariant 2.

- **Evidence:** on 23 Nov it showed `SESSION_VALID` from the 09:30 check all evening. After the next 06:00 IST token rollover it would still have said READY and `execution_ready`.
- **Fix:** a passed check now counts only until the next 06:00 IST after it. Before that rollover the lane shows `CHECKING` / `STALE`, not ready, and an unreadable time fails closed.
- **History:** the same bug was fixed on 18 Nov in `autonomous_loop._live_broker`, but not here.
- **Tests:** `tests/test_broker_readiness.py`.

## 6h. A Claude subagent's review of `2787a690..2cf44fc0`, re-run against `4bc96989`

The subagent reported 8 findings and verified much more: no stranded recovery state at any crash point, no open-by-name under the lock, and no deletion. I ran its scripts against the current head.

| Finding | Severity | At `4bc96989` | Result |
|---|---|---|---|
| **1** Freeze writes were not durable at return | P1 | Fixed by C8.6-01 (6g) | — |
| **2** A symlink to the firmlink spelling escaped `on_live_volume` | P1 | Fixed by C8.6-02 (6g): refused | — |
| **3** Test children stripped `QT_RUNTIME_ROOT`. The checkout's pointer is production's runtime when run from production's checkout | P1 | Real | Children get a temporary root (crash, keeper and runbook tests). `/Volumes` is root-owned, so the mount point could never have been created; my runs used worktrees with no pointer file |
| **4** A forked child's NORMAL exit closed the inherited keeper and deleted another process's WAL: 15 committed rows became 5 | P2, latent: the product never forks | **Still real** | Inherited connections are pinned (`Py_IncRef`) and never deallocated or closed in the child |
| **5** SIGTERM killed the autonomy process (the main freeze writer) without `atexit`, leaving a durable but hot WAL that the F1 plan and byte copies must refuse | P2 | **Still real** | **A cooperative stop** (redesigned in 6i). SIGTERM calls `sup.stop()` and wakes the loop's sleep; the loop exits between ticks, `sup.shutdown()` runs, the interpreter exits normally, and the keepers checkpoint |
| **6** A two-hop symlink chain was stat-ed inside the mount | P2 | Fixed by the component-wise resolver | — |
| **7** A link through an alias inside the anchor back to `/` is refused | P2 | Kept by design | A conservative refusal of a path that passes through `/Volumes` |
| **8** Re-planning to the same `--out` erased the apply's record, so rollback was impossible | P2, pre-existing | **Real** | `plan` refuses to overwrite a manifest past `planned`, or one that records backups |

Tests are in `tests/test_cycle8_part7.py`. Mutants: **5 of 5 killed** (re-run for the 6i redesign).

## 6i. GPT review of the 6h fixes (`4bc96989..07ef9f07`, C8.7-01..07)

The first fix for finding 5 raised `KeyboardInterrupt` from the SIGTERM handler. **GPT showed that this was the wrong tool**, and it is replaced.

- **C8.7-01 (P1):** an interrupt between a job-store `UPDATE` and its commit leaves the transaction open on the connection the heartbeat shares, and the heartbeat's commit then persists a half-finished completion without its follow-ups.
- **C8.7-02 (P1):** the acquisition ladder's deliberate `except BaseException` swallows the interrupt.
- **The old outright kill was crash-consistent by design.** An asynchronous exception at an arbitrary point is not.

**Redesign:** SIGTERM is a **cooperative stop**. The handler only calls `sup.stop()` and sets an event that the loop sleeps on.

- **Idle case (usual):** the loop exits between ticks, so the stop is clean and the keepers checkpoint.
- **Busy case:** a running job is never interrupted. It finishes, or the host's SIGKILL arrives after its 12 s grace, as before.

| Finding | Severity | Fix |
|---|---|---|
| **C8.7-01, C8.7-02** | P1 | No asynchronous exception any more (above). The acquisition ladder's `except BaseException` predates this cycle; it would still swallow a manual Ctrl-C (P2, open) |
| **C8.7-03** Shutdown can exceed the 12 s grace (CA refresh thread, heartbeat join) | P2 | No regression: SIGKILL leaves the durable hot WAL, as before the keeper existed. The real-entry-point test shows the usual idle stop takes well under the grace. Open: one shutdown deadline |
| **C8.7-04** A concurrent apply during a re-plan could be overwritten | P2 | `plan` re-reads the manifest at publication and refuses if anything wrote it. The residual window (check to rename) needs two commands on the same manifest at once |
| **C8.7-05** The fork hook imported `ctypes` in the child | P2 | Bound at import. Without it, no keepers are made at all |
| **C8.7-06** A rolled-back attempt could not be re-planned under the runbook's fixed names | P2 | Allowed; the backup history is carried into `earlier_backups` |
| **C8.7-07** The SIGTERM test had a readiness race and a synthetic loop | P2 | It now drives the real `run_supervisor`: a clean exit within the grace, the WAL checkpointed, 30 rows, the scheduler lock free again |

Tests are in `tests/test_cycle8_part7.py`. Mutants: **7 of 7 killed.**

## 7. Stage 2, evening (23 Nov, read-only)

The F1 plan of the live ledger ran at 20:59. The store re-plan ran 22:14–22:33 with the code exported from `62750347`.

| Gate | Result | Evidence |
|---|---|---|
| Store plan | **PASS** | `ready` true, 11/11 gates, state `planned`, rc 0 |
| Diff against the 22 Nov dry run | **PASS** | −22 / +665 / 0, identical. Today's rows are already in the live pickle |
| `target_sha256` equals the live pickle | **PASS** | `e5accf5a…ffe0dbfa`, before and after the run |
| Today's archive included | **PASS** | `23112026.csv` (sha `2e0ccd57…`) among 1,763 verified archives; `last_day` 2026-11-23 |
| F1 plan of the live ledger | **PASS** | `planned`, ready, 11/11. 4 stale entries (rowids 5684–5687), 0 missing, 6,832 rows. Target sha `7085b4aa…`; size and mtime unchanged by the plan |
| §0 preflight, re-run 24 Nov 01:22 | **FAIL** (host) | Storage unmounted since the 22:40 clamshell sleep, health unanswered, `host_verdict` UNHEALTHY. PASS: outside hours, AC, checkout clean at `125880a6`, FF to the candidate, live LOCKED, launchd jobs |
| Running jobs during the plan | **UNKNOWN**, mitigated | No `jobs.db` snapshot was taken 22:14–22:33. The plan's `target_unchanged_during_plan` gate held |
| Candidate CI | **PASS** | 5/5 at `62750347`, 5/5 at `2787a690` |
| Plans made with the final candidate | **BLOCKED** | Both plans used `62750347`. The F1 manifest is v2, which the final tool refuses by design. Re-run stage 2 (read-only) with the final SHA once the volume is mounted |

The aborted 21:00 run's staging (`$M/staging`) is kept as evidence; the re-plan used `$M/staging_r2`.

## 8. F1 on the real ledger with the final code (24 Nov, isolated SSD copies)

**Source:** the 20:59 plan's byte copy of the live ledger (sha `7085b4aa…`). The decision ledger is the 20 Nov copy, with 6,110 `canonical_freeze_id`s.

| Step | Result |
|---|---|
| Plan | READY, 11/11, 4 stale (rowids 5684–5687), 0 missing |
| Apply | `applied`, proven before COMMIT, byte-identical to the plan, integrity `ok` |
| Rollback | `rolled_back`; content equals the original |
| Re-apply | `applied`; a third apply is a no-op |
| Hard crash inside the transaction (`os._exit`) | Recovers to `applied` |
| Hard crash after COMMIT | Recovers to `applied` |
| Addressability, after every step | 0 keys resolve to another row; 6,110 / 6,110 ledger ids resolve through the index |

## 9. Verification

- Targeted regression (140 files touching the changed stores): 139 passed. The 140th was the repair file, caught mid-rework; it passes now.
- The affected files listed in the commit messages all pass.
- **The full guarded suite at `2787a690`:** 520 files, 0 failing, at least 5,286 tests passed and 2 skipped. The 36 files whose count was masked are counted exactly in the `2cf44fc0` run.
- The full suite at the final head and its CI are in the #333 report.

## Open

- **Owner decisions:**
  - Stages 3–6;
  - stage 3b ("repair freeze index");
  - `apply repair`;
  - the 814 unlearned outcomes;
  - `apply counterfactual migration`;
  - live, which stays HARD-LOCKED.
- **P0 host operations.** Power and lid: three AutoPowerOffs and two lid-close losses in five days. After the 20 Nov AutoPowerOff the USB disk did not remount: the desk failed its storage preflight from 10:57 to 23:37 (18,885 such failures in all), and from 14:37 the host was also in clamshell sleep, during market hours.
- **P0 host (24 Nov):**
  - the USB drive leaves the bus on the AC→battery switch, and the disk-image driver then **panics the kernel**;
  - battery health is "Poor";
  - the desk has been down since the 22:40 clamshell sleep. The recovery is a plain detach, which awaits the owner's words.
- **P1 engineering:**
  - **storage self-recovery cannot clear a stale attach.** The preflight re-attaches an image that is attached but unmounted, and against a stale helper that fails forever. **Proposed:** after `diskutil mount` of the image's volume fails, detach exactly the configured image's device, then attach.
    - **Safe because:** never another image, never a mounted one.
    - **Why it waits:** this changes a deliberately fail-closed gate, races any attach in flight, and needs the owner's decision;
  - group commit for decision freezes (6c);
  - `counterfactual_learning.freeze_decision` re-reads the whole 32 MB ledger to deduplicate every write, which is O(n²) per scan, in production today;
  - a measured durability policy for rollback-journal stores and JSONL ledgers.
- **P2:**
  - `/api/product-readiness` took 12.7 s and `/api/autonomy-desk` 8.4 s during market hours; profile them on a copy;
  - the Kite session check shown as READY was from 09:30;
  - the two Nifty readings disagree (+1.122% on the market panel, −1.406% in the F&O context).
