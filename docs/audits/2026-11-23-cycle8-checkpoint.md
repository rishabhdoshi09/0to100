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

**Decision: the barrier stays.** It is what keeps the decision ledgers from tearing, the F1 failure itself, and the cost is acceptable for paper trading.

**Follow-up (next cycle, with review): a persistent per-thread connection for the freeze writers.** The measurement shows that makes the barrier essentially free (0.1 ms median). It changes the four freeze stores' connection lifecycle: the WAL would persist while the desk runs, and the stage-3b repair already refuses a hot WAL.

## 6d. Host events after the close (23 Nov)

| When (IST) | Event |
|---|---|
| 10:47 | The USB disk unmounted on battery; the desk failed closed until the lid was opened at 14:31 |
| 15:10 | The battery reached 2%: Low Power Sleep, then AutoPowerOff until 20:23 |

**The desk missed the close and the evening jobs.** After the wake the WORKING image was attached through a stale disk-image process (Errno 5). A plain `hdiutil detach` was done, as on 22 Nov. The supervisor re-attached the image within about 60 s, the desk returned at 20:28, and the end-of-day archive landed at 20:28.

**Two more paper cycles were orphaned** (10:47 and 15:02). They were retired correctly as `ORPHANED_PAPER_CYCLE_RESTART`.

## 7. Verification

- Targeted regression (140 files touching the changed stores): 139 passed. The 140th was the repair file, caught mid-rework; it passes now.
- The affected files listed in the commit messages all pass.
- The full suite and CI are in the #333 report.

## Open

- **Owner decisions:**
  - Stages 3–6;
  - stage 3b ("repair freeze index");
  - `apply repair`;
  - the 814 unlearned outcomes;
  - `apply counterfactual migration`;
  - live, which stays HARD-LOCKED.
- **P0 host operations.** Power and lid: three AutoPowerOffs and two lid-close losses in five days. After the 20 Nov AutoPowerOff the USB disk did not remount: the desk failed its storage preflight from 10:57 to 23:37 (18,885 such failures in all), and from 14:37 the host was also in clamshell sleep, during market hours.
- **P1 engineering:**
  - measure the barrier's cost on the USB stack;
  - a measured durability policy for rollback-journal stores and JSONL ledgers.
- **P2:** `/api/product-readiness` took 12.7 s and `/api/autonomy-desk` 8.4 s during market hours; profile them on a copy.
