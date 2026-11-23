# F1: damaged index in the decision-freeze ledger (investigation, 23 Nov 2026)

This document records facts only and contains no secrets.

**Work done:**

- The production database was read only through byte copies on the internal SSD.
- Nothing in production was changed.
- The repair needs the owner's words (runbook stage 3b).

## 1. Finding (reproduced independently)

**File:** `logs/product/counterfactuals.jsonl.freeze.db`. It is the identity store `product.counterfactual_learning` writes through `product.decision_freeze.freeze()`.

| Check | Result |
|---|---|
| sha256 of production, the Stage 1 backup, a 23 Nov 08:50 copy and both copier manifests | `e3625033…` in every case; the file has not been written since 20 Nov 10:39:11 |
| `integrity_check` (on copies) | `wrong # of entries in index sqlite_autoindex_freezes_1` |
| `quick_check` | `ok`: it skips the index/table comparison by design (see §5) |
| Table scan vs primary-key index | 6,112 rows vs 6,116 entries; no duplicate keys; no table row is missing from the index |
| Stale entries | `dec_991c357c…`, `dec_7cf72023…`, `dec_4160250e…`, `dec_52914c72…` |
| Rows they point at | Rowids 5684–5687 (UNITEDPOLY, BBOX, HMAAGRO, PRIVISCL; `as_of` 2026-11-19; frozen 20 Nov 09:35 IST; WATCH). Each row also has its own, correct entry |
| Effect | A lookup of a stale key returns a different decision's row. `freeze()` of that key would raise a spurious `DecisionIdentityCollision`, or return the wrong saved payload |

## 2. Root cause

**The writer is insert-only.**

- No code deletes from the freeze tables.
- Each freeze is one connection and one INSERT, in WAL mode with `synchronous=NORMAL`.

So for stale index entries to point at rowids that newer rows occupy, earlier rows at 5684–5687 must have been lost from the table after their index entries were persisted. The table's highest rowid then fell back to 5683, and the next inserts re-used it.

**The cross-check against the counterfactual ledger** (`counterfactuals.jsonl`, appended after every successful freeze):

- All 6,110 ledger rows that carry an id resolve to a table row.
- Two table rows have no ledger row:
  - NYKAA, frozen 16 Nov 10:20 IST;
  - APOLLOPIPE, rowid 5683, frozen 19 Nov 11:05:46 IST. It was the last surviving freeze of that day's batch; the ledger's 19 Nov rows end two seconds earlier.
- The 4 stale keys appear in neither store.

**Conclusion.** The tail of the 19 Nov batch was lost from both the ledger file and the table, and only index pages survived. Whether those 4 freezes had been acknowledged cannot be established, because the ledger's own tail was lost too. The data lost is 4 counterfactual WATCH decisions of 19 Nov. Nothing was traded on them.

**Both losses happened seconds before a lid-close sleep** (`pmset -g log`):

| Lost write (IST) | What followed |
|---|---|
| 16 Nov 10:19:59, NYKAA's ledger line | 10:21:23 clamshell sleep on battery (12%), **84 s later** |
| 19 Nov 11:05:46–11:05:5x, APOLLOPIPE's ledger line and the 4 stale-key freezes | 11:06:05 charger removed; 11:06:26 clamshell sleep on battery, **about 30 s later** |

**Corroboration.** The paper cycle that started at 11:03:04 on 19 Nov was found RUNNING by the next supervisor, and retired as `ORPHANED_PAPER_CYCLE_RESTART`. Lid-close sleep drops the USB runtime (see the host notes); writes still in the disk-image layer's cache were lost, and not in order.

**Separate hazard: running out of battery.**

| When (IST) | Event |
|---|---|
| 17 Nov 08:15 | Low Power Sleep at 7%, then AutoPowerOff until 10:31 |
| 19 Nov 19:55 | Low Power Sleep at 0%, then AutoPowerOff until 21:58 |
| 20 Nov 10:38 | Clamshell sleep at 2%, then AutoPowerOff until 10:57 (the ledger's last write was 10:39:11) |

After the 20 Nov wake the USB disk did not remount. The host entrypoint then failed its storage preflight (`external volume is not mounted: /Volumes/Expansion`) until 23:37. It fails closed, as designed; 18,885 such failures are in `launchd.err.log` across these events.

**Why the order matters: every freeze is a checkpoint.**

- `decision_freeze.freeze()` opens and closes a connection per decision.
- Closing the last connection to a WAL database checkpoints its pages into the database file and deletes the WAL.
- SQLite issues a real flush (`F_FULLFSYNC`) only when `fullfsync` or `checkpoint_fullfsync` is set, and QuantTerm's shared connection policy set neither.
- So the checkpoint's writes to the database and the deletion of the WAL could reach the disk in any order.

A drop in that window keeps some pages (the index page) and loses others (the table page), with the WAL already gone. That is F1.

## 3. Repair (prepared, not applied): `product/sqlite_index_repair.py` (v2)

The first version swapped in a repaired copy by rename. The cycle-8 GPT review showed this was unsafe (`docs/audits/2026-11-23-cycle8-checkpoint.md`); v2 repairs in place inside SQLite's own locking.

1. **`plan`** works on a stable byte copy: one read through one descriptor, and the device, inode, size, mtime and ctime must not move, with no hot WAL or journal before or after. It runs the same repair routine `apply` runs on a second copy, and must prove all of the following:
   - the damage is index-only;
   - `integrity_check` is ok;
   - every index holds exactly its table's (key, rowid) entries, read through a scan SQLite reports as covering;
   - every unique key resolves, with the index's own collation, to its own row;
   - every row is unchanged, across every rowid table in SQLite's `table_list` (rowids through an unshadowed alias, plus a content digest);
   - the schema, the header bytes `REINDEX` must not touch, and the full foreign-key list are unchanged.

   Its outputs may never be in the database's directory, and symlinked targets are refused.
2. **`apply`** needs the owner's words, a stopped desk, no open handle and no other SQLite connection. Then it:
   - opens the ledger with `locking_mode=EXCLUSIVE`, so no other connection can read or write until it closes;
   - checkpoints and verifies the content is the planned original;
   - writes a synced backup;
   - records `applying`;
   - runs `REINDEX` in one transaction;
   - checkpoints and proves the result against the plan's digests.

   It never renames or replaces the file. Every byte read of the live file goes through one descriptor opened before SQLite and closed after it, because on POSIX closing any descriptor drops all of the process's locks. A crash rolls the transaction back; re-running finishes or redoes it, and after success it is a no-op.
3. **`rollback`** restores the backup through SQLite's backup API under the same lock. It acts only while the ledger holds exactly the repaired content: in every state it refuses anything else, so writes made after the repair are never discarded.

**On a copy of the production file** (23 Nov):

- the plan is READY with all 11 gates true, including `expected_stale` 4;
- `apply` produced the byte-identical repaired file the plan proved (`bytes_match_plan`);
- 6,112 rows, with an identical digest;
- all 6,110 ledger ids resolve to their own freeze;
- the 4 stale keys resolve to nothing;
- rollback restores the original content, and re-apply works.

**Tests.** `tests/test_sqlite_index_repair.py` (49) and the reviewers' sketches in `tests/test_cycle8_review.py` (review) and `tests/test_cycle8_verification.py` (verification pass). They cover:

- the plan proofs;
- every refusal;
- a writer in another process held off through the backup, the repair and the verification;
- crashes before the `REINDEX`, inside its transaction, and after its commit;
- rollback never discarding newer writes;
- the runbook's own commands, run as written, including a stage-3 copy that fails on a mismatch.

**Mutants.** 33 of 35 guards, reverted on their own, fail a test. There are two defensive survivors:

- the covering-scan check: a scan of only the key columns plus an unshadowed rowid alias is always covering for an ordinary index;
- the proof's lookup totals: `integrity_check` sees every lookup failure that could be constructed.

**After the verification pass**, rollback restores only from a private snapshot proven by sha256 and content, and `apply` proves the trial repair BEFORE `COMMIT`.

## 4. Prevention: checkpoint barriers (`core.atomic_io.make_sqlite_checkpoint_safe`)

**What it sets.** On macOS, every WAL store now sets `checkpoint_fullfsync=ON`, and also `fullfsync=ON` where `synchronous` ≤ NORMAL.

**What it costs.** One full flush per checkpoint.

- **Persistent connections** (for example the job ledger): checkpoints are rare, so the cost is negligible, and `synchronous=FULL` commits keep their per-commit cost unchanged.
- **Connections opened per write** (for example `decision_freeze.freeze()`): every close is a checkpoint, so every write pays one full flush, measured at about 0.4 s on this disk. That is the cost of the barrier F1 lacked; it is to be measured end to end on the real USB stack.
- **Commits that each pay a full flush** remain the deliberate choice of `make_sqlite_durable` (the paper F&O book, OMS, protection).

**Where it applies.**

- The shared `product.sqlite_runtime.connect()`, which covers F1's writer and the other product stores opened through it.
- The 14 other WAL stores that open their own connections.
- A guard test fails if any WAL store in the repository opens without a barrier.

**Not covered.** Rollback-journal stores (for example `feature_store.db`) need `fullfsync` on every commit. That change is a measured follow-up.

## 5. Detection: the SQLite audit used `quick_check`

**The defect.** `product/sqlite_audit.py` (`scripts/audit_sqlite_runtime.py`) probed only `quick_check`, which never compares index entries with table rows. On F1's database it reported `ok` and classified the store `READONLY_OK`.

**The fix.** It now runs the full `integrity_check` by default. On F1 it reports `BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE` and exits 2. `--quick` remains, labelled with its blind spot.

## 6. Open (owner decisions and follow-ups)

- **Stage 3b, "repair freeze index":** apply the planned repair after the Stage 3 backup.
- **Power and lid (P0, host operations).** Both data losses came seconds before a lid-close sleep with the runtime attached, and the host drained to 0–7% three times in four days. Keep the Mac on AC with the lid open; `caffeinate -i` does not prevent clamshell sleep. The battery Telegram watch exists; whether it fired on 19 Nov is not yet checked.
- **Rollback-journal stores and JSONL ledgers.** Durable appends and commits need a measured `F_FULLFSYNC` policy (cost about 0.4 s per flush on this disk).
- **Storage (owner decision).** exFAT is not journaled. An APFS-formatted runtime disk would remove the metadata-loss window.
