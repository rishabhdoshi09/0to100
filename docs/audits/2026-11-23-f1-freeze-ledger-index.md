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

**The host's power log** (`pmset -g log`) shows five power events in five days:

| When (IST) | Event |
|---|---|
| 16 Nov 10:40 | Software shutdown (NYKAA's ledger row from 10:20 lost) |
| 17 Nov 08:15 | Low Power Sleep at 7%, then AutoPowerOff until 10:31 |
| 18 Nov 20:49 | Software shutdown |
| **19 Nov 19:55** | **Low Power Sleep at 0%, then AutoPowerOff until 21:58 (wake on AC at 3%)** |
| 20 Nov 10:38 | Clamshell sleep at 2%, then AutoPowerOff until 10:57 (the ledger's last write was 10:39:11) |

**The storage stack.** The runtime is APFS inside a sparsebundle on an exFAT USB disk. Writes made hours earlier did not survive the cut: on 19 Nov the gap was about 9 h, on 16 Nov about 20 min. So acknowledged writes sat in a cache (the disk-image layer, or exFAT's unjournaled metadata) far longer than macOS's usual roughly 30 s write-back.

**Why the barrier failed.** SQLite asks for a real flush only with `fullfsync` / `checkpoint_fullfsync`, and QuantTerm's shared connection policy set neither. A WAL checkpoint then has no barrier between writing pages into the database and re-using the WAL. A cut in that window persists some pages and loses others.

**Exactly which layer held the writes is inferred, not proven.** The power events and the losses are measured.

## 3. Repair (prepared, not applied): `product/sqlite_index_repair.py`

1. **`plan`** takes a stable byte copy: one read that also hashes it, and the source's size, mtime and ctime must not move. It runs `REINDEX` on a second copy and must prove all of the following:
   - the damage is index-only;
   - the repaired copy passes `integrity_check`;
   - every index holds exactly its table's (key, rowid) entries;
   - every unique key resolves to its own row;
   - every row is unchanged (rowids and a content digest of every column);
   - the schema and header are unchanged.
2. **`apply`** needs the owner's words and a stopped desk. It re-verifies both sha256s, refuses a hot WAL or journal and open handles, writes a synced backup first, records `swapping` durably, then swaps in one atomic rename and re-verifies.
3. **`rollback`** needs words when the target is live. It recovers every interrupted state, and never discards writes made after a verified apply.
4. **Idempotent:** a healthy database plans as `nothing_to_repair`, and re-applying is a no-op.

**On a copy of the production file** (23 Nov):

- the plan is READY with all 11 gates true, including `expected_stale` 4;
- 6,112 rows, with an identical digest (`78b016b8…`) before and after;
- the repaired copy passes `integrity_check`;
- all 6,110 ledger ids resolve to their own freeze (fingerprint and symbol match);
- the 4 stale keys resolve to nothing;
- deterministic: an independent second plan produced the byte-identical repaired file (`6cf24778…`).

**Tests** (`tests/test_sqlite_index_repair.py`) build F1's exact damage deterministically (`tests/sqlite_damage.py`). They cover:

- the plan proofs;
- refusals: table damage, non-index damage, a moving source, a hot WAL, a changed target or copy, a running desk, copy mode under `/Volumes`, missing words;
- crashes before the swap, while recording `swapping` (no file left behind), and after the swap (recover, re-plan, re-apply);
- rollback and its refusals;
- the CLI;
- the runbook's own commands, run as written.

Each guard, reverted on its own, fails a test (17 mutants).

## 4. Prevention: checkpoint barriers (`core.atomic_io.make_sqlite_checkpoint_safe`)

**What it sets.** On macOS, every WAL store now sets `checkpoint_fullfsync=ON`, and also `fullfsync=ON` where `synchronous` ≤ NORMAL.

**What it costs.** In WAL mode with `synchronous=NORMAL`, commits never sync, so only checkpoints and the rare WAL-header sync pay a full flush. Stores at `synchronous=FULL` (SQLite's default, for example the job ledger) get only `checkpoint_fullfsync`, so their per-commit cost does not change. Commits that each pay a full flush remain the deliberate choice of `make_sqlite_durable` (the paper F&O book, OMS, protection).

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
- **Power (P0, host operations).** The host drained to 0–7% three times in four days, with the runtime on USB. Keep the Mac on AC with the lid open. The battery Telegram watch exists; whether it fired on 19 Nov is not yet checked.
- **Rollback-journal stores and JSONL ledgers.** Durable appends and commits need a measured `F_FULLFSYNC` policy (cost about 0.4 s per flush on this disk).
- **Storage (owner decision).** exFAT is not journaled. An APFS-formatted runtime disk would remove the metadata-loss window.
