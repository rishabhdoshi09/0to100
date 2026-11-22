# Stage 1 report: fresh verified production backup (22–23 Nov 2026)

This report records facts only and contains no secrets.

**Authorization.** The owner authorized **Stage 1 only** of `docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md`: a fresh, independently verifiable production backup.

**Nothing else was done to production:**

- the desk was not stopped or restarted;
- no historical-cache repair was applied, and no learning observation was corrected;
- no code was deployed (production stays `125880a6`);
- no broker setting was changed, and live execution stays **HARD-LOCKED**.

**Reads and writes.**

- Production was only read. That includes two read-only SQLite online-backup reads of WAL databases, which never block the desk's writers.
- Writes went only to the new backup image and to the maintenance folder on the internal SSD (`M=/Users/apple/QuantTermMaintenance/20261122`).
- The five existing backup images were not touched; their fingerprints are identical before and after.

## 1. Verdict

**PASS for the backup.** It is complete for its scope, every file is checksum-verified from a read-only mount, and it is recoverable to the exact production state.

**One production finding (F1, §6).** The primary-key index of `logs/product/counterfactuals.jsonl.freeze.db` was already damaged in production before the backup started. The backup holds that file byte for byte, which is what a rollback copy must do. Its repair is a production mutation and needs the owner's approval.

## 2. Image

| | |
|---|---|
| Path | `/Volumes/Expansion/QuantTermStorage_BACKUP_20261122_preUpgrade.sparsebundle` |
| Format | A new APFS sparsebundle with a new container identity: 150 GB capacity, volume `QTBackupPreUpgrade` |
| Size on disk | 8.0 GiB |
| State | **Detached.** Stage 3 re-attaches it (runbook §3) |
| Stage-1 manifest | `$M/copy/manifest.sha256`: 20,156 lines, `sha256  relpath`, the backup-side hash of every file |

## 3. Coverage

**In the image: `QuantTerm/runtime/`, the whole production runtime**, apart from the two exclusions below. That is every production state file, database, ledger, paper book, configuration file under the runtime, the store pickle and the archives:

- 20,156 files, 7.69 GiB;
- 70 SQLite databases;
- 8,700 JSON files;
- `logs/bhav/store_cache.pkl` with sha256 `76133eaf…`, the plan's target.

**Not in the image, on purpose:**

| Path | Size | Why | Effect on rollback |
|---|---|---|---|
| `runtime/logs/cron.log` | 6.7 GB | Static since 13 Sep 09:16. At this disk's 1–2 MB/s, the two logs alone take 1.5–3 h | None. No stage reads or writes them. Being static since 13 Sep, they are also in every runtime backup taken since then (the newest is the 10 Nov image, which was not re-opened to check tonight) |
| `runtime/logs/simplequant.log.1` | 3.3 GB | Static since 13 Sep 12:31; as above | None, as above |
| `perf-testbed-20261110`, `perf-testbed-B-20261110`, `restore-drill-20261110` | about 49 GB | 10 Nov drill copies. That day's runtime is in `QuantTermStorage_BACKUP_20261110_preDeploy` | None. Stages 3–6 never touch them, and they stay on the WORKING volume |
| `rollback/runtime-2026-09-13` | 14 GB | The September rollback tree | None, as above |

**Outside the runtime volume.** None of these is changed by Stages 3–6, so the rollback is unaffected:

| What | Recovered from |
|---|---|
| The code checkout and `venv` | git: `125880a6` is on origin, and the code rollback is `git reset --hard 125880a6` |
| `.env` (configuration and secrets) | The internal SSD, mode 600. Deliberately not duplicated, so secrets stay in one place; the owner keeps a secure copy |
| The launchd plist and its `.previous` | Rewritten by `host_install`; a rollback reinstalls it from the old code |
| `~/Library/Application Support/QuantTerm/runtime` | The symlink to the volume runtime; unchanged |
| `~/Library/Logs/QuantTerm` | launchd logs and the storage-alert state; diagnostics only |
| The review relay | Development tooling, independent of the desk |

**Not covered by any tier.** None: every production-critical file is in the image or recoverable from git. The Stage 3 SSD tier adds independent, desk-stopped copies of the ledgers, the paper state and the store pickle.

## 4. Method and timeline (IST)

| When | What |
|---|---|
| 22 Nov 21:55 | The five existing images were fingerprinted (Info.plist and token sha256, band count, size, newest band) |
| 21:59:06 | The new image created and attached |
| 21:59:17 to 22:10:17 | A whole-tree rsync, stopped as too slow, which led to the runtime-only scope |
| 22:10:27 | Stamp `$M/bulk_started` touched, before the runtime-scope pass. Every file was later read and verified by the copier, which ran wholly after the stamp, so the Stage 3 delta (files written after the stamp) is complete |
| 22:10:33 to 00:10:13 | A runtime-scope rsync, resumed at 23:50 with the two logs excluded. It was stopped as too slow: source reads are seek-bound at 0.6–1.5 MB/s with the desk up |
| 23 Nov 00:10:20 | The verified single-read copier started: one read per source file, a temp write and rename, the backup copy re-read and its sha256 compared. LIVE_CHANGED when the source's size, mtime or ctime moved during the read |
| 02:10:12 | The session's 2-hour job limit stopped it after 17,572 verified files |
| 02:13:13 | **Resume.** Run 1's progress log proves manifest entries 128–17,574 OK: at every progress line, `files_ok` equals the regular manifest lines up to that file, and both non-OK files fall in entries 1–127. The resume carried 17,442 of them (a backup copy still regular and of the source size) and re-processed entries 1–127, the 5 resized live files and everything after. It removed the one temp file the stop had left |
| 02:35:08 | The copy finished |
| 02:36 | The 2 stale files were removed (§5) and the 2 WAL snapshots taken |
| 02:37 to 02:43 | **Read-only re-mount:** detach, attach `-readonly` at `/Volumes/QTBackupVerify`, re-hash every manifest entry |
| 02:43 to 02:48 | Recoverability checks on the read-only mount; existing images re-fingerprinted; storage measured; detach |

**Desk health.** READY throughout, except four brief I/O-contention flaps:

- NO_ANSWER at 02:13:48;
- RECOVERING at 02:14:23 and at 02:16:26 ("market-operations heartbeat is stale");
- each returned to READY within about 30–60 s with no action.

## 5. Verification results

| Check | Result |
|---|---|
| Copier classes (20,156 files) | 20,155 verified OK (17,442 carried from run 1), 0 LIVE_CHANGED, 0 vanished, 2 excluded, 0 failures other than the `-shm` below |
| `-shm` | 1, `logs/autonomy/jobs.db-shm`. Its source still has mtime 22 Nov 15:51:16 (before the stamp), but 153 of its 32,768 bytes differ from the copy. SQLite writes `-shm` through mmap, which never updates the mtime. It is the WAL index, rebuilt from the WAL on the first open, so not a restore input |
| Stale files | `logs/signal_outcomes.db-wal` (0 bytes) and `logs/signal_outcomes.db-shm` were copied in run 1, then deleted at the source when the database closed. They were removed from the backup, as rsync `--delete` would; 0 unexplained |
| Live databases | 3 were written after the stamp. `cases.db` and `jobs.db` (WAL) also have online-backup snapshots in `QuantTerm/stage1_sqlite_snapshots/`, outside `runtime/`: integrity ok, and each sha256 re-checked on the read-only mount. `feature_store.db` (rollback journal) was not snapshotted, because a reader would block the desk's writers |
| `feature_store.db` copy | It did not change during its read, no `-journal` was beside it, and `integrity_check` is ok, so a mid-commit copy is very unlikely but not impossible. The guaranteed copy is Stage 3's SSD `.backup` |
| **Read-only re-mount** | The mount reports `read-only`, and the root is not writable. All 20,156 manifest entries were re-hashed from the image: 8.26 GB in 5.5 min at 23.6 MB/s. **0 mismatches, 0 files outside the manifest** |
| SQLite | 69 of 70 pass `integrity_check`, each copied with its `-wal` to the SSD and checked there. The 70th is F1 (§6) |
| JSON | 8,700 of 8,700 parse |
| Store pickle | sha256 `76133eaf…0afd3` in the backup equals the source and the plan's target. It loads: 3,471 symbols, last day 2026-11-20, 0 pending |
| Independent re-read | A seeded sample of 300 of the 20,130 files unchanged since the stamp, re-read from the **source** in a separate process: 0 mismatches against the manifest |
| Existing backups | The 5 images are identical before and after |
| Headroom | The Expansion disk has 412 GiB free and the internal SSD 11 GiB; the new image has 142 GiB of free capacity |

## 6. Finding F1: a damaged index in production (pre-existing)

**File:** `logs/product/counterfactuals.jsonl.freeze.db`, the decision-freeze ledger written by `product/decision_freeze.py`.

**Symptom:** `integrity_check` reports `wrong # of entries in index sqlite_autoindex_freezes_1`.

**Proof that it is production, not the backup:**

- The source file is static since 20 Nov 10:39:11 (mtime equal to ctime), with no `-wal` or `-shm` beside it.
- Its sha256 `e3625033…` is identical in the source, the backup, both copier manifests and a fresh SSD byte copy of the source.
- That copy shows the same error.

**What is damaged:**

- A table scan finds 6,112 rows; the primary-key index holds 6,116 entries, and there are no duplicate keys.
- Rowids 5684–5687 each carry a second, stale index entry. These are UNITEDPOLY, BBOX, HMAAGRO and PRIVISCL, `as_of` 2026-11-19, frozen 20 Nov 09:35 IST, all with the decision WATCH.
- The stale keys are `dec_991c357cba0df929a72dd76f`, `dec_7cf720230d0ca727633cbfcb`, `dec_4160250e26312969d594827b` and `dec_52914c7222496f68eb85208e`.
- A lookup of a stale key returns a different freeze's row, and inserting one of those keys would wrongly be refused.

**Cause.** The table and index pages come from different transaction versions, the signature of a torn write. A lost flush on the USB runtime is the most likely explanation; it is not proven.

**Repair.** On an SSD copy, `REINDEX` gives `integrity_check ok` with all 6,112 rows.

**Production action: none.** A repair is a production mutation that needs its own approval. It would be best done after the Stage 3 backup, with the desk stopped.

## 7. Limits

- **Same disk.** The new image is on the same USB disk as the WORKING image. It protects against image damage, not disk failure. The Stage 3 SSD copies are the independent tier.
- **Not a point-in-time snapshot.** The bulk image was taken over about 4.5 h with the desk running. The consistent restore point is Stage 3, where the final pass and the SSD `.backup` copies are taken with the desk stopped.
- **The resume trusts run 1's hashes.** It carried run 1's verified hashes rather than re-reading them. The read-only re-mount then re-hashed every file, carried ones included, against the manifest.
- **`.env` is outside the image by design.**

## 8. Runbook corrections (this docs commit)

The code fix comes first, as `7d136a5a`: `scripts/runtime_backup_final_pass.sh --exclude`.

| # | Gap | Correction |
|---|---|---|
| A1 | The Stage 3 final pass named the whole `QuantTerm/` tree: about 63 GB of drill and testbed copies would have been copied inside the window | Runtime paths; the Stage 1 section records the scope and method actually used |
| A2 | The image is detached after Stage 1, but Stage 3 assumed it was attached | Re-attach, then check `hdiutil info` |
| A3 | The post-checks said `report_schema: 5` | 6 |
| A4 | Deploy without the repair: the store `rollback` refuses, and the `&&` chain never reset the code | A code-only rollback chain |
| A5 | `learning_policies.json` was not in `paper_state.sha256` | Added |
| A6 | The downtime estimates were unmeasured | Re-stated from the Stage 1 rates (20–45 min), plus the pre-stop verify-only (about 6 min) |
| A7 | Preflight did not check that live is LOCKED, which launchd jobs exist, SSD space or the stamp | Added |
| A8 | The batch-id wording | Unique per plan instance |
| C7V-02 | The final pass ran without the Stage 1 baseline | `--verify-only --baseline $M/copy/manifest.sha256` before the stop; the in-window pass re-checks the delta |
| **A9 (new)** | The two excluded logs: in the window, rsync `--delete` would copy 10 GB, which is hours, and then fail coverage (UNCOVERED); verify-only fails if either is present | `--exclude logs/cron.log --exclude logs/simplequant.log.1` on both lines (`7d136a5a`, with tests) |

## 9. Stage 3–6 approval checklist (none executed)

**Before asking for Stage 3:**

- [ ] The candidate SHA is re-stated and contains `7d136a5a` and the runbook commit; CI is 5/5 on it.
- [ ] **F1 decided by the owner.** Either a `REINDEX` of the freeze ledger, after the Stage 3 backup and on its own approval, or an explicit decision to defer it.
- [ ] Stage 2 is re-planned on the day: read-only, READY, 11/11 gates, the expected diff (−22, +665, 0 changed), and the production pickle sha still `76133eaf…`.
- [ ] A night window outside 09:15–15:30 IST, with no EOD job running; AC power, lid open.
- [ ] `host_install status` shows live LOCKED; health READY; 0 RUNNING jobs; SSD ≥ 4 GB free.
- [ ] The pre-stop `--verify-only` prints `BASELINE_OK … excluded=2`, with the desk still up.

**Per-stage approval words** (each separate, each in the owner's own words):

- [ ] **Stage 3, "start maintenance":** stop; the final pass prints `FINAL_PASS_OK … excluded=2`; detach; the 7 SQLite `.backup` copies each print `ok`; JSON and the pickle are copied (the pickle pair must match); `paper_state.sha256` is written.
- [ ] **Stage 4, "apply store repair":** only together with Stage 6 in the same window. sha256 == `new_sha256`; `applied True True`; the post-repair audit is clean. Otherwise roll back.
- [ ] **Stage 5, "apply quarantine"** (optional): plan bound to the Stage 4 manifest; complete; binding verified; the proven rows; nothing unexplained. Rollback is by batch, with the owner's words.
- [ ] **Stage 6, "deploy `<SHA>`":** `ff-only`; install `--no-start`; the paper hash check; start; READY within 10 min; live LOCKED; a GET sweep with no 5xx.

**Rollback chains:**

- the store-first chain if Stage 4 was applied;
- the code-only chain if it was not (A4);
- a paper-state restore only on a separate approval.

**Still separately gated:**

- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- any live unlock, which stays HARD-LOCKED.
