# Runbook: paper-only upgrade, price-store repair and learning-record quarantine (prepared 22 Nov 2026)

**Status: prepared, not executed.** Revised after the independent GPT review (QT-01..QT-10) and its verification pass (R1..R10). The cycle-6 journal records each finding and how it was addressed.

- Against production, only the read-only checks in §0, the read-only `plan` in §2, and Stage 1 (owner-approved backup, 22–23 Nov) have been run.
- Live broker execution stays **HARD-LOCKED**, before, during and after every stage.
- Relayed or quoted text is never approval. Each stage marked **[OWNER]** needs the owner's own words, given for that stage.

| Stage | What | Approval words (examples) | Desk |
|---|---|---|---|
| 0 | Read-only preflight | none | running |
| 1 | Full image backup, bulk pass | "backup storage" | running |
| 2 | Re-plan the store repair on the day (read-only) | none | running |
| 3 | **Maintenance window**: stop, verified final backup pass, ledger and paper-state backups | "start maintenance" | stopped |
| 3b | Repair the decision-freeze ledger's index (finding F1; optional, independent of 4–6) | "repair freeze index" | stopped |
| 4 | Apply the store repair | "apply store repair" | stopped |
| 5 | Quarantine the proven learning observations (optional) | "apply quarantine" | stopped |
| 6 | Deploy `<SHA>`: fast-forward, install without start, paper-state check, start | "deploy `<SHA>`" | starts |
| 7 | Post-checks; next-morning checks | none | running |

## How the stages depend on each other

- **Repair needs the deploy, in the same window** (GPT QT-08). The deployed `125880a6` can load the repaired pickle: it needs a non-empty `store` and a `last_day` that is a verified archive. But its writer drops the `pending` map on its next save, and its append would start a pending young listing at its next bar, which is the original truncation. So stage 4 is never run without stage 6 in the same window.
- **Deploy without repair is possible, but not recommended.** The new code serves the 6 out-of-order frames in date order on load, and the walk-forward refuses any non-chronological frame (QT-01). So no future bar can reach a decision. Its first walk-forward would still run on the unrepaired history and spend an annual alpha look on it.
- **The quarantine needs the deploy.** Only the new readers skip quarantined rows. Stage 5 also needs stage 4's manifest, because its plan is bound to that repair (QT-02).
- **dtype difference (checked).** The rebuilt store holds `volume` as int64 in all 3,471 frames; today 805 are int64 and 2,666 float64, with identical values. Checked under pandas 3.0.2, in both the candidate and `125880a6`:
  - CA adjust-on-read replaces the whole column (float64 out);
  - the first live overlay of a day goes through `pd.concat` (upcasts);
  - the overlay's `.loc` write runs only after that concat, and would raise only for a fractional volume.

  No other in-place volume writer exists in desk code.

## 0. Preflight (read-only; any failure stops the runbook)

Run on the Mac, in Terminal, **outside 09:15–15:30 IST**, on a night with no EOD job running (after about 21:00 IST), lid open and charger connected.

```bash
cd /Users/apple/0to100
date                                                     # outside 09:15-15:30 IST
pmset -g batt | head -2                                  # "AC Power"
mount | grep -E "QuantTermStorage|Expansion"             # WORKING image on /Volumes/QuantTermStorage, Expansion mounted
hdiutil info | grep -E "image-path|/Volumes/QuantTermStorage"   # image = QuantTermStorage_WORKING_20261107.sparsebundle
df -h /Volumes/QuantTermStorage /Volumes/Expansion /System/Volumes/Data
#   need: runtime >= 5 GB free; Expansion >= 60 GB free (stage 1); internal SSD >= 4 GB free
git status --porcelain                                   # must print nothing
git rev-parse HEAD                                       # 125880a6...
git fetch origin claude/settlement-batch-20261118
git merge-base --is-ancestor HEAD <SHA> && echo FF_OK    # FF_OK
curl -s http://127.0.0.1:8765/api/health | head -c 300   # "lifecycle":"READY"
R=/Volumes/QuantTermStorage/QuantTerm/runtime/logs
sqlite3 "file:$R/autonomy/jobs.db?mode=ro" "select job_type from jobs where status='RUNNING'"   # must be empty
gh run list --commit <SHA> --limit 10                    # the 5 workflows: success
venv/bin/python -m product.host_install status --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --manager launchd \
  | grep -E "live_execution|host_verdict"                # live_execution ... locked=True; HEALTHY
launchctl list | grep -i quantterm                       # only com.quantterm.desk and com.quantterm.relay
```

- **Never run `scripts/quantterm_storage_preflight.sh` by hand.** It can attach the old `QuantTermStorage.sparsebundle`, which has the same volume UUID.
- **Never attach `QuantTermStorage.sparsebundle`.**

**Verified at 18:02–18:30 IST on 22 Nov:**

- AC at 100%.
- The WORKING image was mounted.
- Free space: runtime 103 GB, Expansion 420 GB, SSD about 8.4 GB.
- The checkout was clean at `125880a6`, an ancestor of the candidate (a fast-forward is possible).
- Health was READY. Jobs: 0 RUNNING (2,415 SUCCEEDED, 429 SKIPPED_IDEMPOTENT, 74 BLOCKED, 62 CANCELLED, 33 PERMANENT_FAILED).
- The deployed walk-forward report was `report_schema` 2, generated 22 Nov 08:55.
- **Newest full backup:** `QuantTermStorage_BACKUP_20261110_preDeploy.sparsebundle` (16 GB on disk). That is 12 days old, so stage 1 is recommended.

## 1. Full image backup, bulk pass **[OWNER: "backup storage"]** (desk running; about 4–5 h of USB IO on this disk, measured 22–23 Nov)

This is the §13.4 method of the 8 Nov audit: a NEW image (new UUID, distinct volume name and mountpoint), a bulk rsync while the desk runs at night, and a VERIFIED final pass with the desk stopped (stage 3).

- **Stamp first.** The stamp is touched BEFORE the bulk pass. The final pass re-copies and byte-compares everything modified after it, because rsync's size+mtime check (whole seconds in macOS rsync 2.6.9) can skip a same-size database rewrite (QT-07).
- **Night only.** Bulk IO on the Expansion disk slows the desk badly.
- **Same disk.** The new image sits on the same USB disk as the WORKING image. It protects against image damage, not disk failure. The stage-3 SSD copies are the independent tier for the ledgers, paper state and store pickle.

```bash
M=/Users/apple/QuantTermMaintenance/$(date +%Y%m%d)            # internal SSD, never under /Volumes
mkdir -p $M && touch $M/bulk_started
B=/Volumes/Expansion/QuantTermStorage_BACKUP_$(date +%Y%m%d)_preUpgrade.sparsebundle
hdiutil create -type SPARSEBUNDLE -fs APFS -size 150g -volname QTBackupPreUpgrade "$B"
hdiutil attach -nobrowse -mountpoint /Volumes/QTBackupPreUpgrade "$B"
# bulk pass of the RUNTIME with the verified single-read copier kept in $M: one read per source file,
# a temp write and rename, then the backup copy is re-read and its sha256 compared -> $M/copy/manifest.sha256
/Users/apple/0to100/venv/bin/python -I $M/copy_and_hash.py /Volumes/QuantTermStorage/QuantTerm/runtime \
  /Volumes/QTBackupPreUpgrade/QuantTerm/runtime $M/bulk_started $M/copy --exclude logs/cron.log --exclude logs/simplequant.log.1
```

**Scope: `QuantTerm/runtime/` only, the production state.** The same scope as the 10 Nov backup. Not copied, on purpose:

| Path | Why | Effect on rollback |
|---|---|---|
| `perf-testbed-20261110`, `perf-testbed-B-20261110`, `restore-drill-20261110` | 10 Nov drill copies; that day's runtime is in `QuantTermStorage_BACKUP_20261110_preDeploy` | None. Stages 3–6 never touch them, and they stay on the WORKING volume |
| `rollback/runtime-2026-09-13` | The September rollback tree | None, as above |
| `runtime/logs/cron.log` (6.7 GB) and `runtime/logs/simplequant.log.1` (3.3 GB) | Static since 13 Sep. At this disk's 1–2 MB/s they alone take 1.5–3 h | None. No stage reads or writes them, and the stage-3 final pass excludes them too (A9) |

**Outside the runtime.** These are not in the image, and none is changed by Stages 3–6, so the rollback is unaffected:

| What | Where it is recovered from |
|---|---|
| The code checkout and `venv` | git: `125880a6` is on origin, and the code rollback is `git reset --hard 125880a6` |
| `.env` (configuration and secrets) | On the internal SSD, mode 600. Deliberately not duplicated, so secrets stay in one place; keep your own secure copy |
| The launchd plist and its `.previous` | Rewritten by `host_install`; a rollback reinstalls it from the old code |
| `~/Library/Application Support/QuantTerm/runtime` | The symlink to the volume runtime; unchanged |
| `~/Library/Logs/QuantTerm` | launchd logs and the storage-alert state; diagnostics only |
| The review relay | Development tooling, independent of the desk |

**What the bulk image is, and is not.** It was taken with the desk running. A file that did not change while it was read is copied exactly. 3 databases were being written: `logs/product/cases/cases.db`, `logs/autonomy/jobs.db`, `logs/feature_store.db`. A file-level copy of a database that is being written is not guaranteed to be transactionally consistent, so:

- the WAL databases also have SQLite online-backup snapshots in `QuantTerm/stage1_sqlite_snapshots/`, outside `runtime/`, which the final pass does not touch (`logs/product/cases/cases.db`: integrity ok, 2 tables, sha256 `8f23416a2aa5…`; `logs/autonomy/jobs.db`: integrity ok, 1 table, sha256 `60f3823c977c…`);
- `logs/feature_store.db` uses a rollback journal, and a reader would block the desk's writers for the whole read, so it has no snapshot. It did not change during its single read (copier class OK), no rollback journal was beside it when the copier reached it, and `integrity_check` is ok. That makes a mid-commit copy very unlikely but not impossible; the guaranteed copy is Stage 3's SSD `.backup` (feature_store.db is in that list).

The consistent restore point for every ledger is Stage 3: the final pass and the SSD `.backup` copies are taken with the desk stopped.

SQLite `-shm` files are written through mmap, so their mtime does not change, and a backup copy can differ from the live file without any visible write (1 at stage 1: `logs/autonomy/jobs.db-shm`). SQLite rebuilds `-shm` from the WAL on the first open, so it is not a restore input.

**Executed 22–23 Nov 2026, owner-approved** (`M=/Users/apple/QuantTermMaintenance/20261122`):

- Image `/Volumes/Expansion/QuantTermStorage_BACKUP_20261122_preUpgrade.sparsebundle`; stamp `$M/bulk_started` at 22 Nov 22:10:27. It was touched after a first whole-tree rsync and before the runtime-scope pass; the copier read and verified every file after it, so the Stage 3 delta is complete.
- An rsync bulk pass (first of the whole tree, then of the runtime) was stopped as too slow: source reads are seek-bound at 0.6–1.5 MB/s with the desk up.
- The verified copier ran from 23 Nov 00:10. The session's 2-hour job limit stopped it at 02:10, after 17,572 verified files. Its progress log proves manifest entries 128–17,574 OK (`files_ok` equals the regular manifest lines at every progress line). The resume (`$M/copy_and_hash_resume.py`) carried 17,442 of them and processed the rest, from 02:13 to 02:35:08.
- **Result:** 20,156 files, 7.69 GiB. 20,155 verified OK (17,442 carried from run 1), 0 changed while being read (LIVE_CHANGED: refreshed by the stage-3 final pass), 1 `-shm` hash difference (explained above), 0 other failures, 0 vanished during the walk, 2 excluded.
- Files deleted from the source after their copy: 2 removed from the backup, as rsync `--delete` would (`logs/signal_outcomes.db-shm`, `logs/signal_outcomes.db-wal`); 0 unexplained.
- **Read-only re-mount:** attached `-readonly` at `/Volumes/QTBackupVerify` (the mount reports `read-only`); every one of the 20,156 manifest entries re-hashed from the image (7.69 GiB in 5.5 min, 23.6 MB/s): 0 mismatches, 0 files outside the manifest.
- **Recoverability:** 69 of 70 SQLite databases pass `integrity_check` (each copied with its `-wal` to the SSD and checked there); the other, `logs/product/counterfactuals.jsonl.freeze.db`, is damaged IN PRODUCTION and the backup holds it byte for byte (see the Stage 1 report, finding F1); 8,700 JSON files parse; the store pickle loads and its sha256 equals the source and `76133eaf…` (3,471 symbols, last day 2026-11-20, 0 pending); an independent seeded sample of 300 unchanged files re-read from the SOURCE matches the manifest (0 mismatches).
- The 5 older backup images are unchanged (fingerprints before and after).
- **Headroom:** the Expansion disk has 412 GiB free; the internal SSD has 11 GiB free; the new image takes 8.0 GiB on disk.
- **Verdict: PASS for the backup**: complete, checksum-verified from a read-only mount, and recoverable to the exact production state. **Finding F1** is a defect IN PRODUCTION that the backup preserves byte for byte (`docs/audits/2026-11-23-stage1-backup-report.md`). The image was left **detached** after the read-only re-mount.

## 2. Re-plan the store repair on the day (read-only, desk running, about 40 min)

The rebuild tool is in the candidate, not in the deployed code, so it runs from an exported copy of `<SHA>` on the internal SSD. The checkout is not changed.

```bash
M=/Users/apple/QuantTermMaintenance/<YYYYMMDD of stage 1>
mkdir -p $M/code && cd /Users/apple/0to100 && git archive <SHA> | tar -x -C $M/code
cd $M/code
for v in KITE_API_KEY KITE_API_SECRET KITE_ACCESS_TOKEN TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID DEEPSEEK_API_KEY; do export $v=""; done
unset QT_RUNTIME_ROOT
taskpolicy -b nice -n 19 /Users/apple/0to100/venv/bin/python -m product.bhav_store_rebuild plan \
  --target /Volumes/QuantTermStorage/QuantTerm/runtime --staging $M/staging --out $M/store_manifest.json
shasum -a 256 /Volumes/QuantTermStorage/QuantTerm/runtime/logs/bhav/store_cache.pkl   # == target_sha256 in the manifest
```

The plan:

- **Changes nothing in production.** It snapshots the archives (each checked before and after its copy), builds in a fresh interpreter bound to the staging root, audits both stores and diffs them row by row.
- **Must report `"ready": true`** with all **11 gates** true.
- **Expected diff:** removes 22 rows, adds 665 rows, changes 0 values. 3,471 symbols are preserved in order.

**Dry run, 22 Nov 17:46–18:23 IST** (2,210 s at background priority, with the desk up):

- READY, all 11 gates true;
- removed 22, added 665, changed 0;
- 3,471 symbols in order: 3,384 built, 87 promoted from below the floor, 37 pending;
- last day 2026-11-20; 1,762 sessions;
- the planned pickle sha256 is `62ee822c…`, and its audit is completely clean (no finding of any kind);
- the production pickle sha256 was `76133eaf…` before and after, and no file was added to production `logs/bhav`.

The rows are identical to the cycle-5 clone:

- **orphan rows:** 22 rows, 19 symbols, on 07-28, 07-30, 08-19 and 08-20;
- **unsorted frames:** HALEOSLABS, LICNETFSEN, MUKESHB, PBMPOLY, SAICAPI, SONAL;
- **truncated listings:** 50 symbols, 665 rows.

**A stale plan.** If an EOD archive arrives between stage 2 and stage 3, stage 4 refuses the plan (`target pickle changed since the plan`). Either:

- re-run this plan inside the window with the desk stopped (adds about 40 min of downtime); or
- end the window: `product.launchd_control start` restarts the old desk unchanged.

Read the manifest's diff before going on; do not just re-run until it passes.

The staging tree needs about 1.3 GB on the SSD.

**F1 (read-only, optional): plan the decision-freeze ledger's index repair on a byte copy.** See `docs/audits/2026-11-23-f1-freeze-ledger-index.md`.

```bash
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.sqlite_index_repair plan \
  --target /Volumes/QuantTermStorage/QuantTerm/runtime/logs/product/counterfactuals.jsonl.freeze.db \
  --staging $M/f1_staging --out $M/f1_manifest.json --expect-stale 4
#   must print "state": "planned", "ready": true, every gate true, stale_total 4, missing_total 0
#   (a scan that freezes a new decision afterwards changes the target: stage 3b then refuses; re-plan)
```

If it prints `"state": "nothing_to_repair"`, the ledger is healthy: skip the F1 copy in stage 3 and stage 3b.

## 3. Maintenance window **[OWNER: "start maintenance"]**: verify the backup, stop the desk, take verified backups (about 6 min before the stop, then about 6–15 min with the desk stopped)

Run stages 3–6 in **one** Terminal session; they share these variables.

```bash
M=/Users/apple/QuantTermMaintenance/<YYYYMMDD of stage 1>; ls $M/store_manifest.json $M/code $M/bulk_started >/dev/null || echo "WRONG M"
R=/Volumes/QuantTermStorage/QuantTerm/runtime/logs
cd /Users/apple/0to100
df -h /System/Volumes/Data | tail -1                      # internal SSD >= 4 GB free
stat -f "%Sm" $M/bulk_started                             # 22 Nov 22:10:27, the stamp from BEFORE the stage-1 bulk pass
# BEFORE the stop (desk still up): re-attach the stage-1 image and re-verify it, file by file, against
# its stage-1 manifest. This re-reads the whole backup (7.69 GiB; 5.5 min at the stage-1
# read-only re-hash), so it stays outside the downtime (Cycle 7).
hdiutil attach -nobrowse -mountpoint /Volumes/QTBackupPreUpgrade /Volumes/Expansion/QuantTermStorage_BACKUP_20261122_preUpgrade.sparsebundle
hdiutil info | grep image-path                            # exactly: the WORKING image and this backup image
bash $M/code/scripts/runtime_backup_final_pass.sh /Volumes/QuantTermStorage/QuantTerm/runtime /Volumes/QTBackupPreUpgrade/QuantTerm/runtime \
  $M/bulk_started --baseline $M/copy/manifest.sha256 --verify-only --exclude logs/cron.log --exclude logs/simplequant.log.1
#   must print BASELINE_OK ... excluded=2; any BASELINE_MISMATCH or UNCOVERED stops the runbook before anything is stopped
venv/bin/python -m product.launchd_control stop
launchctl list | grep com.quantterm.desk                 # must print nothing (a loaded job counts as running)
pgrep -fl "host_entrypoint|host_launchd_entrypoint|host_supervisor|terminal_product_api|report_api|main.py autonomy|market_ops|api.app:app"   # must print nothing
# verified final backup pass on the RUNTIME (the stage-1 scope): rsync --delete, then every file
# written since the stamp is re-copied and cmp'd (the unchanged ones were re-verified just before the stop).
# The two static September logs stay excluded (A9): without it rsync copies 10 GB inside the downtime.
bash $M/code/scripts/runtime_backup_final_pass.sh /Volumes/QuantTermStorage/QuantTerm/runtime /Volumes/QTBackupPreUpgrade/QuantTerm/runtime \
  $M/bulk_started --exclude logs/cron.log --exclude logs/simplequant.log.1
#   must print FINAL_PASS_OK ... excluded=2; any MISMATCH or a non-zero exit stops the runbook
hdiutil detach /Volumes/QTBackupPreUpgrade
# the independent tier on the internal SSD: ledgers, paper and execution state, the store pickle
TS=$(date +%Y%m%d_%H%M%S); mkdir -p $M/backup
# ledgers and every SQLite paper/execution ledger (the live F&O book is product/fo_paper.sqlite3)
for db in signal_outcomes.db feature_store.db trades.db oms/orders.db protection/plans.db \
          product/fo_paper.sqlite3 product/learning_policies.observations.sqlite3; do
  sqlite3 "$R/$db" ".backup '$M/backup/$(echo $db | tr / _).$TS'" || echo "BACKUP FAILED $db"
done
cp -p $R/intelligence/intel_book.json $R/intelligence/paper_config.json $R/intelligence/runtime_state.json \
      $R/autopilot.json $R/us_autopilot.json $R/product/learning_policies.json $R/ca_events.json $M/backup/
cp -p $R/bhav/store_cache.pkl $M/backup/store_cache_pre_repair.pkl
# the paper state, hashed AFTER the backups opened the databases; re-checked before any start
(cd $R && shasum -a 256 intelligence/intel_book.json intelligence/paper_config.json intelligence/runtime_state.json \
   autopilot.json us_autopilot.json trades.db oms/orders.db protection/plans.db product/fo_paper.sqlite3 \
   product/learning_policies.json) > $M/paper_state.sha256
shasum -a 256 $R/bhav/store_cache.pkl $M/backup/store_cache_pre_repair.pkl         # the pair must match
for f in $M/backup/*.$TS; do echo "$f $(sqlite3 "$f" 'pragma integrity_check')"; done   # each prints ok
ls $M/backup/*.$TS | wc -l                                                          # 7; any BACKUP FAILED stops the runbook
```

**F1 only, in the stage 3 SSD tier** (when stage 2 planned a repair). This is a BYTE copy of the damaged freeze ledger, verified against the plan. A `.backup` copy would fail `integrity_check` above, as the original does.

```bash
# One chain. Every run makes and verifies a FRESH copy, so a failed copy or a ledger that changed since
# the plan always fails (GPT V2-08). An existing copy is never deleted or overwritten: re-run after 3b,
# it is the only off-volume pre-repair copy (cycle-8 review, part 4). OK is printed only for a REGULAR
# file that is not a symlink, not the live ledger itself, and hashes to the plan; T starts empty here,
# so an inherited T is never removed (GPT C8.5-08).
F1B=$M/backup/f1_freeze_pre_repair.db; SRC=$R/product/counterfactuals.jsonl.freeze.db; T=""
want=$(/Users/apple/0to100/venv/bin/python -I -c "import json,sys;print(json.load(open(sys.argv[1]))['target_sha256'])" $M/f1_manifest.json) \
  && T=$(mktemp "$M/backup/.f1_copy.XXXXXX") \
  && cp -p "$SRC" "$T" \
  && [ "$(shasum -a 256 "$T" | cut -d' ' -f1)" = "$want" ] \
  && { [ -e "$F1B" ] || [ -L "$F1B" ] || mv -n "$T" "$F1B"; } \
  && [ -f "$F1B" ] && [ ! -L "$F1B" ] && ! [ "$F1B" -ef "$SRC" ] \
  && [ "$(shasum -a 256 "$F1B" | cut -d' ' -f1)" = "$want" ] \
  && rm -f "$T" && echo "F1_BACKUP_OK: $F1B is a separate, verified copy of the planned ledger" \
  || { [ -n "$T" ] && rm -f "$T"; echo "F1_BACKUP_MISMATCH: the copy failed, the ledger changed since the stage 2 plan, or an existing $F1B is not a separate verified copy (it is kept untouched); re-plan, skip 3b"; false; }
```

If anything is still running after the stop, do not kill it blindly. Check its identity (ppid, cwd, user) as in §13.5 of the 8 Nov audit, and stop it cleanly. `apply` refuses on any match, and treats a failed probe as running.

## 3b. Repair the decision-freeze ledger's index **[OWNER: "repair freeze index"]** (optional; desk stopped; about 1 min)

**The finding (F1).** The lid-close sleep of 19 Nov left 4 stale primary-key entries (rowids 5684–5687) in `logs/product/counterfactuals.jsonl.freeze.db`. A lookup of a stale key returns another decision's row.

**How the repair works.** It is the `REINDEX` that `plan` proved on a copy, run IN PLACE on the live ledger, inside SQLite's own exclusive lock. No file is renamed or replaced. Under that one lock, `apply`:

1. checkpoints;
2. verifies the ledger's content is the planned original: rows, index entries, integrity result, schema and fixed header fields;
3. writes a synced backup;
4. records `applying`;
5. runs the `REINDEX` in one transaction and proves the trial result against the plan BEFORE `COMMIT` (a mismatch rolls back with nothing committed);
6. checkpoints and proves the committed file again, header included.

A crash rolls the transaction back; re-running `apply` finishes or redoes the repair.

**Independence.** It does not depend on stages 4–6. The repaired ledger is valid for `125880a6` and for the candidate, so neither rollback chain needs to undo it.

**Apply**

```bash
cd $M/code
/Users/apple/0to100/venv/bin/python -m product.sqlite_index_repair apply --manifest $M/f1_manifest.json \
  --owner-approval "<the owner's words>" \
  && /Users/apple/0to100/venv/bin/python -m product.sqlite_index_repair check \
  --target $R/product/counterfactuals.jsonl.freeze.db --staging $M/f1_check
#   "state": "applied", "applied_ok": true; then check exits 0 (integrity ok, stale_total 0)
#   the original stays beside the ledger as *.pre-reindex-<stamp>
```

**Before touching the ledger, `apply` refuses each of these:**

- no approval words;
- a plan that is not ready;
- a running or loaded desk;
- an open handle;
- any other SQLite connection;
- a ledger whose content differs from the plan, including a committed write still in its WAL.

Re-running it after success is a no-op.

**Rollback** (on the owner's words). It restores the original content through SQLite's backup API under the same exclusive lock, and only while the ledger holds exactly the repaired content. In every state it refuses anything else, so writes made after the repair are never discarded.

**Rollback from the recorded backup**

```bash
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.sqlite_index_repair rollback \
  --manifest $M/f1_manifest.json --owner-approval "<the owner's words>"
```

**Rollback from the stage 3 SSD copy** (when the recorded backup is gone)

```bash
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.sqlite_index_repair rollback \
  --manifest $M/f1_manifest.json --owner-approval "<the owner's words>" --backup $M/backup/f1_freeze_pre_repair.db
```

## 4. Apply the store repair **[OWNER: "apply store repair"]** (about 3–5 min; only together with stage 6)

```bash
cd $M/code
/Users/apple/0to100/venv/bin/python -m product.bhav_store_rebuild apply \
  --manifest $M/store_manifest.json --owner-approval "<the owner's words>"
```

**Before swapping, `apply` refuses each of these:**

- no approval words;
- any failed gate;
- a target pickle whose sha256 changed since the plan;
- any archive whose **content** changed since the plan, or a changed calendar;
- a running or loaded desk;
- an open handle on the pickle;
- another holder of `logs/bhav/.store_publish.lock`.

**How it swaps:**

1. It writes a verified backup next to the target (`store_cache.pkl.pre-rebuild-<stamp>`) and **syncs the directory before anything is replaced**.
2. It writes the transaction record (`state: swapping`) durably (F_FULLFSYNC plus a directory sync).
3. It swaps a verified copy into place with an atomic rename and re-hashes it.

A directory-sync error is raised, never swallowed (QT-06).

**Post-repair data validation (all must pass):**

```bash
P=/Volumes/QuantTermStorage/QuantTerm/runtime/logs/bhav/store_cache.pkl
shasum -a 256 $P                                          # == new_sha256 (the pickle the plan audited clean)
/Users/apple/0to100/venv/bin/python -I -c "import json;m=json.load(open('$M/store_manifest.json'));print(m['state'],m['applied_ok'],all(m['gates'].values()))"
#   applied True True
cp -p $P $M/post_repair.pkl && cd $M/code && /Users/apple/0to100/venv/bin/python -m research.store_integrity_audit \
  --root $M/staging --pickle $M/post_repair.pkl --out $M/audit_post_repair.json
#   contamination_kinds [] and incomplete_kinds [] and counts {} (the plan's audit of this exact pickle; about 5 min)
```

**Rollback triggers.** Roll back if any of these happens:

- `apply` exits non-zero after writing `state: swapping`;
- `applied_ok` is false;
- the sha256 check fails;
- the audit shows any contamination or missing official row.

```bash
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.bhav_store_rebuild rollback --manifest $M/store_manifest.json
#   if the recorded backup next to the target is missing or unreadable, restore the verified SSD copy instead:
#   ... rollback --manifest $M/store_manifest.json --backup $M/backup/store_cache_pre_repair.pkl
shasum -a 256 $P                                          # == target_sha256 (byte-identical pre-repair pickle)
```

**What `rollback` restores:**

- **After an interrupted or unverified swap** (`swapping`, `swap_unverified`), it restores whatever the target holds, including a missing or damaged file (QT-06).
- **After a verified apply** (`applied`), it runs only while the target holds exactly the applied pickle. It never discards a cache the desk saved later.

If `apply` refused **before** swapping, nothing changed. Restart the old desk with `venv/bin/python -m product.launchd_control start`.

## 5. Quarantine the proven learning observations **[OWNER: "apply quarantine"]** (optional; about 2 min)

This is a separate decision; see `docs/plans/2026-11-22-learning-observation-correction.md`. It needs the stage-3 backups and stage 4's manifest, and the desk must still be stopped.

The plan is **bound to that repair** (QT-02):

- the current pickle must equal the manifest's `target_sha256`;
- the repaired pickle must equal its `new_sha256`;
- both must end on the same session.

Otherwise the plan is incomplete, and `apply` refuses.

`apply` also refuses when:

- the live store is not the repaired pickle;
- any observation was frozen or changed since the plan. That digest is checked inside the transaction (R2).

```bash
cd $M/code
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction plan \
  --features-db $M/backup/feature_store.db.$TS --current-pickle $M/backup/store_cache_pre_repair.pkl \
  --repaired-pickle $M/staging/logs/bhav/store_cache.pkl --ca-events $M/backup/ca_events.json \
  --store-manifest $M/store_manifest.json --out $M/quarantine_manifest.json
#   review: "complete": true, store_binding_verified true, unexplained [], not_rederived {}, refusals [];
#   affected = the 10 proven rows of the 22 Nov dry run (VEDPOWER 8, ASHIKAG 2) plus any frozen since;
#   carried_breadth_bound.material_sessions [] (22 Nov dry run: 1,318 values, max shift 0.042 pp)
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction apply \
  --manifest $M/quarantine_manifest.json --features-db $R/feature_store.db \
  --owner-approval "<the owner's words>"
```

**What it writes.** It writes only `observation_quarantine` rows, in **one** SQLite transaction (QT-05):

- every vector is re-verified inside the transaction;
- a refusal or failure writes nothing;
- the rows carry a batch id unique to this plan instance (R5).

No vector, label or other table changes.

**Rerun and rollback.** After an interruption, a rerun of the same `apply` completes the same batch. **Rollback** releases exactly that batch, never an earlier quarantine:

```bash
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction rollback \
  --manifest $M/quarantine_manifest.json --features-db $R/feature_store.db \
  --owner-approval "<the owner's words>"
```

A live rollback needs the owner's words too, because releasing rows changes what learning reads (Cycle 7).

## 6. Deploy `<SHA>` **[OWNER: "deploy `<SHA>`"]** (about 4–9 min to READY)

The service starts only after the paper state is proven untouched. The chain stops at the first failure, so a failed hash check cannot start the desk (QT-10, R10):

```bash
cd /Users/apple/0to100
git status --porcelain                                    # nothing
git merge --ff-only <SHA>
venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd --no-start \
  && (cd $R && shasum -a 256 -c $M/paper_state.sha256) \
  && venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
venv/bin/python -m product.host_install status --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --manager launchd
curl -s http://127.0.0.1:8765/api/health | head -c 400    # "lifecycle":"READY", official history current
```

**Rollback triggers:**

- the install fails;
- health is not READY within 10 min;
- live is not LOCKED in `host_install status`;
- any 5xx on a GET sweep of the desk routes;
- an **unexpected** paper-state change: anything but the documented first-EOD RPEL `bars_held` 3 → 5 reconciliation (18 Nov audit §10.2), which cannot happen in a night window.

**Code rollback** (§10.2/§11.7 of the 18 Nov audit), store first, because the old writer would drop `pending` (QT-08):

```bash
venv/bin/python -m product.launchd_control stop
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.bhav_store_rebuild rollback --manifest $M/store_manifest.json \
  && cd /Users/apple/0to100 && git reset --hard 125880a6 \
  && venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd --no-start \
  && (cd $R && shasum -a 256 -c $M/paper_state.sha256) \
  && venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
#   a failed hash check stops here: see "paper state" below
```

**If Stage 4 was NOT applied** (deploy without the repair), the store `rollback` above refuses ("nothing to roll back") and the chain would stop. Use the code-only chain instead:

```bash
venv/bin/python -m product.launchd_control stop
cd /Users/apple/0to100 && git reset --hard 125880a6 \
  && venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd --no-start \
  && (cd $R && shasum -a 256 -c $M/paper_state.sha256) \
  && venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
```

**If the store rollback refuses** (the new desk already saved a newer pickle, for example after the next EOD archive), **do not start `125880a6` on that pickle** (R9). Its writer would drop `pending` on its first save. Choose one:

- **Fix forward:** stay on the new code. This is preferred, because the rollback trigger is then a code defect to fix, not a data one.
- **Go back:** the pickle is a cache, so the newer session is not lost. On the owner's explicit approval to discard the newer cache:
  1. Verify `$M/backup/store_cache_pre_repair.pkl` against `target_sha256`.
  2. Copy it over the target with the desk stopped.
  3. Run the chain above from `git reset`.

  The old desk re-appends any newer session from its archive on its first refresh. Before upgrading again, a new stage-2 plan is required.

**Paper state.** A file reverts only on the owner's explicit decision; this is a ledger change, separate from "deploy". With the desk stopped:

1. Verify the stage-3 copy's sha256 against `paper_state.sha256`.
2. Restore only the changed file from `$M/backup/` (JSON by `cp -p`; SQLite by `sqlite3 <live> ".restore '<backup>'"`).
3. Re-check, then start.

A reinstall alone never restores paper state.

**What does not roll back.** An alpha look the new walk-forward spent stays spent (`logs/research/alpha_ledger.db`): looks are never refunded. This is by design.

## 7. Post-checks

**Immediately:**

- health READY;
- the desk log shows `bhav_store_loaded_from_cache` with 3,471 symbols and no `bhav_cached_session_rejected`;
- there is no `bhav_store_resorted_on_load` after a repair; it is expected only without one;
- `host_install status` reports `HEALTHY` and live LOCKED;
- `shasum -c $M/paper_state.sha256` is still OK.

**The first night after (background, no action):**

- The walk-forward reruns under report schema 6, replacing the deployed schema 2. Nothing older than 6 can certify a BUY (R1). It runs on the repaired history; allow about 10–15 min. Its report records `non_chronological_frames_refused`, which is 0 after a repair.
- The first CA refresh re-walks its windows once (`corporate_actions_incomplete` shows until it finishes; that is honest).
- The next scan builds a new snapshot. Saved BUYs from the old snapshot stay WATCH until it re-judges them (fails closed).

**Next morning:**

- the outcome job ran;
- `logs/signal_backtest.json` has `report_schema: 6`;
- an EOD append left the store clean. Optionally re-run the §4 audit on a fresh copy: no `ORPHAN_ON_ARCHIVE_DATE`, no `UNSORTED`.

**Expected downtime** (desk stopped, stages 3–6):

| Stage | Time |
|---|---|
| Stop | about 1 min |
| Verified final backup pass | 2–10 min (rsync walks about 20,156 files; the files written since the stamp are re-copied and compared: 26 at the end of stage 1, growing every day until stage 3) |
| SSD backups | 2–4 min (SQLite reads through the USB image are slow) |
| Store apply | 3–5 min |
| Post-repair audit | about 5 min |
| Optional quarantine | about 2 min |
| Install, paper check, start, READY | 5–10 min |
| **Total** | **about 20–45 min** |

- **Stale plan:** add about 40 min to re-plan inside the window, or end the window.
- **Beforehand, with the desk up:** the 40-minute plan and the pre-stop verify-only (about 6 min).
- These are estimates from the dry run, the 8 Nov deploys and the 22–23 Nov stage-1 rates (source reads 0.6–1.5 MB/s with the desk up). They have not been measured end to end.

## 8. Commands that require the owner's approval (only Stage 1 has been run)

1. Stage 1: `touch $M/bulk_started`, `hdiutil create … && hdiutil attach …` and the verified copier (new backup image). **Done 22–23 Nov 2026, owner-approved; see §1.**
2. Stage 3: the pre-stop `--verify-only`, `product.launchd_control stop`, `scripts/runtime_backup_final_pass.sh` (with the two `--exclude`) and the SSD backups.
3. Stage 3b (optional): `product.sqlite_index_repair apply --manifest $M/f1_manifest.json --owner-approval "…"`, and its `rollback`.
4. Stage 4: `product.bhav_store_rebuild apply --manifest … --owner-approval "…"`, and `rollback` (with `--backup` if needed) if triggered.
5. Stage 5: `product.learning_observation_correction apply …`, and its `rollback`.
6. Stage 6: `git merge --ff-only <SHA>` and `product.host_install install …` (with and without `--no-start`).
7. A paper-state restore (separate approval; ledger change).

Still separate decisions, outside this runbook:

- **Stage 1 finding F1** now has its own stage, 3b. It needs its own words ("repair freeze index"), given after the Stage 3 backup;
- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- any live-trading unlock, which stays HARD-LOCKED.
