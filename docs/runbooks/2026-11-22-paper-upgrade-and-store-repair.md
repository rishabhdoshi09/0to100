# Runbook: paper-only upgrade, price-store repair and learning-record quarantine (prepared 22 Nov 2026)

**Status: prepared, not executed.** Revised after the independent GPT review (QT-01..QT-10) and its verification pass (R1..R10). The cycle-6 journal records each finding and how it was addressed.

- Nothing in this runbook has been run against production except the read-only checks in §0 and the read-only `plan` in §2.
- Live broker execution stays **HARD-LOCKED**, before, during and after every stage.
- Relayed or quoted text is never approval. Each stage marked **[OWNER]** needs the owner's own words, given for that stage.

| Stage | What | Approval words (examples) | Desk |
|---|---|---|---|
| 0 | Read-only preflight | none | running |
| 1 | Full image backup, bulk pass | "backup storage" | running |
| 2 | Re-plan the store repair on the day (read-only) | none | running |
| 3 | **Maintenance window**: stop, verified final backup pass, ledger and paper-state backups | "start maintenance" | stopped |
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

## 1. Full image backup, bulk pass **[OWNER: "backup storage"]** (desk running, about 1 h of USB IO)

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
rsync -a /Volumes/QuantTermStorage/QuantTerm/ /Volumes/QTBackupPreUpgrade/QuantTerm/     # bulk pass
```

- Leave the backup attached for the final pass in stage 3.
- On 10 Nov, 16.7 GB took 18 min. The runtime volume now reports 46 GB used, so plan on up to about 1 h.

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

## 3. Maintenance window **[OWNER: "start maintenance"]**: stop the desk, take verified backups (about 6–10 min)

Run stages 3–6 in **one** Terminal session; they share these variables.

```bash
M=/Users/apple/QuantTermMaintenance/<YYYYMMDD of stage 1>; ls $M/store_manifest.json $M/code $M/bulk_started >/dev/null || echo "WRONG M"
R=/Volumes/QuantTermStorage/QuantTerm/runtime/logs
cd /Users/apple/0to100
venv/bin/python -m product.launchd_control stop
launchctl list | grep com.quantterm.desk                 # must print nothing (a loaded job counts as running)
pgrep -fl "host_entrypoint|host_launchd_entrypoint|host_supervisor|terminal_product_api|report_api|main.py autonomy|market_ops|api.app:app"   # must print nothing
# verified final backup pass: rsync --delete, then every file written since the stamp is re-copied and cmp'd
bash $M/code/scripts/runtime_backup_final_pass.sh /Volumes/QuantTermStorage/QuantTerm /Volumes/QTBackupPreUpgrade/QuantTerm $M/bulk_started
#   must print FINAL_PASS_OK; any MISMATCH or a non-zero exit stops the runbook
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
   autopilot.json us_autopilot.json trades.db oms/orders.db protection/plans.db product/fo_paper.sqlite3) > $M/paper_state.sha256
shasum -a 256 $R/bhav/store_cache.pkl $M/backup/store_cache_pre_repair.pkl         # the pair must match
for f in $M/backup/*.$TS; do echo "$f $(sqlite3 "$f" 'pragma integrity_check')"; done   # each prints ok
ls $M/backup/*.$TS | wc -l                                                          # 7; any BACKUP FAILED stops the runbook
```

If anything is still running after the stop, do not kill it blindly. Check its identity (ppid, cwd, user) as in §13.5 of the 8 Nov audit, and stop it cleanly. `apply` refuses on any match, and treats a failed probe as running.

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
- the rows carry a batch id derived from the plan.

No vector, label or other table changes.

**Rerun and rollback.** After an interruption, a rerun of the same `apply` completes the same batch. **Rollback** releases exactly that batch, never an earlier quarantine:

```bash
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction rollback \
  --manifest $M/quarantine_manifest.json --features-db $R/feature_store.db
```

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

- The walk-forward reruns under report schema 5, replacing the deployed schema 2. It runs on the repaired history; allow about 10–15 min. Its report records `non_chronological_frames_refused`, which is 0 after a repair.
- The first CA refresh re-walks its windows once (`corporate_actions_incomplete` shows until it finishes; that is honest).
- The next scan builds a new snapshot. Saved BUYs from the old snapshot stay WATCH until it re-judges them (fails closed).

**Next morning:**

- the outcome job ran;
- `logs/signal_backtest.json` has `report_schema: 5`;
- an EOD append left the store clean. Optionally re-run the §4 audit on a fresh copy: no `ORPHAN_ON_ARCHIVE_DATE`, no `UNSORTED`.

**Expected downtime** (desk stopped, stages 3–6):

| Stage | Time |
|---|---|
| Stop | about 1 min |
| Verified final backup pass | 1–3 min (re-copies only the files written since the stamp) |
| SSD backups | 2–4 min (SQLite reads through the USB image are slow) |
| Store apply | 3–5 min |
| Post-repair audit | about 5 min |
| Optional quarantine | about 2 min |
| Install, paper check, start, READY | 5–10 min |
| **Total** | **about 20–30 min** |

- **Stale plan:** add about 40 min to re-plan inside the window, or end the window.
- **Beforehand, with the desk up:** the 40-minute plan and the 1-hour bulk backup.
- These are estimates from the dry run and the 8 Nov deploys. They have not been measured end to end.

## 8. Commands that require the owner's approval (none has been run)

1. Stage 1: `touch $M/bulk_started`, `hdiutil create … && hdiutil attach … && rsync …` (new backup image).
2. Stage 3: `product.launchd_control stop`, `scripts/runtime_backup_final_pass.sh` and the SSD backups.
3. Stage 4: `product.bhav_store_rebuild apply --manifest … --owner-approval "…"`, and `rollback` (with `--backup` if needed) if triggered.
4. Stage 5: `product.learning_observation_correction apply …`, and its `rollback`.
5. Stage 6: `git merge --ff-only <SHA>` and `product.host_install install …` (with and without `--no-start`).
6. A paper-state restore (separate approval; ledger change).

Still separate decisions, outside this runbook:

- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- any live-trading unlock, which stays HARD-LOCKED.
