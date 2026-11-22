# Runbook: paper-only upgrade, price-store repair and learning-record quarantine (prepared 22 Nov 2026)

**Status: prepared, not executed.**

- Nothing in this runbook has been run against production except the read-only checks in §1 and the read-only `plan` in §3.
- Live broker execution stays **HARD-LOCKED**, before, during and after every stage.
- Relayed or quoted text is never approval. Each stage marked **[OWNER]** needs the owner's own words, given for that stage.

| Stage | What | Approval words (examples) | Desk |
|---|---|---|---|
| 0 | Read-only preflight | none | running |
| 1 | Full image backup, bulk pass | "backup storage" | running |
| 2 | Re-plan the store repair on the day (read-only) | none | running |
| 3 | **Maintenance window**: stop, final backup pass, ledger backups | "start maintenance" | stopped |
| 4 | Apply the store repair | "apply store repair" | stopped |
| 5 | Quarantine the affected learning observations (optional, separate) | "apply quarantine" | stopped |
| 6 | Deploy `<SHA>` (fast-forward, install, start) | "deploy `<SHA>`" | starts |
| 7 | Post-checks; next-morning checks | none | running |

Stages 4, 5 and 6 are independent decisions:

- **Repair without deploy is supported.** The deployed `125880a6` loads the repaired pickle; it needs `store` and a `last_day` that is a verified archive, and it ignores `pending`. However, its append path would add new overlay rows again.
- **dtype difference.** The rebuilt store holds `volume` as int64 in all 3,471 frames; today, 805 are int64 and 2,666 float64, with identical values. Checked under pandas 3.0.2, in both the candidate and `125880a6`:
  - CA adjust-on-read replaces the whole column (float64 out);
  - the first live overlay of a day goes through `pd.concat` (upcasts);
  - the overlay's `.loc` write runs only after that concat. It would raise only for a fractional volume, and quotes are integral.

  No other in-place volume writer exists in desk code.
- **Deploy without repair is supported.** The new code loads the current pickle, but its first walk-forward would then be computed on the unrepaired history.
- **Quarantine without deploy has no effect.** Only the new loaders skip quarantined rows.

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
- The checkout was clean at `125880a6`, an ancestor of `0842e229` (92 commits; a fast-forward is possible).
- Health was READY. Jobs: 0 RUNNING (2,415 SUCCEEDED, 429 SKIPPED_IDEMPOTENT, 74 BLOCKED, 62 CANCELLED, 33 PERMANENT_FAILED).
- The deployed walk-forward report was `report_schema` 2, generated 22 Nov 08:55.
- **Newest full backup:** `QuantTermStorage_BACKUP_20261110_preDeploy.sparsebundle` (16 GB on disk). That is 12 days old, so stage 1 is recommended.

## 1. Full image backup, bulk pass **[OWNER: "backup storage"]** (desk running, about 1 h of USB IO)

This is the §13.4 method of the 8 Nov audit: a NEW image (new UUID, distinct volume name and mountpoint), a bulk rsync while the desk runs at night, and a final pass with the desk stopped (stage 3). Bulk IO on the Expansion disk slows the desk badly, so run it only at night.

```bash
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
M=/Users/apple/QuantTermMaintenance/$(date +%Y%m%d)            # internal SSD, never under /Volumes
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

A different diff on the day means the store changed (for example, a new EOD session was appended). Read the manifest's diff before going on; do not just re-run until it passes.

The staging tree needs about 1.3 GB on the SSD.

## 3. Maintenance window **[OWNER: "start maintenance"]**: stop the desk, take consistent backups (about 5–8 min)

```bash
cd /Users/apple/0to100
venv/bin/python -m product.launchd_control stop
launchctl list | grep com.quantterm.desk                 # must print nothing (a loaded job counts as running)
pgrep -fl "host_entrypoint|host_launchd_entrypoint|host_supervisor|terminal_product_api|report_api|main.py autonomy|market_ops|api.app:app"   # must print nothing
# final backup pass (stage 1), now consistent:
rsync -a --delete /Volumes/QuantTermStorage/QuantTerm/ /Volumes/QTBackupPreUpgrade/QuantTerm/
hdiutil detach /Volumes/QTBackupPreUpgrade
# ledger and store backups on the internal SSD (off the runtime volume):
R=/Volumes/QuantTermStorage/QuantTerm/runtime/logs; TS=$(date +%Y%m%d_%H%M%S); mkdir -p $M/backup
sqlite3 $R/signal_outcomes.db ".backup '$M/backup/signal_outcomes_$TS.db'"
sqlite3 $R/feature_store.db   ".backup '$M/backup/feature_store_$TS.db'"
cp -p $R/intelligence/intel_book.json $R/product/learning_policies.json $R/ca_events.json $M/backup/
cp -p $R/bhav/store_cache.pkl $M/backup/store_cache_pre_repair.pkl
shasum -a 256 $R/intelligence/intel_book.json $M/backup/intel_book.json \
  $R/product/learning_policies.json $M/backup/learning_policies.json \
  $R/bhav/store_cache.pkl $M/backup/store_cache_pre_repair.pkl          # each pair must match
for f in $M/backup/*.db; do sqlite3 "$f" "pragma integrity_check"; done  # each prints ok
```

If anything is still running after the stop, do not kill it blindly. Check its identity (ppid, cwd, user) as in §13.5 of the 8 Nov audit, and stop it cleanly. `apply` refuses on any match, and treats a failed probe as running.

## 4. Apply the store repair **[OWNER: "apply store repair"]** (about 3–5 min)

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

**How it swaps.** It keeps a verified backup next to the target (`store_cache.pkl.pre-rebuild-<stamp>`) and writes the transaction record (`state: swapping`) to the manifest. It then swaps a verified, fsynced copy into place with an atomic rename and re-hashes it.

**Post-repair data validation (all must pass):**

```bash
P=/Volumes/QuantTermStorage/QuantTerm/runtime/logs/bhav/store_cache.pkl
shasum -a 256 $P                                          # == new_sha256 (the pickle the plan audited clean)
/Users/apple/0to100/venv/bin/python -I -c "import json;m=json.load(open('$M/store_manifest.json'));print(m['state'],m['applied_ok'],all(m['gates'].values()))"
#   applied True True
cp -p $P $M/post_repair.pkl && cd $M/code && /Users/apple/0to100/venv/bin/python -m research.store_integrity_audit \
  --root $M/staging --pickle $M/post_repair.pkl --out $M/audit_post_repair.json
#   contamination_kinds [] and incomplete_kinds [] and counts {} (the plan's audit of this exact pickle)
```

**Rollback triggers.** Roll back if `apply` exits non-zero after writing `state: swapping`, if `applied_ok` is false, if the sha256 check fails, or if the audit shows any contamination or missing official row:

```bash
cd $M/code && /Users/apple/0to100/venv/bin/python -m product.bhav_store_rebuild rollback --manifest $M/store_manifest.json
shasum -a 256 $P                                          # == target_sha256 (byte-identical pre-repair pickle)
```

- `rollback` runs only while the target holds exactly the applied pickle. It never discards a cache the desk saved later.
- If `apply` refused **before** swapping, nothing changed. Restart the old desk with `venv/bin/python -m product.launchd_control start`.

## 5. Quarantine the affected learning observations **[OWNER: "apply quarantine"]** (optional; about 2 min)

This is a separate decision; see `docs/plans/2026-11-22-learning-observation-correction.md`. It needs the stage-3 backups and the plan's rebuilt pickle, and the desk must still be stopped.

```bash
cd $M/code
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction plan \
  --features-db $M/backup/feature_store_$TS.db --current-pickle $M/backup/store_cache_pre_repair.pkl \
  --repaired-pickle $M/staging/logs/bhav/store_cache.pkl --ca-events $M/backup/ca_events.json \
  --out $M/quarantine_manifest.json
#   review: "complete": true, unexplained [] and not_rederived {}; affected = the 10 proven rows of the 22 Nov
#   dry run (VEDPOWER 8, ASHIKAG 2) plus any frozen since; exposed = 90 rows (KNACK, AEROPLANE, SBIFUNDS)
/Users/apple/0to100/venv/bin/python -m product.learning_observation_correction apply \
  --manifest $M/quarantine_manifest.json \
  --features-db /Volumes/QuantTermStorage/QuantTerm/runtime/logs/feature_store.db \
  --owner-approval "<the owner's words>"
```

**What it writes.** Only rows in a new `observation_quarantine` table. No vector, label or other table changes.

**Rollback** is deleting exactly those rows:

```bash
sqlite3 $R/feature_store.db "delete from observation_quarantine where reason='TRUNCATED_HISTORY_PIT_FILL'"
```

Alternatively, restore `$M/backup/feature_store_$TS.db`, but only if nothing was written after the stage-3 backup.

## 6. Deploy `<SHA>` **[OWNER: "deploy `<SHA>`"]** (about 2–6 min, plus about 3 min to READY)

```bash
cd /Users/apple/0to100
git status --porcelain                                    # nothing
git merge --ff-only <SHA>
venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
venv/bin/python -m product.host_install status --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --manager launchd
curl -s http://127.0.0.1:8765/api/health | head -c 400    # "lifecycle":"READY", official history current
```

**Code rollback** (§10.2/§11.7 of the 18 Nov audit):

```bash
venv/bin/python -m product.launchd_control stop
git reset --hard 125880a6
venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
```

**Rollback triggers:**

- the install fails;
- health is not READY within 10 min;
- live is not LOCKED in `host_install status`;
- any 5xx on a GET sweep of the desk routes;
- a paper state file changed between the stage-3 backup and the first start. Paper state is `intel_book.json`, the legacy `trades.db` and the OMS stores; compare by sha256.

**Store and code together.** If both must go back:

1. Stop the desk.
2. Run the store `rollback` (§4).
3. Reset and install the old code.

The store rollback refuses if the new desk has already saved a newer pickle (for example, after the next EOD archive). In that case keep the repaired store: the old code loads it and it is strictly closer to the archives.

## 7. Post-checks

**Immediately:**

- health READY;
- the desk log shows `bhav_store_loaded_from_cache` with 3,471 symbols and no `bhav_cached_session_rejected`;
- `host_install status` reports `HEALTHY` and live LOCKED;
- the paper state hashes are unchanged.

**The first night after (background, no action):**

- The walk-forward reruns under report schema 5, replacing the deployed schema 2. It runs on the repaired history; allow about 10–15 min.
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
| Final backup pass | 1–2 min |
| Ledger backups | 1–3 min (SQLite reads through the USB image are slow) |
| Store apply and validation | 4–8 min |
| Optional quarantine | about 2 min |
| Install and READY | 4–9 min |
| **Total** | **about 15–25 min** |

The 40-minute plan and the 1-hour bulk backup run beforehand with the desk up.

## 8. Commands that require the owner's approval (none has been run)

1. Stage 1: `hdiutil create … && hdiutil attach … && rsync …` (new backup image).
2. Stage 3: `product.launchd_control stop` and the backups.
3. Stage 4: `product.bhav_store_rebuild apply --manifest … --owner-approval "…"` (and `rollback` if triggered).
4. Stage 5: `product.learning_observation_correction apply …`.
5. Stage 6: `git merge --ff-only <SHA>` and `product.host_install install …`.

Still separate decisions, outside this runbook:

- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- any live-trading unlock, which stays HARD-LOCKED.
