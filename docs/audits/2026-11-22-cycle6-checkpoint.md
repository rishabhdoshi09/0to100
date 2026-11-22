# Cycle 6 checkpoint: preparing a safe paper-only production upgrade (22 Nov 2026)

This journal records facts only and contains no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** not deployed, not restarted, not repaired (`125880a6`).
- **Data:**
  - No production ledger, store pickle or archive was modified.
  - Every production access was a read: storage and service checks, the read-only rebuild `plan`, indexed `SELECT`s, and a plain copy of one closed SQLite file.

## Start state

- **Candidate:** `0842e229` on `claude/settlement-batch-20261118` (cycle 5). 5/5 CI green on `05b518c2`.
- **Directive:**
  - verify storage health and backups;
  - dry-run the store repair on production;
  - have GPT review the repair and rollback;
  - write a staged runbook;
  - plan, but do not apply, the correction of the affected learning observations;
  - continue safe research;
  - test, run CI, and report to #333.

## 1. Storage health (read-only, 18:02–18:30 IST)

| Check | Result |
|---|---|
| Power | AC, 100% |
| Runtime | `QuantTermStorage_WORKING_20261107.sparsebundle` mounted at `/Volumes/QuantTermStorage` (150 GB image, 46 GB used, 103 GB free) |
| USB disk "Expansion" | 931 GB, 420 GB free. SMART: "Not Supported" over USB, so no health counters are available |
| Internal SSD | 8.4 GB free after removing two stale staging trees and two duplicate snapshots |
| Backups | Newest full image `QuantTermStorage_BACKUP_20261110_preDeploy.sparsebundle` (16 GB, 12 days old). Older: `_BACKUP_20261107`, `-BACKUP-20260922`, `-SAFETY-20260924`. The old `QuantTermStorage.sparsebundle` must never be attached |
| Desk | health READY; 0 RUNNING jobs; deployed walk-forward report schema 2 (22 Nov 08:55) |
| Memory | 8 GB RAM, about 5 GB swap in use, so heavy jobs ran at background priority one at a time |

A fresh full backup is recommended before the upgrade (runbook stage 1). Its bulk pass is about 1 h of USB IO, so it is an owner-approved night job.

## 2. Store repair, production dry run (`product.bhav_store_rebuild plan`)

**Run.** 17:46–18:23 IST, 2,210 s at background priority, with the desk up. It ran from the dev worktree at `0842e229` with the production venv; credentials were blank and `QT_RUNTIME_ROOT` was unset.

**Result: READY, all 11 gates true.**

| Item | Value |
|---|---|
| Rows removed | 22 (the stale overlay rows: 19 symbols on 2026-07-28, 07-30, 08-19, 08-20) |
| Rows added | 665 (50 young listings; missing sessions from 2023-10-27 to 2026-07-27) |
| Values changed | 0 |
| Frames re-sorted | 6 (HALEOSLABS, LICNETFSEN, MUKESHB, PBMPOLY, SAICAPI, SONAL) |
| Symbols | 3,471 in order: 3,384 built, 87 promoted from below the floor, 37 pending |
| Last day; sessions | 2026-11-20; 1,762 |
| Planned pickle | `62ee822c…`, audited with **no finding of any kind** |
| Production pickle | `76133eaf…` before and after |
| Production `logs/bhav` | unchanged: no lock, backup or temp file |

**The affected set is identical to the cycle-5 clone**, symbol for symbol and row for row.

- **Orphan rows:** ANSALBU, ASSAMENT, ASTAR, GSEC10ABSL (×3), HALEOSLABS, ICDSLTD, LICNETFSEN, LOYALTEX, MUKESHB, PBMPOLY, SAICAPI, SHRIKRISH, SONAL, SUNLOC, TERAI (×2), THAKDEV, UPHOT, VASUPRADA, WHBRADY.
- **Truncated listings:** QMSMEDI 27, SMALLGROWW 27, RMC 25, SMALLIETF 23, VITAL 23, SICALLOG 21, TURTLEMINT 21, the 8 AXIS ETFs 17 each, VEDPOWER 20, and 32 more listed in the manifest.

## 3. Learning-record correction plan (not applied)

**Mechanism.** It is `research/feature_store.py`'s new write-once `observation_quarantine` table.

- `load_observations` and `load_matrix` skip quarantined rows unless asked.
- Every learner reads through them. Checked by grep, the other `FROM observations` readers are:
  - separate databases (feature002, the claims store);
  - one display path, `research/explainability.explain_rejection`, which shows the latest REJECTION reason. It is not learning, and the rejection did happen.
- `product/learning_observation_correction.py` provides:
  - `plan`: read-only, on copies only; it refuses `/Volumes` and a missing database;
  - `apply`: owner words; a stopped desk for a live target; unchanged vector sha256; a complete plan.

**Dry run.** The inputs were the snapshot plus the 160 DECISION rows frozen today (20,653 DECISION rows, equal to production), the plan's two pickles and the CA copy. Every stored fill is reproduced exactly: `complete: true`.

- **10 proven rows** (none settled):
  - VEDPOWER: 6 on 20 Nov and 2 on 22 Nov, `dist_from_high_pct` 8.68% / 9.52% stored, against 29.79% / 30.43% true;
  - ASHIKAG: 2 on 22 Nov, `dist_20ema_pct` −1.73 stored, against −0.20 true.
- **90 exposed rows** (KNACK 52, AEROPLANE 22, SBIFUNDS 16). Their input prefix reaches the lost sessions, but no re-derivable fill moves. The scanner's own carried features cannot be re-derived.
- **The cycle-5 count was 6.** The desk keeps freezing affected rows until the store is repaired, which is why stage 5 re-plans inside the window.

**New finding: feature002 shadow ledger provenance.** cycle 5 did not cover `logs/feature002/shadow.db`. It holds 67 rows for 18 truncated names, and **64 are not reproducible from either store**.

- AEROPLANE stores `low_52w = 117.02` and a 50-bar SMA, where official history has a low of 170.05 and only 47 bars.
- `research/feature002/observe.py::_hist` reads `scan.bulk_fetcher.get_cached` first. That returns the **active Kite snapshot** (`bars_equity.csv`, broker history) whenever Kite is authoritative; otherwise the official store, and yfinance last. (code read).
- So these rows reflect broker history, and the store defect did not reach them. The 3 store-sourced rows show no change.
- The tool reports this and writes nothing. How to treat the mixed source in feature002's analysis is an owner decision (plan, §4).

**Own review fixes before the GPT pass:**

- The changed-symbol test compared dates only. A corrected value, or a re-sorted frame, now counts as changed history too, because the desk reads frames in stored order.
- Windows were sorted before slicing. GPT's QT-09 showed that this hides ordering defects, so it was reverted in §5: inputs are compared in stored order.
- The PIT import-boundary guard now allowlists the offline CLI. A new test pins that no desk module imports it.
- **The value comparison was first dtype-strict.** It flagged 2,668 symbols as changed, because a full build stores `volume` as int64 where appends left float64. It now compares values dtype-blind, and a test pins this.
- **That raised a desk question:** after the repair, every frame's volume is int64, against 805 today. Checked under pandas 3.0.2 in both builds:
  - CA adjust-on-read replaces the column;
  - the first overlay of a day goes through `concat`, which upcasts;
  - the `.loc` overlay write runs only after it, and would raise only for a fractional volume.

  No other in-place volume writer exists. Recorded in the runbook.

## 4. Runbook

`docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md` has eight stages:

0. read-only preflight;
1. a new full image, bulk pass;
2. a re-plan on the day;
3. stop, final backup pass and ledger backups to the SSD;
4. store apply, with sha and audit validation;
5. the optional quarantine;
6. deploy (fast-forward and `host_install`);
7. post-checks.

It also covers rollback triggers and exact commands for the store, the quarantine and the code. Expected downtime is about 20–30 min after the review revisions; the plan and bulk backup run beforehand with the desk up. Every owner-gated command is listed in §8.

## 5. Independent GPT review (`0842e229..783f342f`, one read-only pass, 514 s)

**Scope.** The whole rebuild module, the runbook, and the correction tool and quarantine.

**Result.** 10 findings, 5 P0 and 5 P1. **All 10 were confirmed:** each of GPT's 13 sketches fails on `783f342f` at its claim-bearing assertion. GPT also verified the following as correct:

- both store readers accept the rebuilt payload;
- stage 4 refuses a changed target, candidate, archive or calendar;
- both loaders exclude quarantined rows;
- the live broker interlock is LOCKED.

| Finding | What it was | Fix |
|---|---|---|
| QT-01 (P0) | Deploying without the repair let the walk-forward cut an out-of-order frame by position and hand a decision its future bars, then spend an alpha look | `_load_pkl` serves every frame (served and pending) in date order, keeping attrs. `_run_backtest_inner` refuses any non-chronological frame and reports `non_chronological_frames_refused` |
| QT-02 (P0) | Stage 5 could compare a newer backup with a stale repaired pickle and "prove" a correct row wrong | The plan is bound to the repair manifest (both pickle shas) and refuses a coverage mismatch. A decision that reaches a session the store lacks is unexplained, checked against the archive calendar |
| QT-03 (P0) | A breadth or RS fill on an unchanged peer, and a value that must disappear, passed as complete | Cross-sectional fills on any row are `not_rederived` (incomplete). A vanished value is a defect |
| QT-04 (P0) | Calibration, the rejection analysis and a fitted challenger still consumed quarantined rows | Those sinks are quarantine-aware. A challenger fitted before a quarantine is not served and is refitted |
| QT-05 (P0) | Apply committed row by row; rollback by reason deleted a prior quarantine; the manifest write was not atomic | One `BEGIN IMMEDIATE` transaction with in-transaction vector checks and a plan-derived batch id; `rollback` by batch; atomic manifest writes |
| QT-06 (P1) | The documented store rollback failed on the damage that triggers it; directory syncs were swallowed | `rollback` recovers `swapping` and `swap_unverified` states, including a missing or damaged target, and takes `--backup <verified SSD copy>`. The backup is synced before the swap. F_FULLFSYNC is used. Directory-sync errors are raised |
| QT-07 (P1) | The final rsync pass could keep a same-size, same-second database rewrite | `scripts/runtime_backup_final_pass.sh`: an rsync `--delete` pass, then every file modified since a pre-bulk stamp is re-copied with `-I` and compared byte for byte |
| QT-08 (P1) | The old desk drops `pending` on its next save | The runbook couples the repair to the deploy. The code rollback rolls the store back first, and a re-plan is required before re-upgrading |
| QT-09 (P1) | Exposure sorted both sides and used a 253-bar cap (an EMA is seeded at the first bar) | Exposure compares the whole input prefix in stored order |
| QT-10 (P1) | The deploy rollback could not restore paper state, and the service started before any check | Install with `--no-start`, check the paper-state hashes, then start. The restore procedure is explicit and separately approved. Spent alpha looks are documented as never refunded |

**Adaptations, documented in the tests:**

- **QT-05 rollback sketch.** It ran the old reason-based DELETE; it now runs `rollback`.
- **QT-07 sketch.** It parsed a single rsync line; it now runs the script on a realistic same-second rewrite.
- **QT-08.** It concerns immutable deployed code, so it is fixed by procedure.
- **QT-10.** It asserted that a reinstall restores files, which is not the installer's job, so it is fixed by procedure.

**Own catch while fixing.** My first coverage check compared the calendar cutoff with the store's last day, so it refused every Sunday decision (cutoff Saturday). It now checks for an unseen **verified session**, and a test pins this.

**Mutation checks.** Each of these, reverted on its own, fails its test:

- the walk-forward guard;
- the load-time sort;
- the calibration, non-event and challenger guards;
- the interrupted-swap rollback;
- the coverage refusal;
- the cross-sectional rule;
- the vanished-value rule;
- the forced re-copy in the final pass.

**Re-run on the production snapshots** with the final tool, bound to the real repair manifest:

- `complete: true`, `store_binding.verified: true`, batch `loc-9e9815dd89c6fe1298c9`;
- **10 proven** (VEDPOWER 8, ASHIKAG 2);
- **90 exposed**;
- 0 unexplained.

## 5b. GPT verification pass (`783f342f..8126d597`, the one pass the protocol allows, 445 s)

**Status of the earlier fixes.**

- **Closed** for its counterexample: QT-09.
- **Partial:** QT-01..QT-07 and QT-10.
- **Open:** QT-08.

**New findings.** The remaining parts are reported as R1..R10 (5 P0, 5 P1). **9 are confirmed:** their sketches fail on `8126d597` at the claim.

**R4 did not reproduce.** Its WAL sketch passes on `8126d597`. The cache key was hardened anyway, and the sketch is kept as a regression test.

There is no further GPT loop. Every fix is verified by test and mutation-checked.

| Finding | What it was | Fix |
|---|---|---|
| R1 (P0) | A schema-5 walk-forward written by code without the chronology guard could still certify a BUY | `REPORT_SCHEMA` 6 and `EVIDENCE_MIN_SCHEMA` 6: no pre-guard report can certify, and the supervisor reruns. There is no extra alpha cost after a deploy, because the deployed report is schema 2 and reruns once either way |
| R2 (P0) | Coverage passed when an archive was missing. Apply accepted a plan after new observations were frozen, and never checked the repair was installed | Expected sessions now come from the verified NSE calendar (an unknown year fails closed). The plan records a digest of every observation it judged (1,418 in production); apply recomputes it **inside** its transaction. A live apply needs the live pickle to be the repaired one |
| R3 (P0) | An already-recorded calibration row of a quarantined observation still counted | `DecisionCalibrationEngine._rows()` drops rows linked (`taken:<id>`) to a quarantined observation in every summary, drift and dossier read. The raw row is kept for audit and rollback |
| R4 (P0, not reproduced) | Fingerprint cache keyed on the main file only (WAL) | Keyed on the main file **and** its `-wal`. The challenger captures the fingerprint before reading rows and refuses to publish if it changed during the fit |
| R5 (P0) | Two plans shared a batch id, so rolling back one released the other's rows | The batch id is unique per plan instance. The approved scope (ids and `include_exposed`) is persisted, and a retry must match it |
| R6 (P1) | Scanner-written breadth in non-event vectors bypassed the plan | Carried market-wide values are handled. Breadth is **bounded** by recomputing it on both stores per session (production: 1,318 values, maximum shift 0.042 pp, under the 0.05 tolerance, so immaterial). Any other carried market-wide value is `not_rederived` |
| R7 (P1) | A failed rollback read-back made the next rollback impossible | `rolling_back` is recorded before the rename; `rollback_unverified` and `rolling_back` are recoverable from a verified backup |
| R8 (P1) | The final pass could miss a same-size file published with a preserved old mtime | Files are selected by mtime **or ctime** (`-cnewer`), which `utime` cannot fake, then force-copied and compared |
| R9 (P1) | The runbook let the old desk start on a pending-bearing newer pickle | Removed. Either fix forward, or, on explicit approval, restore the verified pre-repair copy (a cache: the old desk re-appends newer sessions from its archives) and re-plan before upgrading again |
| R10 (P1) | A failed paper-state check still started the desk; the F&O path was wrong (`options/`, live is `product/fo_paper.sqlite3`); the SQLite paper ledgers were not hashed | `&&`-chained, so a failed check cannot start anything. The correct F&O path, every SQLite paper ledger and the claims store are backed up, integrity-checked and hashed |

**Adaptations.**

- R9's sketch asserts that the immutable deployed writer keeps `pending`, so it is fixed by procedure, as QT-08 was.
- The other sketches are adopted as written.

**CI catch on `8126d597`.** `mktemp -t qt-final-pass` is BSD-only, and GNU `mktemp` needs `XXXXXX`, so the hermetic gate failed on Linux. Fixed with a portable template.

**Re-run on the production snapshots** (final tool, bound):

- `complete: true`;
- 10 proven, 90 exposed, 0 unexplained;
- scope of 1,418 observations;
- carried-breadth bound 0.042 pp.

## 6. Verification

See the report (filled in after the regression run and CI).

## Open (owner decisions)

- Stages 1, 3, 4, 5 and 6 of the runbook: "backup storage", "start maintenance", "apply store repair", "apply quarantine" (with or without the exposed tier), "deploy `<SHA>`".
- feature002 provenance: record each row's input source (Kite snapshot or official store) before its analysis.

Still open:

- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- CA conflicts;
- C-5;
- the 52WH prospective test;
- a full-history run;
- the regime vocabulary;
- an instrument master.
