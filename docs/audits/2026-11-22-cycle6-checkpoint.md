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
- **90 exposed rows** (KNACK 52, AEROPLANE 22, SBIFUNDS 16). Their 253-bar input window reaches the lost sessions, but no re-derivable fill moves. The scanner's own carried features cannot be re-derived.
- **The cycle-5 count was 6.** The desk keeps freezing affected rows until the store is repaired, which is why stage 5 re-plans inside the window.

**New finding: feature002 shadow ledger provenance.** cycle 5 did not cover `logs/feature002/shadow.db`. It holds 67 rows for 18 truncated names, and **64 are not reproducible from either store**.

- AEROPLANE stores `low_52w = 117.02` and a 50-bar SMA, where official history has a low of 170.05 and only 47 bars.
- feature002 reads `scan.bulk_fetcher.get_cached` first, and its last resort is yfinance.
- The tool reports this and writes nothing. The analysis rule is an owner decision (plan, §4).

**Own review fixes before the GPT pass:**

- The changed-symbol test compared dates only. A corrected value, or a re-sorted frame, now counts as changed history too, because the desk reads frames in stored order.
- Windows are sorted before slicing.
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

It also covers rollback triggers and exact commands for the store, the quarantine and the code. Expected downtime is about 15–25 min; the plan and bulk backup run beforehand with the desk up. Every owner-gated command is listed in §8.

## 5. Independent GPT review

See §6 (added after the pass).

## Open (owner decisions)

- Stages 1, 3, 4, 5 and 6 of the runbook: "backup storage", "start maintenance", "apply store repair", "apply quarantine" (with or without the exposed tier), "deploy `<SHA>`".
- feature002 provenance: the analysis-time exclusion rule, or an investigation of the bulk cache first.

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
