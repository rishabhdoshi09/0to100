# Cycle 5 checkpoint: historical data integrity (22 Nov 2026)

This journal records facts only and contains no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** not deployed and not restarted (`125880a6`).
- **Data:**
  - No production data was modified: the store pickle, the archives and the learning ledgers are untouched.
  - The production store directory was copied read-only, at background IO priority, to an isolated runtime on the SSD.
  - Every analysis and every rebuild test ran on that copy.

## Start state

- **Candidate:** `b00eba7b` on `claude/settlement-batch-20261118` (cycle 4). All 5 CI workflows are green, and it was reported as #333 comment 7074804630.
- **Directive:** audit whether intraday quotes contaminated official history; identify the affected symbols, sessions, backtests, learning records and evidence certificates; prepare and test a safe reconstruction procedure.

## 1. Audit and impact

The full method and numbers are in `docs/research/store_integrity_2026-11-22.md`. In summary:

- **Comparison.** The audit compared 3,267,038 store rows against 3,268,040 official rows from 1,762 verified archives.
  - Value mismatches: **0**. Rows on dates with no archive: **0**.
  - **22 stale no-trade overlay rows** for 19 symbols on 4 sessions (2026-07-28, 07-30, 08-19, 08-20). Each has volume 0, no delivery figure and close equal to the previous close.
  - **6 frames out of date order.**
  - **665 official rows missing** from 50 young listings, a result of the 30-row build floor.
- **Impact.**
  - **Walk-forward evidence and BUY certificates:** 0 trades on the affected names, on either store, in both the deployed sample (800 names) and the candidate sample (698 names). Controls produced 123 and 23 trades.
  - **Tracked BUY outcomes, paper book and forward ledger:** untouched.
  - **Learning record:** 6 observations of one decision, **VEDPOWER on 2026-11-20**. Its `dist_from_high_pct` is stored as 8.68%; the true value is 29.79%.
    - The ids are `decision::dec_59a03a866e295170cd03e9f1`, `dec_2f20a387e5cd7bd1e7025206`, `dec_d7e80c0b6235a80d18a16eb1`, `dec_117b1c03b5c060fe38803ecf`, and two more from the same 20 Nov session, all unsettled.
    - Correcting them is a learning-ledger repair, which is the owner's decision.

## 2. Fixes, own pass first (`8a52e49d`)

- **Late archives.** `_incremental_append` now sorts every merged frame.
- **Below-floor names.** `_full_build` keeps names below the floor in a persisted `pending` map. A young listing joins the store with its complete history, and the set of names the store serves does not change.
- **Tools.**
  - `research/store_integrity_audit.py` (the audit) and `product/bhav_store_rebuild.py` (plan, apply, rollback).
  - The first rebuild plan on the clone was correctly **refused** by its own gates: a bare `_full_build` would have dropped 87 young listings that appends admit.
  - The rebuild was therefore made to preserve the symbol set. After that, all gates passed; apply, a clean re-audit and a byte-identical rollback all worked on the clone.
- **Own finding.** The desk check missed the launchd supervisor (`product.host_entrypoint`) and `report_api`. GPT reported the same as ST-01.

## 3. Independent GPT cross-review (`b00eba7b..8a52e49d`, one read-only pass)

GPT reported 9 findings: 5 P0, 3 P1 and 1 P2. **All 9 were confirmed**: GPT's 14 sketch cases fail on `8a52e49d` at their claim-bearing assertions. They are now in `tests/test_store_integrity_20261122.py` and pass on `a229ab6e`/`9bfe1c70`.

| Finding | Fix |
|---|---|
| ST-01 (P0): the desk check missed the installed host (`host_entrypoint`, `host_launchd_entrypoint`, `host_bootstrap --child`) and `uvicorn api.app:app`; a failed probe looked like a stopped desk; the check was a single point in time | Real launch patterns plus the launchd service; a probe failure counts as running. Apply and rollback hold the store's **maintenance lock** from the check to publication, and `_save_pkl` refuses to write while it exists. |
| ST-02 (P0): the plan ran on symlinks to mutable archives and on the staging calendar | An **immutable snapshot** of the target's archives AND runtime calendar, with a per-archive source manifest that apply re-verifies |
| ST-03 (P0): rollback discarded a cache saved after the apply; the temp file was checked only after the replace; there was no transaction record before the swap | Rollback runs only while the target holds exactly the applied pickle. Unique, verified, fsynced temp files; the transaction record is written before the swap. |
| ST-04 (P0): the audit skipped the persisted `pending` map | The pending map is audited (and checked for overlap with the served map) |
| ST-05 (P0, pre-existing race): a quote fetched while today's archive was installed could overwrite the official bar | The overlay re-checks official authority under the publishing lock and never overwrites a non-overlay row. `_save_pkl` snapshots the maps under the lock. |
| ST-06 (P1): the rebuild reordered keys, which the deployed walk-forward samples by | The served key order is preserved, with the gate `symbol_order_preserved` |
| ST-07 (P1): the audit accepted a missing `deliv_per`, non-midnight or tz-aware indexes, unsorted frames and persisted overlay markers | All of these are classified |
| ST-08 (P1): a refresh skipped late or corrected archives behind the newest session | Per-session archive **identities** (`incorporated`). A legacy pickle is migrated from the sessions it holds; a verified session absent from every frame is merged; `_store_last_day` never moves back; the session count comes from incorporated sessions. |
| ST-09 (P2): the gates compared lists truncated to 2,000 rows | Complete multisets. Truncation fails the gate. Value changes are allowed only where they equal the audited mismatches. |

**Process miss (recorded).**

- `a229ab6e` failed the runtime-root isolation gate: the rebuild tool built `logs/...` paths itself.
- The gate scans tracked files only, so my pre-commit run of it on untracked files at `8a52e49d` passed vacuously. A shell chain using `| tail` then hid the failure at `a229ab6e`.
- `9bfe1c70` fixes it: paths are resolved through `core.runtime_paths`.
- From now on the gate runs after `git add`, and its exit code is checked directly.

## 4. GPT verification pass (`8a52e49d..9bfe1c70`, one pass, no further loop)

**Status of the ST fixes:**

- **Closed for the reported scenarios:** ST-03, ST-05 and ST-06.
- **Partly closed:** the others. The remaining parts are reported as new findings (QT-01..QT-11, 5 P0 and 6 P1).

**Design decision.** Most of the new findings came from the ST-08 fix itself: per-session archive identities, a legacy migration, and re-merging history on every refresh. Specifically, that mechanism could:

- certify a quote or a stale price as incorporated (QT-01);
- resurrect a pending row that a correction had deleted (QT-02);
- bind a new file identity to bytes parsed earlier (QT-03);
- after a horizon-limited build, re-read about 1,262 historical archives and widen the served universe (QT-07);
- leave a warm API serving a same-date republication (QT-08).

**That mechanism is removed** (`05b518c2`).

- A periodic refresh again only appends newer sessions.
- It also merges a verified session **inside its own window** that no served or pending frame holds, which covers the missed-EOD catch-up case.
- A corrected archive, or history outside the window, is adopted only by the gated rebuild with the desk stopped. This follows GPT's own recommendation in QT-07.
- The audit detects corrections, served or pending (tested).

| Finding | Resolution |
|---|---|
| QT-01, QT-02, QT-03, QT-07, QT-08 | Resolved by design as above. QT-01's quote variant, and QT-07, pass as GPT wrote them. QT-01's corrected-row variant and QT-02 are pinned as "the audit detects and the rebuild repairs". QT-03 and QT-08 no longer apply. |
| QT-04 (P0): apply accepted an archive corrected while it was being staged | Staging compares each archive's identity before and after its copy; apply re-hashes every target archive's content |
| QT-05 (P0): an in-flight save could overwrite a finished repair; a loaded launchd job with PID `-` passed | **One OS `flock`** on `logs/bhav/.store_publish.lock` is held by every save and by the whole repair, so a save prepared before a repair cannot publish after it. A loaded service counts as running. |
| QT-06 (P1): a late archive merged during market hours dropped today's overlay marker, freezing the next quote | The marker survives the merge of an older archive; the redundant per-symbol guard is gone. GPT's sketch ran the full-build path (sessions = 0); the adapted test runs the claimed path and is mutation-checked. |
| QT-09 (P1): an abandoned marker blocked every save | The OS releases the flock, so it cannot go stale. A deferred save is retried on the next refresh. |
| QT-10 (P1): a string index and an eligible name kept only in pending passed the audit | Both are flagged |
| QT-11 (P1): duplicate multiplicity | Counted by extra rows |

**Own find while fixing.**

- `_store_dates()` compared `asi8` (pandas 3 stores parsed dates in µs) with `Timestamp.value` (ns), so every session in the window looked late and was re-read.
- Fixed with `as_unit("ns")`. Pending dates count as present.

The tests are `tests/test_store_integrity_20261122.py`, 37 in total: GPT's sketches, with the adaptations documented in the file.

## 5. Verification (final code `05b518c2`)

- **Production clone, full procedure** (the plan took about 15 minutes, including the 870 MB snapshot copy and two full audits):
  - **All 11 gates pass.** The plan removes exactly the 22 contaminated rows, adds exactly the 665 missing official rows and changes 0 values. All 3,471 symbols are preserved in their order: 87 young listings were promoted, and 37 names stay pending.
  - The current store audits as 22 orphan rows, 665 missing rows, 6 unsorted frames and 37 below-floor names. The rebuilt store audits **clean**.
  - `apply` on the copy (with approval words, `--target-is-copy`) swapped the pickle, and the post-apply audit is clean.
  - `rollback` restored the **byte-identical** original (sha256 `76133eaf…`).
- **Local, Mac guard, one interpreter per file.** **All 149 affected files pass on `05b518c2`.** That set is every reader of the bhavcopy store, the live overlay, the bulk fetcher, the learning-freeze path and the runtime-root isolation gate. `8a52e49d` also passed 148/148 before the review fixes.
- **CI on `05b518c2` (the exact SHA): 5/5 green.**
  - CI 45015491944;
  - Product Acceptance 45015491993;
  - Terminal UI 45015491960;
  - Host Migration Gate 45015491934;
  - Dependency Vulnerability Advisory 45015492089.

## Open (owner decisions)

- **Repair the production store** with `python -m product.bhav_store_rebuild plan --target <runtime> --staging <SSD dir> --out <manifest>`, then `apply --manifest <manifest> --owner-approval "<words>"` with the desk stopped (`launchd_control stop`).
  - It removes the 22 stale rows, restores the 665 missing rows and sorts the 6 frames.
  - Backup and rollback are tested on the clone.
  - Ideally it runs before the post-deploy walk-forward, so the new evidence is computed on repaired data.
- **Whether to correct the 6 VEDPOWER observations** (a learning-ledger repair).

Still open:

- `deploy <sha>`;
- `apply repair` (learning policies);
- the 814 unlearned outcomes;
- CA conflicts;
- C-5;
- the 52WH prospective test;
- a full-history run;
- the regime vocabulary;
- an instrument master.
