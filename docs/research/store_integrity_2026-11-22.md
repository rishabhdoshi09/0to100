# Historical data integrity: is the saved OHLCV store its official archives? (22 Nov 2026)

**Question.** Did live intraday quotes, saved before the RS-02 fix (`43cba9bb`), contaminate the official price history? And what else in the store differs from the verified official archives it caches?

**Method.**

- **Isolated copy.** The production runtime's `logs/bhav` was copied to the session scratchpad at macOS background IO priority (`taskpolicy -b`, 565 s, 870 MB). The copy holds 1,762 archives, the quarantine folder and the store pickle. The pickle's sha256 equals production's: `76133eaf…0afd3`, saved Fri 20 Nov 23:38 IST.
- **Production untouched.** Nothing in the production runtime was modified.
- **Comparison.** `research/store_integrity_audit.py` checks every (symbol, session) row of the pickle against `data.bhavcopy_store._read_day` of the verified archives (`_dates_on_disk`). The store's own parser is used, so parsing differences cannot create findings.
- **Downstream impact** was measured on the SSD snapshots of 20 Nov ~23:40 IST, which are consistent with the pickle:
  - the feature store;
  - the forward ledger;
  - `signal_outcomes.db`, taken by SQLite backup from a read-only connection;
  - the paper book.

## 1. What the store holds that the archives do not

The audit checked 3,267,038 store rows against 3,268,040 official rows.

| Class | Count | Detail |
|---|---|---|
| Value mismatch on a shared row | **0** | — |
| Row on a date with no archive | **0** | — |
| Duplicate date | **0** | — |
| **Orphan row on an archive date** (the archive does not list the symbol) | **22 rows, 19 symbols, 4 sessions** | 2026-07-28, 07-30, 08-19 and 08-20. Every one has volume 0, no delivery figure and close equal to the previous official close: a **stale no-trade quote** from the live overlay for a symbol that did not trade that day. They include G-Sec and Sensex ETFs (`GSEC10ABSL`, `LICNETFSEN`) and illiquid names (`THAKDEV`, `SONAL`, `TERAI`, …). |
| **Frames out of date order** | **6 symbols** | HALEOSLABS, LICNETFSEN, MUKESHB, PBMPOLY, SAICAPI and SONAL. On 20 Aug the overlay added a row while the 19 Aug archive was still missing. The late 19 Aug archive was then appended after it, and `_incremental_append` never re-sorts. `iloc[-1]` is still the latest date for every frame. |
| **Missing official rows** | **665 rows, 50 young listings** | All on sessions up to 2026-07-27, the date of the last full build. `_full_build` drops names below 30 rows, while appends admit a name from its next row, so each such listing lost its first sessions for good. |
| Below the build floor (expected) | 37 symbols | Names with fewer than 30 rows that are not in the store. |
| Session counter | 1,765 vs 1,762 verified | Counter drift only. No rows are involved. |

## 2. Where it reached (each check is a deterministic run, not an estimate)

| Consumer | Affected? | Evidence |
|---|---|---|
| **Walk-forward evidence (BUY certificates)**, deployed `125880a6` (even stride, 800 names) | **No trade affected** | Its sample contains 5 orphan symbols (3 of them unsorted) and 13 truncated ones. Re-running the deployed `_run_backtest_inner` on those 18 gives **0 trades on both** the contaminated and the rebuilt store. The control is 12 liquid names, which give 123 trades. |
| Walk-forward evidence, candidate `b00eba7b` (hash threshold, 698 names, session-anchored grid) | **No trade affected** | 9 affected symbols are in the sample and give **0 trades on both** stores. The in-sample controls give 23 trades. |
| Tracked BUY outcomes (`signal_log`, 4,952 rows) | **No** | No row is an affected symbol. |
| Paper book | **No** | Its only position is RPEL. |
| Forward ledger (6,594 rows) | **No** | No row of an orphan symbol falls within reach of a contaminated session. |
| **Learning record** (feature store copy, 25,972 observations) | **6 observations, one decision** | Orphan symbols: 0 observations. Truncated symbols: 96 DECISION rows, none settled; 32 carry PIT fills. On a complete history `dist_20ema_pct` moves by at most 0.0035 pp. **VEDPOWER on 2026-11-20** is the exception: the store's history began 20 sessions late, so its IPO high was missing, and the record stores `dist_from_high_pct = 8.68%` where the true value is **29.79%**. That one decision is recorded 6 times (`decision::dec_59a03a86…`, `dec_2f20a387…`, `dec_d7e80c0b…`, `dec_117b1c03…`, and 2 more, listed in the cycle-5 journal). Correcting them would be a learning-ledger repair, which is the owner's decision. |

**Limits.**

- Past live scans and cards of the affected names (mostly illiquid or ETFs, which the stock scanner excludes) could have read a flat zero-volume bar. They are not reconstructable.
- The snapshots end at 20 Nov 23:40. Later observations of truncated names inherit the same history until a rebuild.

## 3. Root causes and fixes (after two GPT passes; journal cycle 5)

1. **Stale overlay rows** (RS-02, `43cba9bb`). An archive is now the authority for its own day: appending it drops that date's row from every symbol it does not list.
2. **Out-of-order frames.** `_incremental_append` re-sorts every merged frame. A live row's overlay marker survives the merge of an older archive.
3. **Truncated young listings.**
   - `_full_build` keeps names below the floor in a persisted `pending` map. When a later archive lists one, it joins the store with its complete history.
   - Which names the store serves does not change.
4. **Late archives.** A refresh merges a verified session inside its window that no frame holds.
5. **Corrected archives and history outside the window.** These are adopted only by the gated rebuild. A periodic refresh never silently re-merges history.
6. **Concurrent publication.** Every save and every repair holds one OS `flock`. The live overlay re-checks, under the lock that publishes quotes, that today's official archive is not already installed.

## 4. Reconstruction procedure (`product/bhav_store_rebuild.py`), tested on the clone only

The store's own `build_from_local()` runs in a fresh interpreter bound to a staging root. That root's archives are read-only links to the target's. Names the current store serves are promoted from the build's pending set, so the symbol set is preserved.

**`plan`** writes a manifest with these gates:

- the target is unchanged during the plan;
- the new store has no contamination;
- its last day is the latest archive;
- its session count equals the verified archives;
- no value changed on a kept row;
- the symbol set is preserved;
- the removed rows are exactly the audit's contamination;
- the added rows are exactly the audit's missing official rows.

**Staging.** The plan works from an immutable snapshot of the target's archives and runtime calendar. Each archive's identity is recorded before and after its copy, so a file that changes mid-copy refuses the plan.

**More gates:**

- the symbol **order** is preserved (the deployed walk-forward samples by it);
- the finding lists are complete;
- changed values are exactly the audited mismatches.

**`apply`** refuses each of the following:

- empty owner-approval words;
- any failed gate;
- a target pickle that changed since the plan;
- any archive whose **content** changed since the plan, and a changed calendar;
- a running or loaded desk (all launch commands and the launchd service; a failed probe counts as running);
- an open handle on the pickle.

It holds the store's publish `flock` from its checks to the swap, writes a transaction record before swapping, and uses verified, fsynced temp files.

**`rollback`** runs only while the target holds exactly the applied pickle, so a cache the desk saved later is never discarded.

**Result on the production clone.** The table below is the first symbol-preserving run. The final code (`05b518c2`) adds three gates (11 in all), all passing, with the same row counts; its rebuilt store and the post-apply audit are completely clean.

| Step | Result |
|---|---|
| `plan` | 436 s, **all 8 gates pass** |
| Rows removed | exactly the 22 contaminated rows |
| Rows added | exactly the 665 missing official rows |
| Rows changed | 0 |
| Symbols | 3,471 preserved: 87 young listings promoted, 37 pending |
| `apply` with an empty approval | refused |
| `apply` with approval words on the copy | swapped; re-audit clean (only the 37 expected below-floor names) |
| `rollback` | restored the **byte-identical** original (sha256 `76133eaf…`) |

The first plan run, before the symbol-set fix, was correctly **refused** by its own gates. It would have dropped those 87 young listings.

**Owner decisions.**

- Running `plan` and `apply` against production, which needs the desk stopped.
- Whether to correct the 6 VEDPOWER observations.

## 5. Addendum (22 Nov evening): production dry run, updated learning impact, feature002

**Production plan (read-only).** `plan --target <production runtime>` ran from 17:46 to 18:23 IST at background priority, with the desk up.

- The result is identical to the clone: READY, all 11 gates true; 22 rows removed, 665 added, 0 changed; 6 unsorted frames; 3,471 symbols in order.
- The planned pickle `62ee822c…` audits with no finding of any kind.
- The production pickle stayed `76133eaf…`, and nothing was added to production `logs/bhav`.
- The procedure is in `docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md`.

**Learning record, re-measured with `product/learning_observation_correction.py`.** The run (`docs/plans/2026-11-22-learning-observation-correction.md`) covers the 20 Nov snapshot plus the 160 DECISION rows frozen on 22 Nov. It reproduces every stored fill exactly.

- **10 proven rows:**
  - VEDPOWER: 6 on 20 Nov and 2 on 22 Nov, `dist_from_high_pct` 8.7–9.5% stored, against 29.8–30.4% true;
  - ASHIKAG: 2 on 22 Nov, `dist_20ema_pct` −1.73 stored, against −0.20 true.
- **90 exposed rows** (KNACK, AEROPLANE, SBIFUNDS) whose input window reaches the lost sessions without a re-derivable fill moving.
- None is settled. The count grows daily until the store is repaired.

**feature002 shadow ledger (not covered above).** It holds 67 rows for 18 of the truncated names. 64 of them are **not reproducible from either store**: for example, AEROPLANE `low_52w` is 117.02 against an official 170.05, with 50+ bars where official history has 47.

- `research/feature002/observe.py::_hist` reads `scan.bulk_fetcher.get_cached` first. That returns the **active Kite snapshot** (`bars_equity.csv`, broker history) whenever Kite is authoritative; otherwise the official store, and yfinance last.
- Those rows are therefore consistent with broker history, and the store defect did not reach them; the 3 store-sourced rows show no change.
- The mixed source is a provenance note for feature002's eventual analysis; see the plan, §4.
