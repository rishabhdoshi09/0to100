# Plan: withholding the learning observations built from truncated price history (22 Nov 2026)

**Status: planned, not applied.** Applying it is an owner decision ("apply quarantine"); runbook stage 5 (`docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md`).

- No production learning record has been modified.
- Every number below comes from read-only runs on SSD copies.

## 1. The defect

The store integrity audit (`docs/research/store_integrity_2026-11-22.md`) found 665 official rows missing from 50 young listings. The 30-row build floor dropped each listing's first sessions for good.

**Re-confirmed on production.** The read-only production plan of 22 Nov 18:23 IST (manifest `target_sha256` `76133eaf…`, `new_sha256` `62ee822c…`) found:

- the same 50 symbols and 665 rows;
- the same 22 orphan rows on 19 symbols;
- the same 6 unsorted frames.

**Why the learning record is affected.** A DECISION observation frozen for one of these names carries point-in-time technical fills (`product.pit_features`) computed from that truncated history. A long window can then miss the listing's real extremes. The clearest case is the 252-bar `dist_from_high_pct`: VEDPOWER's IPO high sat in the lost sessions.

## 2. Principles

1. **A frozen vector is never rewritten.** It is the decision-time record, and the write-once contract of `research.feature_store` stays intact.
2. **Withhold, don't correct.** A proven defect is **quarantined**: a write-once row in a new `observation_quarantine` table (id, cause code, evidence json, time).
   - `load_observations` and `load_matrix`, the two loaders every learner uses (`challenger_learning`, `evidence_intelligence`), skip quarantined rows unless `include_quarantined=True`.
   - Re-freezing "corrected" values was rejected. The decision itself (its setup, levels and the scanner's own features) was made on the same history and cannot be re-derived, so a corrected vector would describe a decision that never happened.
3. **Proof before action.** A row is quarantined by default only when one of its stored fills is **reproduced exactly** from the current store AND differs **materially** on the repaired store.
   - Materiality: more than 0.05 absolute AND more than 1% relative. The 20-EMA seed effect, at most 0.0035 pp, is not a defect.
   - A stored value the current store does not reproduce is reported as `unexplained`, and the plan is then incomplete: `apply` refuses.
4. **Exposure is reported, not assumed.** Rows whose 253-bar input window differs between the two stores, without a proven wrong fill, are listed as `exposed`. They are quarantined only with `--include-exposed`.

## 3. Dry run (`product/learning_observation_correction.py plan`)

**Inputs (SSD copies):**

| Input | Source | Identity |
|---|---|---|
| Feature store | the snapshot of 22 Nov 09:39 IST (newest row 20 Nov 05:01 UTC) plus the 160 DECISION rows the desk froze later on 22 Nov, read from production through the `(kind, ts)` index only | 20,653 DECISION rows, the same count as production |
| Current store | the plan's staged copy | sha256 `76133eaf…` |
| Repaired store | the planned pickle | sha256 `62ee822c…` |
| CA table | copy | sha256 `2a1ef002…`, the same as production |

**Result:** `complete: true`; 0 unexplained; 0 fills not re-derived; 13 s.

**Proven: 10 DECISION observations, none settled.**

| Symbol | Decisions (frozen) | Feature | Stored | Repaired |
|---|---|---|---|---|
| VEDPOWER | 6 (20 Nov, 09:27–10:31 IST) | `dist_from_high_pct` | 8.683 | 29.792 |
| VEDPOWER | 2 (22 Nov) | `dist_from_high_pct` | 9.518 | 30.433 |
| ASHIKAG | 2 (22 Nov) | `dist_20ema_pct` | −1.730 | −0.204 |

ASHIKAG lost only one session. Its history is short enough that the EMA seed itself is material.

**Exposed, not proven: 90 DECISION observations, none settled.**

- They are KNACK (52), AEROPLANE (22) and SBIFUNDS (16).
- Their input windows reach the lost sessions, but every re-derivable fill is within tolerance.
- Their carried scanner features (RSI, quality score, the setup itself) cannot be re-derived here.

**Why the count grows.** The cycle-5 note counted 6 rows on the 20 Nov snapshot. The desk keeps freezing decisions for these names until the store is repaired.

Once the store repair (runbook stage 4) is applied, no new rows are affected. That is why stage 5 re-runs the plan inside the window, on that day's backup.

**Timing.** All 10 rows are unsettled, so none has reached a trained model yet: learners read `require_outcome=True`. Their 10-session horizon closes in early December.

- Quarantining before then means they never enter learning.
- The deployed `125880a6` does not know the table. A quarantine takes effect only with the deploy.

## 4. A second ledger: feature002 shadow (report only; finding)

`logs/feature002/shadow.db`, the R3 prospective shadow ranking, was not covered in cycle 5. A copy (taken by plain `cp` while no WAL was open; `integrity_check` ok) holds **67 rows for 18 of the 50 truncated names**, all `eligible_primary`.

**Re-derivation.** The plan re-derives their trend features (`research.feature001.trend_features`) on both stores:

| Rows | What |
|---|---|
| 3 | Exposed only |
| 0 | Proven |
| **64** | **Not reproducible from either store** |

For example, AEROPLANE on 5 Oct stores `low_52w = 117.02` and a 50-bar SMA. The official complete history has a 52-week low of 170.05 and only 47 bars by then.

**The source.** feature002 reads `scan.bulk_fetcher.get_cached` before the official store, and the bulk fetcher's last resort is yfinance. For these young listings, its stored inputs are therefore **not official history**.

**Recommendation.** This is a provenance finding for feature002's eventual analysis, not something this tool can repair. Nothing in that ledger was or will be written by it. The owner's options:

- exclude feature002 rows whose inputs are not reproducible from official history, as a pre-registered analysis rule; or
- investigate the bulk-cache path first (next engineering item).

## 5. Procedure (runbook stage 5; desk stopped)

1. Run `plan` on the stage-3 backup of `feature_store.db`, the pre-repair pickle copy, the planned pickle and the CA copy (`--shadow-db` is optional).
2. Review the manifest:
   - `complete` must be true;
   - the `affected` list (expected: the 10 rows above, plus any frozen since);
   - the `exposed` list.
3. Run `apply --manifest … --features-db <live db> --owner-approval "…"`, adding `--include-exposed` only if the owner chooses to.

**`apply` refuses:**

- empty approval words;
- an incomplete plan;
- a manifest already applied;
- a running or loaded desk (live target);
- any planned row whose vector sha256 changed or that vanished.

It writes only `observation_quarantine` rows, and no partial apply happens on refusal.

**Rollback.** Delete the quarantine rows by reason:

```sql
DELETE FROM observation_quarantine
WHERE reason IN ('TRUNCATED_HISTORY_PIT_FILL', 'TRUNCATED_HISTORY_INPUT_WINDOW');
```

The vectors were never touched, so this restores the previous learning view exactly.

## 6. Tests

`tests/test_learning_observation_correction.py` (21 tests) covers:

- both loaders skip quarantined rows, and the frozen record is byte-identical;
- write-once quarantine with a required reason;
- the plan is read-only and finds the lost IPO high;
- `unexplained` and `not_rederived` make the plan incomplete;
- frames are CA-adjusted exactly as `get_ohlcv` serves them; without the table, the same rows are not reproduced;
- `/Volumes` paths and a missing feature store are refused;
- materiality;
- input-window exposure across kinds, with history lost beyond the window exposing nothing;
- each apply refusal;
- exposed rows are quarantined only when asked;
- the desk check for a live target;
- the feature002 report finds the lost 52-week high and writes nothing.

Mutation-checked:

- dropping the loader exclusion fails the loader test;
- disabling the reproduction check or the CA adjustment fails the CA test.

`tests/test_pit_context_invariance.py` now also pins that no desk module imports this CLI.
