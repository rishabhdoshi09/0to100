# Plan: withholding the learning observations built from truncated price history (22 Nov 2026)

**Status: planned, not applied.** Revised after the GPT review (QT-02..QT-05, QT-09) and its verification pass (R2..R6). Applying it is an owner decision ("apply quarantine"); runbook stage 5 (`docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md`).

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
   - The evidence sinks that read the store directly skip them too (QT-04):
     - champion calibration on a paper close (`record_resolved_prediction`);
     - `research/non_event`'s rejection analysis, threshold replay and reason trend.
   - **The challenger.** A model fitted before a quarantine is not served (`STALE_TRAINING_INPUTS`) and is refitted at the next `train()`; the model records the quarantine fingerprint it saw.
   - **Calibration rows already recorded** for a quarantined observation (`taken:<id>`) no longer count in any summary, drift or dossier; they stay on disk for audit (R3).
   - **The challenger's fingerprint** is captured before its rows are read, and a fit is not published if a quarantine landed meanwhile. The cache is keyed on the database and its WAL (R4).
   - **Kept as raw reads:** outcome settlement and audit reads, so a quarantined row still settles.
   - Re-freezing "corrected" values was rejected. The decision itself (its setup, levels and the scanner's own features) was made on the same history and cannot be re-derived, so a corrected vector would describe a decision that never happened.
3. **Proof before action.** A row is quarantined by default only when one of its stored fills is **reproduced exactly** from the current store AND differs **materially** on the repaired store.
   - Materiality: more than 0.05 absolute AND more than 1% relative. The 20-EMA seed effect, at most 0.0035 pp, is not a defect.
   - A stored value that **cannot exist** on the complete history (the repaired recompute omits it) is also a defect (QT-03).
   - A stored value the current store does not reproduce is reported as `unexplained`, and the plan is then incomplete: `apply` refuses.
   - A decision whose cutoff reaches a session that the repaired store lacks is also `unexplained`. The sessions come from the **verified NSE calendar**, not the archives held, so a missing Monday archive is caught; a year the calendar does not cover fails closed, and a weekend decision is not refused (QT-02, R2).
4. **Never claim more than was re-derived** (QT-03). Breadth and the RS percentile are computed across all symbols, so a changed listing moves every peer.
   - Any observation carrying one of them, and any changed symbol's own delivery fill, is counted under `not_rederived`, and the plan is incomplete.
   - Production PIT fills carry none of them today (producer `pit_v1_technical`).
   - The scanner also writes breadth straight into non-event vectors (R6). That carried breadth is **bounded**: breadth is recomputed per session on both stores, and only a session where the stores differ by more than 0.05 pp makes those rows `not_rederived`.
   - In production, 1,318 carried values (26–30 July) moved by at most **0.042 pp**, so they are immaterial.
   - Any other carried market-wide value (RS, sector, volatility percentiles) is `not_rederived`; production carries none.
5. **Bound to the repair and to the observations judged** (QT-02, R2).
   - The plan records the store manifest it assumes: the current pickle must be its `target_sha256` and the repaired pickle its `new_sha256`, and both must end on the same session.
   - A live `apply` refuses an unbound plan, and refuses when the live store is not the repaired pickle (the repair is not installed).
   - The plan records a digest of every observation it judged: each changed symbol's row and each row carrying a market-wide value (1,418 in production). `apply` recomputes it inside its transaction, so an observation frozen after the plan refuses the apply.
6. **Exposure is reported, not assumed** (QT-09).
   - A row is `exposed` when its **input prefix** differs between the two stores: every bar up to its cutoff, in the stored order the producer read (an EMA is seeded at the first bar).
   - Exposed rows have no proven wrong fill, and they are quarantined only with `--include-exposed`.
7. **One transaction, one batch** (QT-05, R5).
   - `apply` validates every vector and inserts every row inside one `BEGIN IMMEDIATE` transaction; a refusal or failure writes nothing.
   - The batch id is unique to the plan instance.
   - The approved scope (ids and `--include-exposed`) is persisted before the first attempt, and a retry must match it.
   - A rerun after an interruption completes the same batch.
   - `rollback` releases exactly that batch, never an earlier quarantine or another plan's.

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
- Their carried scanner features (RSI, quality score, the setup itself) cannot be re-derived here. While Kite was authoritative, those came from the Kite snapshot, not the store (§4), so exposure is a risk, not a likely defect.
- **Recommendation:** quarantine the 10 proven rows only. Keep `--include-exposed` for a deliberate owner choice.

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

**The source.** `research/feature002/observe.py::_hist` reads `scan.bulk_fetcher.get_cached` first. That returns the **active Kite snapshot** (`bars_equity.csv`, broker history) whenever Kite is authoritative; otherwise the official store, and yfinance last.

- The 64 rows are consistent with that broker source, and the store repair does not change them.
- The 3 rows that do match the store (KNACK on 10, 11 and 21 Oct) are only exposed: no feature moves.
- So feature002 is **not damaged by the store defect**.
- It does carry a different, pre-existing property: its inputs are broker history when the broker is logged in, and official history otherwise.

**Recommendation.** This is a provenance note for feature002's eventual analysis, not something this tool can repair. Nothing in that ledger was or will be written by it. The owner's options:

- record the input source per row before analysing feature002, so broker-sourced and official-sourced rows can be separated (next engineering item); or
- accept the mixed source as the production scanner's own view, which it mirrors.

## 5. Procedure (runbook stage 5; desk stopped)

1. **Plan.** Run `plan` on:
   - the stage-3 backup of `feature_store.db`;
   - the pre-repair pickle copy;
   - the planned pickle;
   - the CA copy;
   - **`--store-manifest`**, the stage-2/4 repair manifest.

   `--shadow-db` is optional.
2. **Review the manifest:**
   - `complete` is true, and `store_binding.verified` is true;
   - `unexplained`, `not_rederived` and `refusals` are empty;
   - the `affected` list (expected: the 10 rows above, plus any frozen since);
   - the `exposed` list.
3. **Apply.** Run `apply --manifest … --features-db <live db> --owner-approval "…"`. Add `--include-exposed` only if the owner chooses to.

**`apply` refuses:**

- empty approval words;
- an incomplete plan;
- an unbound plan (live target);
- a manifest already applied;
- a running or loaded desk (live target);
- any planned row whose vector sha256 changed or that vanished, checked inside the transaction.

It writes only `observation_quarantine` rows, in one transaction, so nothing is written on a refusal or failure.

**Rollback** releases exactly this plan's batch:

```bash
python -m product.learning_observation_correction rollback --manifest <manifest> --features-db <live db>
```

The vectors were never touched, so this restores the previous learning view exactly. A quarantine that existed before the batch is kept.

## 6. Tests

`tests/test_learning_observation_correction.py` (40 tests) covers:

- **Loaders and quarantine:**
  - both loaders skip quarantined rows, and the frozen record is byte-identical;
  - write-once quarantine with a required reason.
- **Plan:**
  - it is read-only and finds the lost IPO high;
  - `unexplained` and `not_rederived` make the plan incomplete;
  - frames are CA-adjusted exactly as `get_ohlcv` serves them; without the table, the same rows are not reproduced;
  - `/Volumes` paths and a missing feature store are refused;
  - materiality;
  - dtype-blind but order-sensitive history comparison;
  - input-prefix exposure;
  - weekend cutoffs against the session calendar.
- **Apply:**
  - each refusal;
  - exposed rows are quarantined only when asked;
  - the desk check and the repair binding for a live target.
- **feature002 report:** finds the lost 52-week high and writes nothing.
- **GPT cycle-6 sketches:**
  - QT-02: a stale candidate proves nothing;
  - QT-03: a breadth fill on an unchanged peer, and a value that must disappear;
  - QT-04: no calibration and no rejection analysis from a quarantined row; a stale challenger is not served;
  - QT-05: an atomic batch; rollback keeps a prior quarantine; an interrupted apply completes on rerun;
  - QT-09: scanner input order.
- **GPT verification sketches:**
  - R2: a missing weekday archive; observations added after the plan;
  - R3: an already-settled calibration row;
  - R4: a WAL quarantine;
  - R5: a second plan's rollback keeps the first plan's quarantine;
  - R6: scanner breadth on an unchanged peer.

**Mutation checks.** Each of these, reverted on its own, fails its test:

- the loader exclusion;
- the reproduction check;
- the CA adjustment;
- the coverage refusal;
- the cross-sectional rule;
- the vanished-value rule;
- the calibration, non-event and challenger guards.

`tests/test_pit_context_invariance.py` also pins that no desk module imports this CLI.
