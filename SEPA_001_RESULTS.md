# SEPA-001 Results

**Eligibility version:** `sepa-001.v1`  
**Date:** 2026-08-21  
**Live execution:** not wired. Paper/autopilot/broker unchanged.

This file answers the ten research questions from the SEPA-001 brief using
official NSE bhavcopy (768 sessions, cache through 2026-08-20). It is **not**
a promotion memo.

## Sample honesty (read this first)

| Item | Status |
|------|--------|
| Price | Official bhavcopy, sliced `index <= as_of` |
| Universe | `point_in_time_universe` reports `survivorship_complete=true` but note: inferred from local first/last sessions, **not** an official listing archive |
| Corporate actions | **`logs/ca_events.json` absent** — prices are **raw**. `pit_safe` is therefore **false** for this run |
| Fundamentals / news | **Not used** (Yahoo `.info` is not PIT) |
| Symbol set | Top **80** names by recent rupee turnover (not A–Z order) |
| Sampling | `sample_step=10`, `lookback=320`, `horizon=20` |
| Fills A–D | Production scanner entry/stop/target (`signal_backtest._simulate_timed`) + CNC costs |
| Fills E–F | Next-bar open vs structural stop; gap through buy-zone = no fill |
| Overlap | One open trade per variant/symbol (simple embargo) |

Harness `evaluate()` uses the existing `research.harness` gate (floor 30 trades).
A PROMOTE on n≈34 with missing CA is **not** a licence to trade.

Primary run: `docs/overhaul/experiments/SEPA-001/ablation_80liquid.json`  
Replay objects: `docs/overhaul/experiments/SEPA-001/SEPA_001_REPLAY.md`

### Main ablation (80 liquid names)

| Variant | Definition | n | Expectancy (net R) | PF | Max DD (R) | Fail-break % | +1R % | Harness |
|---------|------------|---|---|---|---|---|---|---|
| A | Production scanner signals | 688 | +0.050 | 1.10 | −20.6 | 41.4 | 42.9 | UNDERPOWERED |
| B | A + 7-rule Stage-2 structure (no RS) | 273 | +0.082 | — | — | — | — | UNDERPOWERED |
| C | B + RS percentile ≥ 70 | 200 | +0.123 | 1.26 | −8.4 | 40.5 | 49.5 | UNDERPOWERED |
| D | C + structural VCP (scanner fill/stop) | 34 | +0.360 | 1.94 | −3.7 | 32.4 | 52.9 | PROMOTE* |
| E | D + pivot buy-zone + structural stop | 0† | — | — | — | — | — | no trades |
| F | Core SEPA only (no scanner required) | 0† | — | — | — | — | — | no trades |

\*Harness PROMOTE on D is **not** research-grade: n=34, no CA table, scanner ATR geometry, 80-name liquidity bias.  
†After relaxing ATR-multiple as a *hard* reject (percent risk > 8% remains the reject), a **60-name** re-check produced **E=1 / F=2** trades — still UNDERPOWERED.

Median R for A is **negative** (−0.26) despite a slightly positive mean — a few large winners, many small losses.

---

## 1. Did strict Stage 2 improve the baseline?

**Yes, modestly, as a scanner filter — not as a standalone SEPA system.**

A → B: n 688 → 273 (frequency −60%), expectancy +0.050 → +0.082 R.  
Harness still UNDERPOWERED (needs ~920 trades to trust a +0.10R edge).

The 7 MA/52w rules (price vs 50/150/200, stack, rising 200, 30% off low, within 25% of high) **thinned** scanner noise. They did **not** flip the sign of the edge from negative to robustly positive.

On a 40-name coarser grid, requiring 7-of-8 vs structure-only B was **the same expectancy** (~+0.22 R) with n 121 vs 97 — little extra cost to being strict on the MA stack in that slice.

## 2. Did cross-sectional RS improve it?

**Yes, as an additional scanner filter, with a sample-size cost.**

B → C: n 273 → 200, expectancy +0.082 → +0.123 R, max DD −20.6 (A) vs −8.4 (C), PF 1.10 → 1.26, % reaching +1R 43% → 50%.  
Harness: still UNDERPOWERED (p≈0.08, PSR 0.93).

RS is **not** the old +5pp vs Nifty metric. It is `rs_cs_v1`:  
`0.40*r63 + 0.20*r126 + 0.20*r189 + 0.20*r252`, percentile vs the as-of liquid set.

**RS threshold study (40 liquid names, scanner+structure+RS = variant C):**

| RS ≥ | n | Expectancy R | Harness |
|------|---|--------------|---------|
| 70 | (main run 80 names) 200 | +0.123 | UNDERPOWERED |
| 80 | 45 | +0.409 | PROMOTE (small n) |
| 90 | 22 | +0.871 | UNDERPOWERED |

Higher RS **looks** better and rarer. Do **not** pick 90 from this table — n=22 is a postcard, not a study.

NSE transfer of Minervini’s RS ≥ 70: **compatible as a frequency filter**, not proven as an optimal cutoff.

## 3. Did structural VCP improve it?

**As a scanner add-on (D): the point estimate improved; the sample is too small and the fills are not SEPA fills.**

C → D: n 200 → 34, expectancy +0.123 → +0.360 R, fail-break 40.5% → 32.4%, max DD −8.4 → −3.7 R.  
Harness said PROMOTE. That verdict is **not** acceptable as SEPA evidence because:

- D still enters with the **scanner’s** `entry = last price if through pivot` and **2×ATR stop / 4×ATR target**.
- n=34, CA missing, one liquidity cohort, 2025-heavy.

**As core SEPA (F): VCP + buy-zone produced almost no trades.** Gate counts on 40 liquid names (step 20):

| Gate | Hits |
|------|------|
| Stage-2 structure | 140 |
| RS ≥ 70 | 188 |
| VCP detected | 58 |
| Buy-zone valid | 32 (includes some failed-VCP pivots) |
| Fully eligible (8/8 + VCP + zone + stop) | **0** in that diagnostic |

Among detected VCPs, **median distance to pivot was +9.9%**; 72% were >1.5% extended. The zigzag is finding real pullbacks, then price has **already run**. That is the opposite of a specific entry.

## 4. Did pivot-entry discipline improve it?

**It removed the trades rather than improving them.**

E (buy-zone on the D set) was **0** in the 80-name run.  
Widening the zone to **5%** still gave E=0 / F=0 on 40 names (before the ATR-stop fix). After the fix, E=1, F=2 on 60 names.

**Buy-zone sensitivity:** not statistically estimable — the 1.5% default (and 5%) did not produce a usable n. What we *can* say:

- Synthetic tests prove the invariant: pivot ₹P, price P×1.07 → **NO TRADE — INVALID ENTRY** even with RS 97 and a clean VCP.
- On NSE liquid names, scanner “VCP” hits are typically **already extended vs the structural pivot**. Applying a real buy-zone therefore **does not refine D; it deletes D**.

That is a success for methodology and a failure for frequency.

## 5. Did structural stops improve risk characteristics?

**Not measurable on core SEPA (n≤2).**

D’s better max DD uses **scanner ATR stops**, not contraction lows.  
Diagnostic: 23 names flagged `WIDE_STRUCTURAL_STOP` at the 8% cap.  
An earlier rule that also rejected **>3×ATR** killed synthetic 4% stops on low-ATR trends; that was too strict and was changed to: **hard reject only if stop distance > 8%**. ATR multiple remains diagnostic.

We cannot claim structural stops improved MAE/MFE versus 2×ATR on this sample.

## 6. Which component reduced false breakouts the most?

**On the scanner path:** Stage-2 + RS + VCP (A 41.4% fail-break → D 32.4%).  
**On the SEPA path:** buy-zone reduced false breakouts by **not taking the extended prints** (n→0). That is reduction by refusal, not by a better active trade set.

## 7. Which component cost the most trade frequency?

1. **Buy-zone / specific entry** — 34 D trades → 0 E trades (80-name run).  
2. **Structural VCP** — 200 → 34.  
3. **Stage-2 structure** — 688 → 273.  
4. **RS ≥ 70** — 273 → 200 (smallest cut, useful).

## 8. Which combination produced the best *robust* result?

**None is robust enough to promote.**

If forced to pick a *research* leader among **powered-enough-looking** stacks:

- **C (scanner + Stage-2 structure + RS≥70)** is the only variant with n=200, better expectancy than A, better DD, still UNDERPOWERED by the house gate.
- **D** has the best point estimate and a harness PROMOTE, but n=34 and non-SEPA fills — **do not generalise**.
- **F (full core SEPA)** has no evidence of positive expectancy because it barely trades.

Robustness checklist vs C: year mix is 2025/2026 heavy; CA missing; 80-name bias; overlapping trades; regime bucket UNKNOWN. **Not robust.**

## 9. Does NSE evidence support strict Minervini thresholds?

| Rule | Evidence on this sample |
|------|-------------------------|
| RS ≥ 70 | Useful filter on the **scanner** path. ≥80 looks stronger with n=45 — not confirmed. **Transfer: plausible, not proven.** |
| 30% above 52w low | Packed into B; B helped vs A. No isolated ablation of 25% vs 30%. **Unseparated.** |
| Within 25% of 52w high | Same — part of B. **Unseparated.** |
| 8/8 template | Strict 8/8 (includes RS) is C, which beat A/B on point estimate. 7/8 vs 8/8 on B (40 names) was a tie in expectancy. **8/8 not shown to hurt; not shown to be necessary.** |
| Pivot buy-zone 1.5% | **Too tight for this VCP+liquid cohort** (median +10% above pivot). Needs a designed parameter study *after* VCP timing is closer to the coil, or a different pivot definition (last contraction high vs earliest base high). |

Minervini’s numbers are **discretionary in the books**. We implemented them mechanically. NSE data did **not** validate the full stack as a turnkey rule set.

## 10. Is Full Core-SEPA superior to the current baseline?

**INCONCLUSIVE / not on this sample.**

Full core SEPA (F) did not produce a comparable trade stream. We cannot say it is better or worse than scanner A. We *can* say:

- Pieces of SEPA (Stage-2, RS, swing VCP **as filters on the existing scanner**) **point** to higher expectancy and lower heat.
- The **specific entry point** — the heart of SEPA — currently **conflicts** with how those scanner hits are priced (already extended).
- Without PIT CA, no result is research-grade.

**Final result: INCONCLUSIVE**

---

## Architecture recommendation

**C. MODIFY AND RETEST**

Not A (promote to paper): core eligibility does not yet trade, CA table missing, D’s PROMOTE is the wrong fill model.  
Not B (keep as a dead research filter): Stage-2 + RS are worth keeping as **research gates** to stack on the scanner in SEPA-002 *if* CA and a larger PIT universe are fixed.  
Not D (reject): several components moved the needle in the expected direction.

### What to change before SEPA-002

1. Load `logs/ca_events.json` and refuse to label runs PIT-safe until `verify_ca_adjustment` passes.  
2. Rebuild VCP pivot as **last contraction resistance**, and only evaluate buy-zone on bars **at that coil**, not 10% later (or sample daily, not every 10th bar).  
3. Keep buy-zone as a hard gate; do **not** silently fall back to `entry = last price`.  
4. Re-run A–F on the full liquid universe with embargoed walk-forward years, then harness with `n_trials` = number of variants.  
5. Do not wire `evaluate_sepa_eligibility` into autopilot until F (or E) has n≥30 **and** CA-complete **and** a non-negative block-bootstrap CI.

SEPA-001 delivered the **canonical object and tests**. It did **not** deliver a tradable SEPA edge.

---

## Unit-test evidence (independent of bhavcopy)

Covered in `tests/test_sepa_001_eligibility.py`:

- Each of the 8 template rules pass/fail; missing history is `passed=None` (fail-closed).  
- 30% off 52w low; 25% from 52w high; SMA200 slope.  
- RS percentile has no future bars; universe membership is respected.  
- VCP: 2- and 3-contraction pass; grind / widening / deep base / expanding volume reject.  
- Future bars cannot change as-of JSON.  
- **Excellent stock + 7% extension → `eligible=False`, headline `NO TRADE — INVALID ENTRY`.**  
- Deterministic serialization.

That invariant is implemented even though NSE frequency at 1.5% is currently near zero.
