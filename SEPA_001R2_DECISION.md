# SEPA-001R2 decision

**Revision:** SEPA-001R2.1  
**Run SHA:** `118c087ffe1d1568ccaeef1c22b380289661219e`  
**Freeze:** `SEPA_001R2_RESEARCH_FREEZE.md` (config hash `76acdb2bb188a5f4`)  
**Eval:** 2020-09-28 → 2026-07-24, official NSE bhav, daily observation  
**PIT class:** `PIT_DEGRADED`  
**Global `ca_complete`:** false (verifier unchanged)  
**`ca_research_acceptable`:** true (293 unresolved events enumerated; 269 path crossings censored; no inferred factors)

SEPA-001 and SEPA-001R files are immutable. This decision uses the
predeclared walk-forward in `SEPA_001R2_VALIDATION_PROTOCOL.md`. No RS,
VCP, Stage-2, stop, or buy-zone was changed after seeing results.

Paper shadow, if it were granted, would still be observation / fake-money
only. It is **not** granted.

---

## Verdict

# KEEP RESEARCH-ONLY

Not `PROMOTE TO PAPER SHADOW`: the untouched confirmation block for core F
is **REJECT** (n=975, E[R]=−0.122, block CI [−0.250, +0.014]). Pooled F
`STATISTICAL_SIGNAL` is development-heavy and is not confirmation evidence.
The runner’s first `DEPLOYMENT_ELIGIBLE` label used the pooled sample; that
label is **void**. Deployment must follow the confirmation block.

Not `MODIFY AND RETEST` in this milestone: thresholds are frozen. A future
experiment that changes the spec must write a new protocol *before* seeing
that experiment’s numbers.

Not `REJECT CORE SEPA`: Stage-2 still thins the scanner baseline (A −0.130R
→ B −0.019R), RS≥70 moves the point estimate further (C +0.026R,
INCONCLUSIVE), and the research architecture is now causally specified.
The *trade rule* failed confirmation; the *research question* is not closed.

No paper integration, broker, GTT, or live autopilot follows from this file.

---

## Numeric answers

### 1. Snapshot funnel (symbol × date)

| Stage | Count |
|---|---|
| Candidates | 3,474,133 |
| Investable | 2,110,601 |
| Stage-2 | 627,636 |
| RS pass | 399,311 |
| VCP detected | 215,487 |
| Pivot defined | 627,591 |
| Entry-ready | 59,346 |

### 2. Unique-opportunity funnel

| Stage | Unique count |
|---|---|
| VCP detected | 22,441 |
| Valid pivot | 26,247 |
| Entry-ready | 16,943 |
| Valid next-open fill | 6,734 |
| Gap-through | 3,308 |
| Observed extended | 7,192 |
| Left-censored | 10,801 |
| CA-censored | 215 |
| Stop-too-wide | 1,225 |
| Expired/failed | 1,385 |
| Pivot-retest | 325 |

Ledger unique setups: **19,336**. Unique funnel stages are not a partition
of that number (a setup can hit several refusal classes over its life).

### 3. CA-censored observations

**269** path crossings (`diagnostics.ca_censored_outcomes`). Unique-funnel
CA-censored setups: **215**.

### 4. Old static future quarantine

**184,919** symbol×date investable observations would have been removed by
the static symbol blacklist even though they sat **before** the unresolved
event (causal segments kept them).

### 5. Daily scanner vs `scanner_step=5`

Canonical run is daily. Counterfactual on the same path:

- **688,079** A signal-days fell on sessions that a 5-day scanner clock
  would have skipped
- **8,411** E entry-ready sessions would have been skipped by that clock

Do not treat 5-day sampling as the A–E ladder.

### 6. Session embargo vs calendar-day embargo

**421,841** disagreements between `as_of + Timedelta(days=hold)` and the
actual exit session (weekends, holidays, 1-session and 20-session holds).

Raw vs embargo-deduped n (this is why A cannot be compared to F as the
same `n`):

| Variant | Raw signal-days | Deduped n |
|---|---|---|
| A | 597,235 | 91,549 |
| B | 278,945 | 42,276 |
| C | 160,833 | 25,430 |
| D | 59,192 | 13,560 |
| G | 357,868 | 20,813 |
| E | 5,309 | 3,392 setups/fills |
| F | 6,798 | 4,208 setups/fills |

### 7. Does Stage-2 add value after true as-of universe construction?

**As a scanner filter, modestly, not as a standalone edge.** A −0.130R
(n=91,549, REJECT, CI [−0.150, −0.107]) → B −0.019R (n=42,276, REJECT, CI
[−0.047, +0.008]). The loss shrinks; the CI still includes zero / negative.
Confirmation B is REJECT (−0.069R).

### 8. Does RS≥70 add value?

**Mildly on the pooled scanner path; not in confirmation.** B −0.019R → C
+0.026R (n=25,430, INCONCLUSIVE, CI [−0.010, +0.059]). Ungated 20d forward
% is **not** monotone in RS (50–69 +3.16% mean, 70–79 +3.30%, 80–89 +3.04%,
90–94 +2.80%, 95–99 +6.36% mean / **−1.07% median**). Confirmation C is
REJECT (−0.054R). Do not raise the threshold from this file.

### 9. Does VCP add incremental value over Stage-2+RS?

**No, on the scanner ladder.** C +0.026R INCONCLUSIVE → D −0.017R REJECT
(n=13,560). Causal VCP as a scanner add-on destroyed the thin C edge.

### 10. Does specific entry improve expectancy or primarily destroy frequency?

**Both, and confirmation still fails.** D −0.017R (n=13,560 scanner fills)
→ E +0.095R (n=3,392, INCONCLUSIVE, CI [−0.013, +0.219]). Frequency drops
~4×. Unique funnel: 16,943 entry-ready → 6,734 valid fills, with 7,192
extended and 3,308 gap-through. Confirmation E is REJECT (−0.110R, n=857).

### 11. Core F trades by calendar year (`as_of`)

| Year | n | E[R] |
|---|---|---|
| 2020 | 161 | +0.406 |
| 2021 | 639 | +0.292 |
| 2022 | 507 | **−0.096** |
| 2023 | 998 | +0.407 |
| 2024 | 928 | +0.028 |
| 2025 | 600 | **−0.180** |
| 2026 | 375 | **−0.030** |
| **Total** | **4,208** | **+0.123** |

### 12. Development / validation / confirmation (core F)

| Block | n | E[R] | Block CI | Verdict |
|---|---|---|---|---|
| Development (→2023-12-31) | 2,305 | +0.264 | [+0.116, +0.419] | STATISTICAL_SIGNAL |
| Validation (2024) | 928 | +0.028 | [−0.180, +0.259] | UNDERPOWERED |
| Confirmation (2025-01-01→2026-07-24) | 975 | **−0.122** | [−0.250, +0.014] | **REJECT** |

E mirrors F: development +0.215 SIGNAL; validation +0.032 UNDERPOWERED;
confirmation −0.110 REJECT.

### 13. Is performance dominated by one year, sector, or regime?

**Year, yes. Sector/regime, not measurable here.** F’s pooled plus comes
from 2020, 2021 and 2023. 2022 is already negative *inside* development.
2025 is the worst year. Sector labels are **UNKNOWN for 3,131 / 4,208**
fills (map coverage hole). Regime classifier returned `unknown` for every
F row. Do not claim a sector or regime edge from this run.

### 14. Is core F better than baseline A in a statistically defensible way?

**Not as a deployment claim.** Pooled F +0.123R vs A −0.130R uses different
statistical units (unique setups vs embargoed scanner days) and is driven
by 2020–2023. On the **confirmation** block both are losers (F −0.122, A
−0.303). F loses less than A in 2025–26; that is not a licence to paper
a negative-expectancy rule. n_eff for pooled F is 916 vs 4,208 rows.

### 15. Is core F actually investable frequently enough for a retail system?

**Frequency is not the binding constraint. Expectancy in the untouched
block is.** 4,208 F fills in ~5.96 post-warm-up years is ~700 names/year
across the whole official book — easily frequent enough if the edge were
real. Confirmation says it is not. Unique setups 19,336 with 10,801
left-censored also show that *detection* is common; *confirmed forward
edge* is not.

---

## What this milestone did and did not do

Did: causal CA segments, daily A–G observation, session embargo, honest G
signal study, as-of candidate membership, datetime64-ns as-of screens,
exhaustive unresolved-event audit, predeclared walk-forward, freeze, then
the long-history run.

Did **not**: paper integration, live orders, GTT, autopilot, or any
threshold change after seeing confirmation.
