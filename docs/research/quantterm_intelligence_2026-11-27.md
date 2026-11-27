# QuantTerm intelligence and readiness

Baseline `cf4aad988352ef45b65e0fd45a607e3384324ab9`. Development used the isolated research worktree and the existing verified archive copy. Production recovery is deferred.

## Stock selection and historical coverage

The existing scoped Gauntlet validates official archive bytes and separates missing or invalid market sessions, other series, and corporate-action gaps. The discovery runner consumes validated contiguous bars, runs the production scanner without its live evidence gate, selects on the decision close and uses the shared simulator on the next session. Its cycle records can be rescored without rereading archives or simulating exits. Production recommendations use separate evidence and eligibility gates, and the protected experimental paper lane has its own forward outcome and learning records.

The original execution-geometry study found eight negative-expectancy configurations. The scanner lost after costs on all 15 individual signals and lagged its matched random control. The earlier relative-strength result faded under cost stress and failed the extended replay. The two attributed signal combinations also failed their extended replay. Their historical results and the prospective registration are preserved.

This pass tested two fixed variants of the relative-strength leader filter without tuning execution. Both retain positive 30-session excess and a close above its 50-session average.

- Delivery confirmation ranks by the rise in official delivery share over the latest five validated sessions versus the preceding 25. Missing delivery history excludes a candidate from this rule. Rising delivery share is a participation proxy, with no inference about buyer identity.
- Downside resilience ranks by positive average excess on declining cohort-basket sessions over the latest 30 sessions. A down day requires 50 valid cohort returns. A decision requires at least eight such days. The market series is the same archived cohort equal-weight close return, with no filling across gaps.

The isolated replay uses January–October 2025 alone. All 232 names, 48,244 accepted bars, per-date archive hashes, quarantine results, and original strategy/cycle outputs match the existing discovery study. November 2025 and later remain reserved. Modern cohort selection limits inference. The 1,765 production archive files are inventory only. This pass establishes reproducibility on the 232-stock cohort; broader validated coverage and comprehensive historical computation remain unfinished.

All rows use five slots, next-session open, the shared 3% stop / 5% target / 2% breakeven rule, 10-session exits and 32 bp round-trip friction. Each row has 14 decision cycles. Target, stop and expiry percentages are empirical frequencies conditional on resolved fills. Excess is per cycle in percentage points. Intraday order between barriers uses the existing conservative simulator convention.

| Selection | Opportunities | Resolved | Target / stop / expiry | Net per fill | Net R/fill | Compound | Drawdown | Basket / random excess |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Scanner | 1,894 | 70 | 21.4% / 67.1% / 11.4% | −0.432% | −0.144 | −6.10% | 9.67% | −0.023 / −0.240 |
| Relative strength | 1,374 | 69 | 27.5% / 63.8% / 8.7% | +0.106% | +0.035 | +1.27% | 5.22% | +0.514 / +0.297 |
| Random | 3,192 | 70 | 25.7% / 67.1% / 7.1% | −0.192% | −0.064 | −3.06% | 10.71% | +0.217 / 0 |
| Delivery confirmation | 686 | 70 | 21.4% / 71.4% / 7.1% | −0.460% | −0.153 | −6.39% | 8.59% | −0.050 / −0.267 |
| Downside resilience | 971 | 70 | 21.4% / 68.6% / 10.0% | −0.232% | −0.077 | −3.33% | 7.83% | +0.177 / −0.040 |

The two variants fail the net expectancy and random comparison gates. Their one-sided descriptive lower bounds against random are −1.845 and −1.399 percentage points per cycle, respectively, with insufficient inference cycles. Both lose at the base cost, so adding fixed extra friction cannot rescue them. The runner does not model execution-induced differences in fill probability. Holding time and stock-specific probability are unmeasured in these saved cycle records. No new rule advances to shadow or paper.

## Probability diagnostics

The existing calibration module now scores target-first, stop-first and expiry together from saved discovery records. It matches random training slots by date and allocation, checks that cycle windows do not overlap, and freezes both priors before the later cycles. One joint half-count per class avoids zero-probability predictions. Missing entries and censored holdings are counted separately. Tests include an out-of-order grid, overlapping horizons, excess random slots and an unmeasured training sample.

| Selection | Later resolved fills / cycles | Pooled vs random-prior Brier | Pooled vs random-prior log loss |
|---|---:|---|---|
| Scanner | 35 / 7 | 0.628 / 0.615 | 1.065 / 1.263 |
| Relative strength | 34 / 7 | 0.652 / 0.625 | 1.110 / 1.211 |
| Delivery confirmation | 35 / 7 | 0.487 / 0.510 | 1.047 / 1.081 |
| Downside resilience | 35 / 7 | 0.548 / 0.562 | 0.946 / 1.135 |

Scores are descriptive, with one reliability point per class and only seven later cycles. Across same-date holdings the effective sample cannot be inferred from trade counts. Neither pooled improvement nor the risk ranking establishes stock-specific calibration or improves net expectancy. The discovery estimates stay offline.

Recommendation probability displays now return UNMEASURED when no usable outcomes exist. The previous no-data response exposed a 50% prior as a number in the detail view. Measured sources are described as pooled outcome evidence. Historical price-path simulations are identified separately and their barrier frequencies are no longer compared directly with positive-R frequencies. The persisted-first recommendation route retains its existing background rebuild. This pass establishes no new latency baseline, and production dashboard performance is blocked by the host state.

## Protected paper execution and learning

An isolated synthetic WATCH card passed the controlled exploration checks, entered at half-risk or less through OMS protection, retained its WATCH verdict, and preserved decision and intent identity through strict book restore. The next test-session stop cancelled protection and closed the OMS position. Duplicate marking added no P&L, and learning ingested one forward observation with zero additions on replay. This proves the fixture lifecycle using existing components. It establishes neither prospective profitability nor learning uplift.

Fault injection reproduced a settlement gap with the earlier code. If the OMS close threw after protection was cancelled, the in-memory book had already erased the open position and posted P&L. The new ordering validates bars and close economics, synchronizes the simulated OMS exit, then applies the book mark and P&L. The same-session retry resumes an EXIT_PENDING order and settles once. This does not create a single transaction across OMS, protection, accounting and learning stores.

Independent review found that EOD could quietly advance past a failed older session and that a fresh EOD process could use a raw book. EOD now stops at a thrown failure, preserves earlier completed marks, and reports a retryable failure. It requires the existing adapter for the production singleton when exposure is open and refuses to overwrite an unrecovered book. A failed book save prevents a successful settlement or learning status. The existing calendar loop still needs to distinguish an empty older archive result from an exchange holiday and account for special sessions. Audit projection interrupted by a failed save is not replayed by this change. The historical simulator, recommendation statuses and live interlocks retain their existing behavior.

## Verification and remaining gates

| Assessment | Status | Evidence |
|---|---|---|
| Discovery execution and chronology | PASS | Archive identity and original cycles reproduced. Reserved data and the frozen relative-strength implementation were untouched. |
| New strategy selection | FAIL | Both variants lose before added stress and trail matched random. |
| Stock-specific calibration and learning uplift | UNVERIFIED | No untouched validation or prospective fills, and no demonstrated frozen-baseline improvement. |
| Protected execution and learning fixture | PASS | Controlled WATCH probe, restored lineage, one settlement and one learning update. |
| Production operation | BLOCKED | Read-only working volume, disabled agents, unresolved freeze index, interrupted paper cycle and RPEL risk reconciliation. Current independent full backup remains outstanding. |
| Production action | BLOCKED | Separate approvals required for backup, ledger repair, write access, restart and deployment. Live remains HARD-LOCKED. |

An independent reviewer verified the unchanged archives/cycles, matching of the trading controls, and the absence of reserved-period exposure. The review found general calibration weaknesses around unmatched reference slots and label-window overlap. Both are covered by the revised implementation and negative controls.

## Reproduction record

The bounded replay uses the existing `research.opportunity_discovery` runner, stock list from the frozen specification and isolated archive root. The only changes in the replay process are two in-memory rule additions. The rules above describe the feature windows and ranking; ties use the existing symbol ordering. The raw replay artifact SHA-256 is `39e340528f1f5c1c6b7468ce1b6d99f029b4f34fef013520a11b5a9a41b59370`. Re-score from its `cyclic` entries with `research.calibration.discovery_exit_calibration`, using `random` as the reference and the stored seven-cycle split. Do not resimulate to calculate new summary scores.
