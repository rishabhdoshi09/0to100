# SEPA-001R2 Results

**Revision:** `SEPA-001R2.1`  
**Eligibility:** `sepa-001r2.v1`  
**VCP:** `vcp_causal_v2`  
**Pivot:** `pivot_last_contraction_v1`  
**Config hash:** `76acdb2bb188a5f4`  
**Data:** official_nse_bhavcopy  
**Eval:** 2020-09-28 → 2026-07-24 (warmup 260 sessions)  
**Observation:** `date_step=1` `scanner_step=1` (canonical daily=True)  
**Universe:** as-of investable, `top_n=None`  
**Unique setups:** 19336 (left-censored unique 10801)  

SEPA-001 and SEPA-001R files are immutable. Layer 1 = signal quality. A–D are scanner-path R studies (deduped by exchange-session embargo). G is a forward-% signal study, **not** SEPA R. Harness PROMOTE is never a deployment label. Paper shadow is not live trading.

## Main comparison (deduplicated units)

| Variant | Unit | Raw n | Deduped n | E[R] | PF | Win % | Avg Win | Avg Loss | Max DD R | CI | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | raw=daily scanner rows;  | 597235 | 91549 | -0.1297 | 0.78 | 37.15 | None | None | -13757.683 | [-0.150, -0.107] | REJECT / NOT_DEPLOYMENT_ELIGIBLE |
| B | raw=daily scanner rows;  | 278945 | 42276 | -0.0194 | 0.963 | 39.92 | None | None | -2351.8403 | [-0.047, +0.008] | REJECT / NOT_DEPLOYMENT_ELIGIBLE |
| C | raw=daily scanner rows;  | 160833 | 25430 | 0.0258 | 1.05 | 41.39 | None | None | -931.9515 | [-0.010, +0.059] | INCONCLUSIVE / NOT_DEPLOYMENT_ELIGIBLE |
| D | raw=daily scanner rows;  | 59192 | 13560 | -0.0171 | 0.968 | 39.59 | None | None | -724.5993 | [-0.059, +0.025] | REJECT / NOT_DEPLOYMENT_ELIGIBLE |
| E | persistent setup identit | 5309 | 3392 | 0.0946 | 1.136 | 31.75 | None | None | -235.4694 | [-0.013, +0.219] | INCONCLUSIVE / NOT_DEPLOYMENT_ELIGIBLE |
| F | persistent setup identit | 6798 | 4208 | 0.1228 | 1.178 | 32.03 | None | None | -276.6079 | [+0.004, +0.230] | STATISTICAL_SIGNAL / NOT_DEPLOYMENT_ELIGIBLE |
| G | signal-day | 357868 | 20813 | — | 2.1415 | 69.47 | 46.31 | — | — | — | NOT_SEPA_R / NOT_DEPLOYMENT_ELIGIBLE |

## Diagnostics (R2.1 vs prior runner bugs)

- Static future-CA false removals (symbol×date kept by causal segments): **184919**
- Scanner step-5 would have missed A signal-days: **688079**
- Scanner step-5 would have missed E entry-ready sessions: **8411**
- Calendar-day vs session embargo disagreements: **421841**
- CA-censored outcomes: **269**

## Yearly universe (mean as-of size)

| Year | Mean candidates | Mean investable | As-of points |
|---|---|---|---|
| 2020 | 1703.2 | 812.8 | 69 |
| 2021 | 1817.0 | 1083.4 | 261 |
| 2022 | 2036.6 | 1189.8 | 258 |
| 2023 | 2227.6 | 1352.1 | 260 |
| 2024 | 2489.8 | 1658.1 | 261 |
| 2025 | 2737.3 | 1685.0 | 251 |
| 2026 | 2998.8 | 1752.0 | 147 |

## RS buckets (ungated 20d forward %, as-of universe, CA-uncensored)

| Bucket | n | Mean 20d % | Median 20d % |
|---|---|---|---|
| 50-69 | 398688 | 3.164 | 0.548 |
| 70-79 | 193303 | 3.3 | 0.504 |
| 80-89 | 189334 | 3.044 | 0.501 |
| 90-94 | 91177 | 2.803 | -0.064 |
| 95-99 | 86324 | 6.362 | -1.069 |

## Sample warnings

- PIT class: `PIT_DEGRADED` (as-of metadata, source=bhav_inferred)
- CA complete (global verifier): `False` ; ca_research_acceptable: `True` ; unresolved enumerated: 293
- Coverage: 2019-08-23 → 2026-08-21 (3054 symbols)
