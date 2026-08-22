# EDGE-006 — Red Team

Skeptical review of the frozen L1 / highest-ADV Top20 / monthly run. No spec change.

## Claims

**H1.** Most-liquid Top 20 beat EW.  
Observed: net CAGR **18.13%** vs EW **25.62%** (excess **−7.49%**). vs Nifty **−3.16%**. All three walk-forward blocks negative vs EW. Harness **REJECT**.

**H2.** Positive ADV decile slope.  
Observed: Spearman **−0.87**. D10 (most liquid) 1.77% vs D1 2.33%. Anti-monotonic. Liquid names underperform.

## Defect hunt

| Risk | Finding |
|---|---|
| Sort inverted | Score = +ADV. D10 is most liquid and weakest. L0 (lowest ADV) excess **+1.35%** — the opposite tail is slightly better, not a 2pp protocol MODIFY. |
| ADV look-ahead | Rolling 20 including T. Same series used for the investability floor. |
| Costs | TO 329%/year, drag 1.05%. Gross still loses to EW. |
| L0 rescue | +1.35% is thin, CI not computed as primary, and is a new illiquidity-premium hypothesis (consumed history if ever tested). |
| 2m/q CAGRs | Monthly annualizer; ignored. |
| FEATURE-002 / BUY | No imports. |

## Verdict

**REJECT stands.** Do not switch to L0 inside EDGE-006.
