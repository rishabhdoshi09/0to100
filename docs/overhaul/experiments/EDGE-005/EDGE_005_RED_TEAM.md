# EDGE-005 — Red Team

Skeptical review of the frozen P1 / Top20-near-high / monthly run. No spec change.

## Claims under review

**H1.** Names nearest the 252-session high, Top 20 EW, beat EW after costs.  
Observed: net CAGR **19.45%** vs EW **25.62%** (excess **−6.17%**). vs Nifty proxy **−1.84%**. Gross 23.68% still below EW. Monthly excess mean **−48 bps**. Harness **REJECT**.

**H2.** Positive decile slope.  
Observed: Spearman **0.73**. D1 (farthest) 1.74% vs D9 2.55%. **D10 (nearest) 1.99%** — the extreme near-high bucket underperforms D7–D9. Slope exists in the *middle-high* region; the Top-20 book sits on the D10 dip.

## Defect hunt

| Risk | Finding |
|---|---|
| Look-ahead | `max(close[j-251:j+1])`. Append-future at same j invariant. |
| Sort inverted | D1 is farthest and weakest. LAG control **−19.6%** vs EW. Direction is correct. |
| D10 dip | Material. Do **not** switch to “D8–D9 / 90–99th percentile.” That is a new hypothesis. |
| Costs | TO/year 1102%, drag 3.53%. Gross already loses to EW. Costs are not the reject-of-H1. |
| Confirmation +6.9% vs EW | Same 2025–26 pattern as EDGE-004. Development −10.9%, validation −12.0%. Later-block average **−2.6%**. Do not rewrite as “near-high only after 2025.” |
| Only 63d works | P3 excess **−9.7%**, worse than P1. |
| Scanner demote overlap | Laggards (LAG) are terrible, consistent with the scanner demote. That does **not** make Top20-near-high a portfolio edge. |
| Beat-Nifty technicality | Confirmation vs Nifty +5% keeps the mechanical “no EW and no Nifty” reject from firing. The book still loses to EW over the full sample. RESEARCH-ONLY is the right label, not PROMISING. |
| FEATURE-002 / BUY | No imports. |

## Material defects that would change the verdict

None. RESEARCH-ONLY stands.

The slope is an *explanatory rank* finding, the same family FEATURE-002 is already shadowing. Do not open a second near-high shadow book.
