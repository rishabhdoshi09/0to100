# EDGE-004 — Red Team

Skeptical review of the frozen R1 / Top20-losers / monthly run. No spec change.

## Claims under review

**H1.** Lowest 21-session return Top 20 has positive net excess vs EW.  
Observed: net CAGR **12.55%** vs EW **25.62%** (excess **−13.07%**). vs Nifty proxy **−8.74%**. Monthly excess mean **−77 bps**. Harness **REJECT**.

**H2.** Inverse decile slope (losers beat winners next month).  
Observed: Spearman **−0.14** (wrong sign for reversal). D10 mean 2.01% vs D1 2.03%; D10−D1 **−0.02%**. Deciles are flat (~2.0–2.2%). No reversal surface.

## Defect hunt

| Risk | Finding |
|---|---|
| Sort inverted | Score = −R1. D10 is the lowest prior return and does **not** outperform. If we had bought winners instead, WIN excess is **−13.11%** — both tails lose. Direction is correct; the anomaly is not there. |
| Look-ahead | `incl_momentum(close, j, 21)` uses `close[j]/close[j-21]−1`. Append-future at same j is invariant. |
| Stale universe / fills | Same-session print + next-open reused. Avg filled 19.2/20. |
| Cost units | TO/year 1165%, drag 3.73%. Gross 16.8% still << EW 25.6%. Costs worsen a hole; they do not create it. |
| Confirmation +8.2% vs EW | Real. Development −24.7%, validation −17.3%. Later-block average still **−4.5%**. Do **not** rewrite the hypothesis as “reversal only after 2025.” |
| Only 10d works | R2 excess **−6.9%**. Still negative. Not a MODIFY. |
| WIN control | −13.1%. 21-day extremes (both tails) underperform the middle. Consistent with “don’t buy last-month rockets *or* knives,” not with Jegadeesh losers. |
| 2m/q CAGRs | Monthly annualizer inflates them (noted in the report). Ignored. |
| FEATURE-002 / BUY | No imports. |
| Independence of names | 70 monthly portfolio returns. |

## Material defects that would change the verdict

None. REJECT stands.

## Do not do next

- Switch to R2/R3 or skip-5 inside EDGE-004
- Add a 2025-only gate
- Combine with EDGE-001 12-1 (ensemble of a RESEARCH-ONLY and a REJECT)
