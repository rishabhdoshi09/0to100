# EDGE-003 — Red Team

Skeptical review of the frozen T1 / all-qualifiers / monthly run. No spec change.

## Claims under review

**H1.** T1 inclusion EW beats unconstrained EW after CNC costs.  
Observed: net CAGR **26.89%** vs EW **25.62%** (excess **+1.27%**). Monthly excess mean **+6.8 bps**. Block-bootstrap 95% CI **[−28 bps, +39 bps] includes zero**. Harness **INCONCLUSIVE** (“likely noise”).

**H2.** The qualifier set is not the whole market.  
Observed: mean share **54%** (range 12–92%), mean N **642** of ~1,239 ranked. Content exists. Included next-month mean **2.25%** vs excluded **1.57%** (spread **+68 bps**). Months T1 > exT1: **63%**.

**H3.** Stronger in bull regimes (descriptive).  
Bull-month mean net 2.94% vs sideways 1.67% vs correction 0.08%. n_bear = 2 — ignore. No gate was applied.

## Defect hunt

| Risk | Finding |
|---|---|
| Look-ahead in SMA | `sma_at(close, j, 200)` uses `close[j-199:j+1]`. Slope uses SMA at `j-21`. Unit test: append-future at same j is invariant. Fail-closed on short / non-finite / non-positive closes. |
| Same-close fills | Next-open window reused from EDGE-001. |
| Stale universe | `live_on_session` required. |
| Cost units | 0.32% RT on variable-N EW one-way TO. TO/year 338%, drag **1.08%**. Gross excess vs EW ≈ 2.63%; costs eat about half. They do **not** create the +1.27% (nor do they fully kill it). |
| Variable-N TO | Used ½ Σ\|w_new−w_old\|, not Top-20’s `(added+removed)/(2n)`. Correct for this book. |
| Benchmark | EW investable is the right inclusion bench. Nifty-50 EW proxy also used; official `^NSEI` only from 2024-04-08. |
| 2m / quarterly CAGRs | First write used `_ann_from_monthly` on 2-month and quarterly holds → fake 59% / 87% CAGRs. **Metric bug, sensitivities only.** Re-annualized on calendar span: 26.2% / 23.2%, in line with monthly 26.9%. Those rows were **not** used to pick a winner. Formula excess for 2m/q vs monthly EW dropped. |
| Sector “concentration” | `avg_max_sector` ~72% is **UNKNOWN** on the PIT_DEGRADED map, not a real sector bet. |
| T2 ≈ T1 | Price>SMA200 only: net 26.59%, excess +0.97%. The rising-SMA clause adds ~30 bps of excess. Not a defect; it means T1 is mostly “above the average,” not a distinct slope edge. |
| Only Top20 works | Opposite: T1-Top20 by distance **−3.67%** vs EW. Inclusion is not a stealth extension-rank book. |
| Confirmation reverse | Dev excess +1.24%, confirmation **+0.17%** vs EW and **−1.71%** vs Nifty. Not a protocol reverse (< −2%), but economically the later block is flat. |
| Year dependence | 2021 and 2023 dominate absolute returns (bull years). 2025 net **−2.24%**. Excess is a small tilt on a high-beta book (β vs Nifty proxy **0.93**). |
| Consumed history | FEATURE-001 already used Trend on scanner fires through 2026-07-23. Confirmation is not philosophically pristine for the Trend *idea*. |
| Statistical unit | Inference is 70 monthly **portfolio** excesses, not 45k name-months. Correct. |
| FEATURE-002 / BUY | Study does not import observe / place_trade / Telegram. |
| Mechanical PROMISING | Frozen helper had no CI / harness / confirmation-magnitude gate. It fired PROMISING because later-block average excess was +1.24% and share < 90%. That is a **necessary-condition pass**, not §17 robustness. |

## Material defects that would change the *numbers*

The 2m/q annualizer was fixed and analyze re-run. Primary monthly path unchanged.

No look-ahead, fill, cost-unit, or universe defect that would flip the sign of the +1.27% excess.

## Verdict on the verdict

**Official label: `RESEARCH-ONLY`.**

Mandate §17: PROMISING requires historical evidence *sufficiently robust* to justify future passive validation. A +1.3% excess whose monthly CI includes zero, whose harness is INCONCLUSIVE, whose confirmation excess is +17 bps (and negative vs Nifty), and whose 2025 year is negative, is not that.

Calling this PROMISING would be inconsistent with EDGE-001, which had a *larger* full-sample excess and a real decile slope and was still RESEARCH-ONLY after confirmation stress.

Do **not** open an inclusion shadow book.  
Do **not** switch to T3, 2-month cadence, or Top20 after seeing the tape.  
Do **not** add a breadth/regime gate inside EDGE-003; that is a new ID if ever tested (and would be consumed-history).
