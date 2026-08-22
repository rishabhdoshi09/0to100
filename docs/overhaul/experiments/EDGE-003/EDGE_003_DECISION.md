# EDGE-003 — Decision

## `RESEARCH-ONLY`

Frozen T1 / all-qualifiers / monthly inclusion is **not** a validated trading edge and is **not** robust enough for a passive forward book.

The predeclared mechanical helper fired `PROMISING — FORWARD VALIDATION WARRANTED` because it lacked a CI / harness / confirmation-magnitude gate. Red-team override under mandate §17: robustness is inadequate.

None of the labels authorise paper, live, FEATURE-002, or production BUY changes.

- Mechanical helper: PROMISING (no failures on its frozen checks)
- Reviewer failures vs §17: monthly excess CI includes 0; harness INCONCLUSIVE; confirmation excess +0.17% vs EW and −1.71% vs Nifty; 2025 net −2.24%
- Later-block average excess vs EW (helper): +1.24%
- Live authorised: `False`
- Paper authorised: `False`
- FEATURE-002 change authorised: `False`

Do not rescue with Top-20 distance rank, a shorter SMA, a 2-month cadence, or a regime / breadth gate inside EDGE-003.

### What was learned

- Trend inclusion **has content** (mean share 54%, not the whole market).
- Included names beat excluded names by ~68 bps/month on average — a small *selection* tilt, not a portfolio edge after costs and OOS stress.
- T2 (price>SMA200 only) ≈ T1. The rising-SMA clause is not doing the work.
- Distance-from-SMA Top20 **underperforms**. Extension-rank is not a rescue.

### Next

Move to an independent hypothesis. Do not start an EDGE-003 shadow experiment.
