# EDGE-002 — Red Team

Skeptical review of the frozen V1 / Top20 / monthly run. No spec change.

## Claims under review

H1: lowest 126d realized-vol Top 20 has positive net excess vs EW.  
Observed: net CAGR 6.4% vs EW 25.6% (excess **−19.2%**). Monthly excess mean −1.55%, CI entirely below zero. Harness **REJECT**.

H2: monotonic low-vol → higher next return.  
Observed: Spearman **−0.54**. D10 (lowest vol) mean 1.27% vs D4/D5 ~2.55%. Opposite of the hypothesis.

## Defect hunt

| Risk | Finding |
|---|---|
| Look-ahead in vol | `realized_vol(close, j, 126)` uses `close[j-126:j+1]`. Unit test: append-future at same j is invariant. |
| Sort inverted | Score = −σ. D10 has the lowest vol and the **lowest** next return. If the book had been high-vol, we would have said so. Direction is correct; the anomaly is not. |
| Stale universe | Same-session print reused from EDGE-001. Avg names filled 19.8/20. |
| Same-close fills | Next-open window reused. |
| Cost units | 0.32% RT × ~205% one-way TO/year ≈ 0.66% drag. Gross 7.1% still << EW 25.6%. Costs do not create the reject. |
| Benchmark | EW investable is the right CS bench. Nifty proxy also beaten (excess −14.9%). |
| One year | Every year is below a raging EW bull; 2021 is the only negative absolute year. Not a single-year artifact of the *sign*. |
| Only 20d works | V0/V2/V3 excess vs EW all ≈ −20%. Consumed 20d lookback does not secretly pass. |
| Independence of names | Inference is on 70 monthly portfolio returns, not 80k stock-months. |
| FEATURE-002 leakage | Study does not import observe/BUY/Telegram. |
| Mid-vol looks better | D4–D5 have the highest means. That is a **new** hypothesis (volatility smile / barbell), not a rescue of H1. |

## Material defects that would change the verdict

None found. Re-running the same frozen spec after a code fix is not required.

## Verdict on the verdict

**REJECT stands.** Low-vol Top 20 is a defensive sleeve (7% vol, −11% max DD), not a net excess edge on NSE 2020–2026.
