# EDGE-001 — Results

**Classification: `RESEARCH-ONLY`**

No classification authorises paper, live, FEATURE-002, or production BUY changes.

## Primary (M1 Top20 monthly, net)

- CAGR net 30.04% vs Nifty 21.29% (excess 8.75%) and vs EW 25.62% (excess 4.42%)
- Sharpe 1.102, Sortino 1.794, max DD -37.24%, Calmar 0.807
- Worst month -18.23% (2025-01-31), worst quarter -26.67% from 2025-12-31
- Decile Spearman 0.915, D10−D1 1.48%

## Inference on monthly net excess vs EW

{'mean': 0.004315831979079876, 'ci': {'mean_r': 0.004315831979079876, 'ci_lower': -0.005571552283679897, 'ci_upper': 0.014298470131257195, 'excludes_zero': False, 'block': 4.0, 'n_eff': 50.68494080294807, 'n_boot': 2000}, 'sharpe': 0.3745412438670421, 'psr': 0.9996129050435473, 'dsr': {'dsr': 0.9996129050435473, 'sr0_expected_max_null': 0.0, 'n_trials': 64, 'sharpe_variance': 0.0}, 'p_beat_ew': 0.4857142857142857}

Harness: `INCONCLUSIVE` — Positive (+0.00R over 70 trades) but only 82% confident (need 95%), and the sample was big enough to show a real edge. Likely noise.

## Production MOMENTUM comparison

Production `MOMENTUM` is 5-day time-series + RSI + volume on scanner cards. EDGE-001 is 12-1 cross-sectional, monthly, no stop. The comparison **reuses EDGE-001’s next-open monthly hold** and ranks the PIT universe on 5-session return so expectancy is not mixed with the scanner’s 10–20 day ticket.

- CS M1 net CAGR 30.04% vs 5d-TS Top20 net CAGR 6.93%
- Excess vs EW: M1 4.42% vs 5d-TS -18.69%
- Turnover/year: M1 472.29% vs 5d-TS 1170.00%

They are **not the same phenomenon**. A 5-day TS sort is closer to short-horizon reversal/continuation than to 12-1 relative strength.

Failure flags: `['confirmation_reverses_development']`
