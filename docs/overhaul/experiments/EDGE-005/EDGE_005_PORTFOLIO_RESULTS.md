# EDGE-005 — Portfolio Results (P1 252d near-high, Top20 monthly)

| Metric | P1 Top20 Monthly |
|---|---|
| Years | 5.83 |
| Rebalances | 70 |
| CAGR gross | 23.68% |
| CAGR net | 19.45% |
| EW CAGR | 25.62% |
| Nifty proxy CAGR | 21.29% |
| Excess vs EW | -6.17% |
| Excess vs Nifty | -1.84% |
| Vol | 16.97% |
| Sharpe | 1.139 |
| Sortino | 1.851 |
| Max DD | -21.16% |
| Calmar | 0.919 |
| Win months | 64.29% |
| TO/year | 1102.29% |
| Cost drag/year | 3.53% |
| Worst month | -13.74% |
| Best month | 14.30% |
| Avg names | 19.0 |
| Hit vs EW | 48.57% |

## By year

| Year | Net | Gross |
|---|---|---|
| 2020 | 20.92% | 22.30% |
| 2021 | 40.47% | 45.63% |
| 2022 | -2.28% | 1.30% |
| 2023 | 57.25% | 62.89% |
| 2024 | -9.90% | -6.45% |
| 2025 | 7.76% | 11.14% |
| 2026 | 11.30% | 13.06% |

## Sensitivities (not for winner-picking)

| Spec | n | CAGR net | Sharpe | Max DD |
|---|---|---|---|---|
| diag_LAG_top20_monthly | 70 | 6.03% | 0.342 | -42.80% |
| sens_P1_top10_monthly | 70 | 19.84% | 0.963 | -27.72% |
| sens_P1_top20_2month | 35 | 52.59% | 1.677 | -24.07% |
| sens_P1_top20_4week | 76 | 18.94% | 1.010 | -17.90% |
| sens_P1_top20_quarterly | 23 | 41.82% | 1.042 | -25.36% |
| sens_P1_top30_monthly | 70 | 20.70% | 1.183 | -17.04% |
| sens_P2_top20_monthly | 70 | 16.84% | 0.888 | -22.17% |
| sens_P3_top20_monthly | 70 | 15.89% | 0.922 | -28.54% |

2-month/quarterly CAGRs use the monthly annualizer and are **not** comparable.
Formula excess vs EW: P1=-6.17%, P2=-8.78%, P3=-9.73%, LAG=-19.59%.
