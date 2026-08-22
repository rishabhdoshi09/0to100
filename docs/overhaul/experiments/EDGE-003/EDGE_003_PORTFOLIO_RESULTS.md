# EDGE-003 — Portfolio Results (T1 all-qualifiers monthly)

| Metric | T1 All Monthly |
|---|---|
| Years | 5.83 |
| Rebalances | 70 |
| CAGR gross | 28.25% |
| CAGR net | 26.89% |
| EW CAGR | 25.62% |
| Nifty proxy CAGR | 21.29% |
| Excess vs EW | 1.27% |
| Excess vs Nifty | 5.60% |
| Vol | 19.49% |
| Sharpe | 1.330 |
| Sortino | 2.018 |
| Max DD | -24.01% |
| Calmar | 1.120 |
| Win months | 64.29% |
| TO/year | 337.81% |
| Cost drag/year | 1.08% |
| Worst month | -12.87% |
| Best month | 13.21% |
| Avg names | 641.7 |
| Mean T1 share | 54.15% |
| Hit vs EW | 62.86% |

## By year

| Year | Net | Gross |
|---|---|---|
| 2020 | 15.68% | 16.44% |
| 2021 | 69.11% | 70.10% |
| 2022 | 5.39% | 6.72% |
| 2023 | 69.57% | 70.94% |
| 2024 | 4.93% | 5.84% |
| 2025 | -2.24% | -0.95% |
| 2026 | 11.87% | 12.68% |

## Sensitivities (not for winner-picking)

| Spec | n | CAGR net | Sharpe | Max DD | Ann. |
|---|---|---|---|---|---|
| sens_T1_all_2month | 35 | 26.23% | 1.884 | -18.97% | calendar_span |
| sens_T1_all_4week | 76 | 25.50% | 1.272 | -22.26% | monthly_12 |
| sens_T1_all_quarterly | 23 | 23.20% | 1.830 | -22.52% | calendar_span |
| sens_T1_top20_dist | 70 | 21.95% | 0.862 | -29.89% | monthly_12 |
| sens_T2_all_monthly | 70 | 26.59% | 1.309 | -23.77% | monthly_12 |
| sens_T3_all_monthly | 70 | 27.44% | 1.340 | -24.06% | monthly_12 |

2-month and quarterly CAGRs use calendar span, not 12/year (a monthly annualizer would inflate them). Formula excess vs EW is only computed for monthly-aligned books.

Formula excess vs EW: T1=1.27%, T2=0.97%, T3=1.82%, T1_TOP20=-3.67%, T1_4W=-0.79%.
