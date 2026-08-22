# EDGE-001 — Portfolio Results (primary = M1 Top20 monthly)

Protocol SHA `4a9ac87fc31bbd59`. Fill = next open. No stop. Gross and net both shown. Top 20 was locked before looking at later blocks.

| Metric | M1 Top20 Monthly |
|---|---|
| Years | 5.83 |
| Rebalances | 70 |
| CAGR gross | 31.98% |
| CAGR net | 30.04% |
| Benchmark CAGR (Nifty) | 21.29% |
| Benchmark CAGR (EW universe) | 25.62% |
| Excess CAGR vs Nifty | 8.75% |
| Excess CAGR vs EW | 4.42% |
| Volatility | 27.48% |
| Sharpe | 1.102 |
| Sortino | 1.794 |
| Max drawdown | -37.24% |
| Calmar | 0.807 |
| Win months | 62.86% |
| Turnover/year (one-way) | 472.29% |
| Cost drag/year | 1.51% |
| Worst month | -18.23% |
| Best month | 17.55% |
| Avg names filled | 18.5 |
| Beta vs Nifty | 1.107 |
| Monthly hit vs Nifty | 55.71% |
| Monthly hit vs EW | 48.57% |

## By-year net / gross

| Year | Net | Gross |
|---|---|---|
| 2020 | 24.39% | 25.06% |
| 2021 | 117.94% | 121.59% |
| 2022 | -17.30% | -16.11% |
| 2023 | 97.74% | 100.30% |
| 2024 | 3.43% | 5.10% |
| 2025 | -12.65% | -11.52% |
| 2026 | 15.57% | 16.54% |

## Size / cadence / horizon sensitivities (net CAGR, not for winner-picking)

| Spec | n | CAGR net | Sharpe | Max DD | TO/year |
|---|---|---|---|---|---|
| sens_H3_top20_monthly | 70 | 34.78% | 1.196 | -33.44% | 477.43% |
| sens_M1_top10_monthly | 70 | 32.03% | 1.079 | -35.70% | 480.00% |
| sens_M1_top20_2month | 35 | 53.99% | 1.268 | -33.60% | 663.43% |
| sens_M1_top20_4week | 76 | 24.38% | 0.873 | -42.23% | 450.00% |
| sens_M1_top20_quarterly | 23 | 57.28% | 1.132 | -34.91% | 774.78% |
| sens_M1_top30_monthly | 70 | 30.67% | 1.132 | -42.78% | 456.00% |
| sens_M1_top50_monthly | 70 | 36.70% | 1.329 | -33.48% | 446.06% |
| sens_M2_top20_monthly | 70 | 34.74% | 1.227 | -28.23% | 629.14% |
| sens_M3_top20_monthly | 70 | 29.14% | 1.047 | -31.49% | 544.29% |
| sens_M4_top20_monthly | 70 | 41.50% | 1.324 | -31.69% | 541.71% |

Formula excess CAGR vs EW (net): M1=4.42%, M2=9.12%, M3=3.52%, M4=15.88%.

Sector weights use the contemporaneous NIFTY 500 comment map. Most Top-20 names are **UNKNOWN** (small/mid caps outside that map). Reported average max-sector weight 87.71% is therefore mostly the unmapped bucket, not a single industry bet. Do not add sector caps in this milestone.
