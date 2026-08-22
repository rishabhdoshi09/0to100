# EDGE-001 — Decile Monotonicity

Each month the investable universe is ranked on **M1 12-1** and split into deciles. Forward return is next-session open → next rebalance’s next open. Inference is clustered by rebalance (the monthly mean is the observation).

Spearman(decile, mean next-month return) = **0.915**. D10 − D1 = **1.48%**. D10-only flag = `False`.

| Momentum Decile | n observations | Mean next-month return | Median | Excess vs universe | CI |
|---|---|---|---|---|---|
| D1 (weakest) | 8496 | 1.24% | -0.68% | -0.85% | [-0.53%, 3.37%] |
| D2 | 8563 | 1.69% | -0.29% | -0.40% | [0.06%, 3.32%] |
| D3 | 8551 | 1.82% | 0.45% | -0.27% | [0.33%, 3.30%] |
| D4 | 8566 | 2.23% | 0.33% | 0.14% | [0.91%, 3.76%] |
| D5 | 8585 | 1.86% | 0.27% | -0.23% | [0.46%, 3.21%] |
| D6 | 8545 | 2.32% | 0.65% | 0.23% | [0.97%, 3.61%] |
| D7 | 8553 | 2.05% | 0.32% | -0.04% | [0.67%, 3.46%] |
| D8 | 8567 | 2.31% | 0.51% | 0.22% | [0.80%, 3.88%] |
| D9 | 8521 | 2.70% | 1.18% | 0.61% | [1.00%, 4.26%] |
| D10 (strongest) | 8362 | 2.72% | 0.09% | 0.63% | [0.98%, 4.41%] |

Primary evidence question: does return generally improve as momentum rank improves? Pooled means: mostly yes (D1 weakest, D9/D10 strongest). D10 **median** is weaker than D9 — the extreme bucket is right-tailed, not steadily better. Year-by-year: 2020 and 2026 invert; 2022 and 2024 are flat. The slope is an average, not a reliable annual law.

Year-by-year decile means (monthly average, not compounded):

| Year | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 7.63% | 7.82% | 6.12% | 6.95% | 5.44% | 4.69% | 4.52% | 3.48% | 3.14% | 3.44% |
| 2021 | 2.62% | 3.00% | 3.41% | 4.20% | 3.76% | 4.40% | 3.94% | 5.19% | 6.29% | 6.91% |
| 2022 | -1.01% | -0.01% | -0.09% | 1.24% | -0.18% | 0.63% | 0.74% | 0.67% | 0.23% | -0.06% |
| 2023 | 3.21% | 3.38% | 3.46% | 3.79% | 3.79% | 4.53% | 4.12% | 4.67% | 5.67% | 5.56% |
| 2024 | -0.52% | 0.45% | 0.84% | 0.66% | 0.71% | 0.88% | 0.94% | 0.58% | 0.85% | 0.62% |
| 2025 | -1.05% | -1.06% | -0.26% | -0.24% | -0.24% | 0.06% | -0.24% | 0.04% | 0.11% | 0.45% |
| 2026 | 2.88% | 3.01% | 2.48% | 2.11% | 2.41% | 2.89% | 1.88% | 2.39% | 3.12% | 2.48% |
