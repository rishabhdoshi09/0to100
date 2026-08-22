# EDGE-001 — Walk-Forward

Blocks were frozen in `EDGE_001_RESEARCH_PROTOCOL.md` before backtests. M1 / Top 20 / monthly were **not** changed after opening validation or confirmation.

| Block | Rebalance dates | Role |
|---|---|---|
| Warm-up | until 252 sessions exist | no claim |
| Development | first valid rebalance → 2022-12-31 | specification lock |
| Validation | 2023-01-01 → 2024-12-31 | robustness |
| Confirmation | 2025-01-01 → 2026-08-21 | confirmation |

| Block | n | CAGR net | Excess vs EW | Excess vs Nifty | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| development | 28 | 41.34% | 6.22% | 12.52% | 1.485 | -21.94% |
| validation | 24 | 43.01% | 11.94% | 19.54% | 1.463 | -15.22% |
| confirmation | 18 | 0.63% | -5.35% | -7.23% | 0.170 | -31.90% |

2019–2026 was already mined by SEPA / FEATURE-001. Confirmation is held-out for **this** protocol only. No period is philosophically pristine.
