# NSE opportunity discovery on inspected history

Prior work found −0.17% ten-session unconditional scanner excess against the equal-weight market with a 95% interval of [−0.49%, +0.15%]. The earlier 12–1 momentum test was inconclusive, and low-volatility alpha failed its later test. This study left those specifications and their reserved datasets alone.

The study reads only January 1 through October 31, 2025 from the existing validated NSE archive. The manually named cohort expanded from 130 to 232 stocks. All 232 had bars. The study admitted 48,244 stock-session bars and quarantined 10 documented action sessions and two unresolved moves. Static sector labels cover 119 names. Modern cohort construction and incomplete sector labels limit inference.

Each policy gets five equal portfolio slots on the same non-overlapping 10-session decision grid. Stocks qualify using 60 consecutive accepted bars, a ₹20 price floor and ₹1 crore trailing mean daily turnover. Every fill uses next-session open, a 3% initial stop, 5% target, the same breakeven rule, and 0.32% round-trip costs. These constants were held fixed to compare selection. Idle slots hold cash. For a selected holding that cannot be resolved, the portfolio uses zero recovery as a stress convention, and per-resolved expectancy excludes that holding.

The market control applies the identical fill and exit rules to the eligible basket, scaled daily to match the strategy’s chosen allocation. The random control draws from the same decision-date names with a fixed seed and the same allocation. Market exposure is matched by timing and intended invested fraction. Realized beta can still differ. Figures are exploratory across 14 cycles. Exit forecasts fit the first seven cycles and are scored on the later seven for calibration diagnostics.

| Selection | Opportunities | Resolved fills | Net R/fill | Portfolio return | Drawdown | Excess vs market/cycle | Excess vs random/cycle | Later Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scanner | 1,894 | 70 | -0.144 | -6.10% | 9.67% | -0.02 pp | -0.24 pp | 0.628 |
| liquidity | 3,192 | 69 | -0.072 | -22.94% | 26.28% | -1.23 pp | -1.45 pp | 0.542 |
| relative_strength | 1,374 | 69 | +0.035 | +1.27% | 5.22% | +0.51 pp | +0.30 pp | 0.652 |
| trend_pullback | 467 | 70 | -0.210 | -8.65% | 12.77% | -0.22 pp | -0.44 pp | 0.534 |
| mean_reversion | 123 | 52 | -0.313 | -9.60% | 13.31% | -0.41 pp | -0.54 pp | 0.372 |
| volume_breakout | 23 | 23 | -0.481 | -6.48% | 8.88% | -0.40 pp | -0.39 pp | 0.343 |
| sector_leadership | 415 | 69 | -0.162 | -25.33% | 28.25% | -1.50 pp | -1.71 pp | 0.476 |
| cross_sectional_momentum | 3,192 | 70 | -0.109 | -4.76% | 10.87% | +0.08 pp | -0.14 pp | 0.425 |
| scanner_risk_on | 1,613 | 55 | -0.270 | -8.70% | 11.21% | +0.09 pp | +0.11 pp | 0.490 |
| scanner_confluence | 1,079 | 70 | -0.144 | -6.10% | 9.67% | -0.02 pp | -0.24 pp | 0.628 |
| random | 3,192 | 70 | -0.064 | -3.06% | 10.71% | +0.22 pp | +0.00 pp | 0.528 |

The scanner and its two-signal confluence variant picked the same top five names on every decision date. The scanner’s paired market difference was −0.023 percentage points per cycle with a one-sided 95% lower bound of −1.557 points. Relative strength was the only positive portfolio and resolved-fill result. Its paired market lower bound was −6.729 points. The sample does not establish a reliable edge.

All 15 observed individual scanner signals had negative net returns in attribution. Two exact combinations had positive returns among groups with at least 20 resolved fills. Ascending triangle with pocket pivot had +0.30% across 31 and ascending triangle with pre-breakout had +0.20% across 34. Those overlapping descriptive groups were examined together and have no standalone portfolio confirmation.

Only the predeclared relative-strength rule advanced to a frozen prospective hypothesis. The other rows, including the attractive-looking combinations, remain exploratory. The frozen spec has no trading authority.

## Reproduction and review

Run `python -m research.opportunity_discovery --root <isolated research runtime> --stocks <comma-separated stock list> --out <new JSON file>`. The frozen relative-strength specification records the exact cohort and the source and result digests. The research command hard-caps the archive before the reserved period. Cash sleeves and market and random controls are assigned before outcomes and charged the same modeled costs.

The scoped Gauntlet review checked archive-byte identity, checkpoint binding and atomic per-symbol writes, contiguous feature history and exit prefixes, and mutually exclusive target, stop, and expiry counts. The paper-lane review traced source evidence through the shared empirical caution, historical integrity gate, final portfolio authority, account mutation lock and daily risk quota. The protected paper-fill test verifies that the position is tagged as exploration and the source remains WATCH. Missing measured pairwise correlations use the existing sector proxy. A separate reviewer sign-off and actual prospective paper evidence are outstanding.
