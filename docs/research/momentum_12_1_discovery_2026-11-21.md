# PR-2026-11-21-MOM12-1: discovery result (21 Nov 2026)

**Research only.** This result creates no BUY and changes no gate or risk limit.

The spec was frozen and committed in `ab4b1690` (SHA-256 `86a959d2…`) before any return of this score was computed on this data. **The held-out year (entries Nov 2025 to Oct 2026) was not evaluated.** It is reserved for one look, with no rule changed.

## What was tested

A diversified, medium-term alternative to the desk's short-horizon breakout selection.

- **Selection:** every month, rank liquid, point-in-time-eligible NSE equities by 12-1 momentum (`close[c-21] / close[c-252] - 1`, using bars up to the previous session only).
- **Portfolio:** hold the top 20% equal-weighted until the next month's first session.
- **Costs:** 0.16% per side on turnover.
- **Benchmark:** the equal-weight portfolio of every eligible name, gross of costs.
- **Universe:** membership from the official bhavcopy history, delisted names included.
- **Filters:** traded-price floor of ₹20 on the price that actually traded, and median traded value of at least ₹5 crore.
- **Coverage:** every formation window and holding period lies inside the corporate-action coverage (from 2021-08-28).

## Reproduce

```
nice -n 19 venv/bin/python -m research.momentum_portfolio --root <SSD copy root> --accept-unbound-coverage \
    --out docs/research/momentum_12_1_discovery_2026-11-21.json
```

`--accept-unbound-coverage` is needed because the copy's coverage file predates the QT-12 digest binding. The report records `ca_coverage_binding: UNBOUND`.

## Result (discovery: 37 monthly periods, entries Oct 2022 to Oct 2025)

**Primary.** Mean monthly NET excess over the equal-weight universe is **+0.31%** (HAC SE 0.37%, t = 0.85, one-sided p = 0.20, 95% CI [−0.41%, +1.03%]).

**Verdict: NOT_SIGNIFICANT.** The spec's power analysis says this was expected: about 223 months are needed to detect 0.5%/month.

| Secondary | Value |
|---|---|
| Mean monthly net return | +2.25% |
| Benchmark mean | +1.94% (a strong market) |
| Alpha vs. benchmark (regression) | **−0.04% / month** |
| Beta vs. equal-weight market | **1.18** |
| Maximum drawdown (net) | −28.7% |
| Worst month | −14.6% |
| Hit rate (excess > 0) | 57% |
| Mean one-way turnover per rebalance | 31% |
| Names held | about 143 |
| Held names with no bar at their exit | 66 (50 resumed later; 16 never did) |

**Reading.**

- **No selection alpha.** The raw outperformance is market exposure: beta 1.18 against an equal-weight universe that rose about 2% a month. The regression alpha is negative and indistinguishable from zero. On this evidence, 12-1 momentum in liquid NSE names is a higher-beta version of the market, not an edge.
- **The result is inconclusive, not proven null.** The interval is wide (about ±0.7%/month) and the sample is short. A 0.5%/month edge could neither be confirmed nor excluded.
- **Missing bars.**
  - 50 of the 66 "stopped" holdings resumed trading later. Those are data gaps or short suspensions in the store, not losses, so the pre-declared 50% and 100% loss sensitivities (excess +0.01% and −0.29%) overstate the downside.
  - The 16 that never resumed are delistings or symbol renames; the store has no rename map.
  - The frozen primary rule (the return to the last traded close, then cash) is reported as specified, unchanged after seeing the results.
- **Sector concentration could not be measured honestly.** The sector map covers only 337 symbols, so most holdings are UNMAPPED. The 0.73 "largest sector share" mostly counts the UNMAPPED bucket.
- **Momentum-crash exposure is real:** a −14.6% month and a −28.7% drawdown in three years.

## Decision (per the frozen spec)

The discovery result did not PASS, so the held-out year is **not** evaluated and no paper experiment is proposed. The spec stays frozen. A future, separately pre-registered variant would need its own spec. Examples:

- a beta-hedged or market-neutral construction, if shorting becomes possible;
- a volatility-scaled portfolio.

That spec must be written before its data is looked at.
