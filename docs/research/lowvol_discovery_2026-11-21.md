# PR-2026-11-21-LOWVOL: discovery result (21 Nov 2026)

**Research only.** This result creates no BUY and changes no gate or risk limit.

The spec was frozen and committed in `660ccc89` (SHA-256 `e1e21ebb…`) BEFORE any return of this score was computed on this data. This is the single discovery analysis. The held-out year (entries November 2025 to October 2026) is evaluated next, ONCE, with no rule changed, as the spec requires after a discovery PASS.

## What was tested

- **Selection:** every month, take the 20% of liquid, point-in-time-eligible NSE equities with the LOWEST trailing 252-session volatility of daily log returns (bars up to the previous session only).
- **Portfolio:** equal weight, held to the next month's first session.
- **Costs:** 0.16% per side on turnover.
- **Benchmark:** equal weight over every eligible name, gross of costs.
- **Primary outcome:** monthly alpha (the intercept of net portfolio returns on benchmark returns), Newey-West lag 1, one-sided, m = 1, outside the walk-forward BUY family.

Reproduce: `nice -n 19 venv/bin/python -m research.lowvol_portfolio --root <SSD copy root> --accept-unbound-coverage --out <json>`

## Result (discovery: 37 monthly periods, entries October 2022 to October 2025)

**Primary.** Alpha is **+0.43% per month** (NW SE 0.26%, t = 1.70, one-sided p = **0.049**, 95% CI [−0.07%, +0.94%]).

**Verdict per the frozen rule: PASS (p < 0.05).**

| Secondary | Value |
|---|---|
| Beta to the equal-weight market | **0.59** |
| Mean monthly net return | +1.58% |
| Mean monthly benchmark | +1.94% |
| Mean monthly net excess | −0.36% (lags a strong market, as the mechanism predicts) |
| Net Sharpe (annualised) | **1.65**, against 1.35 for the benchmark |
| Maximum drawdown (net) | −15.5% (12-1 momentum: −28.7%) |
| Worst month | −7.5% |
| Mean one-way turnover per rebalance | 15% |
| Names held | about 143 |
| Held names with no bar at their exit | 12 |

**Stopped-name sensitivity.**

- With a further 50% loss on the remainder, alpha is +0.51% (p = 0.028).
- With a 100% loss, alpha is +0.59% (p = 0.015).
- Low-volatility names stop trading less often than the benchmark's, so the harsher assumption helps the portfolio.

## Reading (honest limits)

- **This is a weak pass.** The one-sided p of 0.049 sits at the threshold, and the two-sided interval includes zero.
- **Two portfolio hypotheses were pre-registered this week.** 12-1 momentum was NOT_SIGNIFICANT, and this one passed. Each was declared at m = 1, but across the two a Bonferroni α of 0.025 would not be met in discovery. The held-out year is the real test. That year has not been looked at for this score.
- **The mechanism's prediction matched.** Lower beta, lower raw return in a rising market, and higher risk-adjusted return with a shallower drawdown. This is the first result this week that is not market exposure in disguise. The 12-1 momentum outperformance was beta 1.18.
- **Not a BUY and not a trading system.** At best it is a candidate for an owner-approved paper experiment. Any proposal waits for the held-out look.
