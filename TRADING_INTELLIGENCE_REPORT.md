# QuantTerm trading intelligence — evidence inventory
Date: 2026-11-07 | Status: PARTIAL / NO PERFORMANCE CERTIFICATION.

## Implemented surface versus measured evidence
The repository contains backtest/backtester.py, backtest/walk_forward.py, product/fno_ranking.py, product/fno_evidence_fusion.py, product/evolution/promotion.py, product/paper_learning_loop.py and execution/us_autopilot.py. Their existence is evidence of implementation, not proof of predictive ability or profitability. Runtime outcome datasets were unavailable to this audit.

The README describes a strategy-to-outcome loop with expected-value ranking, Wilson-shrunk forward evidence, risk gates and daily learning. Those are architecture claims only until a reconciled, timestamped ledger and point-in-time runs verify their inputs and settled outcomes.

## Scientific gates to audit
- Reconstruct every prediction with feature availability timestamps and the exact scan snapshot; prohibit leakage from future candles, post-trade outcome labels and future universe membership.
- Test strategy families separately: momentum/trend, relative strength, breakout/volume, regime and NSE F&O underlying + contract selection.
- Partition by decision time and market regime. Freeze parameters before each held-out period. Account for repeated searches and candidate promotions.
- Include brokerage, fees, spread, slippage, fill timing, liquidity, gaps, expiry, partial fills and capital concurrency in net performance.
- Compare against simple eligible-universe momentum, buy-and-hold benchmark where suitable, and an abstain/no-trade baseline.
- Publish sample counts, effective exposure, drawdown, expectancy and bootstrap intervals; report periods that underperform.
- Distinguish score ranks from calibrated P(win), with reliability curves / bucket support and Brier score where probabilistic outputs exist.
- Keep historical counterfactuals, shadow decisions and settled forward paper outcomes in separately identified cohorts.

## Evolution governance
product/evolution/promotion.py has AUTO_PROMOTION_ENABLED=False in the reviewed source and describes paired forward evidence, multiple-testing correction, drawdown and regime breadth. Whether the actual champion/challenger path enforces these conditions against real settled outcomes remains unverified. Do not unlock auto-promotion because a dashboard or synthetic test says eligible.

## NSE F&O and US lanes
F&O ranking and option-contract evidence are distinct modules; show exact instrument ID, expiry, strike, underlying data cutoff, Greeks source/time, fill model and risk calculations for every proposed paper trade. If genuine option-chain inputs are absent, skip contract selection instead of guessing.
US autopilot is explicitly paper-only and has its own journal and supervision logic; no proof of recent US position exits or settlement was available from code review alone.

## Required proof package
Immutable market-input manifest; decision ledger with as-of timestamps and rejected candidates; outcome ledger with realized fees and slippage; strategy/model version; champion/challenger promotion proofs; walk-forward fold definitions and results; calibration diagnostics; and reproducible commands with pinned Git SHA.

## Verdict
No independently reproduced net alpha, confidence calibration or continually improving behavior has been established in this pass. More simulation counts are not proof of learning.
