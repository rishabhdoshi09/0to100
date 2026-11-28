# Outcome-first research, first bounded cycle

**NO VALIDATED EDGE.** Development branch `codex/quantterm-alpha-discovery` starts at the preserved paper release `537e22902b32ac61a91f38fd7acf7cba57569cb4`. The research job is opt-in and isolated. No production state or trading authority is changed.

## Knowledge carried forward

The scanner scored −0.322% gross per previously modeled BUY fill with Spearman −0.005, and all eight execution variants lost money net. Extended relative strength and cost stress failed. Delivery confirmation and downside resilience lost after costs. Monthly low volatility failed to replicate. Monthly 12–1 momentum remains frozen and inconclusive with its reserve intact. The six-month absolute-trend idea was power-blocked and wasn't rerun. Entry-fill forecasts are promising diagnostics, and the adaptive-versus-frozen replay produced too few fills to establish learning efficacy. [Prior research registry](strategic_intelligence_reset_2026-11-27.md) · [Realistic adaptive replay](../release/2026-11-28-paper-operation-consolidation.md)

## Registered discovery

Hypothesis `14d1234ac19c7bad` tests whether temporary selling pressure reverses in liquid NSE cash stocks. It ranks the entire available eligible cross-section by trailing 20-session return at the decision close. It selects the lowest quintile, requires ₹100m trailing mean daily turnover and 50 names, and measures next official open to the tenth official close. The cost model deducts 32 bp round trip and 52 bp under stress. Size, price and session rules are proxies from the existing scoped archive. These observations do not establish executable portfolio fills.

The identical eligible basket and idle cash are the comparators. One look, no parameter sweep. The preregistered discovery screen requires an overlap-aware one-sided 95% lower excess bound of at least 0.35 percentage points and positive payoff under stress. Regime splits and five fixed quintiles are descriptors, with no significance or strategy claims attached. Dates are discovery only, strictly before November 2025. Any prospective validation requires separately frozen rules and authority.

| Result | Measure |
| --- | ---: |
| Existing observations read | 18,283 |
| Usable matched observations | 4,171 |
| Usable observations with no scanner signals | 2,916 |
| Complete dates / crossed liquidity filter | 13 / 53 |
| Portfolio-independent mean net payoff | −1.469% |
| Matched eligible basket | −1.276% |
| Net payoff under stress | −1.669% |
| Matched excess | −0.193 pp |
| Dependence-aware 95% interval | [−1.423, +1.038] pp |
| Nonoverlapping blocks | 10 |
| Approximate effect detectable at 80% power | 1.35 pp |
| Mean within-date momentum / forward return Spearman | +0.034 |

The test is **INCONCLUSIVE**. The point estimate is unfavorable, and the protocol permits no repeat on these observations. The no-signal rows demonstrate that the test did not depend on scanner picks. Forty dates had at least one missing outcome after liquidity eligibility. Ten observations had no next open, and 71 had an opening observation but lacked the full ten-session window. The latter can reflect verified session, name continuity or action exclusions and require the Gauntlet admission metadata to attribute. The complete-case subset may differ systematically from excluded dates. The 13 matched dates span 2024-10-28 to 2025-09-02. The descriptive negative and nonnegative formation regimes had 10 and three dates respectively. Neither supplies a credible new regime strategy. The current scoped cohort also omits some historical members and delistings. Sector-relative forecasts, other horizons, executable payoff, drawdown and probability calibration remain unmeasured.

## Autonomous operation and next dependency

The existing closed-market scheduler can enqueue the registered look under its own idempotency key when an isolated checkpoint is configured and the historical resource governor allows it. It uses the existing research job retry policy and experiment registry. The 64 MiB source and 30,000-row limits bound each attempt. A failed attempt before commit reruns the same bounded observation read. A completed look returns its immutable result without running the experiment again. The existing Research Director surfaces the persisted result and next step.

This cycle reused the read-only Gauntlet checkpoint SHA-256 `d10b5bb5ce0d22a9e173bf8d2ecbfef5d42702a224fa8eb6282704c8e56d3882`. Analysis takes about one second after startup. Reproduce in a dedicated scratch root under `/tmp`, with a sealed copy named `gauntlet.sqlite` and `QT_RUNTIME_ROOT` set before importing QuantTerm. Run `python -m research.autonomy.outcome_discovery <scratch>/gauntlet.sqlite`, or set `QT_OUTCOME_DISCOVERY_CHECKPOINT` in an isolated supervisor process. The study output includes source and implementation hashes.

The next recorded step is to audit the missing paired outcomes and distinguish archive gaps, action quarantines and horizon truncation. More complete point-in-time observations, particularly outside the modern cohort, are needed to justify another look. The branch implements the first deterministic protocol and stop decisions. General hypothesis generation, automatic follow-on experiments, independent shadow validation and real paper trials are pending.
