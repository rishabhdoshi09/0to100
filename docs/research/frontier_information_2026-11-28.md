# Information frontier · first two-look programme

**NO VALIDATED EDGE.** This isolated programme starts from the completed reversal and three follow-ons on `31c0002a0f42b0b94c4f4c27da2be5a4e47c5e9a`. It preserves every earlier result and spends two new information-stage looks. This is discovery on previously used calendar history with broader observation and feature coverage. Reserved dates begin November 1, 2025.

## Data recovered

The incremental Gauntlet extension verified every daily payload and the NSE corporate-action ledger against the original scoped checkpoint. For split, bonus and consolidation events, the dated authoritative factor must reconcile with consecutive EQ bars. The archive sometimes carries the old share-basis previous close into the ex-date file. In that case, the ex-date open must be closer to the adjusted reference than the raw reference, and all OHLC values must be within 25% of the adjusted basis. An ambiguous event remains quarantined. This admits 30 previously excluded share-action sessions, increasing admitted bars from 159,692 to 159,722, and adds 573 eligible feature rows.

| Measure | Strict action quarantine | Corroborated share actions |
| --- | ---: | ---: |
| Stock-date feature rows | 125,122 | 125,695 |
| Verified three-session payoff rows | 122,035 | 122,681 |
| Verified twenty-session payoff rows | 108,518 | 109,520 |

The new feature surface covers 358 securities and 379 formation dates. Fixed 20-session past paths use only realized share changes through formation. An open-to-close holding payoff includes only share multipliers earned after entry. Missing official sessions, absent rows, BE, other series and unresolved discontinuities still break the path. The checkpoint commits one symbol at a time and skips completed symbols on restart. A completed cache avoids another archive scan.

These are split/bonus/consolidation price returns. Cash dividends, demergers, delisting proceeds, renamed-security identity, and historical sector and index membership remain unresolved. The cohort is still subject to selection bias. The delivery field is taken from the dated official file; publication by the next actionable open was not independently verified. A future execution study would have to prove availability or use a predefined lag. No present-day classification is used as historical membership.

## Director choices and results

After the absolute-turnover hypothesis failed, the director prioritized whether a five-session rise in delivery percentage relative to the preceding 15 identifies cash participation with predictive information over three sessions. After weak continuation and negligible next-open gaps, it tested whether the difference between overnight and regular-session formation returns predicts a separate twenty-session outcome. It used the whole archive-eligible cross-section with no scanner gate. Neither related family had an earlier look with these fields and horizons. Each was registered once; the one-sided acceptance bound adjusts for five related follow-on looks, including the three previous trials.

Both tests formed ranks before checking future data and required 98% coverage in selected and basket groups. The relative primary statistic is same-date top-quintile excess; the adverse missing-return scenario is a sensitivity. Net figures apply 32 bp and 52 bp round-trip modeling as diagnostics. Partial rank association controlling for past total momentum is descriptive. The year and median-market-direction splits are also descriptive, and no regime or interaction was selected from them.

| Test | Dates / nonoverlapping windows | Basket-relative excess | Descriptive 95% interval | Net / stressed diagnostic return | Partial rank | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Delivery participation, 3 sessions | 369 / 124 | +0.074 pp | [−0.001, +0.149] pp | −0.295% / −0.495% | +0.021 | INCONCLUSIVE |
| Overnight versus intraday formation, 20 sessions | 304 / 17 | +0.020 pp | [−0.994, +1.033] pp | +0.102% / −0.098% | −0.031 | INCONCLUSIVE |

The delivery point estimate is economically small and the adjusted one-sided lower limit including censoring stress is negative. The twenty-session test did not meet the independent-window floor. Neither advances to Stage B. Portfolio performance, calibrated forecasts, liquidity capacity and independent efficacy remain unmeasured.

## Reproduction and stopping rule

Copy the checked original Gauntlet database (SHA-256 `d10b5bb5ce0d22a9e173bf8d2ecbfef5d42702a224fa8eb6282704c8e56d3882`) and completed experiment registry into a dedicated scratch root under `/tmp`. Set `QT_RUNTIME_ROOT`, `QT_OUTCOME_DISCOVERY_CHECKPOINT` and `QT_FRONTIER_ARCHIVE` before running the isolated scheduler. The archive directory is the verified isolated copy containing `bhav` and `ca_events.json`. The scheduler chooses the next frontier key after the previous programme finishes. The two-look budget does not renew. The programme persists each code hash, source digest and trial result. Preliminary fixture and corroboration checks reused the identical test specifications; the reported return estimates are from the final corroborated archive generation, with no selection based on a pilot estimate.

The next data priority is dated identity and corporate-action coverage plus earlier independent market periods. Sector and benchmark-relative strategies need dated membership rather than a static map. Independent validation and prospective activation remain separately authorized. No production change occurred.
