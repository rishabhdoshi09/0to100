# Strategic intelligence reset

Development baseline `e5a46878a753123fa2db8a7d786f78adc96df2be`. **NO VALIDATED EDGE.** The scanner remains a way to generate research candidates. Prior gross loss and near-zero rank correlation give no reason to let its score determine portfolio allocation.

## Research registry

| Hypothesis | Status | Evidence and stopping decision |
|---|---|---|
| Scanner patterns and score | REJECTED as a selection authority | Gross −0.322% per modeled BUY fill and score Spearman −0.005. No demonstrated paired selection advantage. |
| Existing eight short-horizon execution geometries | REJECTED | All eight had negative net expectancy. No further execution-grid search. |
| Monthly 12–1 equity momentum | INCONCLUSIVE, FROZEN | Initial test not significant and underpowered. Its reserved test is untouched by this cycle. |
| Monthly low-volatility equity | REJECTED | Discovery alpha did not replicate in the existing held-out result. |
| Relative strength | SHADOW ONLY, FROZEN | Extended replay and cost stress failed. Preserve the existing prospective protocol without activating it. |
| Two scanner combinations | REJECTED for promotion | Both lost in the extended replay. |
| Delivery confirmation and downside resilience | REJECTED for promotion | Both negative after friction and behind matched random. |
| Entry-fill forecast | PROMISING DIAGNOSTIC | Prior frozen Brier 0.437 vs pooled 0.498. No confirmed payoff or allocation uplift. |
| Six-month absolute trend exposure | POWER-BLOCKED | Defined below. No strategy outcomes inspected for this rule. |
| Risk-adjusted cross-sectional and sector rotation | UNREGISTERED | Require point-in-time sector and membership history plus independent regimes before granting a research look. |
| Conditional mean reversion | UNREGISTERED | A concrete economic mechanism and leverage-free cash-equity comparison are required. Prior short-horizon scanner variant failed. |

## Next hypothesis and stopping gate

Consider one medium-horizon exposure rule, economically distinct from ranking prior 12–1 winners. The decision is whether to hold a liquid equity basket or cash during persistent market trends. At each month-end an eligible, point-in-time equal-weight NSE operating-company basket supplies a six-month trailing market return. A positive return permits equity exposure at the next official open. Otherwise hold cash at zero modeled yield. Use the existing monthly portfolio simulator and liquidity checks, 16 bp per side on actual turnover, and report sensitivity to 26 bp per side. Cash, passive exposure to the same eligible basket, and a static cash/equity benchmark with its risk scale set on development data are competing baselines. The primary statistic is monthly after-cost alpha against the risk-matched benchmark with one-sided 5% error, dependence-aware inference, and an economically material minimum of 0.35 percentage points. Require positive after-cost payoff and lower drawdown than passive, plus an untouched chronological evaluation. Budget one fixed hypothesis and one look. No parameter search.

**Preflight stopped the experiment.** From the prior frozen low-volatility discovery artifact, select rows with entry on or after January 1, 2023 and exit before November 1, 2025. Regress net return on the passive benchmark with an intercept. The 33 paired monthly residuals have a standard deviation of 1.541%. The optimistic normal approximation `(z_0.95 + z_0.80) × residual_sd / sqrt(n)` gives 0.667 percentage points of detectable alpha at 80% power. An explicit sensitivity with variance inflation 1.5 gives about 22 effective months and 0.817 points. About 120 independent months are needed for the 0.35-point threshold at that proxy variance. The volatility estimate comes from a different strategy and serves as a planning proxy. Even assigning all 12 additional 2022 months as independent would fall short. No new alpha result, promotion or access to reserved observations is justified by this preflight.

The user-facing probability label remains UNMEASURED where evidence is insufficient. The prior entry model's improvement cannot establish economic fill value. A later allocation test needs a fixed decision prior, realized fill cost and opportunity cost, and a truly independent benchmark.

## 2022 archive coverage

The isolated 2022 audit reused the existing 383-name scoped cohort and corporate-action archive. The official 13 weekday closures were January 26, March 1 and 18, April 14 and 15, May 3, August 9, 15 and 31, October 5, 24 and 26, and November 8. October 24 is a documented live special session and is counted in the expected audit dates. [NSE CM 2022 circular](https://archives.nseindia.com/content/circulars/CMTR50560.pdf)

247 of 248 expected files were present and validated, including the special session. August 8 remains missing. The scoped validator admitted 82,551 bars across 346 stocks and quarantined 543 BE rows, 247 other-series rows, 19 action dates, two unresolved moves and 11,239 absent symbol-sessions. The separately audited 2022–October 2025 total is 327,398 admitted bars. Contiguous coverage and the modern selected cohort still limit conclusions about delistings and renames. This is a bar audit, with no new strategy outcomes. Reproduce with `gauntlet.validator.scoped_equity_frames`, the fixed cohort and action ledger in the prior report, and the dated daily archive bytes selected by the calendar above. Audit SHA-256 `1909917475af19a93c7f1c53e16443bb393ddb15c929542bf0a6ec2b514b745b`.

## Engineering closure

EOD now persists outstanding audit projections in the same paper-book snapshot as their already-settled rows, without a new data store. It uses the original audit session and deterministic event identities. Recovery replays a partial append after reload without repeating a fill, P&L or journal event. Book-save failure leaves a retryable outstanding projection. Pending audit state must be drained or preserved before a rollback to software that predates this field.

Special-session dispatch uses the verified NSE manifest and validates the exact official archive. It starts no regular-clock paper entries for a verified special date. After 20:05 IST the current date may queue retrieval or settlement, and the following date can retry. The worker pins the special date and refuses fallback to a different market session. A validated recovered archive requeues its exact blocked outcome job. A longer missing archival outage still requires the ordinary fail-closed recovery process. The 2026 Budget and Muhurat dates both closed before the conservative polling time. [NSE Budget circular](https://nsearchives.nseindia.com/content/circulars/CMTR72349.pdf) · [NSE Muhurat circular](https://nsearchives.nseindia.com/content/circulars/CMTR76941.pdf)

Independent review confirmed incremental rows and report identity remain scope-bound. Durable entry recovery still requires original matching intent and book evidence. Existing GOOD_WAIT markers replay idempotently, but same-session ordering cannot be reconstructed from date-only fills. Automatic entry-evidence replay remains bounded to 45 days and intentionally cannot restore absent source context. This cycle does not claim recovery of older lost taken rows.

## Readiness

The existing historical paper loop freezes production-style decisions and uses PaperBook risk and settlement. Its historical policy writer sets `affects_selection=False`. Genuine forward selection never learns as though historical results were actual paper fills. A full matched adaptive/frozen replay with an explicit historical-only policy context, matured observations and both branches actually changing their simulated decisions remains UNVERIFIED. The prior adaptive forecast experiment does not satisfy it.

Production remains deferred pending backup, ledger and position reconciliation and separate approvals. The working volume and disabled services were untouched. Live trading is HARD-LOCKED.
