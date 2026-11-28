# Isolated adaptive policy replay

**Research verdict: NO VALIDATED EDGE.** The frozen approach remains the production baseline. This cycle tested whether isolated historical learning can change selection. It registered no strategy and used no reserved data.

| Verification | Result |
| --- | --- |
| Selection and learning mechanics | PASS in synthetic controls. Mature settled net outcomes changed a later decision through the existing policy store and selector. |
| Chronology and restart | PASS in fault injection. A crash after maturity and decision freeze but before the daily checkpoint recovered the same trades and metrics as a clean run. Rerunning the completed comparison returned the same results. |
| Archived-data smoke | PASS through scanning, decision freeze, selection, historical book settlement and the learning sink. |
| Market effectiveness and calibration | UNMEASURED. The smoke produced one settled trade per branch and no changes to selection. |
| Production operations | BLOCKED on independent backup capacity, RPEL reconciliation and interrupted ledger recovery. Drain or preserve pending audit projections before rollback. |

## Archived-data results

A ten-name liquid NSE cohort used 105 warm-up sessions, 12 decision sessions starting April 1, 2025 and 20 further sessions for outcome observation. The scanner evaluated 120 symbol-sessions and returned 24 frozen cards. The comparison consumes the historical replay's BUY candidates and reevaluates them under the actual paper selector, allocation and per-book risk controls. Both arms considered two candidate decisions and took one trade.

| Measure | Frozen | Adaptive |
| --- | ---: | ---: |
| Settled trades | 1 | 1 |
| Mean settled net R | −0.04975R | −0.04975R |
| Mean settled net return | −0.2522% | −0.2522% |
| Portfolio return and excess over idle cash | −0.0241% | −0.0241% |
| Maximum marked drawdown | 0.3191% | 0.3191% |
| Turnover as a fraction of starting capital | 0.1914 | 0.1914 |
| Cash-equity costs on settled trades | ₹21.27 | ₹21.27 |
| Mean capital deployed | 1.496% | 1.496% |
| Peak open risk | 0.4853% | 0.4853% |

Zero decisions changed and the portfolio return difference was 0.0000 percentage points. The paired daily inference had two effective observations and degenerate dispersion. Equity-index and risk-matched alpha, general incorrect-rejection rates and forecast calibration remain unmeasured. The paired positive-rejection diagnostic counts only candidates with an observed settlement in the opposing arm.

The historical book uses the historical candidate's intended entry and next-session-or-later exits. That convention lets both policies share execution assumptions. This study does not establish fill quality, realistic execution alpha or forward profitability. The current selected cohort is a smoke-test fixture and cannot establish point-in-time universal membership. The 2022–October 2025 audit admitted 327,398 bars. The September–December 2021 extension adds 26,264 admitted bars across 329 names, for 353,662 across the audited periods. Of 84 expected 2021 sessions, 83 archives were present. The November 4 Muhurat archive is missing. The strict validator quarantined 4,619 absent symbol-sessions, 810 BE rows, 83 other-series rows, eight documented action rows and five unresolved moves. This extension uses the same 383-name cohort and corporate-action ledger, so incomplete delisted or renamed membership still limits inference. The 2021 audit artifact SHA-256 is `186821d2e465e42b8ba297b0ad5f0b3207d1a855ebd0b5767c598e830c26e214`. [Historical archive audit](strategic_intelligence_reset_2026-11-27.md) · [NSE cash calendar](https://archives.nseindia.com/content/circulars/CMTR46623.pdf) · [NSE Muhurat circular](https://nsearchives.nseindia.com/content/circulars/CMTR50050.pdf)

## Reproduction

Run the synthetic controls in `tests/test_historical_paper_loop.py`. In the persistent-regime control a positive setup receives SUPPORT and a losing setup receives BLOCK. Reversing the latter setup's next payoff makes that same block miss a profitable trade and reduces adaptive performance. This tests both the learner's influence and its vulnerability to stale evidence.

Run `python -m scripts.replay_adaptive_policy_smoke ARCHIVE_LOGS GEOMETRY_CHECKPOINT` with an explicit fresh isolated `QT_RUNTIME_ROOT` containing `logs/nse_holidays.json` and `logs/nse_calendar_manifest.json`. Use copies of the existing validated Gauntlet inputs. The pilot used the action ledger SHA-256 `93f25c6c66a0f3bb25d984c82d2ad95cddbd25ba0a3dda4df4c616f9f40505c1`. Each daily source and the action ledger are checked against the checkpoint, and each selected card is hashed into its immutable decision freeze. The comparison binds the resulting prices, cards, run configuration and code to its checkpoint. The script prints the binding and branch results as JSON.

Only the isolated adaptive selector projects mature comparison outcomes into simulated policy authority. Persisted comparison records retain historical evidence provenance and `affects_selection=False`. The production policy evaluator excludes them. Correlation inputs in this pilot are unavailable, so allocation uses existing fallback controls. Neither the reserved evaluation periods nor the frozen relative-strength experiment was used.

## Independent review

Review reproduced exposure to the current challenger and account through shared selector helpers, input-binding gaps and overstated correlation coverage. The isolated selector now bypasses those current-state reads, the bound card must match its existing decision fingerprint, an empty correlation input records zero coverage, and resumption requires the same effective breakeven threshold. The reference smoke independently checks cash archive bytes against the read-only checkpoint before it constructs the bar mapping. The lower-level comparison accepts caller-validated bars, so use the checked smoke procedure for reproducing its archive result.
