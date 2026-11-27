# Intelligence closure cycle

Development baseline `83ec8e6de9dad69d0f3f89cbed4320d116bcd484`. All measured returns are exploratory. This cycle does not spend the November 2025 onward reserve or alter the frozen relative-strength experiment.

## Coverage and incremental Gauntlet

The 2023 cash-market audit applies the exchange's 15 weekday closures with Bakri Id moved from June 28 to June 29, plus the November 12 special session. The audited dates were January 26, March 7 and 30, April 4, 7 and 14, May 1, June 29, August 15, September 19, October 2 and 24, November 14 and 27 and December 25. [NSE annual calendar](https://archives.nseindia.com/content/circulars/CMTR54757.pdf) · [NSE Bakri Id amendment, issuer copy](https://geplcapital.com/fundamental-research/649a873a2dbe6.1687848762.CMTR57285.pdf) · [NSE Muhurat circular](https://nsearchives.nseindia.com/content/circulars/CMTR59124.pdf)

The isolated 383-name cohort supplied 245 of 246 expected archives. The special-session archive was absent. The existing strict scoped validator admitted 85,155 bars across 357 stocks. It also flagged 8,320 absent symbol-sessions, 145 BE, 201 other series, 10 documented actions and four unresolved large moves. Adding that interval to the prior 2024–October 2025 audit gives 244,847 admitted stock-session bars. This modern cohort still does not represent delisted membership accurately, and no 2023 strategy outcome or untouched validation claim is made. The 2023 audit artifact SHA-256 is `7b1e772884f911cc118e2bdcbdab9d5bc6cad4b9a5b2369f3b1bb1b2fd6dd743`.

The existing observation checkpoint can accept appended sessions, additional stocks and a recovered missing archive while source, verified calendar, cost and split bindings stay unchanged. Recovery invalidates decisions starting one maximum simulator horizon before the changed archive. Existing present archive rewrites, action/calendar amendments, symbol removal, and partially completed prior scopes are rejected. Writes are guarded against a stale worker and reports use a read snapshot. Archive bytes are still hashed and validation is refreshed after binding changes.

The measured pilot used the first 24 symbols from the frozen cohort, 2024-01-01 through 2025-08-01 exclusive, cutoff 2025-01-01 and official stride five. It then extended that same database to the first 36 cohort symbols and 2025-11-01 exclusive. The 44-second first pass produced 954 observations, all retained by the 40-second append. A 47-second clean run produced precisely the same 1,882 observation and 6,616 outcome rows. This wall-clock pilot preceded the report race guard. The final fixture checks row equality, skipped historical scanner calls, missing-archive invalidation and stale reports on the same mechanism. Differences in disk cache and archive validation limit the wall-clock comparison.

## Scanner thesis and probabilities

The existing 18,283 observations and 62,840 outcomes were rescored with `python -m research.scanner_attribution <completed-prior-checkpoint>`. Result SHA-256 `29168ceda4f9a728592e3037d79d0f194e2a88621a7ab3a8b3cb8e7fb52b93af`.

| Diagnostic | Result |
|---|---|
| Mean same-date Spearman score versus next-open to terminal close | −0.005, descriptive 95% interval −0.079 to +0.070 |
| Mean gross returns in score quintiles, low to high | −0.251%, −0.519%, −0.283%, −0.661%, −0.475% |
| Scanner BUY versus rejected signal setups, identical simulator and costs | +0.016 pp per setup, 95% interval −0.840 to +0.871 pp |
| Existing 20-session variant versus original on pairs that both filled | −0.079 pp per fill, 95% interval −0.693 to +0.535 pp |

All paired tests have insufficient independent blocks. These data give no economic case for increasing scanner score weight. The previous finding of gross and net losses remains. These are short-horizon scanner tests. Lower-turnover portfolio strategies and medium-term trend or regime exposure still require longer verified simulation and appropriate controls.

Entry forecasts now exclude unresolved setups and distinguish no-fill from a resolved entry. Exit models remain conditional on resolved fills and distinguish scratches using the simulator's existing barrier semantics. Frozen conditional entry Brier is 0.437 versus 0.498 for the frozen pooled prior. Expanding refits score 0.436 versus 0.504 for their pooled prior. Exit Brier is weaker after adaptation (0.495 versus 0.492 frozen). The payoff probe now holds cash unless expected net payoff is positive. The adaptive probe chose no trades. The frozen probe returned −0.064% versus −0.382% for the matched allocation basket. These diagnostic scores do not demonstrate stock-specific accuracy or production policy improvement. The production policy replay and matched-market probabilities remain UNVERIFIED.

## Paper entry recovery

The EOD path reloads a checksum-verified committed book before replaying missing taken-entry effects. It joins on the immutable selection intent, decision, symbol, strategy, original entry date and context. It recovers only facts supported by those records and marks their origin. Legacy positions without the matching intent remain unmodified. No position or order is created. The existing GOOD_WAIT routine runs before a recovered taken row is written. Durable taken writes are deduplicated by decision identity.

One fault fixture interrupts the taken write after the credit call, reloads committed state and verifies one taken row and no invented legacy row. Another interrupts before the first GOOD_WAIT marker, restarts from a committed book and original intent log, and replays twice against the real disposable counterfactual and policy stores. It confirms one taken row, one GOOD_WAIT credit in each eligible policy and no duplicate effect during later official settlement. Recovery also retries credit for already-recorded entries in the bounded 45-day window. Independent review identified the need to exclude same-session WAITs from newly reconstructed credits, since the intent records only the fill date. The negative fixture verifies this exclusion. Standalone special-session dispatch, interrupted audit projection and all-store atomicity remain open.

The working production volume, frozen experiment and service state were untouched. No model qualifies for promotion. Live trading is HARD-LOCKED.
