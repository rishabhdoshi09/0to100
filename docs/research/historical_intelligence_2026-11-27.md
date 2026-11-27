# Historical intelligence cycle

Baseline `0532b7a3cb1d2e2b2828181deea1b2e67e3e8306`. Everything below is discovery from isolated archives. The frozen relative-strength implementation and November 2025 onward are untouched by this study.

## Coverage and reuse

The isolated manifest has verified NSE CM calendar years 2024–2026. The archives contain earlier files, but this manifest does not verify their session continuity. The study starts January 2024 and ends before November 2025. The candidate cohort is the sorted union of the frozen discovery cohort and bundled operating-company list, 383 names. Current list construction can omit delisted names and the results do not represent the full historical NSE population.

- 454 of 457 expected archives were available. January 20, March 2 and May 18, 2024 were documented special sessions with unavailable files. Official NSE CSV and alternate ZIP requests did not supply verified bytes.
- 360 stocks supplied 159,692 validated stock-session bars, versus 232 stocks and 48,244 bars in the earlier study. The windows differ. Quarantines include 1,057 BE rows, 32 documented action dates, four unresolved moves and 68 other-series rows. An additional 13,029 symbol-sessions were absent from available archives.
- A fixed stride of five official sessions produced 18,283 observation rows for 358 stocks on 53 decision dates. There are 18,202 complete 10-session close measurements, 62,840 simulator outcomes across eight existing configurations and 35,278 filled outcomes. Processed coverage includes zero-row symbols, which are separately distinguishable.
- Two batches finished in 78 and 81 seconds, reusing 192 completed symbols on the second. A third invocation reused all 383, finished in 16 seconds and produced identical observation counts. The final SQLite file is 14.4 MiB and passed integrity check. An invocation still rereads and hashes source archives. Batching prevents repeat scanner computation on an identical bound snapshot. Appending dates or changing covered symbols still requires a new checkpoint. Concurrent symbol writes are atomic, while progress reports are intended for a single orchestrator.

## Scanner failure attribution

Ten-session close-to-close and next-open to terminal close comparisons use equal-weight same-date controls screened at ₹20 and ₹1 crore average traded value. Each comparison uses observed values for its own outcome. There is no hypothetical stop, cost or liquidity execution in these directional controls.

| Comparison | Close to close spread | Next-open to terminal close spread | Next-open gap spread |
|---|---:|---:|---:|
| Scanner BUY vs other eligible rows | −0.189 pp | −0.176 pp | −0.009 pp |
| Scanner signal vs no scanner signal | −0.276 pp | −0.227 pp | −0.044 pp |
| Top five scores vs remaining signals | | −0.126 pp | |

The BUY next-open spread was measured on 53 dates with 14 effective horizon blocks. Its descriptive 95% interval is −1.217 to +0.865 percentage points. On the scanner's simulated BUY fills, 1,592 trades averaged −0.322% gross and −0.642% net, or −0.133R, with a mean hold of 5.44 sessions. The modeled friction is 32 bp, and gross was already negative. The exit counts were 78 target first, 981 stop or scratch, and 533 expiry. The descriptive comparison provides no case for adding technical filters or for deleting a specific signal. Sector, regime, volatility and turnover interaction attribution and matched execution for rejected names remain incomplete.

## Conditional probabilities and adaptation

Four fixed decision features are standardized using training data alone. Ridge-regularized multinomial fits estimate the exit class and positive net result conditional on a resolved fill. Ridge payoff includes resolved no-fills as cash. The frozen model fits 2024. The adaptive model refits before each 2025 calendar quarter using prior labels with a 20-session purge. The pooled reference fits the corresponding earlier fills. Reliability bins and dependence-aware paired inference are in the analysis output.

| Forecast | Exit Brier / log loss | Positive-return Brier / log loss | Payoff RMSE |
|---|---|---|---:|
| Frozen conditional | 0.4922 / 0.7870 | 0.3629 / 0.5486 | 3.364 pp |
| Frozen pooled | 0.5049 / 0.8038 | 0.3653 / 0.5520 | |
| Adaptive conditional | 0.4951 / 0.7895 | 0.3640 / 0.5504 | 3.370 pp |
| Adaptive pooled | 0.5063 / 0.8045 | 0.3650 / 0.5517 | |

Scoring uses 3,401 resolved fills over 38 dates and only 10 effective horizon blocks. The paired one-sided lower bounds for exit Brier improvement over the corresponding priors are negative (−0.00187 frozen, −0.00163 adaptive). The adaptive model worsened the frozen model's point scores. The matched-basket execution control below is a return comparison; there is no separately calibrated market prior in this pass.

A forced top-five payoff ranking probe on nine non-overlapping dates resulted in +0.79% adaptive, −16.02% frozen and −5.81% exposure-matched basket. Realized drawdowns were 1.50% and 20.17% for adaptive and frozen. The frozen rank includes one censored sleeve, charged a 100% sleeve loss under the common stress convention. The adaptive results cannot demonstrate learning uplift. These are offline forecasts from timestamps and maturing simulator labels. Replay of the complete production learning policy, frozen decision ledger and restart flows is not established. Stock-specific production probabilities stay UNMEASURED.

## Reproduction

Run from the isolated checkout with the existing audited runtime in `QT_RUNTIME_ROOT`. Use the sorted union of `cohort` in `docs/research/opportunity_relative_strength_frozen.json` and the `SYMBOL` column in `data/nse_symbols_fallback.csv`. Supply the comma-separated result to `--stocks`.

```sh
python -m gauntlet --geometry-checkpoint "$checkpoint" \
  --research-start 2024-01-01 --development-before 2025-01-01 \
  --evaluation-before 2025-11-01 --observations --sample-step 5 \
  --max-symbols 192 --stocks "$stocks"
python -m research.scanner_attribution "$checkpoint"
```

Repeat the Gauntlet command until COMPLETE. Analysis requires a completed checkpoint. Binding SHA-256 `d5a06b7fb29614c4a13ede5d5a37d2094c3ac4b0dac8859db58f4a71dc6712ca` includes source and archive identities, official sessions and model costs. Checkpoint file SHA-256 `d10b5bb5ce0d22a9e173bf8d2ecbfef5d42702a224fa8eb6282704c8e56d3882`. Analysis SHA-256 `1a9d327251a4b224aab643771f11af21aeedf759a43dff755d880ba8915c14e1`.

Production was left read-only. Separate backup and deployment approvals remain outstanding and live trading remains HARD-LOCKED.
