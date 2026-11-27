# QuantTerm research handoff, 26 November 2026

Development started from `426994a795fdeb26775d50d6991c70bbf06dc7d8` plus the uncommitted stock filter, schema gate, regression and Cycle 11 checkpoint in Claude's `c9` worktree. The earlier worktrees and scratch artifacts were preserved. The scratchpad contained the copied research runtime, the original and stock-filtered trade ledgers, and the probability analysis script. Older dirty worktrees include independent changes outside this research handoff.

## Historical evidence

Replaying Claude's analysis on the recovered ledgers gave a holdout expectancy of −2.183R before the fund filter and −0.084R after it. Each holdout had 293 filled trades. The filtered sample's observed target-hit rate was 6.1%. Those are exploratory results from an already inspected period. The two samples can contain different symbols after filtering. The target-hit event also differs from the new study's positive net-return event.

The copied store loaded 3,471 symbols across 1,763 sessions. Regenerating point-in-time membership from the same isolated archive passed the survivorship check. Independent review reproduced a 28% rate in a 150-symbol sample and identified DHARAN as the unmatched corporate-action symbol. The full Gauntlet preflight flagged 22.1% of 3,471 symbols and aborted. It also identified unadjusted-looking consecutive moves and uncovered calendar years. The verified full run requires a corrected snapshot, verified session calendar coverage, and a frozen evaluation window previously unseen by the researchers.

The Gauntlet geometry mode uses the existing scanner's session-anchored walk-forward and the canonical gap-aware fill/exit simulator, including delivery costs. It checkpoints per symbol in SQLite. The variant grid covers entry buffer, minimum stop distance, targets and holding horizons. Selection uses development outcomes only, with a maximum-horizon purge at both boundaries. Evaluation reports trades, censoring, mean net return, holding time, probability calibration error and a paired comparison with the current geometry on matched resolved setups. Its output is historical research and cannot authorize execution.

Run the existing Gauntlet entry point on an approved research machine with a frozen isolated runtime:

```sh
QT_RUNTIME_ROOT=/path/to/frozen-research-runtime python -m gauntlet \
  --geometry-checkpoint /path/to/research.sqlite \
  --development-before YYYY-MM-DD --evaluation-before YYYY-MM-DD
```

The same checkpoint resumes the same study. The bound snapshot, code, settings and date boundaries must match. Keep old checkpoints with their respective evidence.

## Paper and Home

Replaying the 26 November saved cards found 15 technically eligible probes rejected by the EV and case-memory consensus veto. Both summaries can draw from the same tracked signal log. The experimental lane can handle them as a single caution only when provenance confirms that the case sample contains zero independent desk outcomes. Production keeps its veto. The lane now searches past the first three failures and sends its selected probe through the existing correlation authority, account gates and daily half-risk limit. A replay or a regression pass does not imply a forward fill.

Home and recommendation cards show entry, stop, target, a short reason, measured or estimated profit probability when evidence exists, the holding limit, and expected net return when available. A holding limit describes the rule. The average realized holding period for the new geometry study requires completed verified data.

## Follow-up archive audit

The 3,471-symbol snapshot contains 1,763 validated file dates from 23 August 2019 through 23 November 2026. The adjusted-close detector flagged 1,162 moves across 768 symbols. Of those moves:

- 908 spanned at least one intervening archived date. These include suspensions and series interruptions. The aggregate alone cannot verify continuity for any given name.
- 169 exceeded the threshold even against the NSE file's own previous close. This includes penny-price tick moves, but the reported reference can also remain on the pre-action scale.
- 85 exhibited a material change in the exchange reference and a sub-threshold move from that reference. Seventy-four followed the archive-wide August–October 2019 hole. Consecutive file ordinals cannot establish consecutive trading sessions across a market-wide hole.

The unmatched, 4 April 2025 bonus was recorded under DHARAN. NSE changed the trading symbol from KBCGLOBAL to DHARAN effective 26 June 2025. The copied bhavcopies show both symbols in BE on the relevant sides of that date. Mapping the bonus to KBCGLOBAL in an alternate isolated action ledger cleared the symbol-mismatch check. The source ledger was preserved. [NSE listing circular CML68648](https://nsearchives.nseindia.com/content/circulars/CML68648.pdf)

The annual CM circulars and identified election and Bakri Id amendments expose 24 candidate missing regular weekdays between 23 August and 1 October 2019 and an absent 8 August 2022 file. A SEBI-hosted filing corroborates trading on the latter date. The direct archived-file fetch for 26 August 2019 returned a stream error and the legacy ZIP endpoint timed out. The calendar manifest remains verified for 2026. Historical amendments and special trading sessions need full reconciliation before attesting prior years. [2019 calendar](https://nsearchives.nseindia.com/content/circulars/CMTR39612.pdf) · [2019 election circular](https://www.steelcitynettrade.com/Circulars/Trading%20Holiday%20on%20October%2021%2C%202019%20on%20account%20of%20General%20Assembly%20Elections.pdf) · [2022 calendar](https://nsearchives.nseindia.com/content/circulars/CMTR50560.pdf) · [SEBI-hosted filing](https://www.sebi.gov.in/sebi_data/attachdocs/sep-2022/1663916996573.pdf) · [2023 Bakri Id amendment](https://www.niftyindices.com/Press_Release/ind_prs27062023.pdf) · [2024 election amendment](https://nsearchives.nseindia.com/content/circulars/CMTR61518.pdf)

The isolated preflight with the corrected ledger still stops on phantom gaps and incomplete calendar coverage. The recovered trades already consumed outcomes through 19 November 2026, and the remaining archive is too short for a new 20-session evaluation. No geometry rows or holdout results were produced. The existing report now scores all eight configurations. Its development-only probability forecasts distinguish target, stop, and expiry, including target-triggered gap scratches and stop-triggered breakeven scratches. It reports per-event Brier scores, calibration errors, and overlap-aware uncertainty where estimable. A successful dataset preflight and finished symbols are reused on an exact checkpoint binding.
