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
