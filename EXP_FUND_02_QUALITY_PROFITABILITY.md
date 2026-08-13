# EXP-FUND-02 — Quality / Profitability

> Scientific report. Production unchanged. Not a live trading authorization.
> Global trust `OPERATIONAL_ONLY`. No ML. Closed OHLCV branches not reopened.

## WHAT WE TESTED

Do financially stronger and more profitable companies produce better future risk-adjusted returns?

## WHAT HAPPENED

The evidence was mixed or too weak to decide cleanly under the frozen rules.

## WHAT IT MEANS

Still uncertain.

## WHAT QUANTTERM WILL DO

No tuning. No live use.

---

## Technical evidence

- Experiment ID: `EXP-FUND-02`
- Hypothesis ID: `9907302dd4d61ec5`
- Type: `ALPHA_FUNDAMENTAL`
- Foundation package: `46ff79f58ee21c9e`
- Parent OHLCV snapshot: `2f683be0c73eaa33`
- Partitions: discovery `2023-01-01→2024-06-30`; confirm `2024-07-01→2025-03-18`
- Cost: CNC 0.32 pct pts; drag/rebalance=0.0032
- Discovery verdict: `INCONCLUSIVE`
- Confirmation: `None`
- Final verdict: **INCONCLUSIVE**
- Next action: `HOLD_NO_TUNING`
- Production authority: `False`
- Registry: `REJECTED`
- Result hash: `af8c50881a7252f9`
- Multiple-testing: harness n_trials=4

### Discovery detail

```json
{
  "pack": {
    "verdict": "INCONCLUSIVE",
    "n": 85,
    "n_eff": 23.81269780333877,
    "mean_r": 0.0018,
    "sharpe": 0.0748,
    "psr": 0.7513,
    "dsr": 0.7513,
    "p_value": 0.24617727203006615,
    "insight": "Positive (+0.00R over 85 trades) but only 75% confident (need 95%), and the sample was big enough to show a real edge. Likely noise.",
    "hit_rate": 0.5647,
    "profit_factor": 1.2214,
    "max_drawdown": -0.3391,
    "ci_95": [
      -0.0033,
      0.0069
    ],
    "mean_gross": 0.004988,
    "cost_drag": 0.0032,
    "mean_net": 0.001788
  },
  "verdict": "INCONCLUSIVE",
  "n_rebalances": 85,
  "median_names": 823,
  "ew_mean_gross": 0.036985,
  "ls_minus_ew_mean_gross": -0.031997
}
```

### Confirmation detail

```json
null
```

## Protocol notes

- Quality metric: net margin = PAT / Revenue (PIT AVAILABLE_AT)
- Rebalance every 5 sessions; hold 21
- No sector neutralization (PIT sectors unavailable)

_Generated 2026-08-11T18:15:38.992360+00:00_
_git_sha `378cd4895c6b00fa116a3076e1f0eedac5d84324`_
