# EXP-FUND-03 — Earnings Growth

> Scientific report. Production unchanged. Not a live trading authorization.
> Global trust `OPERATIONAL_ONLY`. No ML. Closed OHLCV branches not reopened.

## WHAT WE TESTED

Do companies whose earnings are improving faster subsequently outperform?

## WHAT HAPPENED

The effect appeared in discovery and held up in an untouched later period after costs.

## WHAT IT MEANS

The idea looks real on this certified history — not approved for live trading.

## WHAT QUANTTERM WILL DO

Eligible for follow-up research only. QuantTerm will not trade on it yet.

---

## Technical evidence

- Experiment ID: `EXP-FUND-03`
- Hypothesis ID: `ef129d36b8f8378b`
- Type: `ALPHA_FUNDAMENTAL`
- Foundation package: `46ff79f58ee21c9e`
- Parent OHLCV snapshot: `2f683be0c73eaa33`
- Partitions: discovery `2023-01-01→2024-06-30`; confirm `2024-07-01→2025-03-18`
- Cost: CNC 0.32 pct pts; drag/rebalance=0.0032
- Discovery verdict: `PASS`
- Confirmation: `PASS`
- Final verdict: **CONFIRMED**
- Next action: `ELIGIBLE_FOR_FOLLOWUP_RESEARCH`
- Production authority: `False`
- Registry: `PROMOTED`
- Result hash: `3d508563a5d81633`
- Multiple-testing: harness n_trials=4

### Discovery detail

```json
{
  "pack": {
    "verdict": "PROMOTE",
    "n": 66,
    "n_eff": 17.430536610731366,
    "mean_r": 0.0071,
    "sharpe": 0.3931,
    "psr": 0.9956,
    "dsr": 0.9956,
    "p_value": 0.0010832231850911203,
    "insight": "Real edge: +0.01R over 66 trades, 100% confident it beats zero (deflated for 4 trials).",
    "hit_rate": 0.7121,
    "profit_factor": 2.6558,
    "max_drawdown": -0.1728,
    "ci_95": [
      0.0027,
      0.0114
    ],
    "mean_gross": 0.010273,
    "cost_drag": 0.0032,
    "mean_net": 0.007073
  },
  "verdict": "PASS",
  "n_rebalances": 66,
  "median_names": 809,
  "ew_mean_gross": 0.047326,
  "ls_minus_ew_mean_gross": -0.037053
}
```

### Confirmation detail

```json
{
  "pack": {
    "verdict": "PROMOTE",
    "n": 37,
    "n_eff": 8.354717962163202,
    "mean_r": 0.0042,
    "sharpe": 0.3098,
    "psr": 0.9625,
    "dsr": 0.9625,
    "p_value": 0.03378651908703819,
    "insight": "Real edge: +0.00R over 37 trades, 96% confident it beats zero (deflated for 4 trials).",
    "hit_rate": 0.5946,
    "profit_factor": 2.0994,
    "max_drawdown": -0.1078,
    "ci_95": [
      -0.0002,
      0.0086
    ],
    "mean_gross": 0.007397,
    "cost_drag": 0.0032,
    "mean_net": 0.004197
  },
  "verdict": "PASS",
  "n_rebalances": 37,
  "median_names": 825,
  "ew_mean_gross": -0.012486,
  "ls_minus_ew_mean_gross": 0.019883
}
```

## Protocol notes

- Growth metric: YoY basic EPS from PIT fundamentals
- Rebalance every 5 sessions; hold 21

_Generated 2026-08-11T18:15:44.310818+00:00_
_git_sha `378cd4895c6b00fa116a3076e1f0eedac5d84324`_
