# EXP-FUND-04 — Value / Trailing PE

> Scientific report. Production unchanged. Not a live trading authorization.
> Global trust `OPERATIONAL_ONLY`. No ML. Closed OHLCV branches not reopened.

## WHAT WE TESTED

Do cheaper companies, measured using information actually known at the time, outperform more expensive companies?

## WHAT HAPPENED

The evidence was mixed or too weak to decide cleanly under the frozen rules.

## WHAT IT MEANS

Still uncertain.

## WHAT QUANTTERM WILL DO

No tuning. No live use.

---

## Technical evidence

- Experiment ID: `EXP-FUND-04`
- Hypothesis ID: `30259407bbca3443`
- Type: `ALPHA_FUNDAMENTAL`
- Foundation package: `46ff79f58ee21c9e`
- Parent OHLCV snapshot: `2f683be0c73eaa33`
- Partitions: discovery `2023-01-01→2024-06-30`; confirm `2024-07-01→2025-03-18`
- Cost: CNC 0.32 pct pts; drag/rebalance=0.0032
- Discovery verdict: `PASS`
- Confirmation: `INCONCLUSIVE`
- Final verdict: **INCONCLUSIVE**
- Next action: `HOLD_NO_TUNING`
- Production authority: `False`
- Registry: `REJECTED`
- Result hash: `fa8d2c3e8cda0027`
- Multiple-testing: harness n_trials=4

### Discovery detail

```json
{
  "pack": {
    "verdict": "PROMOTE",
    "n": 85,
    "n_eff": 19.834898582657868,
    "mean_r": 0.0114,
    "sharpe": 0.4528,
    "psr": 0.9998,
    "dsr": 0.9998,
    "p_value": 3.6279509191919155e-05,
    "insight": "Real edge: +0.01R over 85 trades, 100% confident it beats zero (deflated for 4 trials).",
    "hit_rate": 0.7059,
    "profit_factor": 2.959,
    "max_drawdown": -0.2817,
    "ci_95": [
      0.0061,
      0.0168
    ],
    "mean_gross": 0.014634,
    "cost_drag": 0.0032,
    "mean_net": 0.011434
  },
  "verdict": "PASS",
  "n_rebalances": 85,
  "median_names": 748,
  "ew_mean_gross": 0.037929,
  "ls_minus_ew_mean_gross": -0.023295
}
```

### Confirmation detail

```json
{
  "pack": {
    "verdict": "INCONCLUSIVE",
    "n": 37,
    "n_eff": 9.777466356526082,
    "mean_r": 0.0027,
    "sharpe": 0.1385,
    "psr": 0.7984,
    "dsr": 0.7984,
    "p_value": 0.20246628762495608,
    "insight": "Positive (+0.00R over 37 trades) but only 80% confident (need 95%), and the sample was big enough to show a real edge. Likely noise.",
    "hit_rate": 0.5135,
    "profit_factor": 1.4051,
    "max_drawdown": -0.1554,
    "ci_95": [
      -0.0036,
      0.009
    ],
    "mean_gross": 0.005902,
    "cost_drag": 0.0032,
    "mean_net": 0.002702
  },
  "verdict": "INCONCLUSIVE",
  "n_rebalances": 37,
  "median_names": 772,
  "ew_mean_gross": -0.011838,
  "ls_minus_ew_mean_gross": 0.017741
}
```

## Protocol notes

- Valuation: trailing PE with available_ts <= formation
- Outlier rule: exclude PE > 200.0; PE<=0 excluded
- No sector-neutral value design

_Generated 2026-08-11T18:15:45.910237+00:00_
_git_sha `378cd4895c6b00fa116a3076e1f0eedac5d84324`_
