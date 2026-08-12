# EXP-FUND-01 — Post-Earnings Drift

> Scientific report. Production unchanged. Not a live trading authorization.
> Global trust `OPERATIONAL_ONLY`. No ML. Closed OHLCV branches not reopened.

## WHAT WE TESTED

After a company reports materially better or worse business results, does the stock continue moving in the same direction for some time?

## WHAT HAPPENED

The evidence was mixed or too weak to decide cleanly under the frozen rules.

## WHAT IT MEANS

Still uncertain.

## WHAT QUANTTERM WILL DO

No tuning. No live use.

---

## Technical evidence

- Experiment ID: `EXP-FUND-01`
- Hypothesis ID: `736c936f510eeac0`
- Type: `ALPHA_EVENT`
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
- Result hash: `e4b1bdf032643b2f`
- Multiple-testing: harness n_trials=4

### Discovery detail

```json
{
  "pack": {
    "verdict": "UNDERPOWERED",
    "n": 12,
    "n_eff": 8.806736784949296,
    "mean_r": 0.0135,
    "sharpe": 0.4802,
    "psr": 0.867,
    "dsr": 0.867,
    "p_value": 0.06221348053210667,
    "insight": "Only 12 trades \u2014 below the 30-trade floor for any claim (a great-looking edge on a handful of trades is usually luck). Keep tracking.",
    "hit_rate": 0.75,
    "profit_factor": 3.4505,
    "max_drawdown": -0.0663,
    "ci_95": [
      -0.0024,
      0.0295
    ],
    "mean_gross": 0.016744,
    "cost_drag": 0.0032,
    "mean_net": 0.013544
  },
  "verdict": "INCONCLUSIVE",
  "n_events": 9497,
  "n_rebalances": 12,
  "median_names": 326,
  "ew_mean_gross": 0.052835
}
```

### Confirmation detail

```json
null
```

## Protocol notes

- Signal: YoY basic EPS growth at earnings AVAILABLE_AT
- Entry: next session after AVAILABLE_AT (conservative)
- Hold: 21 sessions
- Only EARNINGS_RESULT (no generic announcement mining)
- Matched events discovery/confirm: 10150 / 4553

_Generated 2026-08-11T18:15:37.091479+00:00_
_git_sha `378cd4895c6b00fa116a3076e1f0eedac5d84324`_
