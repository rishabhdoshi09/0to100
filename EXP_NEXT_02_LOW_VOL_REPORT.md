# EXP-NEXT-02 — Low-Volatility Effect

> Scientific report. Production unchanged. Not a live trading authorization.

## WHAT WE TESTED / WHAT HAPPENED / WHAT IT MEANS / WHAT QUANTTERM WILL DO

**What we tested:** Whether quieter (lower-volatility) stocks produce better risk-adjusted results than high-volatility stocks after costs.

**What happened:** The evidence was mixed or underpowered under the frozen criteria.

**What it means:** We cannot claim the idea works or fails cleanly yet.

**What QuantTerm will do:** No tuning. No live use. Hold.


---

## Technical layer

- Experiment ID: `EXP-NEXT-02`
- Hypothesis ID: `5eb01b27fc75b885`
- Type: `ALPHA`
- Snapshot: `a7a9828ec37e09e4`
- Discovery verdict: `INCONCLUSIVE`
- Confirmation: `None`
- Final verdict: **INCONCLUSIVE**
- Next action: `HOLD_NO_TUNING`
- Production authority: `False`
- Registry: `REJECTED`
- Result hash: `af8c50881a7252f9`

### Discovery detail

```json
{
  "pack": {
    "verdict": "UNDERPOWERED",
    "n": 13,
    "n_eff": 7.873106149249786,
    "mean_r": 0.0076,
    "sharpe": 0.2087,
    "psr": 0.7594,
    "dsr": 0.7594,
    "p_value": 0.23312482398419296,
    "insight": "Only 13 trades \u2014 below the 30-trade floor for any claim (a great-looking edge on a handful of trades is usually luck). Keep tracking.",
    "hit_rate": 0.5385,
    "profit_factor": 1.6776,
    "max_drawdown": -0.1093,
    "ci_95": [
      -0.0122,
      0.0273
    ],
    "mean_gross": 0.0108,
    "cost_drag": 0.0032,
    "mean_net": 0.0076
  },
  "long_only_sharpe_proxy": 0.9667,
  "ew_sharpe_proxy": -0.2327,
  "n_rebalances": 13,
  "verdict": "INCONCLUSIVE"
}
```

### Confirmation detail

```json
null
```

_Generated 2026-08-11T17:14:50.572753+00:00_
