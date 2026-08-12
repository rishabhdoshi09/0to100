# EXP-NEXT-01 — Short-Horizon Reversal

> Scientific report. Production unchanged. Not a live trading authorization.

## WHAT WE TESTED / WHAT HAPPENED / WHAT IT MEANS / WHAT QUANTTERM WILL DO

**What we tested:** Whether stocks that fall sharply over a few days tend to bounce back after trading costs.

**What happened:** After trading costs and the frozen checks, the effect was not reliable (or did not survive independent confirmation).

**What it means:** The idea does not currently show a proven advantage QuantTerm should use.

**What QuantTerm will do:** Nothing. The strategy / risk rule will not be used. Branch closed.


---

## Technical layer

- Experiment ID: `EXP-NEXT-01`
- Hypothesis ID: `be12db7a0d764c98`
- Type: `ALPHA`
- Snapshot: `a7a9828ec37e09e4`
- Discovery verdict: `FAIL`
- Confirmation: `None`
- Final verdict: **FAIL**
- Next action: `REJECT_CLOSE_BRANCH`
- Production authority: `False`
- Registry: `REJECTED`
- Result hash: `52dfce69de2ce0c5`

### Discovery detail

```json
{
  "cells": {
    "f1_h5": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 230.73644583759256,
      "mean_r": -0.0023,
      "sharpe": -0.1018,
      "psr": 0.0596,
      "dsr": 0.0596,
      "p_value": 0.9467121116566072,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4269,
      "profit_factor": 0.7593,
      "max_drawdown": -0.58,
      "ci_95": [
        -0.005,
        0.0005
      ],
      "mean_gross": 0.0009,
      "cost_drag": 0.0032,
      "mean_net": -0.0023,
      "formation": 1,
      "hold": 5,
      "n_oos": 253
    },
    "f1_h10": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 253.0,
      "mean_r": -0.0018,
      "sharpe": -0.0601,
      "psr": 0.1703,
      "dsr": 0.1703,
      "p_value": 0.8301471454611118,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4822,
      "profit_factor": 0.8585,
      "max_drawdown": -0.667,
      "ci_95": [
        -0.0055,
        0.0019
      ],
      "mean_gross": 0.0014,
      "cost_drag": 0.0032,
      "mean_net": -0.0018,
      "formation": 1,
      "hold": 10,
      "n_oos": 253
    },
    "f3_h5": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 129.19711489719165,
      "mean_r": -0.0008,
      "sharpe": -0.0373,
      "psr": 0.2812,
      "dsr": 0.2812,
      "p_value": 0.7234422381064103,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4308,
      "profit_factor": 0.9044,
      "max_drawdown": -0.4899,
      "ci_95": [
        -0.0036,
        0.0019
      ],
      "mean_gross": 0.0024,
      "cost_drag": 0.0032,
      "mean_net": -0.0008,
      "formation": 3,
      "hold": 5,
      "n_oos": 253
    },
    "f3_h10": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 118.765067258246,
      "mean_r": -0.0007,
      "sharpe": -0.0241,
      "psr": 0.3508,
      "dsr": 0.3508,
      "p_value": 0.6490391928835333,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4822,
      "profit_factor": 0.9392,
      "max_drawdown": -0.8932,
      "ci_95": [
        -0.0044,
        0.0029
      ],
      "mean_gross": 0.0025,
      "cost_drag": 0.0032,
      "mean_net": -0.0007,
      "formation": 3,
      "hold": 10,
      "n_oos": 253
    },
    "f5_h5": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 73.87993571123984,
      "mean_r": -0.001,
      "sharpe": -0.0444,
      "psr": 0.244,
      "dsr": 0.244,
      "p_value": 0.7596259587530805,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4545,
      "profit_factor": 0.8908,
      "max_drawdown": -0.7022,
      "ci_95": [
        -0.0036,
        0.0017
      ],
      "mean_gross": 0.0022,
      "cost_drag": 0.0032,
      "mean_net": -0.001,
      "formation": 5,
      "hold": 5,
      "n_oos": 253
    },
    "f5_h10": {
      "verdict": "REJECT",
      "n": 253,
      "n_eff": 56.61589058592418,
      "mean_r": -0.0002,
      "sharpe": -0.0053,
      "psr": 0.4662,
      "dsr": 0.4662,
      "p_value": 0.5337952656981941,
      "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
      "hit_rate": 0.4862,
      "profit_factor": 0.9869,
      "max_drawdown": -1.112,
      "ci_95": [
        -0.0037,
        0.0034
      ],
      "mean_gross": 0.003,
      "cost_drag": 0.0032,
      "mean_net": -0.0002,
      "formation": 5,
      "hold": 10,
      "n_oos": 253
    }
  },
  "fdr": {
    "rejected": [],
    "detail": {
      "f1_h5": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "f1_h10": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "f3_h5": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "f3_h10": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "f5_h5": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "f5_h10": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      }
    },
    "threshold": 0.0
  },
  "best_key": "f5_h10",
  "best": {
    "verdict": "REJECT",
    "n": 253,
    "n_eff": 56.61589058592418,
    "mean_r": -0.0002,
    "sharpe": -0.0053,
    "psr": 0.4662,
    "dsr": 0.4662,
    "p_value": 0.5337952656981941,
    "insight": "No edge \u2014 -0.00R over 253 trades. Not worth acting on.",
    "hit_rate": 0.4862,
    "profit_factor": 0.9869,
    "max_drawdown": -1.112,
    "ci_95": [
      -0.0037,
      0.0034
    ],
    "mean_gross": 0.003,
    "cost_drag": 0.0032,
    "mean_net": -0.0002,
    "formation": 5,
    "hold": 10,
    "n_oos": 253
  },
  "momentum_contrast_net": {
    "verdict": "REJECT",
    "n": 253,
    "n_eff": 35.69033064013164,
    "mean_r": -0.0131,
    "sharpe": -0.4294,
    "psr": 0.0,
    "dsr": 0.0,
    "p_value": 0.99999999996827,
    "insight": "No edge \u2014 -0.01R over 253 trades. Not worth acting on.",
    "hit_rate": 0.336,
    "profit_factor": 0.3343,
    "max_drawdown": -3.3891,
    "ci_95": [
      -0.0169,
      -0.0094
    ],
    "mean_gross": null,
    "cost_drag": 0.0032,
    "mean_net": -0.0131
  },
  "verdict": "FAIL"
}
```

### Confirmation detail

```json
null
```

_Generated 2026-08-11T17:14:50.528887+00:00_
