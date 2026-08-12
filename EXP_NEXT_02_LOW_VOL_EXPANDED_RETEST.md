# EXP-NEXT-02B — Low-Volatility Expanded Retest

> Scientific report. Production unchanged. Not a live trading authorization.
> Global trust remains `OPERATIONAL_ONLY`. Phase B not started. No ML.

## WHAT WE TESTED

Do calmer stocks give investors better returns for the amount of risk taken?

## WHAT HAPPENED

After realistic costs, quieter stocks did not deliver a useful edge over louder stocks under the frozen rules.

## WHAT IT MEANS

The evidence rejects the low-volatility hypothesis on this dataset.

## WHAT QUANTTERM WILL DO

Nothing. QuantTerm will not use this idea.

---

## 1. Previous 29-name result

| Field | Value |
|-------|--------|
| Experiment | `EXP-NEXT-02` |
| Hypothesis ID | `5eb01b27fc75b885` |
| Snapshot | `a7a9828ec37e09e4` |
| Final | **INCONCLUSIVE** |
| Next | `HOLD_NO_TUNING` |
| Discovery n_rebalances | 13 (UNDERPOWERED) |
| Discovery mean_net | +0.0076 |

## 2. New experiment ID

- **`EXP-NEXT-02B`** (does not overwrite EXP-NEXT-02)
- Hypothesis ID: `aa7b98515af8a5f1`
- Protocol version: `lowvol_expanded_retest.v1`
- Result hash: `a3b6732ee2b8787b`

## 3. Expanded snapshot certification

- Snapshot: `2f683be0c73eaa33`
- Scoped certification: `SCOPED_RESEARCH_READY`
- Global trust: `OPERATIONAL_ONLY`
- Securities: **870**
- Sessions (full): **1900**
- Security-years: **5940.8**
- Date range: 2020-01-01 → 2026-08-11
- PIT smoke: `True`

## 4. Frozen hypothesis

- Plain: Do calmer stocks give investors better returns for the amount of risk taken?
- Technical: Lowest trailing 20d realized-volatility quintile outperforms the highest-vol quintile on a 21d hold after CNC costs, with positive net mean return and harness PROMOTE on discovery; confirmation requires the same gates on the untouched confirmation partition.
- Null: No positive cost-aware OOS low-minus-high vol edge after CNC costs; any gross gap is economically destroyed by costs, statistically underpowered, or fails confirmation.
- Frozen file: `docs/overhaul/EXP_NEXT_02B_FROZEN_PROTOCOL.json`
- Frozen before outcomes: `True`

## 5. Protocol

- Vol lookback: **20d** realized vol
- Cohorts: lowest vs highest **20%** by inverse-vol rank
- Rebalance: every **21** sessions
- Hold: **21** sessions
- Costs: CNC round-trip **0.32** pct points; turnover one-way=1.0
- Primary portfolio: long low-vol / short high-vol
- Multiple testing: single primary specification (n_trials=1)
- Methodological deltas vs EXP-NEXT-02: see frozen JSON `methodological_versioning_vs_EXP_NEXT_02` (universe + partitions only; not tuned on the prior inconclusive result)

## 6. Discovery / confirmation partitions

- Discovery: `2021-01-01→2023-12-31` (915 sessions)
- Confirmation: `2024-01-01→panel_end` (671 sessions)
- Method: chronological; confirmation untouched during discovery
- Pre-registered approx rebalances: discovery ~42, confirm ~31 (both ≥ harness min_n=30) — stated before outcome inspection

## 7. Sample size

- Discovery rebalances: **44**
- Discovery n_eff: **35.249278406689655**
- Discovery median cohort size (per leg): **157**
- Discovery mean names scored: **779**
- Confirmation rebalances: **n/a (not opened / not PASS)**

## 8–11. Gross / costs / net / risk-adjusted

### Discovery

```json
{
  "mean_gross": -0.013426,
  "cost_drag": 0.0032,
  "mean_net": -0.016626,
  "sharpe_net": -1.6996,
  "sortino_net": -2.6282,
  "downside_std": 0.021914,
  "max_drawdown": -0.7865,
  "hit_rate": 0.2955,
  "long_only_sharpe_proxy_gross": 1.4629,
  "ew_sharpe_proxy_gross": 1.842,
  "long_only_minus_ew_mean_gross": -0.009709,
  "pack": {
    "verdict": "REJECT",
    "n": 44,
    "n_eff": 35.249278406689655,
    "mean_r": -0.0166,
    "sharpe": -0.4906,
    "psr": 0.0018,
    "dsr": 0.0018,
    "p_value": 0.9988918382555799,
    "insight": "No edge \u2014 -0.02R over 44 trades. Not worth acting on.",
    "hit_rate": 0.2955,
    "profit_factor": 0.3029,
    "max_drawdown": -0.7865,
    "ci_95": [
      -0.0266,
      -0.0066
    ],
    "mean_gross": -0.013426,
    "cost_drag": 0.0032,
    "mean_net": -0.016626
  },
  "verdict": "FAIL"
}
```

### Confirmation

```json
null
```

## 12. Statistical evidence

- Discovery harness verdict: `REJECT`
- Discovery p_value: `0.9988918382555799`
- Discovery DSR/PSR: `0.0018` / `0.0018`
- Discovery CI95 (mean net): `[-0.0266, -0.0066]`
- Confirmation harness: `n/a`
- Multiple-testing: single frozen specification

## 13. Economic evidence

- Discovery gross mean: **-0.013426**
- Cost drag / rebalance: **0.0032**
- Discovery net mean: **-0.016626**
- Economic gate mean_net>0: **FAIL**

## 14. Confirmation evidence

- Opened: **False**
- Confirmation verdict: **n/a**
- Final after confirm mapping: **FAIL**

## 15. Subperiod stability (discovery calendar years)

```json
[
  {
    "year": 2021,
    "n_rebalances": 15,
    "mean_gross": -0.01872,
    "mean_net": -0.02192,
    "sharpe_net": -2.0508,
    "verdict": "FAIL"
  },
  {
    "year": 2022,
    "n_rebalances": 15,
    "mean_gross": -0.00929,
    "mean_net": -0.01249,
    "sharpe_net": -1.3336,
    "verdict": "FAIL"
  },
  {
    "year": 2023,
    "n_rebalances": 14,
    "mean_gross": -0.00941,
    "mean_net": -0.01261,
    "sharpe_net": -2.1962,
    "verdict": "FAIL"
  }
]
```

## 16. Path A secondary robustness

> SECONDARY ROBUSTNESS EVIDENCE — cannot override primary verdict.

- Snapshot: `a7a9828ec37e09e4` (29 names)
- Partitions: `{'discovery': '2024-08-01→2025-07-31', 'confirm': '2025-08-01→end'}`
- Discovery verdict: `INCONCLUSIVE` (n=13, net=0.007575)
- Confirmation: `None`
- Path A final: **INCONCLUSIVE**

```json
{
  "role": "SECONDARY_ROBUSTNESS_EVIDENCE",
  "experiment_ref": "EXP-NEXT-02 reproducibility surface",
  "snapshot_id": "a7a9828ec37e09e4",
  "n_securities": 29,
  "partitions": {
    "discovery": "2024-08-01\u21922025-07-31",
    "confirm": "2025-08-01\u2192end"
  },
  "discovery": {
    "label": "path_a_discovery",
    "n_rebalances": 13,
    "median_cohort_size": 5,
    "mean_names_scored": 29,
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
      "mean_gross": 0.010775,
      "cost_drag": 0.0032,
      "mean_net": 0.007575
    },
    "mean_gross": 0.010775,
    "cost_drag": 0.0032,
    "mean_net": 0.007575,
    "sharpe_net": 0.723,
    "sortino_net": 1.18,
    "downside_std": 0.022238,
    "max_drawdown": -0.1093,
    "hit_rate": 0.5385,
    "long_only_sharpe_proxy_gross": 0.9667,
    "ew_sharpe_proxy_gross": -0.2327,
    "long_only_minus_ew_mean_gross": 0.010776,
    "turnover_one_way_assumed": 1.0,
    "verdict": "INCONCLUSIVE",
    "rebalance_dates_first_last": [
      "2024-08-01",
      "2025-07-31"
    ]
  },
  "confirmation": null,
  "final_verdict": "INCONCLUSIVE",
  "cannot_override_primary": true,
  "scoped_certification": "READY_FOR_SCIENTIFIC_RERUN"
}
```

## 17. Scientific-memory update

- Registry status: `REJECTED`
- Next action: `REJECT_CLOSE_BRANCH`
- Closed branches (momentum/reversal/structure/network/logistic/vol-compression) remain closed and were not reopened.

## 18. Production behaviour confirmation

| Surface | Status |
|---------|--------|
| Brain / ranking / risk / sizing | Unchanged |
| Execution / broker / autopilot / alerts | Unchanged |
| Production authority | `False` |

## 19. Plain-English conclusion

The evidence rejects the low-volatility hypothesis on this dataset.

Nothing. QuantTerm will not use this idea.

## 20. Final verdict

**FAIL**

---

## Status card

| Field | Value |
|-------|--------|
| PREVIOUS RESULT | INCONCLUSIVE (EXP-NEXT-02 / 29-name) |
| EXPANDED DISCOVERY | `FAIL` (net=-0.016626, n=44) |
| INDEPENDENT CONFIRMATION | `not opened` |
| STATISTICAL VERDICT | disc `REJECT` / conf `None` |
| ECONOMIC VERDICT | FAIL (gross=-0.013426, drag=0.0032, net=-0.016626) |
| FINAL VERDICT | **FAIL** |
| NEXT ACTION | **REJECT_CLOSE_BRANCH** |

_Generated 2026-08-11T17:42:59.190310+00:00_
_git_sha `0dc85354e18a51d8c2dc3edfd987487c45ceb389`_
