# EXP-FUND-03-FOLLOWUP — Earnings Growth Follow-up Validation

> Scientific follow-up only. Does **not** overwrite EXP-FUND-03.
> Production unchanged. No ML. No Phase B. Not a live trading authorization.

## WHAT WE ALREADY KNEW

Earnings growth was the first QuantTerm idea to pass both discovery and an independent historical confirmation.

## WHAT WE CHECKED NOW

We checked whether that result was broad, repeatable, practical after costs, and dependent on only a few lucky stocks.

## WHAT HAPPENED

Follow-up checks were mixed: some supportive, some weak. Not enough to clear a robust-confirmed bar, and not a clean failure on one gate.

## WHAT QUANTTERM WILL DO

Hold as inconclusive follow-up. No tuning. No live use.

---

## Technical evidence

- Follow-up experiment ID: `EXP-FUND-03-FOLLOWUP`
- Parent: `EXP-FUND-03` (hypothesis `ef129d36b8f8378b`, result hash `3d508563a5d81633`)
- Follow-up hypothesis ID: `2617d6e4022e4ddb`
- Foundation package: `46ff79f58ee21c9e`
- Parent OHLCV snapshot: `2f683be0c73eaa33`
- Frozen protocol: `docs/overhaul/EXP_FUND_03_FOLLOWUP_FROZEN_PROTOCOL.json`
- Reproduction: **PASS**
- Final follow-up verdict: **INCONCLUSIVE_FOLLOWUP**
- Next action: `RECORD_EVIDENCE_NO_TUNING`
- Production authority: `False`
- Git SHA: `5f4198b086de12d1`
- Result hash: `9b4956698f1e6023`

### F1 — Frozen confirmed effect

Parent definition, AVAILABLE_AT treatment, universe, quintile L/S, reb=5, hold=21, CNC costs, and confirmation metrics are frozen in the follow-up protocol. Parent EXP-FUND-03 registry entry was **not** overwritten.

### F2 — Reproduction

```json
{
  "pass": 1,
  "package": "46ff79f58ee21c9e",
  "ohlcv_snapshot": "2f683be0c73eaa33",
  "discovery": {
    "label": "discovery",
    "ok": 1,
    "checks": {
      "n_match": 1,
      "mean_net_match": 1,
      "mean_gross_match": 1,
      "parent_partition_pass": 1
    },
    "reproduced": {
      "n": 66,
      "mean_gross": 0.010273,
      "mean_net": 0.007073,
      "harness_verdict": "PROMOTE"
    },
    "stored": {
      "n": 66,
      "mean_gross": 0.010273,
      "mean_net": 0.007073,
      "harness_verdict": "PROMOTE",
      "partition_verdict": "PASS"
    }
  },
  "confirmation": {
    "label": "confirmation",
    "ok": 1,
    "checks": {
      "n_match": 1,
      "mean_net_match": 1,
      "mean_gross_match": 1,
      "parent_partition_pass": 1
    },
    "reproduced": {
      "n": 37,
      "mean_gross": 0.007397,
      "mean_net": 0.004197,
      "harness_verdict": "PROMOTE"
    },
    "stored": {
      "n": 37,
      "mean_gross": 0.007397,
      "mean_net": 0.004197,
      "harness_verdict": "PROMOTE",
      "partition_verdict": "PASS"
    }
  },
  "same_config": {
    "reb": 5,
    "hold": 21,
    "q": 0.2,
    "cost_drag": 0.0032,
    "n_trials": 4
  }
}
```

### F3 — Subperiod stability

```json
{
  "SP1_2023H1": {
    "start": "2023-01-01",
    "end": "2023-06-30",
    "gross_effect": 0.014077,
    "net_effect": 0.010877,
    "sample_size": 13,
    "turnover_assumption": 1.0,
    "drawdown": -0.0113,
    "effect_direction": "POSITIVE",
    "direction": "POSITIVE",
    "hit_rate": 0.7692
  },
  "SP2_2023H2": {
    "start": "2023-07-01",
    "end": "2023-12-31",
    "gross_effect": 0.006867,
    "net_effect": 0.003667,
    "sample_size": 28,
    "turnover_assumption": 1.0,
    "drawdown": -0.1542,
    "effect_direction": "POSITIVE",
    "direction": "POSITIVE",
    "hit_rate": 0.5357
  },
  "SP3_2024H1": {
    "start": "2024-01-01",
    "end": "2024-06-30",
    "gross_effect": 0.014597,
    "net_effect": 0.011397,
    "sample_size": 26,
    "turnover_assumption": 1.0,
    "drawdown": -0.005,
    "effect_direction": "POSITIVE",
    "direction": "POSITIVE",
    "hit_rate": 0.8846
  },
  "SP4_CONFIRM": {
    "start": "2024-07-01",
    "end": "2025-03-18",
    "gross_effect": 0.007397,
    "net_effect": 0.004197,
    "sample_size": 37,
    "turnover_assumption": 1.0,
    "drawdown": -0.1078,
    "effect_direction": "POSITIVE",
    "direction": "POSITIVE",
    "hit_rate": 0.5946
  }
}
```

### F4 — Cross-sectional monotonicity

```json
{
  "Q1": {
    "label": "low_growth",
    "n_rebalances": 102,
    "mean_fwd_gross": 0.020021
  },
  "Q2": {
    "label": "mid",
    "n_rebalances": 102,
    "mean_fwd_gross": 0.022688
  },
  "Q3": {
    "label": "mid",
    "n_rebalances": 102,
    "mean_fwd_gross": 0.021226
  },
  "Q4": {
    "label": "mid",
    "n_rebalances": 102,
    "mean_fwd_gross": 0.027125
  },
  "Q5": {
    "label": "high_growth",
    "n_rebalances": 102,
    "mean_fwd_gross": 0.029828
  },
  "spearman_bucket_vs_return": 0.9,
  "broadly_monotonic": 1,
  "high_minus_low": 0.009807
}
```

### F5 — Cost / turnover robustness

```json
{
  "gross_edge": 0.00962,
  "base_cost_edge": 0.00642,
  "higher_cost_edge_0.50": 0.00462,
  "higher_cost_edge_0.75": 0.00212,
  "higher_cost_edge_1.00": -0.00038,
  "turnover": 1.0,
  "break_even_round_trip_pct_points": 0.962,
  "scenarios": {
    "0.32": {
      "round_trip_pct_points": 0.32,
      "cost_drag": 0.0032,
      "mean_gross": 0.00962,
      "mean_net": 0.00642,
      "n": 103,
      "direction": "POSITIVE"
    },
    "0.50": {
      "round_trip_pct_points": 0.5,
      "cost_drag": 0.005,
      "mean_gross": 0.00962,
      "mean_net": 0.00462,
      "n": 103,
      "direction": "POSITIVE"
    },
    "0.75": {
      "round_trip_pct_points": 0.75,
      "cost_drag": 0.0075,
      "mean_gross": 0.00962,
      "mean_net": 0.00212,
      "n": 103,
      "direction": "POSITIVE"
    },
    "1.00": {
      "round_trip_pct_points": 1.0,
      "cost_drag": 0.01,
      "mean_gross": 0.00962,
      "mean_net": -0.00038,
      "n": 103,
      "direction": "NEGATIVE"
    }
  },
  "economically_fragile": 0
}
```

### F6 — Liquidity / implementability

```json
{
  "adv_window": 21,
  "median_selected_dollar_volume_proxy": 128999723.03690475,
  "median_universe_dollar_volume_proxy": 193357585.0,
  "selected_vs_universe_median_ratio": 0.6672,
  "mean_share_of_selected_in_bottom_adv_quintile": 0.2744,
  "median_long_count": 164,
  "median_short_count": 164,
  "capacity_estimate": null,
  "capacity_note": "No scientifically supportable capacity estimate from current data (ADV proxy only; no depth/impact model).",
  "liquidity_concern": 0
}
```

### F7 — Concentration

```json
{
  "n_rebalances": 103,
  "mean_ls_gross": 0.00962,
  "median_ls_gross": 0.012819,
  "mean_vs_median_gap": -0.003199,
  "share_positive_pnl_from_top5_rebalances": 0.1411,
  "top5_rebalances": [
    {
      "date": "2023-07-06",
      "ls_gross": 0.041291
    },
    {
      "date": "2023-06-30",
      "ls_gross": 0.037806
    },
    {
      "date": "2024-10-09",
      "ls_gross": 0.037595
    },
    {
      "date": "2023-10-25",
      "ls_gross": 0.035642
    },
    {
      "date": "2023-07-18",
      "ls_gross": 0.0349
    }
  ],
  "share_positive_contrib_from_top5_names": 0.0665,
  "top5_names": [
    {
      "symbol": "PREMEXPLN",
      "contrib_sum": 0.071263
    },
    {
      "symbol": "CYIENT",
      "contrib_sum": 0.055107
    },
    {
      "symbol": "TRENT",
      "contrib_sum": 0.046312
    },
    {
      "symbol": "JWL",
      "contrib_sum": 0.04271
    },
    {
      "symbol": "RIIL",
      "contrib_sum": 0.041224
    }
  ],
  "lottery_like": 0
}
```

### F8 — Fundamental-data / AVAILABLE_AT audit

```json
{
  "restatement_lookahead_rows": 0,
  "yoy_rows_total": 6408,
  "leak_fraction_of_yoy": 0.0,
  "material_pit_issue": 0,
  "any_lookahead": 0,
  "amended_period_ends": 0,
  "duplicate_same_available_at_rows": 0,
  "examples": [],
  "note": "Parent _yoy_eps_map keeps last filing per period_end then tags growth with current available_at; if prior-year EPS was restated later than current filing, prev_eps can leak backward."
}
```

### F9 — Placebo / negative controls

```json
[
  {
    "id": "PLACEBO_PREMATURE_PERIOD_END",
    "summary": {
      "n": 106,
      "mean_gross": 0.02952,
      "mean_net": 0.02632,
      "cost_drag": 0.0032,
      "max_drawdown": -0.1471,
      "hit_rate": 0.8208,
      "p_value": 0.0,
      "harness_verdict": "PROMOTE",
      "direction": "POSITIVE",
      "n_rebalances": 106,
      "median_names": 823,
      "ew_mean_gross": 0.025711,
      "turnover_assumption": 1.0
    },
    "suspicious_if": "mean_net clearly positive and similar to true-signal edge",
    "flag_suspicious": 1
  },
  {
    "id": "PLACEBO_PRE_RELEASE_WINDOW",
    "summary": {
      "n": 17,
      "mean_gross": 0.054594,
      "mean_net": 0.051394,
      "cost_drag": 0.0032,
      "max_drawdown": -0.0402,
      "hit_rate": 0.9412,
      "p_value": 9.94e-06,
      "harness_verdict": "UNDERPOWERED",
      "direction": "POSITIVE"
    },
    "overall_ls_pre_ret": 0.067162,
    "flag_suspicious": 1,
    "suspicious_if": "growth predicts returns BEFORE AVAILABLE_AT"
  }
]
```

### F10 — Benchmark incrementality

```json
{
  "raw_growth": {
    "n": 103,
    "mean_gross": 0.00962,
    "mean_net": 0.00642,
    "cost_drag": 0.0032,
    "max_drawdown": -0.1728,
    "hit_rate": 0.699,
    "p_value": 9.997e-05,
    "harness_verdict": "PROMOTE",
    "direction": "POSITIVE",
    "n_rebalances": 103,
    "median_names": 821,
    "ew_mean_gross": 0.024285,
    "turnover_assumption": 1.0
  },
  "momentum_60d_ls": {
    "n": 122,
    "mean_gross": 0.006994,
    "mean_net": 0.003794,
    "cost_drag": 0.0032,
    "max_drawdown": -0.2839,
    "hit_rate": 0.5,
    "p_value": 0.08166703,
    "harness_verdict": "INCONCLUSIVE",
    "direction": "POSITIVE",
    "n_rebalances": 122,
    "median_names": 785,
    "ew_mean_gross": 0.020134,
    "turnover_assumption": 1.0
  },
  "growth_residualized_vs_mom60": {
    "n": 102,
    "mean_gross": 0.002323,
    "mean_net": -0.000877,
    "cost_drag": 0.0032,
    "max_drawdown": -0.5601,
    "hit_rate": 0.5294,
    "p_value": 0.62491521,
    "harness_verdict": "REJECT",
    "direction": "NEGATIVE",
    "n_rebalances": 102,
    "median_names": 733,
    "ew_mean_gross": 0.023929,
    "turnover_assumption": 1.0
  },
  "contains_incremental_info": 0,
  "note": "Diagnostic only \u2014 not a new factor; PE/quality not used as optimizers."
}
```

### Verdict rationale

- placebo controls not clean — cannot clear robust bar
- little incremental information vs 60d momentum

### Status card

| Field | Value |
|---|---|
| ORIGINAL RESULT | CONFIRMED (EXP-FUND-03; net≈+0.71% discovery) |
| REPRODUCIBLE? | YES |
| TIME-STABLE? | YES |
| MONOTONIC? | YES |
| COST-ROBUST? | YES |
| LIQUIDITY ACCEPTABLE? | YES |
| CONCENTRATION ACCEPTABLE? | YES |
| PIT/AVAILABLE_AT CLEAN? | YES |
| PLACEBO CLEAN? | NO |
| FINAL FOLLOW-UP VERDICT | INCONCLUSIVE_FOLLOWUP |
| NEXT ACTION | RECORD_EVIDENCE_NO_TUNING |
