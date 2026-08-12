# Phase A.5 Research-Grade Scientific Rerun

> Scientific rerun of the **frozen** Phase A.5 protocols against scoped certified snapshot `a7a9828ec37e09e4`. **Global trust remains `OPERATIONAL_ONLY`.** Production behaviour unchanged. Phase B not started.

## 1. Executive summary

QuantTerm re-tested five frozen research ideas using verified NSE history for the exact 29-name panel. Global database quality is still not fully certified; this rerun only uses the scoped certified snapshot.

### EXP-A5-01 — **FAIL**

We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Next action: `REJECT`

### EXP-A6-01 — **FAIL**

We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Next action: `REJECT`

### EXP-A2-01 — **FAIL**

We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Next action: `REJECT`

### EXP-A3-01 — **FAIL**

We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Next action: `REJECT`

### EXP-A5A6-01 — **PASS**

This idea showed a reliable advantage in the frozen historical test. It is worth further validation, but it is not approved for real trading yet.

- Next action: `ADVANCE_TO_CONFIRMATION`

## 2. Scoped certification reference

- Source: `PHASE_A5_SCOPED_DATA_CERTIFICATION.md`
- Hash: `78e079a08e2953d16ad8c5b43c9ad3b39383eee1b39a74eac9b6c8e542b26638`
- Scoped status: `READY_FOR_SCIENTIFIC_RERUN`
- Global trust: `OPERATIONAL_ONLY` (unchanged)
- Identity 29/29 VERIFIED · CA unresolved consecutive 0 · universe FIXED_PREREGISTERED_29 · sector static map · price unresolved rate 0.0

## 3. Snapshot / provenance

| Field | Value |
|---|---|
| snapshot_id | `a7a9828ec37e09e4` |
| manifest_checksum | `d38e458411754db1c3c63988e1152691153b81421c3a582b2b7c490a985404e0` |
| equity_sha256 | `9fd4550df76a23fdecd199058f49c1d17eda8d716d64bbef93b78e5f500ffc20` |
| securities | 29 |
| date range | 2023-08-23 → 2026-08-11 |
| sessions | 764 |
| oos_start (frozen) | 2025-09-19 |
| cost model | CNC round_trip_cost_pct = 0.32% |
| seed | 42 |
| git_sha | `e0be4a842c58b1e40dcbc00153d753bb15b71fd2` |
| protocol | `PHASE_A5_FROZEN_PROTOCOLS@2026-08-11` |
| frozen_protocols_sha256 | `710af91d5592766bd29a77cdd7a573d94fb141abb29439b5c808180a95cbd4c9` |
| rerun_result_hash | `f7f391940d261836` |
| evaluated_at | 2026-08-11T16:56:27.099255+00:00 |

## 4. EXP-A5-01 result

**Plain English:** We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Hypothesis ID: `81b8889792f53113`
- Internal verdict tag: `FAIL`
- Scientific verdict: **FAIL**
- Reason: no incremental future co-movement vs baselines
- Registry status: `REJECTED`
- Result hash: `99e219452c183da8`
- Production authority: `False`
- Next action: `REJECT`

### Technical evidence

```json
{
  "metrics": {
    "research_grade": 1,
    "stability_ari": 0.5426,
    "incremental_r2": -0.2181,
    "sector_baseline_score": 0.2323,
    "corr_cluster_baseline_score": 0.3856,
    "best_method": "hierarchical",
    "fdr_rejected_methods": [],
    "live_behaviour_changed": 0
  },
  "fdr": {
    "rejected": [],
    "detail": {
      "hierarchical": {
        "p": 0.9999876639322747,
        "rejected": false,
        "q": 0.9999991914165175
      },
      "kmeans": {
        "p": 0.9999991914165175,
        "rejected": false,
        "q": 0.9999991914165175
      },
      "pca_kmeans": {
        "p": 0.9999991914165175,
        "rejected": false,
        "q": 0.9999991914165175
      }
    },
    "threshold": 0.0
  },
  "baselines": {
    "sector_static": 0.23230105483156802,
    "correlation_clusters": 0.3855741766354674
  },
  "methods": {
    "hierarchical": {
      "future_comovement_score": 0.1675,
      "incremental_vs_best_baseline": -0.2181,
      "stability_ari_mean": 0.5426,
      "membership_turnover": 0.4574,
      "n_eval": 11,
      "p_incremental": 1.0
    },
    "kmeans": {
      "future_comovement_score": 0.1232,
      "incremental_vs_best_baseline": -0.2624,
      "stability_ari_mean": 0.4956,
      "membership_turnover": 0.5044,
      "n_eval": 11,
      "p_incremental": 1.0
    },
    "pca_kmeans": {
      "future_comovement_score": 0.1232,
      "incremental_vs_best_baseline": -0.2624,
      "stability_ari_mean": 0.4956,
      "membership_turnover": 0.5044,
      "n_eval": 11,
      "p_incremental": 1.0
    }
  }
}
```

## 5. EXP-A6-01 result

**Plain English:** We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Hypothesis ID: `590571a11ee06fc2`
- Internal verdict tag: `FAIL`
- Scientific verdict: **FAIL**
- Reason: no conditioned improvement vs pairwise/sector
- Registry status: `REJECTED`
- Result hash: `299d350f7beade6a`
- Production authority: `False`
- Next action: `REJECT`

### Technical evidence

```json
{
  "metrics": {
    "research_grade": 1,
    "incremental_auc_proxy": -0.0007471694565310493,
    "conditioned_improvement": -0.0007471694565310493,
    "partial_p": 0.9938333852431914,
    "fdr_rejected": [],
    "n_rows": 110,
    "live_behaviour_changed": 0,
    "auto_block": 0
  },
  "fdr": {
    "rejected": [],
    "detail": {
      "incr_risk": {
        "p": 0.5760096342556122,
        "rejected": false,
        "q": 0.6788255290315388
      },
      "degree": {
        "p": 0.45649826391069803,
        "rejected": false,
        "q": 0.6788255290315388
      },
      "eig": {
        "p": 0.09945830101554688,
        "rejected": false,
        "q": 0.3481040535544141
      },
      "btw": {
        "p": 0.5818504534556047,
        "rejected": false,
        "q": 0.6788255290315388
      },
      "net_conc": {
        "p": 0.0646015457930327,
        "rejected": false,
        "q": 0.3481040535544141
      },
      "joins": {
        "p": 0.5760096342556118,
        "rejected": false,
        "q": 0.6788255290315388
      },
      "incr_risk_partial": {
        "p": 0.9938333852431914,
        "rejected": false,
        "q": 0.9938333852431914
      }
    },
    "threshold": 0.0
  },
  "partial_incr_risk": {
    "r": -0.0007471694565310493,
    "p": 0.9938333852431914
  },
  "feature_stats": {
    "incr_risk": {
      "corr_cand_loss": -0.02,
      "p_cand_loss": 0.836,
      "corr_sim_loss": -0.054,
      "p_sim_loss": 0.576
    },
    "degree": {
      "corr_cand_loss": -0.0056,
      "p_cand_loss": 0.9534,
      "corr_sim_loss": 0.0719,
      "p_sim_loss": 0.4565
    },
    "eig": {
      "corr_cand_loss": -0.0431,
      "p_cand_loss": 0.6555,
      "corr_sim_loss": -0.1579,
      "p_sim_loss": 0.0995
    },
    "btw": {
      "corr_cand_loss": -0.0364,
      "p_cand_loss": 0.7067,
      "corr_sim_loss": 0.0532,
      "p_sim_loss": 0.5819
    },
    "net_conc": {
      "corr_cand_loss": 0.1768,
      "p_cand_loss": 0.0646,
      "corr_sim_loss": -0.106,
      "p_sim_loss": 0.2711
    },
    "joins": {
      "corr_cand_loss": -0.02,
      "p_cand_loss": 0.836,
      "corr_sim_loss": -0.054,
      "p_sim_loss": 0.576
    }
  }
}
```

## 6. EXP-A2-01 result

**Plain English:** We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Hypothesis ID: `775b4a0fce7d5b83`
- Internal verdict tag: `FAIL`
- Scientific verdict: **FAIL**
- Reason: no positive net edge in preregistered family
- Registry status: `REJECTED`
- Result hash: `2abdc4fbb1ce4466`
- Production authority: `False`
- Next action: `REJECT`

### Technical evidence

```json
{
  "metrics": {
    "research_grade": 1,
    "best_dsr": 0.248,
    "best_horizon": "5d",
    "fdr_any_horizon": 0,
    "n_trials": 4,
    "live_behaviour_changed": 0
  },
  "fdr": {
    "rejected": [],
    "detail": {
      "5d": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "10d": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "22d": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      },
      "66d": {
        "p": 1.0,
        "rejected": false,
        "q": 1.0
      }
    },
    "threshold": 0.0
  },
  "horizons": {
    "5d": {
      "verdict": "REJECT",
      "n": 227,
      "n_eff": 51.55711678061902,
      "mean_r": -0.0013,
      "sharpe": -0.0457,
      "psr": 0.248,
      "dsr": 0.248,
      "p_value": 0.7541350197163716,
      "insight": "No edge \u2014 -0.00R over 227 trades. Not worth acting on.",
      "bars": 5,
      "cost_drag": 0.0032,
      "turnover_proxy": 1.0,
      "n_oos": 227,
      "mean_gross": 0.0019,
      "mean_net": -0.0013,
      "expectancy": -0.0013,
      "ci_95": [
        -0.0052,
        0.0025
      ],
      "profit_factor": 0.8847,
      "max_drawdown_R": -0.6912,
      "alpha_proxy_mean_net": -0.0013,
      "statistical_verdict": "REJECT",
      "economic_verdict": "NON_POSITIVE"
    },
    "10d": {
      "verdict": "REJECT",
      "n": 222,
      "n_eff": 29.994835783654242,
      "mean_r": -0.0028,
      "sharpe": -0.0681,
      "psr": 0.1558,
      "dsr": 0.1558,
      "p_value": 0.8442342754950617,
      "insight": "No edge \u2014 -0.00R over 222 trades. Not worth acting on.",
      "bars": 10,
      "cost_drag": 0.0032,
      "turnover_proxy": 1.0,
      "n_oos": 222,
      "mean_gross": 0.0004,
      "mean_net": -0.0028,
      "expectancy": -0.0028,
      "ci_95": [
        -0.0081,
        0.0026
      ],
      "profit_factor": 0.8333,
      "max_drawdown_R": -1.0499,
      "alpha_proxy_mean_net": -0.0028,
      "statistical_verdict": "REJECT",
      "economic_verdict": "NON_POSITIVE"
    },
    "22d": {
      "verdict": "REJECT",
      "n": 210,
      "n_eff": 19.4575475269436,
      "mean_r": -0.0119,
      "sharpe": -0.2158,
      "psr": 0.0008,
      "dsr": 0.0008,
      "p_value": 0.9989931494049857,
      "insight": "No edge \u2014 -0.01R over 210 trades. Not worth acting on.",
      "bars": 22,
      "cost_drag": 0.0032,
      "turnover_proxy": 1.0,
      "n_oos": 210,
      "mean_gross": -0.0087,
      "mean_net": -0.0119,
      "expectancy": -0.0119,
      "ci_95": [
        -0.0193,
        -0.0044
      ],
      "profit_factor": 0.552,
      "max_drawdown_R": -2.6913,
      "alpha_proxy_mean_net": -0.0119,
      "statistical_verdict": "REJECT",
      "economic_verdict": "NON_POSITIVE"
    },
    "66d": {
      "verdict": "REJECT",
      "n": 166,
      "n_eff": 16.666329076308894,
      "mean_r": -0.0076,
      "sharpe": -0.0902,
      "psr": 0.1183,
      "dsr": 0.1183,
      "p_value": 0.8765107351976639,
      "insight": "No edge \u2014 -0.01R over 166 trades. Not worth acting on.",
      "bars": 66,
      "cost_drag": 0.0032,
      "turnover_proxy": 1.0,
      "n_oos": 166,
      "mean_gross": -0.0044,
      "mean_net": -0.0076,
      "expectancy": -0.0076,
      "ci_95": [
        -0.0205,
        0.0053
      ],
      "profit_factor": 0.7973,
      "max_drawdown_R": -3.8684,
      "alpha_proxy_mean_net": -0.0076,
      "statistical_verdict": "REJECT",
      "economic_verdict": "NON_POSITIVE"
    }
  }
}
```

## 7. EXP-A3-01 result

**Plain English:** We tested this idea again using verified historical NSE data. It still did not show a reliable advantage, so QuantTerm should not use it.

- Hypothesis ID: `7842a46ee335685a`
- Internal verdict tag: `FAIL`
- Scientific verdict: **FAIL**
- Reason: no incremental economic value vs rank incumbent
- Registry status: `REJECTED`
- Result hash: `5da065eb79acc8f1`
- Production authority: `False`
- Next action: `REJECT`

### Technical evidence

```json
{
  "metrics": {
    "research_grade": 1,
    "economic_value_delta": -0.06997,
    "challenger_harness_promote": 0,
    "vs_naive_verdict": "FAIL",
    "vs_rank_verdict": "FAIL",
    "pred_corr_vs_rank": -0.129,
    "live_behaviour_changed": 0
  },
  "vs_naive_verdict": "FAIL",
  "vs_rank_verdict": "FAIL",
  "economic_value_delta": -0.06997
}
```

## 8. EXP-A5A6-01 result

**Plain English:** This idea showed a reliable advantage in the frozen historical test. It is worth further validation, but it is not approved for real trading yet.

- Hypothesis ID: `3734b8a0a9124a60`
- Internal verdict tag: `PASS_RISK`
- Scientific verdict: **PASS**
- Reason: FDR-significant interactions: ['signal_x_network_concentration']
- Registry status: `PROMOTED`
- Result hash: `d5371f08a271c4bb`
- Production authority: `False`
- Next action: `ADVANCE_TO_CONFIRMATION`

### Technical evidence

```json
{
  "metrics": {
    "research_grade": 1,
    "n_fdr_interactions": 1,
    "n_rows": 319,
    "live_behaviour_changed": 0
  },
  "fdr": {
    "rejected": [
      "signal_x_network_concentration"
    ],
    "detail": {
      "signal_x_cluster_stability": {
        "p": 0.1209,
        "rejected": false,
        "q": 0.18134999999999998
      },
      "signal_x_network_concentration": {
        "p": 0.0091,
        "rejected": true,
        "q": 0.0273
      },
      "signal_x_incremental_community_risk": {
        "p": 0.6385,
        "rejected": false,
        "q": 0.6385
      }
    },
    "threshold": 0.0091
  },
  "interactions": {
    "signal_x_cluster_stability": {
      "corr_low": 0.0379,
      "corr_high": 0.2108,
      "delta_corr": 0.1729,
      "p": 0.1209,
      "n_low": 174,
      "n_high": 145
    },
    "signal_x_network_concentration": {
      "corr_low": 0.0368,
      "corr_high": 0.4005,
      "delta_corr": 0.3637,
      "p": 0.0091,
      "n_low": 261,
      "n_high": 58
    },
    "signal_x_incremental_community_risk": {
      "corr_low": 0.1081,
      "corr_high": 0.0,
      "delta_corr": -0.1081,
      "p": 0.6385,
      "n_low": 296,
      "n_high": 23
    }
  }
}
```

## 9. DISPLAY_ONLY vs certified comparison

| EXPERIMENT | DISPLAY_ONLY | CERTIFIED (raw) | SCIENTIFIC | DIRECTION CHANGED |
|---|---|---|---|---|
| EXP-A5-01 | INCONCLUSIVE | FAIL | FAIL | True |
| EXP-A6-01 | INCONCLUSIVE | FAIL | FAIL | True |
| EXP-A2-01 | INCONCLUSIVE | FAIL | FAIL | True |
| EXP-A3-01 | INCONCLUSIVE | FAIL | FAIL | True |
| EXP-A5A6-01 | INCONCLUSIVE | PASS_RISK | PASS | True |

Disagreement with the exploratory DISPLAY_ONLY result is not treated as an error — the point of this rerun is to see what survives certified data.

## 10. Statistical evidence

- **EXP-A5-01**: scientific=`FAIL`; FDR rejected=`[]`; metrics=`{"research_grade": 1, "stability_ari": 0.5426, "incremental_r2": -0.2181, "sector_baseline_score": 0.2323, "corr_cluster_baseline_score": 0.3856, "best_method": "hierarchical", "fdr_rejected_methods": [], "live_behaviour_changed": 0}`
- **EXP-A6-01**: scientific=`FAIL`; FDR rejected=`[]`; metrics=`{"research_grade": 1, "incremental_auc_proxy": -0.0007471694565310493, "conditioned_improvement": -0.0007471694565310493, "partial_p": 0.9938333852431914, "fdr_rejected": [], "n_rows": 110, "live_behaviour_changed": 0, "auto_block": 0}`
- **EXP-A2-01**: scientific=`FAIL`; FDR rejected=`[]`; metrics=`{"research_grade": 1, "best_dsr": 0.248, "best_horizon": "5d", "fdr_any_horizon": 0, "n_trials": 4, "live_behaviour_changed": 0}`
- **EXP-A3-01**: scientific=`FAIL`; FDR rejected=`None`; metrics=`{"research_grade": 1, "economic_value_delta": -0.06997, "challenger_harness_promote": 0, "vs_naive_verdict": "FAIL", "vs_rank_verdict": "FAIL", "pred_corr_vs_rank": -0.129, "live_behaviour_changed": 0}`
- **EXP-A5A6-01**: scientific=`PASS`; FDR rejected=`['signal_x_network_concentration']`; metrics=`{"research_grade": 1, "n_fdr_interactions": 1, "n_rows": 319, "live_behaviour_changed": 0}`

## 11. Economic evidence

- **EXP-A5-01**: economic label `NO_INCREMENT`
- **EXP-A6-01**: economic label `NO_CONDITIONED_IMPROVEMENT`
- **EXP-A2-01**: economic label `NON_POSITIVE_FAMILY`
- **EXP-A3-01**: economic label `NO_INCREMENTAL_VALUE`
- **EXP-A5A6-01**: economic label `FDR_INTERACTION`

Horizon family (cost-aware OOS):

| Horizon | n | n_eff | expectancy | CI95 | PF | Sharpe | DD | cost_drag | stat | econ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 5d | 227 | 51.55711678061902 | -0.0013 | [-0.0052, 0.0025] | 0.8847 | -0.0457 | -0.6912 | 0.0032 | REJECT | NON_POSITIVE |
| 10d | 222 | 29.994835783654242 | -0.0028 | [-0.0081, 0.0026] | 0.8333 | -0.0681 | -1.0499 | 0.0032 | REJECT | NON_POSITIVE |
| 22d | 210 | 19.4575475269436 | -0.0119 | [-0.0193, -0.0044] | 0.552 | -0.2158 | -2.6913 | 0.0032 | REJECT | NON_POSITIVE |
| 66d | 166 | 16.666329076308894 | -0.0076 | [-0.0205, 0.0053] | 0.7973 | -0.0902 | -3.8684 | 0.0032 | REJECT | NON_POSITIVE |

## 12. Cost impact

- Frozen cost model: CNC `round_trip_cost_pct` = **0.32%**
- EXP-A2-01 applies conservative ~100% one-way turnover cost drag per rebalance.
- EXP-A3-01 bake-off uses the same cost-aware economic delta machinery as frozen.

## 13. Multiple-testing / FDR

- **EXP-A5-01**: rejected=[]; detail keys=['hierarchical', 'kmeans', 'pca_kmeans']
- **EXP-A6-01**: rejected=[]; detail keys=['incr_risk', 'degree', 'eig', 'btw', 'net_conc', 'joins', 'incr_risk_partial']
- **EXP-A2-01**: rejected=[]; detail keys=['5d', '10d', '22d', '66d']
- **EXP-A3-01**: rejected=None; detail keys=[]
- **EXP-A5A6-01**: rejected=['signal_x_network_concentration']; detail keys=['signal_x_cluster_stability', 'signal_x_network_concentration', 'signal_x_incremental_community_risk']

## 14. Positive evidence

- `EXP-A5A6-01` → PASS → `ADVANCE_TO_CONFIRMATION`

## 15. Negative evidence

- `EXP-A5-01` → FAIL → `REJECT` (recorded in scientific memory; do not escalate model complexity)
- `EXP-A6-01` → FAIL → `REJECT` (recorded in scientific memory; do not escalate model complexity)
- `EXP-A2-01` → FAIL → `REJECT` (recorded in scientific memory; do not escalate model complexity)
- `EXP-A3-01` → FAIL → `REJECT` (recorded in scientific memory; do not escalate model complexity)

## 16. Inconclusive evidence

None.

## 17. Scientific-memory updates

FAIL outcomes were written as negative evidence; INCONCLUSIVE as WATCH beliefs in the isolated Phase A.5 scientific memory DB (`logs/phase_a5/scientific_memory.db`). Registry results updated on the frozen hypothesis IDs in `logs/phase_a5/experiments.db`.

## 18. Reproducibility

```
snapshot_id=a7a9828ec37e09e4
protocol_version=PHASE_A5_FROZEN_PROTOCOLS@2026-08-11
git_sha=e0be4a842c58b1e40dcbc00153d753bb15b71fd2
seed=42
cost_pct=0.32
oos_start=2025-09-19
rerun_result_hash=f7f391940d261836
runner=python -m research.phase_a5.scientific_rerun
```

## 19. Production behaviour unchanged

- `production_behaviour_changed`: `False`
- `phase_b_started`: `False`
- Brain / CycleContext / ranking / portfolio authority / risk limits / execution / broker / live signals: **not modified**.
- Even PASS only means eligible for confirmation review, not live use.

## 20. What QuantTerm learned (plain English)

We re-checked five frozen research ideas with cleaner, verified history for this specific test panel. The full market database is still not certified overall. For these five ideas:

- **EXP-A5-01**: still no reliable advantage — do not use it.
- **EXP-A6-01**: still no reliable advantage — do not use it.
- **EXP-A2-01**: still no reliable advantage — do not use it.
- **EXP-A3-01**: still no reliable advantage — do not use it.
- **EXP-A5A6-01**: showed a reliable historical advantage — worth further confirmation, not live trading yet.

---

## Final matrix

| CAPABILITY | DISPLAY_ONLY RESULT | CERTIFIED RESULT | STATISTICAL VERDICT | ECONOMIC VERDICT | FINAL SCIENTIFIC VERDICT | NEXT ACTION |
|---|---|---|---|---|---|---|
| EXP-A5-01 | INCONCLUSIVE | FAIL | FAIL | NO_INCREMENT | **FAIL** | `REJECT` |
| EXP-A6-01 | INCONCLUSIVE | FAIL | FAIL | NO_CONDITIONED_IMPROVEMENT | **FAIL** | `REJECT` |
| EXP-A2-01 | INCONCLUSIVE | FAIL | FAIL | NON_POSITIVE_FAMILY | **FAIL** | `REJECT` |
| EXP-A3-01 | INCONCLUSIVE | FAIL | FAIL | NO_INCREMENTAL_VALUE | **FAIL** | `REJECT` |
| EXP-A5A6-01 | INCONCLUSIVE | PASS_RISK | PASS | FDR_INTERACTION | **PASS** | `ADVANCE_TO_CONFIRMATION` |

STOP. Do not begin Phase B. Do not implement production changes from PASS. Do not escalate models from FAIL.
