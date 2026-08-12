# Phase A.5 Frozen Experiment Protocols

**Status:** FROZEN — do not alter hypotheses, horizons, benchmarks,
transaction-cost assumptions, success criteria, or multiple-testing treatment
based on DISPLAY_ONLY exploratory results.

**Frozen at:** 2026-08-11
**Source DB:** `logs/phase_a5/experiments.db`

Phase A.5 experiment protocols are FROZEN. Hypotheses, horizons, benchmarks, transaction-cost assumptions, success criteria, and multiple-testing treatment must not be altered based on DISPLAY_ONLY exploratory results. Research-grade rerun must reuse these definitions.

## Hypothesis IDs

- `81b8889792f53113` — EXP-A5-01 — 81b8889792f53113
- `590571a11ee06fc2` — EXP-A6-01 — 590571a11ee06fc2
- `775b4a0fce7d5b83` — EXP-A2-01 — 775b4a0fce7d5b83
- `7842a46ee335685a` — EXP-A3-01 — 7842a46ee335685a
- `3734b8a0a9124a60` — EXP-A5A6-01 — 3734b8a0a9124a60

## Canonical protocol dump (authoritative)

```json
{
  "experiments": {
    "3734b8a0a9124a60": {
      "code_hash": "phase_a5",
      "data_window_template_keys": [
        "interactions",
        "research_grade",
        "response",
        "signal",
        "snapshot_id",
        "trust_class"
      ],
      "evaluated_at": "2026-08-11T15:56:53",
      "experiment_id": "EXP-A5A6-01",
      "hypothesis": "Market-structure stability and network concentration modulate momentum payoffs as context (interaction), rather than as standalone alpha.",
      "hypothesis_id": "3734b8a0a9124a60",
      "name": "EXP-A5A6-01:Market-structure stability and network concentration modulate momentum payoffs a",
      "null_hypothesis": "Preregistered interactions have no FDR-significant effect on momentum forward returns after multiple-testing control.",
      "protocol": {
        "context_not_standalone_alpha": true,
        "interactions": [
          "signal_x_cluster_stability",
          "signal_x_network_concentration",
          "signal_x_incremental_community_risk"
        ],
        "known_limitations": [
          "DISPLAY_ONLY panel"
        ],
        "multiple_testing": "BH-FDR across preregistered interactions",
        "no_unrestricted_feature_mining": true
      },
      "registered_at": "2026-08-11T15:56:53",
      "registered_data_window": {
        "interactions": [
          "signal_x_cluster_stability",
          "signal_x_network_concentration",
          "signal_x_incremental_community_risk"
        ],
        "research_grade": false,
        "response": "10d_forward_return",
        "signal": "60d_momentum_rank",
        "snapshot_id": "050c77ea71b73001",
        "trust_class": "DISPLAY_ONLY"
      },
      "seed": 42,
      "status_at_freeze": "REJECTED",
      "success_criteria": {
        "n_fdr_interactions": {
          "gte": 1
        },
        "research_grade": {
          "eq": 1
        }
      }
    },
    "590571a11ee06fc2": {
      "code_hash": "phase_a5",
      "data_window_template_keys": [
        "lookback",
        "oos_start",
        "research_grade",
        "snapshot_id",
        "trust_class",
        "universe"
      ],
      "evaluated_at": "2026-08-11T15:56:46",
      "experiment_id": "EXP-A6-01",
      "hypothesis": "Correlation-graph network metrics (community concentration, centrality, incremental community risk) identify future correlated portfolio losses beyond pairwise correlation clusters and sector caps.",
      "hypothesis_id": "590571a11ee06fc2",
      "name": "EXP-A6-01:Correlation-graph network metrics (community concentration, centrality, incremen",
      "null_hypothesis": "Network metrics add no incremental explanatory power for simultaneous losses / drawdowns after conditioning on pairwise \u03c1 clusters and sectors.",
      "protocol": {
        "auto_block": false,
        "baseline": [
          "pairwise_corr_clusters",
          "sector_herfindahl"
        ],
        "challenger": [
          "community_exposure",
          "degree",
          "eigenvector",
          "betweenness",
          "network_concentration",
          "incremental_community_risk"
        ],
        "failure_criterion": "no improvement after conditioning on pairwise/sector",
        "known_limitations": [
          "advisory evaluation only \u2014 no trade blocking",
          "DISPLAY_ONLY / non-RESEARCH_GRADE inputs"
        ],
        "multiple_testing": "BH-FDR across network feature predictors",
        "primary_metric": "incremental explanatory power for simultaneous loss events",
        "success_criterion": "research_grade + positive conditioned improvement",
        "transaction_costs_pct": 0.32
      },
      "registered_at": "2026-08-11T15:56:46",
      "registered_data_window": {
        "lookback": 60,
        "oos_start": "2025-09-19",
        "research_grade": false,
        "snapshot_id": "050c77ea71b73001",
        "trust_class": "DISPLAY_ONLY",
        "universe": [
          "RELIANCE",
          "ONGC",
          "BPCL",
          "TCS",
          "INFY",
          "WIPRO",
          "HCLTECH",
          "HDFCBANK",
          "ICICIBANK",
          "SBIN",
          "KOTAKBANK",
          "AXISBANK",
          "ITC",
          "HINDUNILVR",
          "NESTLEIND",
          "SUNPHARMA",
          "DRREDDY",
          "CIPLA",
          "M&M",
          "MARUTI",
          "TATASTEEL",
          "JSWSTEEL",
          "HINDALCO",
          "NTPC",
          "POWERGRID",
          "LT",
          "ADANIENT",
          "BAJFINANCE",
          "BAJAJFINSV"
        ]
      },
      "seed": 42,
      "status_at_freeze": "REJECTED",
      "success_criteria": {
        "conditioned_improvement": {
          "gt": 0.0
        },
        "incremental_auc_proxy": {
          "gt": 0.0
        },
        "research_grade": {
          "eq": 1
        }
      }
    },
    "775b4a0fce7d5b83": {
      "code_hash": "phase_a5",
      "data_window_template_keys": [
        "cost_pct",
        "horizons",
        "oos_start",
        "research_grade",
        "snapshot_id",
        "strategy",
        "train_end",
        "trust_class"
      ],
      "evaluated_at": "2026-08-11T15:56:51",
      "experiment_id": "EXP-A2-01",
      "hypothesis": "Cross-sectional momentum has a preferred economic horizon within the preregistered set ('5d', '10d', '22d', '66d'); effect is not horizon-invariant.",
      "hypothesis_id": "775b4a0fce7d5b83",
      "name": "EXP-A2-01:Cross-sectional momentum has a preferred economic horizon within the preregister",
      "null_hypothesis": "No horizon in the preregistered family shows a positive cost-aware edge after multiple-testing control; or all are indistinguishable from noise.",
      "protocol": {
        "entry": "close",
        "exit": "close",
        "known_limitations": [
          "DISPLAY_ONLY panel"
        ],
        "multiple_testing": "BH-FDR across 4 horizons; n_trials=4 in DSR",
        "no_post_hoc_horizon_selection_for_success": true,
        "portfolio": "long top 20% / short bottom 20% equal weight",
        "primary_metric": "deflated Sharpe / harness verdict on OOS R stream",
        "secondary_metrics": [
          "mean_r",
          "sharpe",
          "n_eff",
          "cost_drag",
          "turnover_proxy"
        ],
        "signal": "60d return cross-sectional rank"
      },
      "registered_at": "2026-08-11T15:56:46",
      "registered_data_window": {
        "cost_pct": 0.32,
        "horizons": [
          "5d",
          "10d",
          "22d",
          "66d"
        ],
        "oos_start": "2025-09-19",
        "research_grade": false,
        "snapshot_id": "050c77ea71b73001",
        "strategy": "cross_sectional_momentum_60d_rank",
        "train_end": "2025-09-18",
        "trust_class": "DISPLAY_ONLY"
      },
      "seed": 42,
      "status_at_freeze": "REJECTED",
      "success_criteria": {
        "best_dsr": {
          "gte": 0.95
        },
        "fdr_any_horizon": {
          "eq": 1
        },
        "research_grade": {
          "eq": 1
        }
      }
    },
    "7842a46ee335685a": {
      "code_hash": "phase_a5",
      "data_window_template_keys": [
        "cost_pct",
        "features",
        "n_rows",
        "research_grade",
        "snapshot_id",
        "target",
        "trust_class"
      ],
      "evaluated_at": "2026-08-11T15:56:53",
      "experiment_id": "EXP-A3-01",
      "hypothesis": "A simple logistic regression on QuantTerm-style momentum/vol features extracts incremental OOS economic value over naive and rank-rule incumbents.",
      "hypothesis_id": "7842a46ee335685a",
      "name": "EXP-A3-01:A simple logistic regression on QuantTerm-style momentum/vol features extracts i",
      "null_hypothesis": "Logistic challenger does not improve cost-aware OOS expectancy vs naive/rank incumbents after evidence gating.",
      "protocol": {
        "challenger": "logistic_regression",
        "identical_splits": true,
        "incumbents": [
          "naive_baseline",
          "momentum_rank_sign"
        ],
        "known_limitations": [
          "DISPLAY_ONLY panel"
        ],
        "multiple_testing": "single primary challenger; n_trials=1 vs each incumbent",
        "no_deep_learning": true,
        "primary_metric": "economic_value_delta (mean OOS R)"
      },
      "registered_at": "2026-08-11T15:56:53",
      "registered_data_window": {
        "cost_pct": 0.32,
        "features": [
          "mom_5",
          "mom_10",
          "mom_20",
          "mom_60",
          "vol_20"
        ],
        "n_rows": 19488,
        "research_grade": false,
        "snapshot_id": "050c77ea71b73001",
        "target": "10d classification (+1/-1/0 at \u00b11%)",
        "trust_class": "DISPLAY_ONLY"
      },
      "seed": 42,
      "status_at_freeze": "REJECTED",
      "success_criteria": {
        "challenger_harness_promote": {
          "eq": 1
        },
        "economic_value_delta": {
          "gt": 0.0
        },
        "research_grade": {
          "eq": 1
        }
      }
    },
    "81b8889792f53113": {
      "code_hash": "phase_a5",
      "data_window_template_keys": [
        "date_end",
        "date_start",
        "lookback",
        "n_symbols",
        "oos_start",
        "research_grade",
        "snapshot_id",
        "trust_class",
        "universe"
      ],
      "evaluated_at": "2026-08-11T15:56:46",
      "experiment_id": "EXP-A5-01",
      "hypothesis": "Dynamically discovered market structure (hierarchical/k-means/PCA) provides stable incremental future co-movement information beyond static sectors and pairwise correlation clusters.",
      "hypothesis_id": "81b8889792f53113",
      "name": "EXP-A5-01:Dynamically discovered market structure (hierarchical/k-means/PCA) provides stab",
      "null_hypothesis": "Discovered clusters add no incremental future co-movement information and/or are unstable relative to sector and correlation baselines.",
      "protocol": {
        "benchmark": "static NSE sector map + correlation clusters",
        "failure_criterion": {
          "or_incremental_r2": {
            "lte": 0.0
          },
          "stability_ari": {
            "lt": 0.1
          }
        },
        "known_limitations": [
          "DISPLAY_ONLY yfinance panel",
          "no CA ledger",
          "no PIT sector history",
          "survivorship-biased universe"
        ],
        "methods": [
          "sector_static",
          "correlation_clusters",
          "hierarchical",
          "kmeans",
          "pca_kmeans"
        ],
        "multiple_testing": "BH-FDR across methods on incremental_r2 p-proxy",
        "no_cluster_tuning_on_future_returns": true,
        "primary_metric": "incremental_future_comovement_r2_vs_sector",
        "secondary_metrics": [
          "cluster_stability_ari",
          "membership_turnover",
          "sector_overlap_ari",
          "diversification_herfindahl_improvement"
        ],
        "structure_discovery_separated_from_eval": true,
        "success_criterion": {
          "incremental_r2": {
            "gt": 0.0
          },
          "research_grade": 1,
          "stability_ari": {
            "gte": 0.3
          }
        },
        "transaction_costs_pct": 0.32
      },
      "registered_at": "2026-08-11T15:56:46",
      "registered_data_window": {
        "date_end": "2026-08-11",
        "date_start": "2023-08-11",
        "lookback": 60,
        "n_symbols": 29,
        "oos_start": "2025-09-19",
        "research_grade": false,
        "snapshot_id": "050c77ea71b73001",
        "trust_class": "DISPLAY_ONLY",
        "universe": [
          "RELIANCE",
          "ONGC",
          "BPCL",
          "TCS",
          "INFY",
          "WIPRO",
          "HCLTECH",
          "HDFCBANK",
          "ICICIBANK",
          "SBIN",
          "KOTAKBANK",
          "AXISBANK",
          "ITC",
          "HINDUNILVR",
          "NESTLEIND",
          "SUNPHARMA",
          "DRREDDY",
          "CIPLA",
          "M&M",
          "MARUTI",
          "TATASTEEL",
          "JSWSTEEL",
          "HINDALCO",
          "NTPC",
          "POWERGRID",
          "LT",
          "ADANIENT",
          "BAJFINANCE",
          "BAJAJFINSV"
        ]
      },
      "seed": 42,
      "status_at_freeze": "REJECTED",
      "success_criteria": {
        "incremental_r2": {
          "gt": 0.0
        },
        "research_grade": {
          "eq": 1
        },
        "stability_ari": {
          "gte": 0.3
        }
      }
    }
  },
  "freeze_statement": "Phase A.5 experiment protocols are FROZEN. Hypotheses, horizons, benchmarks, transaction-cost assumptions, success criteria, and multiple-testing treatment must not be altered based on DISPLAY_ONLY exploratory results. Research-grade rerun must reuse these definitions.",
  "frozen_at": "2026-08-11",
  "source_db": "logs/phase_a5/experiments.db"
}
```

