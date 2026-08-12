"""Phase A.6 — freeze rejected A.5 branches + surviving confirmation hypothesis.

Created BEFORE confirmation evaluation. Do not edit after seeing confirmation results.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Rejected discovery branches (Phase A.5 certified scientific rerun)
# ---------------------------------------------------------------------------
REJECTED_BRANCHES = [
    {
        "experiment_id": "EXP-A5-01",
        "hypothesis_id": "81b8889792f53113",
        "branch": "dynamic_market_structure_incrementality",
        "scientific_verdict": "FAIL",
        "next_action": "REJECT",
        "freeze": "Do not retune, add clustering algorithms, or re-open.",
    },
    {
        "experiment_id": "EXP-A6-01",
        "hypothesis_id": "590571a11ee06fc2",
        "branch": "standalone_portfolio_network_predictability",
        "scientific_verdict": "FAIL",
        "next_action": "REJECT",
        "freeze": "Do not add graph complexity or re-open as alpha.",
    },
    {
        "experiment_id": "EXP-A2-01",
        "hypothesis_id": "775b4a0fce7d5b83",
        "branch": "multi_horizon_momentum_family",
        "scientific_verdict": "FAIL",
        "next_action": "REJECT",
        "freeze": "Do not add horizons or escalate models to rescue momentum.",
    },
    {
        "experiment_id": "EXP-A3-01",
        "hypothesis_id": "7842a46ee335685a",
        "branch": "simple_logistic_challenger",
        "scientific_verdict": "FAIL",
        "next_action": "REJECT",
        "freeze": "Do not escalate to RF/GBM/XGBoost/DL/ensembles.",
    },
]

# ---------------------------------------------------------------------------
# Sole surviving discovery finding (EXP-A5A6-01)
# ---------------------------------------------------------------------------
DISCOVERY = {
    "experiment_id": "EXP-A5A6-01",
    "hypothesis_id": "3734b8a0a9124a60",
    "snapshot_id": "a7a9828ec37e09e4",
    "scientific_verdict": "PASS",
    "raw_verdict": "PASS_RISK",
    "surviving_interaction": "signal_x_network_concentration",
    "fdr_rejected": ["signal_x_network_concentration"],
    "discovery_delta_corr": 0.3637,
    "discovery_p": 0.0091,
    "discovery_corr_low": 0.0368,
    "discovery_corr_high": 0.4005,
    "discovery_n_low": 261,
    "discovery_n_high": 58,
    "discovery_n_rows": 319,
    "discovery_eval_date_first": "2025-09-19",
    "discovery_eval_date_last": "2026-07-13",
    "oos_start": "2025-09-19",
    "interpretation": (
        "PASS_RISK / context effect only — NOT standalone return prediction. "
        "Confirmation asks whether elevated portfolio-network concentration "
        "worsens the risk profile of otherwise comparable signals."
    ),
}

# ---------------------------------------------------------------------------
# Frozen confirmation protocol (locked before confirmation evaluation)
# ---------------------------------------------------------------------------
CONFIRMATION_PROTOCOL = {
    "experiment_id": "EXP-A6-CONF-01",
    "name": "Independent confirmation of signal_x_network_concentration (risk/context)",
    "parent_discovery_id": "3734b8a0a9124a60",
    "do_not_overwrite": "EXP-A5A6-01",
    "signal": {
        "name": "60d_momentum_rank",
        "definition": (
            "Cross-sectional percentile rank of 60-session close-to-close return "
            "across the frozen 29-name panel (research.phase_a5.metrics."
            "cross_sectional_momentum_scores, lookback=60)."
        ),
    },
    "network_concentration": {
        "name": "portfolio_network_concentration",
        "definition": (
            "Herfindahl index of equal-weighted portfolio community weights from "
            "correlation-graph communities at rho>=0.70 over the prior 60 sessions; "
            "portfolio = top-5 names by 60d momentum rank on the evaluation date "
            "(research.portfolio_network.analyze_network)."
        ),
        "rho_threshold": 0.70,
        "lookback": 60,
        "portfolio_rule": "top_5_by_60d_momentum_rank",
        "no_post_hoc_threshold": (
            "Do NOT invent network_concentration > X from confirmation data. "
            "Within-sample median split only (same as discovery)."
        ),
    },
    "interaction": {
        "name": "signal_x_network_concentration",
        "formula": (
            "Split observations by within-sample median of portfolio_network_"
            "concentration; compute corr(signal, 10d_forward_return) in low vs "
            "high halves; delta_corr = corr_high - corr_low; Fisher-z two-sided "
            "test identical to EXP-A5A6-01."
        ),
        "continuous": True,
        "split": "within_sample_median",
        "expected_direction": {
            "primary_stat": "delta_corr > 0",
            "note": (
                "Discovery observed corr_high > corr_low (delta_corr=+0.3637). "
                "Confirmation requires the same directional statistical effect."
            ),
        },
    },
    "outcome_variable": {
        "name": "10d_forward_return",
        "definition": "close.pct_change(10).shift(-10) on the certified panel",
        "bars": 10,
    },
    "horizons": {"primary": "10d", "lookback": 60, "step": 21},
    "portfolio_assumptions": {
        "universe": "FIXED_PREREGISTERED_29",
        "panel_snapshot": "a7a9828ec37e09e4",
        "network_portfolio": "top_5_momentum",
        "equal_weight_communities": True,
    },
    "cost_assumptions": {
        "primary_interaction_test": "not cost-adjusted (matches discovery)",
        "opportunity_cost_layer": "CNC round_trip_cost_pct reporting only",
    },
    "statistical_test": {
        "name": "fisher_z_delta_corr_two_sided",
        "alpha": 0.05,
        "min_per_split": 30,
        "multiple_testing": (
            "Single pre-specified survivor — no family FDR. "
            "Original FDR treated three interactions; only this survivor advances."
        ),
        "original_fdr_treatment": "BH-FDR alpha=0.05 across three interactions in discovery",
    },
    "effect_size_criterion": {
        "statistical_replication": "delta_corr > 0 and p < 0.05",
        "economic_risk_meaning": {
            "cohort": "signal_rank >= 0.5 (signal-present / above-median signal)",
            "require_any": [
                {
                    "metric": "loss_rate_gap",
                    "def": "P(fwd<0|high_conc) - P(fwd<0|low_conc)",
                    "gte": 0.05,
                },
                {
                    "metric": "left_tail_gap",
                    "def": (
                        "5th percentile of fwd in high_conc minus 5th percentile "
                        "in low_conc (more negative = worse)"
                    ),
                    "lte": -0.01,
                },
            ],
            "rationale": (
                "A risk/context finding is useful only if elevated concentration "
                "is associated with materially worse subsequent outcomes for "
                "comparable signals — not merely a correlational curiosity."
            ),
        },
    },
    "incrementality": {
        "controls": [
            "pairwise_corr_cluster_concentration_of_top5",
            "sector_herfindahl_of_top5",
        ],
        "rule": (
            "After residualizing signal and fwd on controls (or stratified "
            "partial association), the net_conc interaction delta_corr must "
            "remain > 0 with p < 0.05. If not, FAIL — simpler incumbents win."
        ),
    },
    "success_fail_rules": {
        "CONFIRMED": (
            "statistical_replication AND economic_risk_meaning AND incrementality"
        ),
        "FAILED_CONFIRMATION": (
            "primary delta_corr <= 0 OR p >= 0.05 OR incrementality fails "
            "(effect explained by existing pairwise/sector controls)"
        ),
        "INCONCLUSIVE": (
            "statistical replication holds but economic risk meaning fails, "
            "OR sample underpowered (either split n < 30)"
        ),
    },
    "sample": {
        "mode": "untouched_historical_holdout_excluded_from_discovery_eval",
        "discovery_eval_locs": "oos_start .. end-12 step 21",
        "confirmation_eval_locs": "lookback+5 .. oos_start-12 step 21",
        "independence_rule": (
            "Confirmation evaluation dates and discovery evaluation dates must "
            "be disjoint; confirmation forward windows end before oos_start."
        ),
        "forbidden": [
            "resample of discovery OOS observations",
            "yfinance / DISPLAY_ONLY",
            "global uncertified mix-in outside scoped panel",
        ],
    },
    "production_authority": False,
    "not_standalone_buy_sell": True,
    "phase_b_forbidden": True,
}

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "overhaul"
    / "PHASE_A6_FROZEN_CONFIRMATION.json"
)


def write_frozen_protocol() -> Path:
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rejected_branches": REJECTED_BRANCHES,
        "discovery": DISCOVERY,
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "frozen_before_confirmation_evaluation": True,
    }
    PROTOCOL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PROTOCOL_PATH
