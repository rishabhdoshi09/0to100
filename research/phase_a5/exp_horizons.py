"""EXP-A2-01 — Horizon term structure for a preregistered strategy family."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.horizons.catalog import CAPABILITY_HORIZONS, absolute_return_target
from research.horizons.labels import build_forward_return_labels
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg


# Preregistered horizon family for cross-sectional momentum (not every capability blindly)
MOMENTUM_HORIZONS = ("5d", "10d", "22d", "66d")


def run_exp_a2_01(*, closes: pd.DataFrame, manifest: dict,
                  oos_start_frac: float = 0.7) -> dict:
    gate = M.gate_research_grade(manifest)
    dates = list(closes.index)
    n = len(dates)
    oos_start = int(n * oos_start_frac)
    cost = round_trip_cost_pct("CNC")

    hid = prereg.preregister(
        experiment_id="EXP-A2-01",
        hypothesis=(
            "Cross-sectional momentum has a preferred economic horizon within the "
            f"preregistered set {MOMENTUM_HORIZONS}; effect is not horizon-invariant."
        ),
        null_hypothesis=(
            "No horizon in the preregistered family shows a positive cost-aware edge "
            "after multiple-testing control; or all are indistinguishable from noise."
        ),
        success_criteria={
            "research_grade": {"eq": 1},
            "best_dsr": {"gte": 0.95},
            "fdr_any_horizon": {"eq": 1},
        },
        data_window={
            "snapshot_id": manifest.get("snapshot_id"),
            "trust_class": manifest.get("trust_class"),
            "research_grade": False,
            "strategy": "cross_sectional_momentum_60d_rank",
            "horizons": list(MOMENTUM_HORIZONS),
            "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
            "train_end": str(pd.Timestamp(dates[oos_start - 1]).date()),
            "cost_pct": cost,
        },
        protocol={
            "entry": "close",
            "exit": "close",
            "signal": "60d return cross-sectional rank",
            "portfolio": "long top 20% / short bottom 20% equal weight",
            "primary_metric": "deflated Sharpe / harness verdict on OOS R stream",
            "secondary_metrics": [
                "mean_r", "sharpe", "n_eff", "cost_drag", "turnover_proxy",
            ],
            "multiple_testing": f"BH-FDR across {len(MOMENTUM_HORIZONS)} horizons; "
                                f"n_trials={len(MOMENTUM_HORIZONS)} in DSR",
            "no_post_hoc_horizon_selection_for_success": True,
            "known_limitations": [
                "DISPLAY_ONLY panel" if not gate["may_promote"] else "none",
            ],
        },
    )

    scores = M.cross_sectional_momentum_scores(closes, lookback=60)
    horizon_results = {}
    named_p = {}
    sharpes = []

    for hname in MOMENTUM_HORIZONS:
        bars = CAPABILITY_HORIZONS[hname].bars
        fwd = M.forward_returns(closes, bars)
        # Align scores at t with forward return from t (label uses future — evaluation OK)
        port = M.long_short_from_scores(scores.iloc[: n - bars], fwd)
        # Split IS/OOS by date location
        oos_mask = np.array([dates.index(dt) >= oos_start for dt in port.index if dt in dates])
        # port.index are timestamps from closes
        oos = port[[pd.Timestamp(dt) >= pd.Timestamp(dates[oos_start]) for dt in port.index]]
        # Cost drag: assume ~100% one-way turnover per rebalance (conservative)
        net = oos - M.cost_drag(1.0, cost)
        pack = M.harness_pack(net.to_numpy(), n_trials=len(MOMENTUM_HORIZONS), min_n=30)
        sharpes.append(pack["sharpe"])
        named_p[hname] = pack["p_value"] if pack["mean_r"] > 0 else 1.0
        horizon_results[hname] = {
            **pack,
            "bars": bars,
            "cost_drag": round(M.cost_drag(1.0, cost), 4),
            "n_oos": int(len(net)),
            "mean_gross": round(float(oos.mean()) if len(oos) else 0.0, 4),
            "mean_net": round(float(net.mean()) if len(net) else 0.0, 4),
        }

    fdr = M.fdr_on_pvalues(named_p, alpha=0.05)
    # Family-level DSR using all horizon sharpes as the trial set
    best_name = max(MOMENTUM_HORIZONS, key=lambda h: horizon_results[h]["mean_net"])
    best = horizon_results[best_name]
    metrics = {
        "research_grade": 1 if gate["may_promote"] else 0,
        "best_dsr": best["dsr"],
        "best_horizon": best_name,
        "fdr_any_horizon": 1 if fdr["rejected"] else 0,
        "n_trials": len(MOMENTUM_HORIZONS),
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    # Classify effect region without declaring success from max alone
    positive = [h for h in MOMENTUM_HORIZONS if horizon_results[h]["mean_net"] > 0]
    if not gate["may_promote"]:
        verdict, reason = "INCONCLUSIVE", gate["reason"]
        effect = "unknown_pending_research_grade_data"
    elif not positive:
        verdict, reason = "FAIL", "no positive net edge in preregistered family"
        effect = "no_effect"
    elif fdr["rejected"]:
        # map rejected horizons to bands
        bands = []
        for h in fdr["rejected"]:
            b = CAPABILITY_HORIZONS[h].bars
            if b <= 10:
                bands.append("short-horizon")
            elif b <= 44:
                bands.append("intermediate-horizon")
            else:
                bands.append("long-horizon")
        effect = ",".join(sorted(set(bands)))
        verdict, reason = "PASS_ALPHA", f"FDR-significant horizons: {fdr['rejected']}"
    else:
        verdict, reason = "INCONCLUSIVE", "positive point estimates without FDR clearance"
        effect = "possible_but_unconfirmed"

    if verdict in ("FAIL", "INCONCLUSIVE"):
        prereg.remember_watch(
            f"EXP-A2-01 {verdict}: best={best_name} net={best['mean_net']}",
            signal="horizon_term_structure", evidence_n=best["n"],
            ev_r=best["mean_net"], hypothesis_id=hid, notes=reason,
        )

    return {
        "experiment_id": "EXP-A2-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "verdict": verdict,
        "reason": reason,
        "effect_region": effect,
        "preregistered_horizons": list(MOMENTUM_HORIZONS),
        "horizons": horizon_results,
        "fdr": fdr,
        "gate": gate,
        "metrics": metrics,
        "production_authority": False,
    }
