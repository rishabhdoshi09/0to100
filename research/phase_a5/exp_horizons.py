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
                  oos_start_frac: float = 0.7,
                  oos_start_date: str | None = None,
                  frozen_hypothesis_id: str | None = None) -> dict:
    gate = M.gate_research_grade(manifest)
    dates = list(closes.index)
    n = len(dates)
    if oos_start_date:
        oos_start = next(
            (i for i, d in enumerate(dates)
             if str(pd.Timestamp(d).date()) >= str(oos_start_date)[:10]),
            int(n * oos_start_frac),
        )
    else:
        oos_start = int(n * oos_start_frac)
    cost = round_trip_cost_pct("CNC")

    if frozen_hypothesis_id:
        hid = frozen_hypothesis_id
    else:
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
        oos = port[[pd.Timestamp(dt) >= pd.Timestamp(dates[oos_start]) for dt in port.index]]
        # Cost drag: assume ~100% one-way turnover per rebalance (conservative)
        net = oos - M.cost_drag(1.0, cost)
        pack = M.harness_pack(net.to_numpy(), n_trials=len(MOMENTUM_HORIZONS), min_n=30)
        sharpes.append(pack["sharpe"])
        named_p[hname] = pack["p_value"] if pack["mean_r"] > 0 else 1.0
        # Extended economic diagnostics (reporting only — not used to retune protocol)
        arr = net.to_numpy(dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        gross_win = float(wins.sum()) if wins.size else 0.0
        gross_loss = float(abs(losses.sum())) if losses.size else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        equity = np.cumsum(arr) if arr.size else np.array([])
        if equity.size:
            peak = np.maximum.accumulate(equity)
            dd = float((equity - peak).min())
        else:
            dd = 0.0
        # Normal approx CI for mean (reporting); harness remains authoritative for verdict
        if arr.size >= 2 and arr.std(ddof=1) > 0:
            se = arr.std(ddof=1) / np.sqrt(arr.size)
            ci_lo = float(arr.mean() - 1.96 * se)
            ci_hi = float(arr.mean() + 1.96 * se)
        else:
            ci_lo = ci_hi = float(arr.mean()) if arr.size else 0.0
        econ = "POSITIVE" if pack["mean_r"] > 0 else "NON_POSITIVE"
        stat = pack["verdict"]
        horizon_results[hname] = {
            **pack,
            "bars": bars,
            "cost_drag": round(M.cost_drag(1.0, cost), 4),
            "turnover_proxy": 1.0,
            "n_oos": int(len(net)),
            "mean_gross": round(float(oos.mean()) if len(oos) else 0.0, 4),
            "mean_net": round(float(net.mean()) if len(net) else 0.0, 4),
            "expectancy": pack["mean_r"],
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "profit_factor": None if pf == float("inf") else round(pf, 4),
            "max_drawdown_R": round(dd, 4),
            "alpha_proxy_mean_net": round(float(net.mean()) if len(net) else 0.0, 4),
            "statistical_verdict": stat,
            "economic_verdict": econ,
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

    if verdict == "FAIL":
        prereg.remember_negative(
            f"EXP-A2-01 FAIL: momentum family net-negative best={best_name} net={best['mean_net']}",
            signal="horizon_term_structure", evidence_n=best["n"], notes=reason,
        )
    elif verdict == "INCONCLUSIVE":
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
        "scientific_verdict": M.scientific_verdict(verdict),
        "reason": reason,
        "effect_region": effect,
        "preregistered_horizons": list(MOMENTUM_HORIZONS),
        "horizons": horizon_results,
        "fdr": fdr,
        "gate": gate,
        "metrics": metrics,
        "cost_pct": cost,
        "evaluation_snapshot_id": manifest.get("snapshot_id"),
        "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
        "production_authority": False,
    }
