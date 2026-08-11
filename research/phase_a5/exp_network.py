"""EXP-A6-01 — Network risk incrementality (advisory evaluation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.portfolio_network import analyze_network, incremental_candidate_risk
from research.portfolio_network.engine import corr_from_returns
from risk.correlation import clusters_from_corr
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg


def run_exp_a6_01(*, closes: pd.DataFrame, sectors: dict, manifest: dict,
                  lookback: int = 60, step: int = 21,
                  oos_start_frac: float = 0.7) -> dict:
    gate = M.gate_research_grade(manifest)
    rets = M.returns_panel(closes)
    dates = list(closes.index)
    n = len(dates)
    oos_start = int(n * oos_start_frac)

    hid = prereg.preregister(
        experiment_id="EXP-A6-01",
        hypothesis=(
            "Correlation-graph network metrics (community concentration, centrality, "
            "incremental community risk) identify future correlated portfolio losses "
            "beyond pairwise correlation clusters and sector caps."
        ),
        null_hypothesis=(
            "Network metrics add no incremental explanatory power for simultaneous "
            "losses / drawdowns after conditioning on pairwise ρ clusters and sectors."
        ),
        success_criteria={
            "research_grade": {"eq": 1},
            "incremental_auc_proxy": {"gt": 0.0},
            "conditioned_improvement": {"gt": 0.0},
        },
        data_window={
            "snapshot_id": manifest.get("snapshot_id"),
            "trust_class": manifest.get("trust_class"),
            "research_grade": False,
            "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
            "lookback": lookback,
            "universe": list(closes.columns),
        },
        protocol={
            "baseline": ["pairwise_corr_clusters", "sector_herfindahl"],
            "challenger": [
                "community_exposure", "degree", "eigenvector", "betweenness",
                "network_concentration", "incremental_community_risk",
            ],
            "auto_block": False,
            "primary_metric": "incremental explanatory power for simultaneous loss events",
            "success_criterion": "research_grade + positive conditioned improvement",
            "failure_criterion": "no improvement after conditioning on pairwise/sector",
            "multiple_testing": "BH-FDR across network feature predictors",
            "transaction_costs_pct": round_trip_cost_pct("CNC"),
            "known_limitations": protocol_limitations(gate),
        },
    )

    rows = []
    eval_points = list(range(max(lookback + 5, oos_start), n - 21, step))
    for loc in eval_points:
        window = rets.iloc[max(0, loc - lookback): loc + 1]
        fut = rets.iloc[loc + 1: loc + 1 + 21]
        if window.shape[0] < 30 or fut.shape[0] < 10:
            continue
        # Random-ish but deterministic pseudo-portfolio: top 5 by 60d momentum
        mom = closes.pct_change(60).iloc[loc].dropna()
        if len(mom) < 8:
            continue
        port = list(mom.nlargest(5).index)
        candidates = [s for s in mom.index if s not in port]
        corr = corr_from_returns({s: window[s].to_numpy() for s in window.columns}, min_overlap=30)
        net = analyze_network(corr, list(window.columns), as_of=str(pd.Timestamp(dates[loc]).date()),
                              threshold=0.70, portfolio=port)
        # Pairwise cluster baseline: largest cluster size share
        clusters = clusters_from_corr(port, corr, threshold=0.70)
        pairwise_conc = max(len(g) for g in clusters) / max(len(port), 1)
        # Sector concentration
        from collections import Counter
        sec_counts = Counter(sectors.get(s, "UNK") for s in port)
        sector_hhi = sum((c / len(port)) ** 2 for c in sec_counts.values())

        # Simultaneous loss event: fraction of portfolio with fut 21d return < -5%
        cum = (1.0 + fut[port]).prod() - 1.0
        sim_loss = float((cum < -0.05).mean())
        port_dd = float(cum.mean())

        # For each candidate, incremental risk + subsequent contribution if added
        for cand in candidates[:10]:
            info = incremental_candidate_risk(net, cand, portfolio=port)
            if cand not in fut.columns:
                continue
            cand_ret = float((1.0 + fut[cand]).prod() - 1.0)
            rows.append({
                "pairwise_conc": pairwise_conc,
                "sector_hhi": sector_hhi,
                "net_conc": net.portfolio_network_concentration,
                "incr_risk": info["incremental_cluster_risk"] or 0.0,
                "degree": net.centrality_degree.get(cand, 0.0),
                "eig": net.centrality_eigenvector.get(cand, 0.0),
                "btw": net.centrality_betweenness.get(cand, 0.0),
                "joins": 1.0 if info.get("joins_existing_community") else 0.0,
                "sim_loss": sim_loss,
                "port_dd": port_dd,
                "cand_ret": cand_ret,
                "cand_loss": 1.0 if cand_ret < -0.05 else 0.0,
            })

    df = pd.DataFrame(rows)
    feature_names = ["incr_risk", "degree", "eig", "btw", "net_conc", "joins"]
    baseline_names = ["pairwise_conc", "sector_hhi"]
    named_p = {}
    feature_stats = {}

    def _corr_p(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if x.size < 20 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 1.0
        r = float(np.corrcoef(x, y)[0, 1])
        # rough p via Fisher
        z = 0.5 * np.log((1 + r) / (1 - r + 1e-12))
        se = 1.0 / np.sqrt(max(x.size - 3, 1))
        from scipy.stats import norm
        p = float(2 * norm.sf(abs(z / se)))
        return r, p

    if df.empty:
        metrics = {"research_grade": 0, "incremental_auc_proxy": 0.0, "conditioned_improvement": 0.0}
        reg = prereg.record(hid, metrics)
        return {
            "experiment_id": "EXP-A6-01", "hypothesis_id": hid,
            "registry_status": reg.get("status"), "verdict": "INCONCLUSIVE",
            "reason": "insufficient evaluation rows", "gate": gate,
            "metrics": metrics, "production_authority": False, "auto_block": False,
        }

    # Target: candidate loss and portfolio simultaneous loss
    for feat in feature_names:
        r_loss, p_loss = _corr_p(df[feat], df["cand_loss"])
        r_sim, p_sim = _corr_p(df[feat], df["sim_loss"])
        feature_stats[feat] = {
            "corr_cand_loss": round(r_loss, 4), "p_cand_loss": round(p_loss, 4),
            "corr_sim_loss": round(r_sim, 4), "p_sim_loss": round(p_sim, 4),
        }
        named_p[feat] = min(p_loss, p_sim)

    # Conditioned improvement: partial correlation of incr_risk with cand_loss
    # after regressing out pairwise_conc + sector_hhi
    def _partial(y, x, controls):
        Y = np.asarray(y, float)
        X = np.column_stack([np.asarray(x, float), np.asarray(controls, float)])
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
        Y, X = Y[mask], X[mask]
        if Y.size < 30:
            return 0.0, 1.0
        # residualize
        ones = np.ones((X.shape[0], 1))
        C = np.hstack([ones, X[:, 1:]])
        xc = X[:, [0]]
        try:
            beta_y = np.linalg.lstsq(C, Y, rcond=None)[0]
            beta_x = np.linalg.lstsq(C, xc, rcond=None)[0]
            ry = Y - C @ beta_y
            rx = (xc - C @ beta_x).ravel()
            return _corr_p(rx, ry)
        except Exception:
            return 0.0, 1.0

    controls = df[baseline_names].to_numpy()
    incr_r, incr_p = _partial(df["cand_loss"], df["incr_risk"], controls)
    named_p["incr_risk_partial"] = incr_p
    fdr = M.fdr_on_pvalues(named_p, alpha=0.05)

    metrics = {
        "research_grade": 1 if gate["may_promote"] else 0,
        "incremental_auc_proxy": float(incr_r),  # signed corr proxy
        "conditioned_improvement": float(incr_r),
        "partial_p": float(incr_p),
        "fdr_rejected": fdr["rejected"],
        "n_rows": int(len(df)),
        "live_behaviour_changed": 0,
        "auto_block": 0,
    }
    reg = prereg.record(hid, metrics)

    if not gate["may_promote"]:
        verdict, reason = "INCONCLUSIVE", gate["reason"]
    elif incr_r > 0 and "incr_risk_partial" in fdr["rejected"]:
        verdict, reason = "PASS_RISK", "incremental community risk predicts losses after controls"
    elif incr_r <= 0:
        verdict, reason = "FAIL", "no conditioned improvement vs pairwise/sector"
    else:
        verdict, reason = "INCONCLUSIVE", "point improvement without FDR clearance"

    if verdict in ("FAIL", "INCONCLUSIVE"):
        prereg.remember_watch(
            f"EXP-A6-01 {verdict}: partial_r={incr_r:.4f}",
            signal="portfolio_network", evidence_n=len(df), ev_r=incr_r,
            hypothesis_id=hid, notes=reason,
        )

    return {
        "experiment_id": "EXP-A6-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "verdict": verdict,
        "reason": reason,
        "gate": gate,
        "feature_stats": feature_stats,
        "partial_incr_risk": {"r": incr_r, "p": incr_p},
        "fdr": fdr,
        "metrics": metrics,
        "production_authority": False,
        "auto_block": False,
    }


def protocol_limitations(gate: dict) -> list[str]:
    lim = ["advisory evaluation only — no trade blocking"]
    if not gate.get("may_promote"):
        lim.append("DISPLAY_ONLY / non-RESEARCH_GRADE inputs")
    return lim
