"""EXP-A5-01 — Market structure incrementality (research only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from core.costs import round_trip_cost_pct
from research.market_structure import discover_structure, compare_to_labels
from research.market_structure.benchmarks import correlation_clusters_from_returns
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg


def _closes_window(closes: pd.DataFrame, end_loc: int, lookback: int) -> dict:
    start = max(0, end_loc - lookback)
    window = closes.iloc[start:end_loc + 1]
    return {c: window[c].tolist() for c in window.columns}


def run_exp_a5_01(*, closes: pd.DataFrame, sectors: dict, manifest: dict,
                  lookback: int = 60, step: int = 21, n_clusters: int = 5,
                  oos_start_frac: float = 0.7,
                  oos_start_date: str | None = None,
                  frozen_hypothesis_id: str | None = None) -> dict:
    """Preregister then evaluate structure methods without tuning on future returns."""
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

    data_window = {
        "snapshot_id": manifest.get("snapshot_id"),
        "trust_class": manifest.get("trust_class"),
        "research_grade": False,
        "date_start": str(pd.Timestamp(dates[0]).date()),
        "date_end": str(pd.Timestamp(dates[-1]).date()),
        "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
        "lookback": lookback,
        "universe": list(closes.columns),
        "n_symbols": int(closes.shape[1]),
    }
    protocol = {
        "methods": ["sector_static", "correlation_clusters", "hierarchical", "kmeans", "pca_kmeans"],
        "structure_discovery_separated_from_eval": True,
        "no_cluster_tuning_on_future_returns": True,
        "primary_metric": "incremental_future_comovement_r2_vs_sector",
        "secondary_metrics": [
            "cluster_stability_ari", "membership_turnover", "sector_overlap_ari",
            "diversification_herfindahl_improvement",
        ],
        "success_criterion": {
            "research_grade": 1,
            "stability_ari": {"gte": 0.3},
            "incremental_r2": {"gt": 0.0},
        },
        "failure_criterion": {
            "stability_ari": {"lt": 0.1},
            "or_incremental_r2": {"lte": 0.0},
        },
        "multiple_testing": "BH-FDR across methods on incremental_r2 p-proxy",
        "transaction_costs_pct": round_trip_cost_pct("CNC"),
        "benchmark": "static NSE sector map + correlation clusters",
        "known_limitations": (
            ["none"] if gate["may_promote"]
            else ["DISPLAY_ONLY yfinance panel", "no CA ledger",
                  "no PIT sector history", "survivorship-biased universe"]
        ),
    }
    # Success criteria require research_grade==1 so exploratory runs cannot "PROMOTE"
    success_criteria = {
        "research_grade": {"eq": 1},
        "stability_ari": {"gte": 0.3},
        "incremental_r2": {"gt": 0.0},
    }
    if frozen_hypothesis_id:
        hid = frozen_hypothesis_id
    else:
        hid = prereg.preregister(
            experiment_id="EXP-A5-01",
            hypothesis=(
                "Dynamically discovered market structure (hierarchical/k-means/PCA) provides "
                "stable incremental future co-movement information beyond static sectors and "
                "pairwise correlation clusters."
            ),
            null_hypothesis=(
                "Discovered clusters add no incremental future co-movement information and/or "
                "are unstable relative to sector and correlation baselines."
            ),
            success_criteria=success_criteria,
            data_window=data_window,
            protocol=protocol,
        )

    rets = M.returns_panel(closes)
    methods = ["hierarchical", "kmeans", "pca_kmeans"]
    method_rows = []
    stability_series = {m: [] for m in methods}
    prev_labels = {m: None for m in methods}

    # Walk structure discovery on expanding calendar; evaluate OOS co-movement fit
    eval_points = list(range(max(lookback + 5, oos_start), n - 5, step))
    future_corr_scores = {m: [] for m in methods}
    future_corr_scores["sector_static"] = []
    future_corr_scores["correlation_clusters"] = []

    for loc in eval_points:
        win = _closes_window(closes, loc, lookback)
        # Baselines
        sector_ids = {}
        codes = {s: i for i, s in enumerate(sorted(set(sectors.values())))}
        for sym in win:
            if sym in sectors:
                sector_ids[sym] = codes[sectors[sym]]
        corr_clusters = correlation_clusters_from_returns(
            {s: rets[s].iloc[max(0, loc - lookback):loc + 1].to_numpy()
             for s in win if s in rets.columns},
            rho=0.70,
        )
        # Future 21d realised correlation matrix (evaluation only — not used to fit clusters)
        fut = rets.iloc[loc + 1: loc + 1 + 21]
        if fut.shape[0] < 10:
            continue
        fut_corr = fut.corr().fillna(0.0)

        def _label_comovement(labels: dict[str, int]) -> float:
            # Mean within-cluster future corr minus mean between-cluster future corr
            syms = [s for s in labels if s in fut_corr.columns]
            if len(syms) < 4:
                return 0.0
            within, between = [], []
            for i, a in enumerate(syms):
                for b in syms[i + 1:]:
                    rho = float(fut_corr.loc[a, b])
                    if labels[a] == labels[b]:
                        within.append(rho)
                    else:
                        between.append(rho)
            if not within or not between:
                return 0.0
            return float(np.mean(within) - np.mean(between))

        future_corr_scores["sector_static"].append(_label_comovement(sector_ids))
        future_corr_scores["correlation_clusters"].append(_label_comovement(corr_clusters))

        for method in methods:
            res = discover_structure(
                win, as_of=str(pd.Timestamp(dates[loc]).date()), method=method,
                n_clusters=n_clusters, lookback=lookback, seed=42,
                prior_labels=prev_labels[method],
            )
            labels = dict(res.cluster_id_by_symbol)
            future_corr_scores[method].append(_label_comovement(labels))
            if prev_labels[method] is not None:
                common = sorted(set(labels) & set(prev_labels[method]))
                if len(common) >= 3:
                    ari = float(adjusted_rand_score(
                        [prev_labels[method][s] for s in common],
                        [labels[s] for s in common],
                    ))
                    stability_series[method].append(ari)
            prev_labels[method] = labels

    # Aggregate
    sector_base = float(np.mean(future_corr_scores["sector_static"]) or 0.0)
    corr_base = float(np.mean(future_corr_scores["correlation_clusters"]) or 0.0)
    baseline = max(sector_base, corr_base)

    results_by_method = {}
    named_p = {}
    for method in methods:
        series = np.asarray(future_corr_scores[method], float)
        stab = np.asarray(stability_series[method], float)
        mean_score = float(series.mean()) if series.size else 0.0
        incremental = mean_score - baseline
        # simple one-sided t vs baseline series
        if series.size >= 5:
            diff = series - baseline
            se = diff.std(ddof=1) / np.sqrt(diff.size) if diff.std(ddof=1) > 0 else 1.0
            t = diff.mean() / se
            from scipy.stats import t as student_t
            p = float(student_t.sf(t, df=diff.size - 1))
        else:
            p = 1.0
        named_p[method] = p
        turnover = 1.0 - float(stab.mean()) if stab.size else None
        results_by_method[method] = {
            "future_comovement_score": round(mean_score, 4),
            "incremental_vs_best_baseline": round(incremental, 4),
            "stability_ari_mean": None if not stab.size else round(float(stab.mean()), 4),
            "membership_turnover": None if turnover is None else round(turnover, 4),
            "n_eval": int(series.size),
            "p_incremental": round(p, 4),
        }

    fdr = M.fdr_on_pvalues(named_p, alpha=0.05)
    best = max(methods, key=lambda m: results_by_method[m]["incremental_vs_best_baseline"])
    best_row = results_by_method[best]
    metrics = {
        "research_grade": 1 if gate["may_promote"] else 0,
        "stability_ari": best_row["stability_ari_mean"] or 0.0,
        "incremental_r2": best_row["incremental_vs_best_baseline"],  # named for criteria
        "sector_baseline_score": round(sector_base, 4),
        "corr_cluster_baseline_score": round(corr_base, 4),
        "best_method": best,
        "fdr_rejected_methods": fdr["rejected"],
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    # Verdict mapping — cannot PASS without research_grade
    if not gate["may_promote"]:
        verdict = "INCONCLUSIVE"
        reason = gate["reason"]
    elif best_row["incremental_vs_best_baseline"] > 0 and (best_row["stability_ari_mean"] or 0) >= 0.3:
        if best in fdr["rejected"]:
            verdict = "PASS_RISK"  # co-movement / diversification info, not alpha
        else:
            verdict = "INCONCLUSIVE"
        reason = "see metrics"
    elif best_row["incremental_vs_best_baseline"] <= 0:
        verdict = "FAIL"
        reason = "no incremental future co-movement vs baselines"
    else:
        verdict = "INCONCLUSIVE"
        reason = "unstable or FDR-insignificant"

    if verdict == "FAIL":
        prereg.remember_negative(
            f"EXP-A5-01 FAIL: dynamic structure no incremental vs baseline "
            f"(best={best}, incr={best_row['incremental_vs_best_baseline']})",
            signal="market_structure",
            evidence_n=best_row["n_eval"],
            notes=reason,
        )
    elif verdict == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-A5-01 {verdict}: {best} incremental={best_row['incremental_vs_best_baseline']}",
            signal="market_structure",
            evidence_n=best_row["n_eval"],
            ev_r=best_row["incremental_vs_best_baseline"],
            hypothesis_id=hid,
            notes=reason,
        )

    return {
        "experiment_id": "EXP-A5-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "verdict": verdict,
        "scientific_verdict": M.scientific_verdict(verdict),
        "reason": reason,
        "gate": gate,
        "baselines": {"sector_static": sector_base, "correlation_clusters": corr_base},
        "methods": results_by_method,
        "fdr": fdr,
        "metrics": metrics,
        "evaluation_snapshot_id": manifest.get("snapshot_id"),
        "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
        "production_authority": False,
    }
