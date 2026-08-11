"""EXP-A5A6-01 — Context interaction tests (preregistered, FDR-controlled)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.market_structure import discover_structure
from research.portfolio_network import analyze_network
from research.portfolio_network.engine import corr_from_returns
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg


INTERACTIONS = (
    "signal_x_cluster_stability",
    "signal_x_network_concentration",
    "signal_x_incremental_community_risk",
)


def run_exp_a5a6_01(*, closes: pd.DataFrame, sectors: dict, manifest: dict,
                    lookback: int = 60, step: int = 21,
                    oos_start_frac: float = 0.7,
                    oos_start_date: str | None = None,
                    frozen_hypothesis_id: str | None = None) -> dict:
    gate = M.gate_research_grade(manifest)
    rets = M.returns_panel(closes)
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
    scores = M.cross_sectional_momentum_scores(closes, lookback=60)
    fwd = M.forward_returns(closes, 10)

    if frozen_hypothesis_id:
        hid = frozen_hypothesis_id
    else:
        hid = prereg.preregister(
            experiment_id="EXP-A5A6-01",
            hypothesis=(
                "Market-structure stability and network concentration modulate momentum "
                "payoffs as context (interaction), rather than as standalone alpha."
            ),
            null_hypothesis=(
                "Preregistered interactions have no FDR-significant effect on momentum "
                "forward returns after multiple-testing control."
            ),
            success_criteria={
                "research_grade": {"eq": 1},
                "n_fdr_interactions": {"gte": 1},
            },
            data_window={
                "snapshot_id": manifest.get("snapshot_id"),
                "trust_class": manifest.get("trust_class"),
                "research_grade": False,
                "interactions": list(INTERACTIONS),
                "signal": "60d_momentum_rank",
                "response": "10d_forward_return",
            },
            protocol={
                "interactions": list(INTERACTIONS),
                "no_unrestricted_feature_mining": True,
                "multiple_testing": "BH-FDR across preregistered interactions",
                "context_not_standalone_alpha": True,
                "known_limitations": [
                    "DISPLAY_ONLY panel" if not gate["may_promote"] else "none",
                ],
            },
        )

    rows = []
    prev_labels = None
    for loc in range(max(lookback + 5, oos_start), n - 12, step):
        dt = dates[loc]
        win = {c: closes[c].iloc[max(0, loc - lookback): loc + 1].tolist() for c in closes.columns}
        struct = discover_structure(
            win, as_of=str(pd.Timestamp(dt).date()), method="hierarchical",
            n_clusters=5, lookback=lookback, seed=42, prior_labels=prev_labels,
        )
        stability = struct.cluster_stability if struct.cluster_stability is not None else 0.0
        prev_labels = dict(struct.cluster_id_by_symbol)

        window_rets = rets.iloc[max(0, loc - lookback): loc + 1]
        corr = corr_from_returns({s: window_rets[s].to_numpy() for s in window_rets.columns},
                                 min_overlap=30)
        # portfolio = current top momentum names
        srow = scores.iloc[loc].dropna()
        if len(srow) < 8:
            continue
        port = list(srow.nlargest(5).index)
        net = analyze_network(corr, list(window_rets.columns),
                              as_of=str(pd.Timestamp(dt).date()),
                              threshold=0.70, portfolio=port)
        # Per-symbol observations
        frow = fwd.iloc[loc]
        for sym in srow.index:
            if sym not in frow.index or np.isnan(frow[sym]):
                continue
            incr = net.incremental_cluster_risk.get(sym, 0.0)
            rows.append({
                "signal": float(srow[sym]),
                "fwd": float(frow[sym]),
                "stability": float(stability),
                "net_conc": float(net.portfolio_network_concentration),
                "incr_risk": float(incr or 0.0),
            })

    df = pd.DataFrame(rows)
    named_p = {}
    interaction_stats = {}

    def _interaction_test(df, context_col, split_median=True):
        """Compare signal→fwd correlation in high vs low context regimes."""
        if df.empty or df[context_col].nunique() < 2:
            return {"delta_corr": 0.0, "p": 1.0, "n_low": 0, "n_high": 0}
        med = float(df[context_col].median())
        low = df[df[context_col] <= med]
        high = df[df[context_col] > med]

        def _r(sub):
            if len(sub) < 30 or sub["signal"].std() == 0 or sub["fwd"].std() == 0:
                return 0.0
            return float(np.corrcoef(sub["signal"], sub["fwd"])[0, 1])

        r_low, r_high = _r(low), _r(high)
        # Approximate p via correlation difference Fisher z
        def _z(r, n):
            r = max(min(r, 0.999), -0.999)
            return 0.5 * np.log((1 + r) / (1 - r)), max(n - 3, 1)

        z1, n1 = _z(r_low, len(low))
        z2, n2 = _z(r_high, len(high))
        se = np.sqrt(1 / n1 + 1 / n2)
        from scipy.stats import norm
        p = float(2 * norm.sf(abs((z2 - z1) / se))) if se > 0 else 1.0
        return {
            "corr_low": round(r_low, 4),
            "corr_high": round(r_high, 4),
            "delta_corr": round(r_high - r_low, 4),
            "p": round(p, 4),
            "n_low": int(len(low)),
            "n_high": int(len(high)),
        }

    if not df.empty:
        mapping = {
            "signal_x_cluster_stability": "stability",
            "signal_x_network_concentration": "net_conc",
            "signal_x_incremental_community_risk": "incr_risk",
        }
        for name, col in mapping.items():
            stats = _interaction_test(df, col)
            interaction_stats[name] = stats
            named_p[name] = stats["p"]

    fdr = M.fdr_on_pvalues(named_p, alpha=0.05) if named_p else {"rejected": [], "detail": {}}
    metrics = {
        "research_grade": 1 if gate["may_promote"] else 0,
        "n_fdr_interactions": len(fdr.get("rejected") or []),
        "n_rows": int(len(df)),
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    if not gate["may_promote"]:
        verdict, reason = "INCONCLUSIVE", gate["reason"]
    elif fdr.get("rejected"):
        verdict, reason = "PASS_RISK", f"FDR-significant interactions: {fdr['rejected']}"
    elif not df.empty and any(interaction_stats[i]["delta_corr"] != 0 for i in interaction_stats):
        verdict, reason = "INCONCLUSIVE", "interaction point differences without FDR clearance"
    else:
        verdict, reason = "FAIL", "no interaction effects detected"

    if verdict == "FAIL":
        prereg.remember_negative(
            f"EXP-A5A6-01 FAIL: no FDR-significant interactions",
            signal="context_interaction", evidence_n=len(df), notes=reason,
        )
    elif verdict == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-A5A6-01 {verdict}: fdr={fdr.get('rejected')}",
            signal="context_interaction", evidence_n=len(df), ev_r=0.0,
            hypothesis_id=hid, notes=reason,
        )

    return {
        "experiment_id": "EXP-A5A6-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "verdict": verdict,
        "scientific_verdict": M.scientific_verdict(verdict),
        "reason": reason,
        "gate": gate,
        "interactions": interaction_stats,
        "fdr": fdr,
        "metrics": metrics,
        "evaluation_snapshot_id": manifest.get("snapshot_id"),
        "oos_start": str(pd.Timestamp(dates[oos_start]).date()),
        "production_authority": False,
    }
