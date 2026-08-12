"""EXP-A3-01 — Simple logistic/linear challenger vs incumbent rank rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.challenger_lab import (
    BakeOffConfig,
    LogisticChallenger,
    NaiveBaseline,
    run_bakeoff,
)
from research.challenger_lab.models import ModelIdentity
from research.horizons.catalog import absolute_return_target, get_legacy_mh_target
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg


def _feature_matrix(closes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Transparent features already in the QuantTerm spirit (no deep learning)."""
    rets = closes.pct_change()
    feats = {}
    feats["mom_5"] = closes.pct_change(5)
    feats["mom_10"] = closes.pct_change(10)
    feats["mom_20"] = closes.pct_change(20)
    feats["mom_60"] = closes.pct_change(60)
    feats["vol_20"] = rets.rolling(20).std()
    # Stack cross-sectionally per date then flatten for panel logistic on 10d direction
    feature_names = list(feats)
    # Build panel rows: each (date, symbol)
    rows_X, rows_y, index = [], [], []
    fwd = closes.pct_change(10).shift(-10)
    for dt in closes.index[60: -10]:
        for sym in closes.columns:
            x = [float(feats[f].loc[dt, sym]) for f in feature_names]
            y = float(fwd.loc[dt, sym])
            if any(np.isnan(x)) or np.isnan(y):
                continue
            # classification label for logistic
            if y > 0.01:
                lab = 1.0
            elif y < -0.01:
                lab = -1.0
            else:
                lab = 0.0
            rows_X.append(x)
            rows_y.append(lab)
            index.append(dt)
    return np.asarray(rows_X, float), np.asarray(rows_y, float), feature_names, pd.DatetimeIndex(index)


class RankIncumbent:
    """Wrap momentum-rank as a 'model' with fit/predict for bakeoff interface."""

    algorithm = "cross_sectional_momentum_rank"

    def __init__(self, scores_by_row: np.ndarray):
        self._scores = scores_by_row
        self._pred = 0.0

    def fit(self, X, y):
        # Incumbent is rule-based; ignore X/y fit — use precomputed score sign on OOS via predict
        y = np.asarray(y, float)
        # fallback majority for rows without score alignment
        vals, counts = np.unique(y[~np.isnan(y)], return_counts=True)
        self._pred = float(vals[int(np.argmax(counts))]) if vals.size else 0.0
        return self

    def predict(self, X):
        # Without row ids, approximate incumbent as sign of mom_60 feature (column 3)
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] < 4:
            return np.full(len(X), self._pred)
        return np.sign(X[:, 3])


def run_exp_a3_01(*, closes: pd.DataFrame, manifest: dict,
                  frozen_hypothesis_id: str | None = None) -> dict:
    gate = M.gate_research_grade(manifest)
    X, y, feature_names, idx = _feature_matrix(closes)
    target = get_legacy_mh_target("10d")
    cost = round_trip_cost_pct("CNC")

    if frozen_hypothesis_id:
        hid = frozen_hypothesis_id
    else:
        hid = prereg.preregister(
            experiment_id="EXP-A3-01",
            hypothesis=(
                "A simple logistic regression on QuantTerm-style momentum/vol features "
                "extracts incremental OOS economic value over naive and rank-rule incumbents."
            ),
            null_hypothesis=(
                "Logistic challenger does not improve cost-aware OOS expectancy vs naive/"
                "rank incumbents after evidence gating."
            ),
            success_criteria={
                "research_grade": {"eq": 1},
                "economic_value_delta": {"gt": 0.0},
                "challenger_harness_promote": {"eq": 1},
            },
            data_window={
                "snapshot_id": manifest.get("snapshot_id"),
                "trust_class": manifest.get("trust_class"),
                "research_grade": False,
                "features": feature_names,
                "target": "10d classification (+1/-1/0 at ±1%)",
                "n_rows": int(len(y)),
                "cost_pct": cost,
            },
            protocol={
                "incumbents": ["naive_baseline", "momentum_rank_sign"],
                "challenger": "logistic_regression",
                "identical_splits": True,
                "primary_metric": "economic_value_delta (mean OOS R)",
                "multiple_testing": "single primary challenger; n_trials=1 vs each incumbent",
                "no_deep_learning": True,
                "known_limitations": [
                    "DISPLAY_ONLY panel" if not gate["may_promote"] else "none",
                ],
            },
        )

    # Bake-off 1: logistic vs naive
    r_naive = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=42),
        X=X, y=y, target=absolute_return_target("10d"),
        feature_names=feature_names,
        incumbent_id=ModelIdentity("naive", "1", "naive_baseline"),
        challenger_id=ModelIdentity("logistic", "1", "logistic_regression"),
        config=BakeOffConfig(
            role="phase_a5_exp_a3",
            persist_champion=False,
            write_scientific_memory=False,
            min_oos=30,
            seed=42,
        ),
        code_hash="phase_a5_exp_a3",
    )
    # Bake-off 2: logistic vs rank incumbent
    r_rank = run_bakeoff(
        incumbent_model=RankIncumbent(np.zeros(len(y))),
        challenger_model=LogisticChallenger(random_state=42),
        X=X, y=y, target=absolute_return_target("10d"),
        feature_names=feature_names,
        incumbent_id=ModelIdentity("mom_rank", "1", "cross_sectional_momentum_rank"),
        challenger_id=ModelIdentity("logistic", "1", "logistic_regression"),
        config=BakeOffConfig(
            role="phase_a5_exp_a3_rank",
            persist_champion=False,
            write_scientific_memory=False,
            min_oos=30,
            seed=42,
        ),
        code_hash="phase_a5_exp_a3",
    )

    delta = float(r_rank.economic_value_delta or 0.0)
    chal_promote = 1 if (
        r_rank.evidence_result.get("challenger", {}).get("verdict") == "PROMOTE"
    ) else 0
    metrics = {
        "research_grade": 1 if gate["may_promote"] else 0,
        "economic_value_delta": delta,
        "challenger_harness_promote": chal_promote,
        "vs_naive_verdict": r_naive.verdict,
        "vs_rank_verdict": r_rank.verdict,
        "pred_corr_vs_rank": r_rank.prediction_correlation,
        "live_behaviour_changed": 0,
    }
    # Use isolated phase_a5 registry (bakeoff used default registry — also record here)
    reg = prereg.record(hid, metrics)

    if not gate["may_promote"]:
        verdict, reason = "INCONCLUSIVE", gate["reason"]
    elif delta > 0 and chal_promote == 1 and r_rank.verdict == "PROMOTE":
        verdict, reason = "PASS_ALPHA", "logistic cleared evidence vs rank incumbent"
    elif delta <= 0:
        verdict, reason = "FAIL", "no incremental economic value vs rank incumbent"
    else:
        verdict, reason = "INCONCLUSIVE", "mixed bake-off / evidence gate"

    if verdict == "FAIL":
        prereg.remember_negative(
            f"EXP-A3-01 FAIL logistic vs rank delta={delta:.4f}",
            signal="simple_challenger",
            evidence_n=r_rank.oos_period.get("n") or 0,
            notes=reason,
        )
    elif verdict == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-A3-01 {verdict}: delta={delta:.4f} vs_rank={r_rank.verdict}",
            signal="simple_challenger", evidence_n=r_rank.oos_period.get("n") or 0,
            ev_r=delta, hypothesis_id=hid, notes=reason,
        )

    return {
        "experiment_id": "EXP-A3-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "verdict": verdict,
        "scientific_verdict": M.scientific_verdict(verdict),
        "reason": reason,
        "gate": gate,
        "vs_naive": r_naive.to_dict(),
        "vs_rank": r_rank.to_dict(),
        "metrics": metrics,
        "evaluation_snapshot_id": manifest.get("snapshot_id"),
        "production_authority": False,
    }
