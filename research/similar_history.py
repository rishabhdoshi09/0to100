"""
🕒 Similar History — "what happened to setups that looked like this before?"

The 🕒 window behind the scanner. The accumulating Feature Store IS the corpus
now, so this retrieves the k most similar PAST observations (by Mahalanobis
distance over the canonical numeric features — scale- and correlation-aware, the
honest metric for heterogeneous features) and summarises what they went on to do:
median outcome, worst drawdown, win rate, and the plain-English environment the
query sits in.

Pure retrieval math is reused from `market_memory` (robust_inv_cov +
mahalanobis_knn); the corpus is read from `feature_store`. Fail-open — too little
history yields a truthful "not enough similar setups yet", never a guess.
"""
from __future__ import annotations

import numpy as np

from research import feature_schema as _S

# numeric canonical features used for the distance (categoricals excluded)
_NUMERIC = tuple(n for n in _S.FEATURE_NAMES
                 if _S.FEATURE_REGISTRY[n].dtype != "categorical")
_MIN_CORPUS = 20            # below this, don't claim an analog read
_WIN_PCT = 2.0


def _environment_tags(feats: dict) -> list[str]:
    """Plain-English read of the query's environment from its own features —
    what a user recognises ('Healthy Breadth', 'Low VIX'), not raw numbers."""
    tags = []
    reg = feats.get("regime")
    if reg:
        tags.append(str(reg).replace("_", " ").title())
    b = feats.get("breadth_pct_above_50dma")
    if b is not None:
        tags.append("Healthy Breadth" if b >= 55 else
                    "Narrow Breadth" if b <= 35 else "Mixed Breadth")
    v = feats.get("vix")
    if v is not None:
        tags.append("Low VIX" if v < 15 else "High VIX" if v > 22 else "Normal VIX")
    liq = feats.get("liquidity_cr")
    if liq is not None:
        tags.append("High Liquidity" if liq >= 25 else
                    "Thin Liquidity" if liq < 5 else "Moderate Liquidity")
    return tags


def similar(query_features: dict, kind: str | None = "TRADE", k: int = 50) -> dict:
    """Retrieve the k nearest settled observations to `query_features` and
    summarise their outcomes. `kind` filters the corpus (TRADE by default; None =
    all settled). Returns median/worst/best outcome %, win rate, count, and the
    query's environment tags. Fail-open → {'found': False}."""
    from research.feature_store import load_matrix
    try:
        m = load_matrix(kind=kind, feature_names=list(_NUMERIC),
                        require_outcome=True)
    except Exception:
        return {"found": False, "note": "history unavailable"}
    X, y = m.get("X"), m.get("y")
    if X is None or y is None or X.shape[0] < _MIN_CORPUS:
        return {"found": False,
                "note": f"only {0 if X is None else X.shape[0]} settled analogs — "
                        f"need ~{_MIN_CORPUS}."}

    q = np.array([_num(query_features.get(n)) for n in _NUMERIC], float)
    # a column can inform the distance only if the query knows it AND the corpus
    # populates it widely — otherwise requiring it would drop the whole corpus.
    col_cov = 1.0 - np.isnan(X).mean(axis=0)
    keep = (~np.isnan(q)) & (col_cov >= 0.5)
    if keep.sum() < 2:
        return {"found": False, "note": "too few shared features to compare."}
    Xk, qk = X[:, keep], q[keep]
    # rows with any NaN in the kept columns are unusable
    good = ~np.isnan(Xk).any(axis=1)
    Xk, yk = Xk[good], np.asarray(y)[good]
    if Xk.shape[0] < _MIN_CORPUS:
        return {"found": False, "note": "not enough complete analogs yet."}

    from research.market_memory import mahalanobis_knn, robust_inv_cov
    inv = robust_inv_cov(Xk)
    idx, dist = mahalanobis_knn(qk, Xk, inv, k)
    outs = yk[idx].astype(float)
    outs = outs[~np.isnan(outs)]
    if outs.size == 0:
        return {"found": False, "note": "analogs had no settled outcomes."}
    win_rate = float((outs >= _WIN_PCT).mean())
    return {
        "found": True,
        "n_similar": int(outs.size),
        "median_outcome_pct": round(float(np.median(outs)), 2),
        "worst_pct": round(float(outs.min()), 2),        # drawdown proxy
        "best_pct": round(float(outs.max()), 2),
        "win_rate": round(win_rate, 3),
        "environment": _environment_tags(query_features),
        "summary": (f"{outs.size} similar setups: median {np.median(outs):+.1f}% "
                    f"forward, {win_rate*100:.0f}% cleared +{_WIN_PCT:g}%, worst "
                    f"{outs.min():+.1f}%."),
    }


def _num(v):
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")
