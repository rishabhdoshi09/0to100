"""
Market structure discovery — hierarchical + PCA + k-means (research only).

Inputs must already be point-in-time safe (e.g. built via PitContract/Snapshot
with ``through=as_of``). This module never fetches live data and never places
trades.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketStructureResult:
    as_of: str
    method: str
    symbols: tuple[str, ...]
    cluster_id_by_symbol: Mapping[str, int]
    cluster_membership: Mapping[int, tuple[str, ...]]
    cluster_stability: float | None
    outlier_score_by_symbol: Mapping[str, float]
    structural_shift_score: float | None
    latent_factor_exposure: Mapping[str, Mapping[str, float]]
    n_clusters: int
    lookback: int
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"

    def to_cycle_view(self):
        from research.intelligence.runtime.research_seams import MarketStructureView

        return MarketStructureView(
            as_of=self.as_of,
            method=self.method,
            cluster_id_by_symbol=dict(self.cluster_id_by_symbol),
            cluster_stability=self.cluster_stability,
            structural_shift_score=self.structural_shift_score,
            outlier_score_by_symbol=dict(self.outlier_score_by_symbol),
            latent_factor_exposure={k: dict(v) for k, v in self.latent_factor_exposure.items()},
            meta=dict(self.meta),
            evidence_status=self.evidence_status,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def returns_matrix_from_closes(
    closes: Mapping[str, Any],
    *,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Build a date×symbol simple-return matrix from close series.

    ``closes`` maps symbol → sequence/Series of closes already truncated to as_of.
    """
    series = {}
    for sym, raw in closes.items():
        s = pd.Series(raw, dtype=float) if not isinstance(raw, pd.Series) else raw.astype(float)
        series[str(sym).upper()] = s.reset_index(drop=True)
    df = pd.DataFrame(series)
    rets = df.pct_change().dropna(how="all")
    if lookback is not None and lookback > 0:
        rets = rets.iloc[-int(lookback):]
    return rets


def _corr_distance(rets: pd.DataFrame) -> np.ndarray:
    corr = rets.corr().fillna(0.0).to_numpy()
    # Numerical safety: clip correlation
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    return dist


def _hierarchical_labels(dist: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    # sklearn ≥1.2 uses metric=; older uses affinity=
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )
    return model.fit_predict(dist)


def _kmeans_labels(feats: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(feats)


def _pca_exposures(rets: pd.DataFrame, n_components: int) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.decomposition import PCA

    X = rets.fillna(0.0).to_numpy().T  # symbols × time
    n_components = max(1, min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X)  # symbols × components
    return scores, pca.explained_variance_ratio_, {
        "n_components": n_components,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }


def _outlier_scores(feats: np.ndarray) -> np.ndarray:
    # Mahalanobis-ish via z-score distance from centroid (transparent).
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (feats - mu) / sd
    return np.sqrt((z ** 2).sum(axis=1))


def _stability(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index in [≈-1, 1]; 1 = identical partitions."""
    from sklearn.metrics import adjusted_rand_score

    if labels_a.size == 0 or labels_b.size == 0:
        return 0.0
    return float(adjusted_rand_score(labels_a, labels_b))


def discover_structure(
    closes: Mapping[str, Any],
    *,
    as_of: str,
    method: str = "hierarchical",
    n_clusters: int = 5,
    lookback: int = 60,
    n_components: int = 3,
    seed: int = 42,
    prior_labels: Mapping[str, int] | None = None,
) -> MarketStructureResult:
    """Discover clusters/factors from PIT closes.

    ``method`` ∈ {hierarchical, kmeans, pca_kmeans}.
    """
    method = str(method).lower().strip()
    if method not in ("hierarchical", "kmeans", "pca_kmeans"):
        raise ValueError(f"unsupported method '{method}'")

    rets = returns_matrix_from_closes(closes, lookback=lookback)
    # Drop symbols with insufficient observations
    rets = rets.dropna(axis=1, thresh=max(5, lookback // 3))
    symbols = tuple(rets.columns.tolist())
    if len(symbols) < max(2, n_clusters):
        return MarketStructureResult(
            as_of=as_of,
            method=method,
            symbols=symbols,
            cluster_id_by_symbol={s: 0 for s in symbols},
            cluster_membership={0: symbols},
            cluster_stability=None,
            outlier_score_by_symbol={s: 0.0 for s in symbols},
            structural_shift_score=None,
            latent_factor_exposure={},
            n_clusters=1 if symbols else 0,
            lookback=lookback,
            meta={"reason": "insufficient_symbols", "n_symbols": len(symbols)},
        )

    n_clusters = max(2, min(int(n_clusters), len(symbols)))
    scores, var_ratio, pca_meta = _pca_exposures(rets, n_components)
    dist = _corr_distance(rets)

    if method == "hierarchical":
        labels = _hierarchical_labels(dist, n_clusters, seed)
    elif method == "kmeans":
        labels = _kmeans_labels(scores, n_clusters, seed)
    else:  # pca_kmeans
        labels = _kmeans_labels(scores, n_clusters, seed)

    # Half-window stability
    mid = max(5, len(rets) // 2)
    rets_a, rets_b = rets.iloc[:mid], rets.iloc[mid:]
    stability = None
    if rets_a.shape[0] >= 5 and rets_b.shape[0] >= 5:
        try:
            if method == "hierarchical":
                la = _hierarchical_labels(_corr_distance(rets_a), n_clusters, seed)
                lb = _hierarchical_labels(_corr_distance(rets_b), n_clusters, seed)
            else:
                sa, _, _ = _pca_exposures(rets_a, n_components)
                sb, _, _ = _pca_exposures(rets_b, n_components)
                # Align symbol order
                la = _kmeans_labels(sa, n_clusters, seed)
                lb = _kmeans_labels(sb, n_clusters, seed)
            # Compare labels on shared symbol positions (same column order)
            stability = _stability(la, lb)
        except Exception:
            stability = None

    outliers = _outlier_scores(scores)
    cluster_id_by_symbol = {symbols[i]: int(labels[i]) for i in range(len(symbols))}
    membership: dict[int, list[str]] = {}
    for sym, cid in cluster_id_by_symbol.items():
        membership.setdefault(cid, []).append(sym)
    cluster_membership = {k: tuple(v) for k, v in sorted(membership.items())}

    latent = {
        symbols[i]: {f"pc{j+1}": float(scores[i, j]) for j in range(scores.shape[1])}
        for i in range(len(symbols))
    }
    outlier_score_by_symbol = {symbols[i]: float(outliers[i]) for i in range(len(symbols))}

    shift = None
    if prior_labels:
        aligned_prior = np.array([int(prior_labels.get(s, -1)) for s in symbols])
        if np.any(aligned_prior >= 0):
            # Only score symbols present in prior
            mask = aligned_prior >= 0
            if mask.sum() >= 2:
                shift = 1.0 - _stability(labels[mask], aligned_prior[mask])

    return MarketStructureResult(
        as_of=as_of,
        method=method,
        symbols=symbols,
        cluster_id_by_symbol=cluster_id_by_symbol,
        cluster_membership=cluster_membership,
        cluster_stability=None if stability is None else round(float(stability), 4),
        outlier_score_by_symbol=outlier_score_by_symbol,
        structural_shift_score=None if shift is None else round(float(shift), 4),
        latent_factor_exposure=latent,
        n_clusters=n_clusters,
        lookback=lookback,
        meta={
            "pca": pca_meta,
            "seed": seed,
            "evidence_status": "RESEARCH_ONLY",
            "production_authority": False,
        },
    )
