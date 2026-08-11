"""Compare discovered clusters to existing label systems (sectors / corr clusters)."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _ari(a: Mapping[str, int], b: Mapping[str, int]) -> float | None:
    from sklearn.metrics import adjusted_rand_score

    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    return float(adjusted_rand_score(
        [int(a[s]) for s in common],
        [int(b[s]) for s in common],
    ))


def compare_to_labels(
    discovered: Mapping[str, int],
    *,
    sector_labels: Mapping[str, str] | None = None,
    correlation_clusters: Mapping[str, int] | None = None,
    regime_labels: Mapping[str, str] | None = None,
) -> dict:
    """Agreement stats vs incumbent lenses. Pure research reporting."""
    out: dict = {"n_discovered": len(discovered)}
    if sector_labels:
        # Map sector strings to integer codes
        codes = {s: i for i, s in enumerate(sorted(set(sector_labels.values())))}
        sector_ids = {sym: codes[sec] for sym, sec in sector_labels.items() if sym in discovered}
        out["vs_sectors_ari"] = _ari(discovered, sector_ids)
        out["sector_coverage"] = len(sector_ids)
    if correlation_clusters:
        out["vs_correlation_clusters_ari"] = _ari(discovered, correlation_clusters)
    if regime_labels:
        # Regime is usually market-wide; per-symbol regime rare — report coverage only.
        out["regime_label_coverage"] = sum(1 for s in discovered if s in regime_labels)
    return out


def correlation_clusters_from_returns(
    returns_by_symbol: Mapping[str, np.ndarray | list],
    *,
    rho: float = 0.70,
) -> dict[str, int]:
    """Thin wrapper mirroring risk.correlation union-find intent on a returns panel."""
    symbols = sorted(returns_by_symbol)
    if len(symbols) < 2:
        return {s: 0 for s in symbols}
    # Align lengths
    n = min(len(returns_by_symbol[s]) for s in symbols)
    mat = np.column_stack([np.asarray(returns_by_symbol[s], float)[-n:] for s in symbols])
    # Drop NaN rows
    mask = ~np.isnan(mat).any(axis=1)
    mat = mat[mask]
    if mat.shape[0] < 3:
        return {s: i for i, s in enumerate(symbols)}
    corr = np.corrcoef(mat, rowvar=False)
    parent = {s: s for s in symbols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, si in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            sj = symbols[j]
            if corr[i, j] >= rho:
                union(si, sj)
    roots = {}
    out = {}
    next_id = 0
    for s in symbols:
        r = find(s)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        out[s] = roots[r]
    return out
