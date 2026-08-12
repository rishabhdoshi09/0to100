"""
Correlation-graph portfolio network analysis (research / advisory).

Complements ``risk.correlation`` — does not replace pairwise guards or
``portfolio_risk.check_new_trade``. No automatic trade blocking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

# Default edge threshold matches risk.correlation.CLUSTER_RHO intent.
DEFAULT_RHO = 0.70


@dataclass(frozen=True)
class NetworkRiskResult:
    as_of: str
    symbols: tuple[str, ...]
    community_id_by_symbol: Mapping[str, int]
    centrality_degree: Mapping[str, float]
    centrality_eigenvector: Mapping[str, float]
    centrality_betweenness: Mapping[str, float]
    portfolio_network_concentration: float
    incremental_cluster_risk: Mapping[str, float]
    contagion_score: float
    n_communities: int
    n_edges: int
    threshold: float
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"

    def to_cycle_view(self, *, incremental: Mapping[str, float] | None = None):
        from research.intelligence.runtime.research_seams import NetworkRiskView

        return NetworkRiskView(
            as_of=self.as_of,
            community_id_by_symbol=dict(self.community_id_by_symbol),
            centrality_by_symbol=dict(self.centrality_degree),
            portfolio_network_concentration=self.portfolio_network_concentration,
            incremental_cluster_risk=dict(
                incremental if incremental is not None else self.incremental_cluster_risk
            ),
            contagion_score=self.contagion_score,
            meta={
                **dict(self.meta),
                "eigenvector": dict(self.centrality_eigenvector),
                "betweenness": dict(self.centrality_betweenness),
                "n_communities": self.n_communities,
                "n_edges": self.n_edges,
                "threshold": self.threshold,
            },
            evidence_status=self.evidence_status,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_correlation_graph(
    corr: Mapping[tuple[str, str], float],
    symbols: Iterable[str],
    *,
    threshold: float = DEFAULT_RHO,
) -> tuple[list[str], np.ndarray, int]:
    """Return (symbols, adjacency, n_edges) from a pairwise corr lookup.

    ``corr`` keys are (a, b) as produced by ``risk.correlation.pairwise_corr``.
    """
    syms = sorted({str(s).upper() for s in symbols})
    idx = {s: i for i, s in enumerate(syms)}
    n = len(syms)
    adj = np.zeros((n, n), dtype=float)
    edges = 0
    for (a, b), rho in (corr or {}).items():
        a, b = str(a).upper(), str(b).upper()
        if a not in idx or b not in idx:
            continue
        if float(rho) >= threshold:
            i, j = idx[a], idx[b]
            if i == j:
                continue
            if adj[i, j] == 0:
                edges += 1
            adj[i, j] = adj[j, i] = float(rho)
    return syms, adj, edges


def _communities(adj: np.ndarray) -> np.ndarray:
    """Connected components on thresholded edges (transparent community proxy)."""
    n = adj.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                union(i, j)
    roots = {}
    labels = np.zeros(n, dtype=int)
    next_id = 0
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        labels[i] = roots[r]
    return labels


def _degree_centrality(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    if n <= 1:
        return np.zeros(n)
    deg = (adj > 0).sum(axis=1).astype(float)
    return deg / (n - 1)


def _eigenvector_centrality(adj: np.ndarray, *, iters: int = 100) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    A = (adj > 0).astype(float)
    v = np.ones(n) / n
    for _ in range(iters):
        w = A @ v
        norm = np.linalg.norm(w)
        if norm < 1e-15:
            return np.zeros(n)
        v = w / norm
    return v


def _betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """Brandes betweenness on the unweighted threshold graph (small-n OK)."""
    n = adj.shape[0]
    if n <= 2:
        return np.zeros(n)
    A = (adj > 0)
    cb = np.zeros(n, dtype=float)
    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            nbrs = np.where(A[v])[0]
            for w in nbrs:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(int(w))
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    # Normalise for undirected graph
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    return cb * scale


def _herfindahl(weights: Mapping[Any, float]) -> float:
    vals = np.asarray(list(weights.values()), float)
    if vals.size == 0:
        return 0.0
    s = vals.sum()
    if s <= 0:
        return 0.0
    p = vals / s
    return float((p ** 2).sum())


def analyze_network(
    corr: Mapping[tuple[str, str], float],
    symbols: Iterable[str],
    *,
    as_of: str,
    threshold: float = DEFAULT_RHO,
    portfolio: Iterable[str] | None = None,
) -> NetworkRiskResult:
    """Build network metrics for ``symbols``; concentration uses ``portfolio`` subset."""
    syms, adj, n_edges = build_correlation_graph(corr, symbols, threshold=threshold)
    labels = _communities(adj)
    deg = _degree_centrality(adj)
    eig = _eigenvector_centrality(adj)
    btw = _betweenness_centrality(adj)

    community_id = {syms[i]: int(labels[i]) for i in range(len(syms))}
    centrality_degree = {syms[i]: round(float(deg[i]), 4) for i in range(len(syms))}
    centrality_eig = {syms[i]: round(float(eig[i]), 4) for i in range(len(syms))}
    centrality_btw = {syms[i]: round(float(btw[i]), 4) for i in range(len(syms))}

    port = [str(s).upper() for s in (portfolio or syms) if str(s).upper() in community_id]
    # Community exposure weights (equal weight within portfolio)
    comm_w: dict[int, float] = {}
    for s in port:
        cid = community_id[s]
        comm_w[cid] = comm_w.get(cid, 0.0) + 1.0
    concentration = _herfindahl(comm_w)

    # Contagion proxy: mean degree among portfolio nodes (0..1)
    if port:
        contagion = float(np.mean([centrality_degree[s] for s in port]))
    else:
        contagion = 0.0

    # Incremental risk for every symbol vs current portfolio communities
    port_comms = {community_id[s] for s in port}
    incremental = {}
    for s in syms:
        cid = community_id[s]
        if s in port:
            incremental[s] = 0.0
        elif cid in port_comms and len(port) > 0:
            # Joining an existing community: share of that community in the book
            incremental[s] = round(comm_w.get(cid, 0.0) / max(len(port), 1), 4)
        else:
            incremental[s] = 0.0  # opens a new community — diversification friendly

    return NetworkRiskResult(
        as_of=as_of,
        symbols=tuple(syms),
        community_id_by_symbol=community_id,
        centrality_degree=centrality_degree,
        centrality_eigenvector=centrality_eig,
        centrality_betweenness=centrality_btw,
        portfolio_network_concentration=round(concentration, 4),
        incremental_cluster_risk=incremental,
        contagion_score=round(contagion, 4),
        n_communities=int(len(set(labels))) if len(labels) else 0,
        n_edges=int(n_edges),
        threshold=float(threshold),
        meta={
            "production_authority": False,
            "advisory_only": True,
            "auto_block": False,
            "complements": ["risk.correlation", "risk.portfolio_risk"],
        },
    )


def incremental_candidate_risk(
    result: NetworkRiskResult,
    candidate: str,
    *,
    portfolio: Iterable[str],
) -> dict:
    """Explicit answer: incremental network/community risk of adding ``candidate``."""
    cand = str(candidate).upper()
    port = [str(s).upper() for s in portfolio]

    if cand not in result.community_id_by_symbol:
        return {
            "candidate": cand,
            "status": "UNKNOWN_SYMBOL",
            "incremental_cluster_risk": None,
            "joins_existing_community": None,
            "advisory": True,
            "blocks_trade": False,
        }
    cid = result.community_id_by_symbol[cand]
    port_comms = {
        result.community_id_by_symbol[s]
        for s in port
        if s in result.community_id_by_symbol
    }
    joins = cid in port_comms
    if cand in port:
        incr = 0.0
        joins = True
    elif joins and port:
        same = sum(
            1 for s in port
            if result.community_id_by_symbol.get(s) == cid
        )
        incr = round(same / len(port), 4)
    else:
        incr = 0.0

    return {
        "candidate": cand,
        "community_id": cid,
        "joins_existing_community": joins,
        "incremental_cluster_risk": incr,
        "portfolio_network_concentration": result.portfolio_network_concentration,
        "contagion_score": result.contagion_score,
        "vs_pairwise_note": (
            "Complement to risk.correlation.clusters_from_corr — does not replace it"
        ),
        "advisory": True,
        "blocks_trade": False,
        "evidence_status": "RESEARCH_ONLY",
    }


def corr_from_returns(
    returns_by_symbol: Mapping[str, np.ndarray | list],
    *,
    min_overlap: int = 30,
) -> dict[tuple[str, str], float]:
    """Build a pairwise corr dict from an in-memory returns panel (tests / PIT)."""
    symbols = sorted(str(s).upper() for s in returns_by_symbol)
    out: dict[tuple[str, str], float] = {}
    arrays = {s: np.asarray(returns_by_symbol[s], float) for s in returns_by_symbol}
    # normalise keys
    arrays = {str(k).upper(): v for k, v in arrays.items()}
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            ra, rb = arrays[a], arrays[b]
            n = min(len(ra), len(rb))
            if n < min_overlap:
                continue
            x, y = ra[-n:], rb[-n:]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < min_overlap:
                continue
            rho = float(np.corrcoef(x[mask], y[mask])[0, 1])
            if rho == rho:
                out[(a, b)] = round(rho, 3)
    return out
