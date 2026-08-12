"""Phase A / A6 — portfolio network advisory tests."""
from __future__ import annotations

import numpy as np
import pytest

from risk.correlation import clusters_from_corr
from research.portfolio_network import (
    analyze_network,
    incremental_candidate_risk,
)
from research.portfolio_network.engine import corr_from_returns


def _corr_two_clusters():
    # A,B,C tightly correlated; D,E tightly correlated; cross ~0
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=80)
    f2 = rng.normal(size=80)
    rets = {
        "A": f1 + rng.normal(scale=0.05, size=80),
        "B": f1 + rng.normal(scale=0.05, size=80),
        "C": f1 + rng.normal(scale=0.05, size=80),
        "D": f2 + rng.normal(scale=0.05, size=80),
        "E": f2 + rng.normal(scale=0.05, size=80),
    }
    return corr_from_returns(rets, min_overlap=30)


def test_network_communities_and_centrality():
    corr = _corr_two_clusters()
    result = analyze_network(corr, ["A", "B", "C", "D", "E"], as_of="d080",
                             threshold=0.5, portfolio=["A", "B"])
    assert result.evidence_status == "RESEARCH_ONLY"
    assert result.meta["auto_block"] is False
    assert result.meta["production_authority"] is False
    assert result.n_communities >= 2
    assert result.n_edges >= 2
    assert "A" in result.centrality_degree
    assert "A" in result.centrality_eigenvector
    assert "A" in result.centrality_betweenness
    assert 0.0 <= result.portfolio_network_concentration <= 1.0


def test_incremental_risk_of_same_community_candidate():
    corr = _corr_two_clusters()
    result = analyze_network(corr, ["A", "B", "C", "D", "E"], as_of="d080",
                             threshold=0.5, portfolio=["A", "B"])
    # C should join A's community → positive incremental risk
    info = incremental_candidate_risk(result, "C", portfolio=["A", "B"])
    assert info["blocks_trade"] is False
    assert info["advisory"] is True
    assert info["joins_existing_community"] is True
    assert info["incremental_cluster_risk"] > 0

    # D should open/diversify relative to A-B community
    info_d = incremental_candidate_risk(result, "D", portfolio=["A", "B"])
    assert info_d["joins_existing_community"] is False
    assert info_d["incremental_cluster_risk"] == 0.0


def test_perfect_twin_increases_concentration_vs_diversifier():
    corr = _corr_two_clusters()
    concentrated = analyze_network(
        corr, ["A", "B", "C", "D", "E"], as_of="t", threshold=0.5,
        portfolio=["A", "B", "C"],
    )
    diversified = analyze_network(
        corr, ["A", "B", "C", "D", "E"], as_of="t", threshold=0.5,
        portfolio=["A", "D"],
    )
    assert concentrated.portfolio_network_concentration >= diversified.portfolio_network_concentration


def test_complements_pairwise_correlation_clusters():
    corr = _corr_two_clusters()
    symbols = ["A", "B", "C", "D", "E"]
    # Incumbent union-find clusters still work on the same corr dict.
    clusters = clusters_from_corr(symbols, corr, threshold=0.5)
    result = analyze_network(corr, symbols, as_of="t", threshold=0.5)
    assert len(clusters) >= 2
    assert result.n_communities >= 2
    # Network does not mutate or replace clusters_from_corr output shape.
    assert all(isinstance(g, list) for g in clusters)


def test_to_cycle_view_seam():
    corr = _corr_two_clusters()
    result = analyze_network(corr, ["A", "B", "C"], as_of="t", threshold=0.5,
                             portfolio=["A"])
    view = result.to_cycle_view()
    assert view.as_of == "t"
    assert view.evidence_status == "RESEARCH_ONLY"
    assert view.community_id_by_symbol
