"""Phase A / A5 — market structure research-only tests."""
from __future__ import annotations

import numpy as np
import pytest

from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.pit_contract import PitContract
from research.market_structure import (
    compare_to_labels,
    discover_structure,
    returns_matrix_from_closes,
)
from research.market_structure.benchmarks import correlation_clusters_from_returns
from research.market_structure.pit_inputs import closes_from_pit


def _synth_closes(n=80, n_sym=8, seed=0):
    rng = np.random.default_rng(seed)
    # Two latent factors → two natural clusters
    f1 = rng.normal(scale=0.01, size=n).cumsum()
    f2 = rng.normal(scale=0.01, size=n).cumsum()
    closes = {}
    for i in range(n_sym):
        base = 100 + (f1 if i < n_sym // 2 else f2)
        noise = rng.normal(scale=0.5, size=n)
        closes[f"S{i}"] = list(100 + np.exp(base / 100.0) + noise)
    return closes


def test_discover_hierarchical_produces_clusters():
    closes = _synth_closes()
    result = discover_structure(
        closes, as_of="d079", method="hierarchical", n_clusters=2, lookback=60, seed=0
    )
    assert result.evidence_status == "RESEARCH_ONLY"
    assert result.meta.get("production_authority") is False
    assert len(result.cluster_id_by_symbol) >= 4
    assert result.n_clusters == 2
    assert set(result.cluster_membership.keys()) <= set(range(2))
    assert result.cluster_stability is None or -1.0 <= result.cluster_stability <= 1.0
    assert result.latent_factor_exposure
    assert result.outlier_score_by_symbol


def test_kmeans_and_pca_kmeans_run():
    closes = _synth_closes(seed=1)
    a = discover_structure(closes, as_of="t", method="kmeans", n_clusters=3, lookback=50)
    b = discover_structure(closes, as_of="t", method="pca_kmeans", n_clusters=3, lookback=50)
    assert a.n_clusters == 3 and b.n_clusters == 3


def test_structural_shift_vs_prior():
    closes = _synth_closes(seed=2)
    first = discover_structure(closes, as_of="t", method="hierarchical", n_clusters=2)
    # Prior equal to current → shift ≈ 0
    shift0 = discover_structure(
        closes, as_of="t", method="hierarchical", n_clusters=2,
        prior_labels=first.cluster_id_by_symbol,
    )
    assert shift0.structural_shift_score == pytest.approx(0.0, abs=1e-9)


def test_compare_to_sector_and_corr_labels():
    closes = _synth_closes(seed=3)
    result = discover_structure(closes, as_of="t", method="hierarchical", n_clusters=2)
    sectors = {s: ("BANK" if i < 4 else "IT") for i, s in enumerate(result.symbols)}
    rets = returns_matrix_from_closes(closes, lookback=60)
    corr_clusters = correlation_clusters_from_returns(
        {c: rets[c].to_numpy() for c in rets.columns}, rho=0.3
    )
    stats = compare_to_labels(
        result.cluster_id_by_symbol,
        sector_labels=sectors,
        correlation_clusters=corr_clusters,
    )
    assert "vs_sectors_ari" in stats
    assert "vs_correlation_clusters_ari" in stats


def test_pit_contract_inputs_only_past_bars(tmp_path):
    store = SnapshotStore(tmp_path)
    rows = []
    for i, c in enumerate([100, 101, 102, 103, 104, 105]):
        rows.append(("AAA", f"d{i:03d}", c - 1, c + 1, c - 2, c, 1000, "EQ"))
    for i, c in enumerate([50, 51, 52, 53, 54, 55]):
        rows.append(("BBB", f"d{i:03d}", c - 1, c + 1, c - 2, c, 1000, "EQ"))
    sid = store.commit_snapshot(rows)
    pit = PitContract.from_store(store, sid)
    closes = closes_from_pit(pit, ["AAA", "BBB"], as_of="d003")
    assert set(closes) == {"AAA", "BBB"}
    assert len(closes["AAA"]) == 4  # d000..d003 only
    # Future as_of blocked → empty
    assert closes_from_pit(pit, ["AAA"], as_of="d999") == {}


def test_to_cycle_view_seam():
    closes = _synth_closes(n=40, n_sym=4, seed=4)
    result = discover_structure(closes, as_of="d039", method="kmeans", n_clusters=2, lookback=30)
    view = result.to_cycle_view()
    assert view.as_of == "d039"
    assert view.evidence_status == "RESEARCH_ONLY"
    assert view.cluster_id_by_symbol
