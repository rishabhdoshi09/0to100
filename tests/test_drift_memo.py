"""QT-PRA-075: research views must not recompute seeded drift tests per GET.

/api/research-director took 8.4 s warm on the owner's Mac, ~4.5 s of it in
``assess_drift`` permutation tests that are deterministic for a fixed seed.
The memo must be exactly equivalent, keyed on the stream content, and must
not let a caller corrupt the cached result.
"""
from __future__ import annotations

import numpy as np
import pytest

from research import drift as D


@pytest.fixture(autouse=True)
def _fresh_memo():
    D._assess_drift_memo.cache_clear()
    yield
    D._assess_drift_memo.cache_clear()


def _streams():
    rng = np.random.default_rng(7)
    yield list(rng.normal(0.2, 1.0, 120))
    yield list(np.concatenate([rng.normal(0.6, 0.8, 80), rng.normal(-0.5, 0.8, 60)]))
    yield list(rng.normal(0.0, 1.0, 12))  # below min_samples


@pytest.mark.parametrize("stream", list(_streams()))
def test_memoised_result_equals_direct_computation(stream):
    assert D.assess_drift(stream) == D._assess_drift(stream)
    assert D.assess_drift(np.asarray(stream)) == D._assess_drift(stream)


def test_unchanged_stream_is_computed_once_and_a_new_outcome_recomputes(monkeypatch):
    calls = {"n": 0}
    real = D._assess_drift

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(D, "_assess_drift", counting)
    stream = list(np.random.default_rng(3).normal(0.1, 1.0, 90))
    first = D.assess_drift(stream)
    for _ in range(4):
        assert D.assess_drift(list(stream)) == first
    assert calls["n"] == 1
    D.assess_drift(stream + [2.5])
    assert calls["n"] == 2


def test_caller_mutation_cannot_poison_the_memo():
    stream = list(np.random.default_rng(5).normal(0.1, 1.0, 90))
    got = D.assess_drift(stream)
    got.status = "TAMPERED"
    got.insight = "x"
    again = D.assess_drift(stream)
    assert again.status != "TAMPERED"
    assert again == D._assess_drift(stream)


def test_unseeded_calls_are_never_memoised(monkeypatch):
    calls = {"n": 0}
    real = D._assess_drift

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(D, "_assess_drift", counting)
    stream = list(np.random.default_rng(9).normal(0.1, 1.0, 90))
    D.assess_drift(stream, seed=None)
    D.assess_drift(stream, seed=None)
    assert calls["n"] == 2


def test_research_overview_scans_the_feature_store_once(monkeypatch):
    """The overview used to run data_health twice (directly and inside
    research_debt), scanning every feature-store observation each time."""
    from research import feature_store as FS
    from research import research_overview as RO

    calls = {"coverage": 0}

    def coverage(kind=None):
        calls["coverage"] += 1
        return [{"feature": "f", "n": 1, "fill_rate": 0.25, "problems": {"STALE": 2}}]

    monkeypatch.setattr(FS, "feature_coverage", coverage)
    monkeypatch.setattr(FS, "observation_counts", lambda: {"total": 1, "on_current_schema": False})
    out = RO.overview()

    assert calls["coverage"] == 1
    assert out["data_health"]["stale_values"] == 2
    assert out["research_debt"]["schemas_awaiting_migration"] == 1
    assert RO.research_debt()["schemas_awaiting_migration"] == 1  # standalone still reads it
