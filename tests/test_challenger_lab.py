"""Phase A / A3 — challenger lab wiring tests (network-free)."""
from __future__ import annotations

import numpy as np
import pytest

from research import registry as REG
from research import scientific_memory as SM
from research.challenger_lab import (
    BakeOffConfig,
    LogisticChallenger,
    NaiveBaseline,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    VERDICT_KEEP_INCUMBENT,
    VERDICT_PROMOTE,
    run_bakeoff,
)
from research.horizons.catalog import get_legacy_mh_target


def _linear_xy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    # Separable-ish directional labels in {-1, 0, 1}
    score = X[:, 0] - 0.2 * X[:, 1]
    y = np.where(score > 0.4, 1.0, np.where(score < -0.4, -1.0, 0.0))
    return X, y


@pytest.fixture
def isolated_research_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(REG, "_DB_PATH", tmp_path / "experiments.db")
    monkeypatch.setattr(SM, "_DB_PATH", tmp_path / "scientific_memory.db")
    return tmp_path


def test_identical_inputs_for_incumbent_and_challenger(isolated_research_dbs):
    X, y = _linear_xy()
    target = get_legacy_mh_target("5d")
    # Use absolute_return target with 5 bars for split sizing (labels already built)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")

    inc = NaiveBaseline()
    chal = LogisticChallenger(random_state=0)
    result = run_bakeoff(
        incumbent_model=inc,
        challenger_model=chal,
        X=X,
        y=y,
        target=target,
        feature_names=["a", "b", "c"],
        config=BakeOffConfig(min_oos=5, write_scientific_memory=True),
    )
    assert result.train_period["n"] > 0
    assert result.oos_period["n"] > 0
    assert result.features == ["a", "b", "c"]
    assert result.target["name"] == "5d"
    assert result.cost_model["round_trip_pct"] >= 0
    assert result.hypothesis_id
    assert result.live_behaviour_changed is False
    assert result.verdict in {
        VERDICT_PROMOTE, VERDICT_KEEP_INCUMBENT, VERDICT_FAIL, VERDICT_INCONCLUSIVE
    }
    # Provenance: experiment registered
    exp = REG.get_experiment(result.hypothesis_id)
    assert exp is not None
    assert exp["status"] in ("PROMOTED", "REJECTED")


def test_deterministic_provenance(isolated_research_dbs):
    X, y = _linear_xy(seed=1)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")
    cfg = BakeOffConfig(seed=7, min_oos=5, write_scientific_memory=False)

    r1 = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=7),
        X=X, y=y, target=target, config=cfg, code_hash="abc",
    )
    r2 = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=7),
        X=X, y=y, target=target, config=cfg, code_hash="abc",
    )
    # Same hypothesis definition → same id (pre-registration idempotency)
    assert r1.hypothesis_id == r2.hypothesis_id
    assert r1.oos_period == r2.oos_period


def test_no_promotion_without_evidence_gate(isolated_research_dbs):
    # Tiny OOS → cannot clear committee min_trades → not PROMOTE
    X, y = _linear_xy(n=40, seed=2)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")
    result = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=2),
        X=X, y=y, target=target,
        config=BakeOffConfig(min_oos=3, write_scientific_memory=False),
    )
    assert result.verdict != VERDICT_PROMOTE


def test_failed_challenger_preserved_in_scientific_memory(isolated_research_dbs, monkeypatch):
    X, y = _linear_xy(n=400, seed=3)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")

    from research.autonomy.challenge import CommitteeDecision, REJECT

    def _reject_committee(ctx, *, producer, committee_actor="promotion_committee"):
        return CommitteeDecision(REJECT, (), "forced reject for unit test (leakage)")

    monkeypatch.setattr(
        "research.challenger_lab.bakeoff.CH.promotion_committee", _reject_committee
    )

    result = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=3),
        X=X, y=y, target=target,
        config=BakeOffConfig(min_oos=5, write_scientific_memory=True, role="unit_test_role"),
    )
    assert result.verdict == VERDICT_FAIL
    beliefs = SM.list_beliefs(status=SM.REJECTED)
    assert any("challenger" in (b.get("statement") or "") for b in beliefs)


def test_live_production_unchanged_by_default(isolated_research_dbs):
    X, y = _linear_xy(seed=4)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")
    result = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=4),
        X=X, y=y, target=target,
        config=BakeOffConfig(
            role="scorer_weights",  # a role that exists in tests — must not auto-write
            persist_champion=False,
            min_oos=5,
            write_scientific_memory=False,
        ),
    )
    assert result.live_behaviour_changed is False
    assert REG.current_champion("scorer_weights") is None


def test_persist_champion_opt_in_research_role_only(isolated_research_dbs):
    """Even with persist_champion, this only touches the research champions table
    for an explicit research role — never scanner live behaviour."""
    X, y = _linear_xy(n=200, seed=5)
    from research.horizons.catalog import absolute_return_target
    target = absolute_return_target("5d")

    # Make a strong challenger signal: y perfectly recoverable from X[:,0] sign
    y = np.sign(X[:, 0])
    y[y == 0] = 1.0

    result = run_bakeoff(
        incumbent_model=NaiveBaseline(),
        challenger_model=LogisticChallenger(random_state=5),
        X=X, y=y, target=target,
        config=BakeOffConfig(
            role="research_unit_test_role",
            persist_champion=True,
            min_oos=5,
            write_scientific_memory=False,
        ),
    )
    assert result.live_behaviour_changed is False
    # Champion table may or may not update depending on evidence clearance;
    # the invariant is live_behaviour_changed stays False.
    if result.verdict == VERDICT_PROMOTE:
        champ = REG.current_champion("research_unit_test_role")
        assert champ is not None
