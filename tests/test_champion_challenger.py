"""Phase 2 — Champion / Challenger research factory."""

from __future__ import annotations

from product.champion_challenger import (
    ELIGIBLE,
    PROMOTED,
    REJECTED,
    SHADOW,
    ChampionChallengerEngine,
    champion_identity,
)
from product.strategy_catalog import ensemble_identity


def _pit():
    return [
        {"symbol": "TCS", "selection_score": 92, "setup_label": "VCP", "reco_tier": "high_conviction", "sector": "Technology"},
        {"symbol": "ONGC", "selection_score": 89, "setup_label": "VCP", "reco_tier": "good_setup", "sector": "Energy"},
        {"symbol": "INFY", "selection_score": 80, "setup_label": "SEPA", "reco_tier": "good_setup", "sector": "Technology"},
    ]


def test_champion_unaffected_by_challenger_registration(tmp_path):
    before = ensemble_identity()["rules_hash"]
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    champ = eng.champion()
    assert champ["role"] == "CHAMPION"
    assert champ["can_execute"] is True
    assert champ["rules_hash"] == before
    eng.register(
        challenger_id="w_vcp",
        hypothesis="VCP names deserve 1.2x rank",
        changed_behavior="ranking_weights.VCP=1.2",
        rules={"ranking_weights": {"VCP": 1.2}},
        oos_data="",
    )
    assert ensemble_identity()["rules_hash"] == before
    assert eng.champion()["rules_hash"] == before
    assert eng.get("w_vcp")["can_execute"] is False
    assert eng.get("w_vcp")["controls_paper_capital"] is False
    assert eng.get("w_vcp")["status"] == SHADOW


def test_challenger_sees_same_pit_evidence(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    eng.register(
        challenger_id="w_vcp",
        hypothesis="boost VCP",
        changed_behavior="VCP weight",
        rules={"ranking_weights": {"VCP": 1.5}},
        oos_data="oos-2024",
    )
    pit = _pit()
    out = eng.evaluate_same_pit("w_vcp", pit, as_of="2026-09-01", champion_ranking=["TCS", "ONGC", "INFY"])
    assert out["same_pit"] is True
    assert out["n_opportunities"] == 3
    assert out["can_execute"] is False
    assert {r["symbol"] for r in out["ranking"]} == {"TCS", "ONGC", "INFY"}
    # Frozen PIT: mutating the original list later must not change stored n
    pit.append({"symbol": "FAKE", "selection_score": 99})
    stored = eng.get("w_vcp")["pit_evaluations"][-1]
    assert stored["n"] == 3


def test_challenger_cannot_execute(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    row = eng.register(
        challenger_id="x",
        hypothesis="h",
        changed_behavior="b",
        rules={"entry_rules": {"min_rs": 80}},
        oos_data="oos",
    )
    assert row["can_execute"] is False
    assert row["controls_paper_capital"] is False
    out = eng.evaluate_same_pit("x", _pit(), as_of="2026-09-01")
    assert out["can_execute"] is False


def test_challenger_evidence_is_separate(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    eng.register(challenger_id="a", hypothesis="h", changed_behavior="b", rules={"x": 1}, oos_data="oos")
    eng.record_forward("a", pnl=1.0, split="oos", sector="Technology")
    champ = champion_identity()
    assert "forward_observations" not in champ
    assert eng.get("a")["forward_observations"][0]["pnl"] == 1.0
    assert eng.compare("a")["evidence_store"] == "challenger"
    assert eng.compare("a")["champion_rules_hash"] == champ["rules_hash"]
    assert eng.compare("a")["challenger_rules_hash"] != "" 


def test_weak_challenger_rejected(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    eng.register(challenger_id="weak", hypothesis="h", changed_behavior="b", rules={"w": 1}, oos_data="oos")
    for _ in range(30):
        eng.record_forward("weak", pnl=-0.5, split="oos")
    cmp = eng.compare("weak")
    assert cmp["status"] == REJECTED
    assert eng.get("weak")["can_execute"] is False
    # Production hash unchanged
    assert eng.champion()["rules_hash"] == ensemble_identity()["rules_hash"]


def test_strong_challenger_remains_shadow_until_promotion_contract(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    eng.register(
        challenger_id="strong",
        hypothesis="better VCP filter",
        changed_behavior="entry_rules",
        rules={"entry_rules": {"min_rs": 70}},
        training_data="train",
        validation_data="val",
        oos_data="",  # in-sample only
    )
    for _ in range(40):
        eng.record_forward("strong", pnl=1.2, split="in_sample")
    blocked = eng.promote("strong")
    assert blocked["promoted"] is False
    assert "IN_SAMPLE_ONLY" in blocked["reasons"]
    assert blocked["champion_unchanged"] is True
    assert eng.get("strong")["status"] == SHADOW
    assert eng.champion()["rules_hash"] == ensemble_identity()["rules_hash"]


def test_production_rules_hash_changes_only_after_explicit_promotion(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    before = eng.champion()["rules_hash"]
    eng.register(
        challenger_id="ready",
        hypothesis="oos-validated overlay",
        changed_behavior="policy",
        rules={"evidence_policies": {"VCP|Technology": "SUPPORT"}},
        training_data="t",
        validation_data="v",
        oos_data="oos-2024",
    )
    for _ in range(35):
        eng.record_forward("ready", pnl=0.8, split="oos", execution_adjusted_pnl=0.6)
    # Still not production — compare does not promote
    assert eng.compare("ready")["status"] != PROMOTED
    assert eng.champion()["rules_hash"] == before
    result = eng.promote("ready", adversarial_status="SURVIVED")
    assert result["promoted"] is True
    assert result["explicit"] is True
    assert result["previous_rules_hash"] == before
    assert result["champion_rules_hash"] != before
    assert eng.champion()["version"] == result["version"]
    assert eng.get("ready")["status"] == PROMOTED
    # Ensemble production identity file is NOT silently rewritten
    assert ensemble_identity()["rules_hash"] == before
