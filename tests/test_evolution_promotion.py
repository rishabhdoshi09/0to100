"""Scientific promotion gate scenarios (sections 13, 14, 21, 36, 40, 41).

Each scenario seeds graded shadow-decision rows directly (the realistic
production path to GRADED rows is exercised end-to-end in
tests/test_evolution_tournament.py and test_evolution_core.py; this file
isolates the scientific decision logic itself with controlled R-streams, the
only way to deterministically exercise "tiny sample", "fails FDR", "worse
drawdown" etc. without depending on real market data).
"""
from __future__ import annotations

from unittest import mock

import pytest

from product.evolution import policy_registry as PR
from product.evolution import promotion as PROMO
from product.evolution import shadow_decisions as SD


def _seed(policy_id, sid, r, *, regime="TRENDING_BULL", decision="ENTER_NOW"):
    row = {
        "schema_version": 1, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
        "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
        "decision": decision, "reason_code": "ELIGIBLE" if decision == "ENTER_NOW" else "REJECT",
        "adjusted_score": 50.0, "breakdown": {}, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "sector": "IT", "setup_label": "VCP", "regime": regime, "frozen_at": "x", "fingerprint": "x",
        "outcome": {"forward_return_pct": r * 5, "not_pnl": True},
        "classification": "WINNER_TAKEN" if r > 0 else "LOSER_TAKEN",
        "graded_at": "x", "not_pnl": True, "is_champion_decision": policy_id.endswith("CHAMP"),
        "counterfactual_R": r,
    }
    SD.save_graded_decision(row)


def _seed_pair(name, champ_rs, chal_rs, regimes=None):
    PR.register_policy(policy_id=f"{name}_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(
        policy_id=f"{name}_CHAL", domain=PR.EQUITY, hypothesis="test hypothesis",
        weights={"relative_strength_mult": 2.0}, status=PR.CHALLENGER, parent_policy_id=f"{name}_CHAMP",
    )
    regs = regimes or [("TRENDING_BULL" if i % 2 else "SIDEWAYS") for i in range(len(champ_rs))]
    for i, (cr, hr, reg) in enumerate(zip(champ_rs, chal_rs, regs)):
        _seed(f"{name}_CHAMP", f"{name}_snap_{i}", cr, regime=reg)
        _seed(f"{name}_CHAL", f"{name}_snap_{i}", hr, regime=reg)


# ── Scenario A: Champion performs better -> no promotion ───────────────────

def test_scenario_a_champion_better_no_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 40
    champ_rs = [1.5 if i % 2 == 0 else -0.5 for i in range(n)]
    chal_rs = [0.2 if i % 2 == 0 else -1.0 for i in range(n)]
    _seed_pair("A", champ_rs, chal_rs)
    result = PROMO.evaluate_promotion(PR.EQUITY, "A_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE
    assert result["incremental_expectancy_R"] < 0


# ── Scenario B: tiny sample looks better -> no promotion ────────────────────

def test_scenario_b_tiny_sample_no_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _seed_pair("B", [-1.0] * 5, [2.0] * 5)
    result = PROMO.evaluate_promotion(PR.EQUITY, "B_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE
    assert "sample" in result["reason"] or result["paired_snapshots"] < PROMO.MIN_PAIRED_SAMPLE
    assert result["paired_snapshots"] == 5


# ── Scenario C: wins historical, loses forward -> no promotion ─────────────

def test_scenario_c_forward_losing_edge_no_promotion_despite_claimed_prior(tmp_path, monkeypatch):
    """A challenger's registered hypothesis can claim anything; only the
    actual forward paired evidence decides. Simulate a "strong historical
    prior" claim with a genuinely losing FORWARD stream."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="C_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="C_CHAL", domain=PR.EQUITY,
        hypothesis="strong in historical walk-forward (claimed) -- real forward test below",
        weights={}, status=PR.CHALLENGER, parent_policy_id="C_CHAMP",
    )
    n = 40
    champ_rs = [0.2] * n
    chal_rs = [-0.3] * n  # forward evidence: genuinely worse than champion
    for i, (cr, hr) in enumerate(zip(champ_rs, chal_rs)):
        _seed("C_CHAMP", f"snap_{i}", cr)
        _seed("C_CHAL", f"snap_{i}", hr)
    result = PROMO.evaluate_promotion(PR.EQUITY, "C_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE


# ── Scenario D: robust forward edge -> promotion eligible ───────────────────

def test_scenario_d_robust_forward_edge_is_promotion_eligible(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    _seed_pair("D", champ_rs, chal_rs)
    result = PROMO.evaluate_promotion(PR.EQUITY, "D_CHAL")
    assert result["status"] == PROMO.PROMOTION_ELIGIBLE
    assert result["incremental_expectancy_R"] > 0
    assert result["harness_verdict"] == "PROMOTE"


# ── Scenario E: looks superior raw, fails FDR correction -> no promotion ──

def test_scenario_e_fails_fdr_correction_across_simultaneous_challengers(tmp_path, monkeypatch):
    """Directly exercise the Benjamini-Hochberg override: a challenger whose
    OWN harness verdict is PROMOTE must still be blocked if the batch-level
    FDR correction marks it not-significant once tested alongside siblings."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="E_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="E_CHAL", domain=PR.EQUITY, hypothesis="marginal", weights={}, status=PR.CHALLENGER, parent_policy_id="E_CHAMP")
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    for i, (cr, hr) in enumerate(zip(champ_rs, chal_rs)):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("E_CHAMP", f"snap_{i}", cr, regime=reg)
        _seed("E_CHAL", f"snap_{i}", hr, regime=reg)

    # Without correction, this looks PROMOTE (same data as scenario D).
    solo = PROMO.evaluate_promotion(PR.EQUITY, "E_CHAL")
    assert solo["status"] == PROMO.PROMOTION_ELIGIBLE

    # With an explicit FDR rejection (as evaluate_promotion_batch would
    # compute when testing many simultaneous challengers), it must flip.
    corrected = PROMO.evaluate_promotion(PR.EQUITY, "E_CHAL", fdr_rejected=False)
    assert corrected["status"] == PROMO.NOT_ELIGIBLE
    assert "FDR" in corrected["reason"]


def test_evaluate_promotion_batch_applies_fdr_correction_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="BATCH_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    for i in range(8):
        PR.register_policy(
            policy_id=f"BATCH_CHAL_{i}", domain=PR.EQUITY, hypothesis=f"variant {i}",
            weights={}, status=PR.CHALLENGER, parent_policy_id="BATCH_CHAMP",
        )
    import random
    random.seed(11)
    n = 35
    for i in range(n):
        champ_r = random.gauss(0, 1)
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("BATCH_CHAMP", f"snap_{i}", champ_r, regime=reg)
        for c in range(8):
            noise = random.gauss(0.02, 1.0)
            _seed(f"BATCH_CHAL_{c}", f"snap_{i}", champ_r + noise, regime=reg)

    batch = PROMO.evaluate_promotion_batch(PR.EQUITY)
    assert len(batch) == 8
    assert all("fdr_rejected" in r for r in batch)
    # Pure noise around a near-zero mean edge: FDR correction must not
    # rubber-stamp every challenger as independently significant.
    assert sum(1 for r in batch if r["status"] == PROMO.PROMOTION_ELIGIBLE) < 8


# ── Scenario F: higher expectancy, worse drawdown -> no promotion ──────────

def test_scenario_f_worse_drawdown_blocks_promotion_despite_higher_expectancy(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    champ_rs = [0.05] * 60
    # Ten-trade losing streak -> a real drawdown, before 50 solid winners
    # bring the challenger's overall mean clearly above the champion's.
    chal_rs = [-0.3] * 10 + [0.3] * 50
    _seed_pair("F", champ_rs, chal_rs)
    result = PROMO.evaluate_promotion(PR.EQUITY, "F_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE
    assert "drawdown" in result["reason"] or "risk" in result["reason"]
    assert result["challenger_max_drawdown_R"] < result["champion_max_drawdown_R"]


# ── Scenario G: promoted Champion deteriorates -> rollback restores prior ──

def test_scenario_g_rollback_restores_the_previous_champion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    _seed_pair("G", champ_rs, chal_rs)
    result = PROMO.evaluate_promotion(PR.EQUITY, "G_CHAL")
    assert result["status"] == PROMO.PROMOTION_ELIGIBLE

    advanced = PROMO.advance_eligible_to_probation(PR.EQUITY)
    assert any(p["policy_id"] == "G_CHAL" for p in advanced)
    assert PR.get_policy("G_CHAL")["status"] == PR.PROBATION
    for i in range(PROMO.MIN_PROBATION_ADDITIONAL_PAIRED):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("G_CHAMP", f"G_prob_{i}", -0.2, regime=reg)
        _seed("G_CHAL", f"G_prob_{i}", 0.3, regime=reg)

    PROMO.promote_to_champion(
        PR.EQUITY, "G_CHAL", actor="operator",
        reason="cleared gates plus probation evidence",
    )
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "G_CHAL"
    assert PR.get_policy("G_CHAMP")["status"] == PR.PROBATION

    PROMO.rollback(PR.EQUITY, actor="operator", reason="deteriorated after promotion")
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "G_CHAMP"
    assert PR.get_policy("G_CHAL")["status"] == PR.PROBATION
    # Nothing destroyed -- both policies keep their full lifecycle history.
    assert len(PR.get_policy("G_CHAMP")["lifecycle_history"]) >= 2
    assert len(PR.get_policy("G_CHAL")["lifecycle_history"]) >= 2


# ── Section 41: AUTO_PROMOTION_ENABLED is False; explicit promotion still works ──

def test_auto_promotion_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    assert PROMO.AUTO_PROMOTION_ENABLED is False
    PR.register_policy(policy_id="AUTO_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="AUTO_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="AUTO_CHAMP")
    with pytest.raises(RuntimeError, match="AUTO_PROMOTION_ENABLED"):
        PROMO.promote_to_champion(PR.EQUITY, "AUTO_CHAL", actor="scheduler", reason="auto", allow_auto=True)
    # Explicit human action cannot bypass scientific qualification either.
    with pytest.raises(RuntimeError, match="PROMOTION_ELIGIBLE"):
        PROMO.promote_to_champion(PR.EQUITY, "AUTO_CHAL", actor="operator", reason="manual review")
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "AUTO_CHAMP"


def test_no_single_metric_triggers_promotion_alone(tmp_path, monkeypatch):
    """A challenger with a PROMOTE harness verdict but insufficient regime
    breadth must still be blocked -- promotion requires ALL gates."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="NR_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="NR_CHAL", domain=PR.EQUITY, hypothesis="one-regime edge", weights={}, status=PR.CHALLENGER, parent_policy_id="NR_CHAMP")
    n = 30
    # Comparable (tied, both flat/monotonic) drawdown profiles, but the edge
    # is concentrated in ONE regime: the challenger does WORSE than the
    # champion in the bear half and much better in the bull half.
    champ_rs = [0.3] * n + [0.0] * n
    chal_rs = [0.0] * n + [0.6] * n
    regimes = ["TRENDING_BEAR"] * n + ["TRENDING_BULL"] * n
    for i, (cr, hr, reg) in enumerate(zip(champ_rs, chal_rs, regimes)):
        _seed("NR_CHAMP", f"snap_{i}", cr, regime=reg)
        _seed("NR_CHAL", f"snap_{i}", hr, regime=reg)
    result = PROMO.evaluate_promotion(PR.EQUITY, "NR_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE
    assert result["regime_breadth"]["regimes_acceptable"] < result["regime_breadth"]["regimes_observed"]


def test_one_regime_only_evidence_cannot_promote(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="ONE_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(
        policy_id="ONE_CHAL", domain=PR.EQUITY, hypothesis="great but one regime only",
        weights={"relative_strength_mult": 1.5}, status=PR.CHALLENGER, parent_policy_id="ONE_CHAMP",
    )
    for i in range(60):
        _seed("ONE_CHAMP", f"one_{i}", -0.2, regime="TRENDING_BULL")
        _seed("ONE_CHAL", f"one_{i}", 0.4, regime="TRENDING_BULL")
    result = PROMO.evaluate_promotion(PR.EQUITY, "ONE_CHAL")
    assert result["status"] == PROMO.NOT_ELIGIBLE
    assert "regime breadth" in result["reason"]


def test_promotion_batch_persists_scientific_proof(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    champ_rs = [(-0.25 if i % 2 == 0 else 0.05) for i in range(60)]
    chal_rs = [(0.30 if i % 2 == 0 else 0.20) for i in range(60)]
    _seed_pair("PROOF", champ_rs, chal_rs)
    proof_file = tmp_path / "proofs.jsonl"
    batch = PROMO.evaluate_promotion_batch(PR.EQUITY, proof_path=proof_file)
    row = next(r for r in batch if r["policy_id"] == "PROOF_CHAL")
    assert row["status"] == PROMO.PROMOTION_ELIGIBLE
    proof = PROMO.latest_promotion_proof("PROOF_CHAL", path=proof_file)
    assert proof is not None
    assert proof["paired_snapshots"] >= 30
    assert proof["policy_manifest_fingerprint"]


def test_recent_promoted_champion_has_hysteresis_before_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="HYS_CHAMP", domain=PR.EQUITY, hypothesis="incumbent",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="HYS_CHAL", domain=PR.EQUITY, hypothesis="challenger",
        weights={}, status=PR.CHALLENGER, parent_policy_id="HYS_CHAMP",
    )
    store = PR.load_registry()
    current = dict(store["policies"]["HYS_CHAMP"])
    current["promotion_history"] = [{
        "at": PROMO.datetime.now(PROMO.timezone.utc).isoformat(),
        "previous_champion_policy_id": "OLDER",
    }]
    store["policies"]["HYS_CHAMP"] = current
    PR.save_registry(store)

    monkeypatch.setattr(
        PROMO,
        "evaluate_promotion_batch",
        lambda *args, **kwargs: [{
            "policy_id": "HYS_CHAL",
            "status": PROMO.PROMOTION_ELIGIBLE,
            "incremental_expectancy_R": 0.5,
        }],
    )
    with pytest.raises(RuntimeError, match="hysteresis"):
        PROMO.promote_to_champion(
            PR.EQUITY, "HYS_CHAL", actor="operator", reason="too soon",
        )
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "HYS_CHAMP"


def test_materially_negative_challenger_becomes_retirement_eligible(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="RET_CHAMP", domain=PR.EQUITY, hypothesis="baseline",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="RET_CHAL", domain=PR.EQUITY, hypothesis="bad challenger",
        weights={}, status=PR.CHALLENGER, parent_policy_id="RET_CHAMP",
    )
    for i in range(80):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("RET_CHAMP", f"ret_{i}", 0.1, regime=reg)
        _seed("RET_CHAL", f"ret_{i}", -0.3, regime=reg)
    rows = PROMO.evaluate_retirement_batch(PR.EQUITY)
    verdict = next(r for r in rows if r["policy_id"] == "RET_CHAL")
    assert verdict["status"] == PROMO.RETIREMENT_ELIGIBLE
    retired = PROMO.retire_qualified_challengers(PR.EQUITY)
    assert any(r["policy_id"] == "RET_CHAL" for r in retired)
    assert PR.get_policy("RET_CHAL")["status"] == PR.RETIRED


def test_former_champion_is_never_auto_retired(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="SAFE_OLD", domain=PR.EQUITY, hypothesis="former champion",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="SAFE_NEW", domain=PR.EQUITY, hypothesis="current champion",
        weights={}, status=PR.CHALLENGER, parent_policy_id="SAFE_OLD",
    )
    PR.set_status("SAFE_OLD", PR.PROBATION, reason="test handoff")
    PR.set_status("SAFE_NEW", PR.CHAMPION, reason="test handoff")

    for i in range(80):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("SAFE_NEW", f"safe_{i}", 0.3, regime=reg)
        _seed("SAFE_OLD", f"safe_{i}", -0.4, regime=reg)

    rows = PROMO.evaluate_retirement_batch(PR.EQUITY)
    protected = next(r for r in rows if r["policy_id"] == "SAFE_OLD")
    assert protected["status"] == PROMO.NOT_ELIGIBLE
    assert protected["rollback_protected"] is True
    assert "rollback-protected" in protected["reason"]

    retired = PROMO.retire_qualified_challengers(PR.EQUITY)
    assert all(r["policy_id"] != "SAFE_OLD" for r in retired)
    assert PR.get_policy("SAFE_OLD")["status"] == PR.PROBATION



def test_qualified_challenger_cannot_skip_probation(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    _seed_pair("PROB", champ_rs, chal_rs)
    assert PROMO.evaluate_promotion(
        PR.EQUITY, "PROB_CHAL"
    )["status"] == PROMO.PROMOTION_ELIGIBLE

    with pytest.raises(RuntimeError, match="PROBATION"):
        PROMO.promote_to_champion(
            PR.EQUITY, "PROB_CHAL", actor="operator",
            reason="attempted direct promotion",
        )
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "PROB_CHAMP"


def test_probation_requires_additional_unseen_forward_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    _seed_pair("CANARY", champ_rs, chal_rs)
    advanced = PROMO.advance_eligible_to_probation(PR.EQUITY)
    assert any(p["policy_id"] == "CANARY_CHAL" for p in advanced)

    with pytest.raises(RuntimeError, match="PROBATION evidence incomplete"):
        PROMO.promote_to_champion(
            PR.EQUITY, "CANARY_CHAL", actor="operator",
            reason="not enough post-probation evidence",
        )

    for i in range(PROMO.MIN_PROBATION_ADDITIONAL_PAIRED):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("CANARY_CHAMP", f"CANARY_prob_{i}", -0.2, regime=reg)
        _seed("CANARY_CHAL", f"CANARY_prob_{i}", 0.3, regime=reg)

    promoted = PROMO.promote_to_champion(
        PR.EQUITY, "CANARY_CHAL", actor="operator",
        reason="post-probation evidence complete",
    )
    assert promoted["status"] == PR.CHAMPION
