"""Phase 7: PAPER_AUTO_PROMOTION_READY is a separate, additional-evidence
readiness state -- it must never itself enable automatic promotion, and it
must demand strictly more than plain PROMOTION_ELIGIBLE (being scientifically
eligible is necessary but not sufficient to be "ready for an unattended
mechanism to act on").
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from product.evolution import policy_registry as PR
from product.evolution import promotion as PROMO
from product.evolution import shadow_decisions as SD
from product.evolution.auto_promotion_readiness import (
    NOT_READY,
    PAPER_AUTO_PROMOTION_READY,
    evaluate_auto_promotion_readiness,
)


def _seed(policy_id, sid, r, *, regime="TRENDING_BULL", sector="IT", decision="ENTER_NOW"):
    row = {
        "schema_version": SD.SCHEMA_VERSION, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
        "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
        "decision": decision, "reason_code": "ELIGIBLE" if decision == "ENTER_NOW" else "REJECT",
        "adjusted_score": 50.0, "breakdown": {}, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "sector": sector, "setup_label": "VCP", "regime": regime,
        "frozen_at": datetime.now(timezone.utc).isoformat(), "fingerprint": "x",
        "outcome": {"forward_return_pct": r * 5, "not_pnl": True},
        "classification": "WINNER_TAKEN" if r > 0 else "LOSER_TAKEN",
        "graded_at": "x", "not_pnl": True, "is_champion_decision": policy_id.endswith("CHAMP"),
        "counterfactual_R": r,
    }
    SD.save_graded_decision(row)


def _seed_pair(name, champ_rs, chal_rs, regimes=None, sectors=None):
    regs = regimes or [("TRENDING_BULL" if i % 2 else "SIDEWAYS") for i in range(len(champ_rs))]
    secs = sectors or [("IT" if i % 2 else "PHARMA") for i in range(len(champ_rs))]
    for i, (cr, hr, reg, sec) in enumerate(zip(champ_rs, chal_rs, regs, secs)):
        _seed(f"{name}_CHAMP", f"{name}_snap_{i}", cr, regime=reg, sector=sec)
        _seed(f"{name}_CHAL", f"{name}_snap_{i}", hr, regime=reg, sector=sec)


def test_not_eligible_at_all_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="RAW_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="RAW_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="RAW_CHAMP")
    result = evaluate_auto_promotion_readiness(PR.EQUITY, "RAW_CHAL")
    assert result["status"] == NOT_READY
    checks = {c["name"]: c["passed"] for c in result["checks"]}
    assert checks["scientific_promotion_eligible"] is False
    assert checks["currently_in_probation"] is False


def test_promotion_eligible_but_not_yet_in_probation_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    PR.register_policy(policy_id="ELIG_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="ELIG_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="ELIG_CHAMP")
    _seed_pair("ELIG", champ_rs, chal_rs)
    assert PROMO.evaluate_promotion(PR.EQUITY, "ELIG_CHAL")["status"] == PROMO.PROMOTION_ELIGIBLE

    result = evaluate_auto_promotion_readiness(PR.EQUITY, "ELIG_CHAL")
    assert result["status"] == NOT_READY
    checks = {c["name"]: c["passed"] for c in result["checks"]}
    assert checks["scientific_promotion_eligible"] is True
    assert checks["currently_in_probation"] is False


def test_full_pipeline_reaches_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    PR.register_policy(policy_id="READY_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="READY_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="READY_CHAMP")
    _seed_pair("READY", champ_rs, chal_rs)
    PROMO.advance_eligible_to_probation(PR.EQUITY)
    assert PR.get_policy("READY_CHAL")["status"] == PR.PROBATION

    # No rollback target yet (READY_CHAMP was never demoted from CHAMPION
    # into PROBATION with CHAMPION in its own lifecycle history).
    result = evaluate_auto_promotion_readiness(PR.EQUITY, "READY_CHAL")
    checks = {c["name"]: c["passed"] for c in result["checks"]}
    assert checks["currently_in_probation"] is True
    assert checks["probation_evidence_genuinely_new_and_not_regressed"] is False
    assert checks["rollback_target_available"] is False  # no prior-champion-in-probation yet

    for i in range(PROMO.MIN_PROBATION_ADDITIONAL_PAIRED):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        sec = "IT" if i % 2 else "PHARMA"
        _seed("READY_CHAMP", f"READY_prob_{i}", -0.2, regime=reg, sector=sec)
        _seed("READY_CHAL", f"READY_prob_{i}", 0.3, regime=reg, sector=sec)

    PROMO.promote_to_champion(PR.EQUITY, "READY_CHAL", actor="operator", reason="test promotion")
    assert PR.current_champion(PR.EQUITY)["policy_id"] == "READY_CHAL"
    assert PR.get_policy("READY_CHAMP")["status"] == PR.PROBATION  # now a rollback target

    # READY_CHAL is now CHAMPION itself, not a challenger anymore -- evaluate
    # whether the OLD champion (now in probation) would be auto-promotion-
    # ready to come BACK, which exercises the rollback_target_available=True
    # path meaningfully (it is itself a former CHAMPION in PROBATION).
    second_result = evaluate_auto_promotion_readiness(PR.EQUITY, "READY_CHAMP")
    second_checks = {c["name"]: c["passed"] for c in second_result["checks"]}
    assert second_checks["rollback_target_available"] is True


def test_auto_promotion_enabled_flag_never_flips(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="FLAG_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="FLAG_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="FLAG_CHAMP")
    before = PROMO.AUTO_PROMOTION_ENABLED
    result = evaluate_auto_promotion_readiness(PR.EQUITY, "FLAG_CHAL")
    assert PROMO.AUTO_PROMOTION_ENABLED == before == False
    assert result["auto_promotion_enabled"] is False


def test_sector_dominated_evidence_blocks_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    PR.register_policy(policy_id="SECTOR_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="SECTOR_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="SECTOR_CHAMP")
    # Every single observation is IT -- one sector dominates completely.
    _seed_pair("SECTOR", champ_rs, chal_rs, sectors=["IT"] * n)

    result = evaluate_auto_promotion_readiness(PR.EQUITY, "SECTOR_CHAL")
    checks = {c["name"]: c["passed"] for c in result["checks"]}
    assert checks["evidence_not_sector_dominated"] is False
    assert result["status"] == NOT_READY
