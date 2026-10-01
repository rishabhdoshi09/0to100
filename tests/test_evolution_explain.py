"""Phase 8: one-click explainability. An operator must be able to answer
"why does this policy deserve PAPER authority" from this single payload --
exact manifests, exact paired snapshot IDs, regime/sector distribution,
and proof that no historical evidence could have leaked in, without
reading any source code.
"""
from __future__ import annotations

from datetime import datetime, timezone

from product.evolution import policy_registry as PR
from product.evolution import promotion as PROMO
from product.evolution import shadow_decisions as SD
from product.evolution.explain import explain_promotion_candidate


def _seed(policy_id, sid, r, *, regime="TRENDING_BULL", sector="IT"):
    row = {
        "schema_version": 1, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
        "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
        "decision": "ENTER_NOW", "reason_code": "ELIGIBLE",
        "adjusted_score": 50.0, "breakdown": {}, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "sector": sector, "setup_label": "VCP", "regime": regime,
        "frozen_at": datetime.now(timezone.utc).isoformat(), "fingerprint": "x",
        "outcome": {"forward_return_pct": r * 5, "not_pnl": True},
        "classification": "WINNER_TAKEN" if r > 0 else "LOSER_TAKEN",
        "graded_at": "x", "not_pnl": True, "is_champion_decision": policy_id.endswith("CHAMP"),
        "counterfactual_R": r,
        "evidence_class": "EVOLUTION_SHADOW",
    }
    SD.save_graded_decision(row)


def test_explain_unqualified_candidate_still_returns_full_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="EX_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="EX_CHAL", domain=PR.EQUITY, hypothesis="test hypothesis", weights={"relative_strength_mult": 1.5}, status=PR.CHALLENGER, parent_policy_id="EX_CHAMP")
    result = explain_promotion_candidate(PR.EQUITY, "EX_CHAL")
    assert result["policy_manifest"]["policy_id"] == "EX_CHAL"
    assert result["policy_manifest"]["hypothesis"] == "test hypothesis"
    assert result["policy_manifest"]["weights"] == {"relative_strength_mult": 1.5}
    assert result["champion_manifest"]["policy_id"] == "EX_CHAMP"
    assert result["scientific_eligibility"]["status"] == PROMO.NOT_ELIGIBLE
    assert result["paired_snapshot_ids"] == []
    assert result["evidence_integrity"]["all_forward_evidence"] is True


def test_explain_reports_paired_snapshots_and_distributions(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="DIST_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="DIST_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="DIST_CHAMP")
    for i in range(10):
        reg = "TRENDING_BULL" if i < 7 else "SIDEWAYS"
        sec = "IT" if i < 5 else "PHARMA"
        _seed("DIST_CHAMP", f"snap_{i}", 0.1, regime=reg, sector=sec)
        _seed("DIST_CHAL", f"snap_{i}", 0.2, regime=reg, sector=sec)

    result = explain_promotion_candidate(PR.EQUITY, "DIST_CHAL")
    assert result["paired_sample_size"] == 10
    assert len(result["paired_snapshot_ids"]) == 10
    assert result["regime_distribution"] == {"TRENDING_BULL": 7, "SIDEWAYS": 3}
    assert result["sector_distribution"] == {"IT": 5, "PHARMA": 5}
    assert result["evidence_integrity"]["all_forward_evidence"] is True
    assert result["evidence_integrity"]["non_forward_evidence_count"] == 0


def test_explain_flags_non_forward_evidence_if_it_somehow_appeared(tmp_path, monkeypatch):
    """Structurally this cannot happen in production (promotion only reads
    the shadow ledger, historical_priors is a separate store) -- but the
    explain surface itself must genuinely detect it if a row's evidence_class
    were ever anything other than the two legitimate forward tags, not just
    assert cleanliness unconditionally."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="LEAK_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="LEAK_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="LEAK_CHAMP")
    _seed("LEAK_CHAMP", "snap_0", 0.1)
    _seed("LEAK_CHAL", "snap_0", 0.2)
    rows = SD.list_shadow_decisions(policy_id="LEAK_CHAL")
    tainted = dict(rows[0])
    tainted["evidence_class"] = "HISTORICAL_REPLAY"
    SD.save_graded_decision(tainted)

    result = explain_promotion_candidate(PR.EQUITY, "LEAK_CHAL")
    assert result["evidence_integrity"]["all_forward_evidence"] is False
    assert result["evidence_integrity"]["non_forward_evidence_count"] == 1


def test_explain_includes_lifecycle_and_probation_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    n = 60
    champ_rs = [(-0.3 if i % 2 == 0 else 0.1) for i in range(n)]
    chal_rs = [(0.3 if i % 2 == 0 else 0.25) for i in range(n)]
    PR.register_policy(policy_id="LIFE_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="LIFE_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="LIFE_CHAMP")
    for i, (cr, hr) in enumerate(zip(champ_rs, chal_rs)):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("LIFE_CHAMP", f"snap_{i}", cr, regime=reg)
        _seed("LIFE_CHAL", f"snap_{i}", hr, regime=reg)
    PROMO.advance_eligible_to_probation(PR.EQUITY)

    result = explain_promotion_candidate(PR.EQUITY, "LIFE_CHAL")
    assert result["probation"] is not None
    assert "started_at" in result["probation"]
    assert any(e.get("status") == PR.PROBATION for e in result["lifecycle_history"])
    assert result["auto_promotion_readiness"] is not None
