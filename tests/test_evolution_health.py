"""Phase 3: the daily Evolution health report. A zero-observation day must
read as healthy, not as a failure; genuine anomalies (missing provenance,
regime/sector concentration, near-duplicate policies, evidence-class leaks)
must be caught using only already-persisted data.
"""
from __future__ import annotations

from datetime import datetime, timezone

from product.evolution import health, policy_registry as PR
from product.evolution import shadow_decisions as SD


def _seed(
    policy_id, sid, r, *, decision="ENTER_NOW", regime="TRENDING_BULL",
    sector="IT", is_champion=False, frozen_at=None, graded_at=None,
    evidence_class="EVOLUTION_SHADOW", missing_fingerprint=False,
):
    if frozen_at is None:
        frozen_at = datetime.now(timezone.utc).isoformat()
    row = {
        "schema_version": 1, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
        "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
        "decision": decision, "reason_code": "ELIGIBLE" if decision == "ENTER_NOW" else "REJECT",
        "adjusted_score": 50.0, "breakdown": {}, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "sector": sector, "setup_label": "VCP", "regime": regime, "frozen_at": frozen_at,
        "fingerprint": ("" if missing_fingerprint else "fp"),
        "outcome": {"forward_return_pct": r * 5, "not_pnl": True} if r is not None else None,
        "classification": ("WINNER_TAKEN" if r and r > 0 else "LOSER_TAKEN") if r is not None else None,
        "graded_at": graded_at or (frozen_at if r is not None else None),
        "not_pnl": True, "is_champion_decision": is_champion,
        "counterfactual_R": r,
        "evidence_class": evidence_class,
    }
    SD.save_graded_decision(row)


def test_fresh_domain_is_healthy_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    report = health.daily_health_report(PR.EQUITY, as_of="2026-09-30")
    assert report["champion_policy_id"] is None
    assert report["champion_made_decisions_today"] is False
    assert report["total_shadow_rows_this_domain"] == 0
    assert report["historical_evidence_leak_check"]["clean"] is True
    assert report["provenance_gap_count_all_time"] == 0


def test_champion_decisions_and_challenger_shadows_counted_for_today(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="H_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="H_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="H_CHAMP")
    today = datetime.now(timezone.utc).date().isoformat()
    _seed("H_CHAMP", "s1", 0.1, is_champion=True, decision="ENTER_NOW")
    _seed("H_CHAL", "s1", 0.2, is_champion=False, decision="ENTER_NOW")
    _seed("H_CHAL", "s2", None, is_champion=False, decision="REJECT")

    report = health.daily_health_report(PR.EQUITY, as_of=today)
    assert report["champion_made_decisions_today"] is True
    assert report["champion_decisions_today"] == 1
    assert report["challenger_shadows_today"] == 2
    assert report["taken_today"] == 2  # champion + challenger both ENTER_NOW
    assert report["rejected_today"] == 1


def test_missing_provenance_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="PROV_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    today = datetime.now(timezone.utc).date().isoformat()
    _seed("PROV_CHAMP", "bad1", 0.1, is_champion=True, missing_fingerprint=True)
    report = health.daily_health_report(PR.EQUITY, as_of=today)
    assert report["provenance_gap_count_all_time"] == 1
    assert report["provenance_gaps_today"][0]["missing"] == ["fingerprint"]


def test_single_regime_dominance_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="REG_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    for i in range(20):
        _seed("REG_CHAMP", f"r{i}", 0.1, is_champion=True, regime="TRENDING_BULL")
    report = health.daily_health_report(PR.EQUITY, persist=False)
    assert report["regime_dominance"] is not None
    assert report["regime_dominance"]["value"] == "TRENDING_BULL"
    assert report["regime_dominance"]["share"] == 1.0


def test_mixed_regimes_are_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="MIX_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    for i in range(20):
        reg = "TRENDING_BULL" if i % 2 else "SIDEWAYS"
        _seed("MIX_CHAMP", f"m{i}", 0.1, is_champion=True, regime=reg)
    report = health.daily_health_report(PR.EQUITY, persist=False)
    assert report["regime_dominance"] is None


def test_near_duplicate_challengers_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="DUP_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="DUP_A", domain=PR.EQUITY, hypothesis="a", weights={}, status=PR.CHALLENGER, parent_policy_id="DUP_CHAMP")
    PR.register_policy(policy_id="DUP_B", domain=PR.EQUITY, hypothesis="b", weights={}, status=PR.CHALLENGER, parent_policy_id="DUP_CHAMP")
    for i in range(15):
        _seed("DUP_A", f"d{i}", 0.1, decision="ENTER_NOW")
        _seed("DUP_B", f"d{i}", 0.2, decision="ENTER_NOW")  # agrees on every decision
    report = health.daily_health_report(PR.EQUITY, persist=False)
    pairs = report["duplicate_policy_pairs"]
    assert any({p["policy_a"], p["policy_b"]} == {"DUP_A", "DUP_B"} for p in pairs)


def test_challenger_with_zero_output_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="ZERO_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="ZERO_CHAL", domain=PR.EQUITY, hypothesis="never runs", weights={}, status=PR.CHALLENGER, parent_policy_id="ZERO_CHAMP")
    report = health.daily_health_report(PR.EQUITY, persist=False)
    assert "ZERO_CHAL" in report["challengers_with_zero_output_ever"]


def test_evidence_class_leak_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="LEAK_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    _seed("LEAK_CHAMP", "leak1", 0.1, is_champion=True, evidence_class="HISTORICAL_REPLAY")
    report = health.daily_health_report(PR.EQUITY, persist=False)
    assert report["historical_evidence_leak_check"]["clean"] is False
    assert report["historical_evidence_leak_check"]["leaked_count"] == 1


def test_storage_growth_needs_history_before_flagging(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="GROW_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    report = health.daily_health_report(PR.EQUITY, as_of="2026-09-30")
    assert report["storage_growth"]["abnormal_growth"] is False
    assert report["storage_growth"]["baseline_days"] == 0


def test_persisted_reports_do_not_duplicate_same_day(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="PERSIST_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    health.daily_health_report(PR.EQUITY, as_of="2026-09-30")
    health.daily_health_report(PR.EQUITY, as_of="2026-09-30")
    rows = health._read_reports()
    same_day = [r for r in rows if r.get("domain") == PR.EQUITY and r.get("as_of") == "2026-09-30"]
    assert len(same_day) == 1
