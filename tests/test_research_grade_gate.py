"""Research-grade gate: earned, never manually stamped."""
from research.intelligence.data.research_grade_gate import (
    evaluate_research_grade,
    stamp_manifest_if_earned,
)


def test_gate_refuses_without_ledgers(monkeypatch, tmp_path):
    # Isolate from any real logs by pointing env overrides if modules honor them
    monkeypatch.setenv("QT_SECURITY_IDENTITY_FILE", str(tmp_path / "missing_id.json"))
    monkeypatch.setenv("QT_UNIVERSE_HISTORY_FILE", str(tmp_path / "missing_u.json"))
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(tmp_path / "missing_ca.json"))
    gate = evaluate_research_grade(run_gauntlet_validate=False, sample=5)
    assert gate["earned"] is False
    assert gate["research_grade"] is False
    assert gate["trust_class"] != "RESEARCH_GRADE"
    assert gate["user_facing"]["state"] == "NOT_READY"
    assert "Research quality" in gate["user_facing"]["headline"]


def test_stamp_refuses_manual_research_grade():
    gate = {
        "earned": False,
        "may_stamp_manifest": False,
        "trust_class": "OPERATIONAL_ONLY",
        "failed": ["corporate_actions"],
        "reason": "ca missing",
        "evaluated_at": "2026-01-01T00:00:00+00:00",
    }
    stamped = stamp_manifest_if_earned(
        {"trust_class": "RESEARCH_GRADE", "research_grade": True, "snapshot_id": "x"},
        gate,
    )
    assert stamped["research_grade"] is False
    assert stamped["trust_class"] != "RESEARCH_GRADE"
    assert stamped["_earned_by_gate"] is False


def test_stamp_allows_when_earned():
    gate = {
        "earned": True,
        "may_stamp_manifest": True,
        "trust_class": "RESEARCH_GRADE",
        "failed": [],
        "reason": "ok",
        "evaluated_at": "2026-01-01T00:00:00+00:00",
    }
    stamped = stamp_manifest_if_earned({"snapshot_id": "abc"}, gate)
    assert stamped["trust_class"] == "RESEARCH_GRADE"
    assert stamped["research_grade"] is True
    assert stamped["_earned_by_gate"] is True
