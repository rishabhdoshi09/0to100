from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from product import counterfactual_learning as CF
from product import decision_freeze as DF
from product.paper_autopilot import (
    DECISION_FINGERPRINT_FAILED,
    run_reco_paper_cycle,
)
from research.auto_research.paper_book import PaperBook


def _record(**over):
    row = {
        "decision_id": "d-1",
        "symbol": "TCS",
        "as_of": "2026-09-21",
        "decision": "ENTER_NOW",
        "reason_code": "ELIGIBLE",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "setup_label": "VCP",
        "sector": "Technology",
        "regime": "BULL_TREND",
        "methods": {"trend": {"status": "pass", "points": 90}},
        "empirical": {"sample_size": 40, "confidence": "MEASURED"},
        "dd_status": "PASS",
        "entry_quality": "ready",
        "family_confirms": 3,
        "missing_evidence": [],
        "selection_score": 92.0,
        "policy_effect": "NEUTRAL",
        "thesis_hash": "thesis-a",
        "calibration_snapshot_id": "cal-a",
        "data_snapshot_id": "market:official_nse:2026-09-21",
        "source_scan_id": "scan-a",
        "evidence_class": "PAPER_FORWARD",
        "versions": {"decision_engine_version": "v1", "risk_policy_version": "r1"},
    }
    row.update(over)
    return row


def _eligible_card():
    return {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": 3,
        "score": 82,
        "primary_thesis": "VCP + quality",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90, "detail": "clean"},
            {"id": "sepa", "status": "pass", "points": 70, "detail": "base"},
            {"id": "funds", "status": "pass", "points": 80, "detail": "quality"},
            {"id": "trend", "status": "pass", "points": 85, "detail": "up"},
            {"id": "rs", "status": "pass", "points": 75, "detail": "leader"},
            {"id": "ev", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "case", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "sector", "status": "pass", "points": 80, "detail": "leader"},
        ],
    }


def test_fingerprint_ignores_later_outcomes_but_tracks_decision_time_truth():
    base = _record()
    same_after_outcome = {**base, "later_outcome": {"return_pct": 20.0}}
    changed_thesis = {**base, "thesis_hash": "thesis-b"}
    changed_calibration = {**base, "calibration_snapshot_id": "cal-b"}
    changed_method = {
        **base,
        "methods": {"trend": {"status": "fail", "points": 10}},
    }

    fp = DF.evidence_fingerprint(base)
    assert DF.evidence_fingerprint(same_after_outcome) == fp
    assert DF.evidence_fingerprint(changed_thesis) != fp
    assert DF.evidence_fingerprint(changed_calibration) != fp
    assert DF.evidence_fingerprint(changed_method) != fp


def test_same_decision_id_is_idempotent_but_identity_collision_fails_closed(tmp_path):
    path = tmp_path / "freezes.db"
    first = DF.freeze(_record(), path=path)
    second = DF.freeze(_record(), path=path)

    assert first["freeze_id"] == second["freeze_id"] == "d-1"
    assert first["fingerprint"] == second["fingerprint"]

    with pytest.raises(DF.DecisionIdentityCollision):
        DF.freeze(_record(entry=101.0), path=path)


def test_counterfactual_freeze_links_canonical_identity_and_is_restart_idempotent(tmp_path):
    ledger = tmp_path / "counterfactuals.jsonl"
    evidence = {
        "decision_id": "rejected-1",
        "rules_hash": "rules-a",
        "thesis_hash": "thesis-a",
        "calibration_snapshot_id": "cal-a",
        "data_snapshot_id": "market:official_nse:2026-09-21",
        "source_scan_id": "scan-a",
        "evidence_class": "PAPER_FORWARD",
        "setup_label": "VCP",
        "sector": "Technology",
        "regime": "BULL_TREND",
        "methods": {"trend": {"status": "pass"}},
        "dd_status": "PASS",
    }

    first = CF.freeze_decision(
        symbol="TCS",
        reason_code="ENTRY_TOO_EXTENDED",
        decision="WAIT",
        entry=100.0,
        stop=94.0,
        target=115.0,
        as_of="2026-09-21",
        evidence=evidence,
        path=ledger,
    )
    second = CF.freeze_decision(
        symbol="TCS",
        reason_code="ENTRY_TOO_EXTENDED",
        decision="WAIT",
        entry=100.0,
        stop=94.0,
        target=115.0,
        as_of="2026-09-21",
        evidence=evidence,
        path=ledger,
    )

    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert first["counterfactual_id"] == second["counterfactual_id"] == "rejected-1"
    assert first["canonical_freeze_id"] == "rejected-1"
    assert first["decision_fingerprint"]
    assert first["not_pnl"] is True


def test_paper_buy_never_opens_when_fingerprint_persistence_fails(monkeypatch):
    card = _eligible_card()
    stamp = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc).isoformat()
    workspace = {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "cards": [card]}],
    }
    monkeypatch.setattr(
        "product.decision_freeze.freeze",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("freeze unavailable")),
    )
    book = PaperBook(capital=100_000)

    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace=workspace,
        as_of="2026-09-21",
        now=datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc),
        entries_allowed=True,
        paper_enabled=True,
        persist_journal=False,
        enforce_history=False,
    )

    assert not book.open
    assert out["taken"] == []
    assert DECISION_FINGERPRINT_FAILED in out["cycle_reasons"]
    assert out["rejections"]
    assert out["rejections"][0]["reason_code"] == DECISION_FINGERPRINT_FAILED



def test_successful_paper_buy_carries_freeze_reference(monkeypatch):
    card = _eligible_card()
    stamp = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc).isoformat()
    workspace = {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "cards": [card]}],
    }
    monkeypatch.setattr(
        "product.decision_freeze.freeze",
        lambda rec, **_k: {
            "freeze_id": str(rec.get("decision_id") or "freeze-test"),
            "fingerprint": "fp-test",
        },
    )
    book = PaperBook(capital=100_000)

    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace=workspace,
        as_of="2026-09-21",
        now=datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc),
        entries_allowed=True,
        paper_enabled=True,
        persist_journal=False,
        enforce_history=False,
    )

    assert len(book.open) == 1
    assert len(out["taken"]) == 1
    assert out["taken"][0]["freeze_id"]
    assert out["taken"][0]["evidence_fingerprint"] == "fp-test"


def test_forward_observation_preserves_canonical_fingerprint_reference(tmp_path):
    from product.forward_evidence import freeze_observation, load_ledger

    path = tmp_path / "forward.jsonl"
    frozen = freeze_observation(
        {
            "symbol": "TCS",
            "decision": "ENTER_NOW",
            "reason_code": "ELIGIBLE",
            "entry": 100.0,
            "stop": 94.0,
            "target": 115.0,
            "freeze_id": "freeze-1",
            "evidence_fingerprint": "fp-1",
            "thesis_hash": "thesis-a",
            "calibration_snapshot_id": "cal-a",
            "data_snapshot_id": "market:2026-09-21",
        },
        cycle_id="cycle-1",
        as_of="2026-09-21",
        rules_hash="rules-a",
        group="TAKEN",
        entered=True,
        path=path,
    )

    assert frozen is not None
    assert frozen["freeze_id"] == "freeze-1"
    assert frozen["decision_fingerprint"] == "fp-1"
    assert frozen["pit_proof"]["freeze_id"] == "freeze-1"
    assert load_ledger(path)[0]["calibration_snapshot_id"] == "cal-a"
