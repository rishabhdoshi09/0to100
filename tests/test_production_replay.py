from __future__ import annotations

import json
from pathlib import Path

from product.production_replay import replay_candidate, replay_tape_status


def _experts():
    return [
        {
            "id": "xs_momentum", "label": "Momentum", "family": "price_leadership",
            "status": "pass", "eligible": True, "thesis": "Cross-sectional momentum",
            "horizon": "swing", "evidence": ["top-ranked momentum"],
        },
        {
            "id": "vcp", "label": "VCP", "family": "structure",
            "status": "pass", "eligible": True, "thesis": "VCP",
            "horizon": "swing", "evidence": ["tight base"],
        },
        {
            "id": "quality", "label": "Quality", "family": "business_quality",
            "status": "pass", "eligible": True, "thesis": "Quality",
            "horizon": "position", "evidence": ["quality evidence"],
        },
    ]


def _candidate(expected_tier: str = "high_conviction"):
    return {
        "input": {
            "symbol": "ABC",
            "status": "Ready to trade",
            "rsi": 65,
            "signals": ["VCP", "MOMENTUM"],
            "classification": "QUALITY_COMPOUNDER",
            "category_id": "momentum_breakouts",
        },
        "expert_snapshot": _experts(),
        "decision_at_capture": {
            "symbol": "ABC",
            "reco_tier": expected_tier,
            "allows_recommend": True,
            "entry_state": "ready",
            "primary_thesis_id": "vcp",
            "family_confirms": 3,
            "category_id": "momentum_breakouts",
        },
    }


def _record(rules_hash: str, candidate: dict) -> dict:
    return {
        "schema_version": 2,
        "captured_at": "2026-01-01T10:02:00+00:00",
        "scan_scanned_at": "2026-01-01T10:00:00+00:00",
        "captured_live": True,
        "replay_scope": "FROZEN_EXPERTS_TO_ENSEMBLE_DECISION",
        "full_expert_replay_available": False,
        "production_strategy": {"rules_hash": rules_hash},
        "candidates": [candidate],
    }


def test_replay_candidate_exact_when_current_ensemble_matches_capture():
    result = replay_candidate(_candidate())
    assert result["replayable"] is True
    assert result["exact"] is True
    assert result["diffs"] == {}


def test_replay_candidate_exposes_behavior_drift():
    result = replay_candidate(_candidate(expected_tier="watch"))
    assert result["replayable"] is True
    assert result["exact"] is False
    assert result["diffs"]["reco_tier"]["captured"] == "watch"
    assert result["diffs"]["reco_tier"]["replayed"] == "high_conviction"


def test_replay_tape_counts_only_current_hash_records(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps(_record("current", _candidate())) + "\n"
        + json.dumps(_record("old", _candidate())) + "\n",
        encoding="utf-8",
    )
    status = replay_tape_status(path, current_rules_hash="current")
    assert status["available"] is True
    assert status["same_hash_records"] == 1
    assert status["replayable"] == 1
    assert status["exact"] == 1
    assert status["mismatched"] == 0
    assert status["integrity_pass"] is True
    assert status["performance_evidence"] is False
    assert status["full_expert_replay_available"] is False


def test_missing_tape_is_honest_empty_state(tmp_path: Path):
    status = replay_tape_status(tmp_path / "missing.jsonl", current_rules_hash="current")
    assert status["available"] is False
    assert status["status"] == "NO_REPLAY_TAPE"
    assert status["integrity_pass"] is False
    assert status["performance_evidence"] is False
