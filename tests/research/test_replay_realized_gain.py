import json

import pytest

from research.autonomy.replay_realized_gain import persist_realized_replay_gain


def _selection():
    return {
        "evidence_origin": "HISTORICAL_REPLAY",
        "request_id": "req-1",
        "session_date": "2026-01-05",
        "strategy_id": "prod",
        "thesis_hash": "thesis-1",
        "universe_snapshot_id": "u-1",
        "data_version": "d-1",
        "feature_version": "f-1",
        "model_version": "m-1",
        "signal_registry_version": "s-1",
        "decision_fingerprint": "dec-1",
        "acquisition_fingerprint": "acq-1",
    }


def _result():
    return {
        **_selection(),
        "status": "COMPLETED",
        "eligible_sample_count": 3,
        "metrics_produced": ["calibration", "regime"],
    }


def test_persists_realized_event_idempotently(tmp_path):
    path = tmp_path / "journal.jsonl"
    first = persist_realized_replay_gain(_selection(), _result(), journal_path=path)
    second = persist_realized_replay_gain(_selection(), _result(), journal_path=path)
    assert first == second
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "REALIZED"
    assert rows[0]["eligible_samples"] == 3
    assert rows[0]["metrics_realized"] == ["calibration", "regime"]


def test_rejects_forward_result(tmp_path):
    result = _result()
    result["evidence_origin"] = "FORWARD_PAPER"
    with pytest.raises(ValueError, match="HISTORICAL_REPLAY"):
        persist_realized_replay_gain(_selection(), result, journal_path=tmp_path / "j")


def test_rejects_identity_mismatch(tmp_path):
    result = _result()
    result["model_version"] = "other"
    with pytest.raises(ValueError, match="model_version"):
        persist_realized_replay_gain(_selection(), result, journal_path=tmp_path / "j")


def test_rejects_nonterminal_result(tmp_path):
    result = _result()
    result["status"] = "RUNNING"
    with pytest.raises(ValueError, match="terminal-success"):
        persist_realized_replay_gain(_selection(), result, journal_path=tmp_path / "j")
