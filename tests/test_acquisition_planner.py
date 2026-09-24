import json

from research.autonomy.acquisition_planner import plan_historical_acquisitions


def _request(**overrides):
    row = {
        "status": "OPEN",
        "allowed_lanes": ["HISTORICAL_REPLAY"],
        "request_id": "evreq_1",
        "strategy_id": "prod-selection",
        "thesis_hash": "thesis-v1",
        "sample_deficit": 4,
        "missing_metrics": ["walk_forward_ok"],
    }
    row.update(overrides)
    return row


def _candidate(day, samples, **overrides):
    row = {
        "session_date": day,
        "strategy_id": "prod-selection",
        "thesis_hash": "thesis-v1",
        "universe_snapshot_id": f"universe-{day}",
        "data_version": f"data-{day}",
        "feature_version": "features-v1",
        "model_version": "model-v1",
        "signal_registry_version": "signals-v1",
        "decision_fingerprint": f"decision-{day}",
        "eligible_sample_count": samples,
        "metrics_available": ["walk_forward_ok"],
    }
    row.update(overrides)
    return row


def test_planner_selects_highest_gain_and_persists_before_execution(tmp_path):
    path = tmp_path / "journal.jsonl"
    result = plan_historical_acquisitions(
        _request(),
        [_candidate("2025-01-02", 1), _candidate("2025-01-03", 4)],
        batch_size=1,
        journal_path=path,
    )
    assert result["sessions"] == ["2025-01-03"]
    assert result["evidence_origin"] == "HISTORICAL_REPLAY"
    assert result["selected_count"] == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["SELECTED"]
    assert rows[0]["session_date"] == "2025-01-03"


def test_planner_is_bounded_and_retry_idempotent(tmp_path):
    path = tmp_path / "journal.jsonl"
    candidates = [_candidate(f"2025-01-0{i}", i) for i in range(2, 6)]
    first = plan_historical_acquisitions(_request(), candidates, batch_size=2, journal_path=path)
    second = plan_historical_acquisitions(_request(), candidates, batch_size=2, journal_path=path)
    assert first["sessions"] == second["sessions"]
    assert len(first["sessions"]) == 2
    assert len(path.read_text().splitlines()) == 2


def test_planner_refuses_forward_only_request(tmp_path):
    path = tmp_path / "journal.jsonl"
    result = plan_historical_acquisitions(
        _request(allowed_lanes=["FORWARD_PAPER"]),
        [_candidate("2025-01-03", 4)],
        journal_path=path,
    )
    assert result["sessions"] == []
    assert result["reason"] == "no_eligible_information_gain"
    assert not path.exists()


def test_planner_refuses_thesis_mismatch_and_incomplete_identity(tmp_path):
    path = tmp_path / "journal.jsonl"
    result = plan_historical_acquisitions(
        _request(),
        [
            _candidate("2025-01-03", 4, thesis_hash="other"),
            _candidate("2025-01-04", 4, feature_version=""),
        ],
        journal_path=path,
    )
    assert result["sessions"] == []
    assert not path.exists()


def test_zero_batch_does_not_rank_or_write(tmp_path):
    path = tmp_path / "journal.jsonl"
    result = plan_historical_acquisitions(
        _request(), [_candidate("2025-01-03", 4)], batch_size=0, journal_path=path
    )
    assert result["reason"] == "batch_size_zero"
    assert result["sessions"] == []
    assert not path.exists()
