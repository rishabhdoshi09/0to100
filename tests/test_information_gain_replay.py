from research.autonomy.information_gain import rank_historical_sessions


def _request(**overrides):
    payload = {
        "status": "OPEN",
        "allowed_lanes": ["HISTORICAL_REPLAY"],
        "request_id": "evreq_test",
        "strategy_id": "breakout",
        "thesis_hash": "thesis-v1",
        "sample_deficit": 10,
        "missing_metrics": ["walk_forward_ok"],
    }
    payload.update(overrides)
    return payload


def _candidate(date="2026-01-02", **overrides):
    payload = {
        "session_date": date,
        "strategy_id": "breakout",
        "thesis_hash": "thesis-v1",
        "universe_snapshot_id": f"universe-{date}",
        "data_version": "data-v1",
        "feature_version": "feature-v1",
        "model_version": "model-v1",
        "signal_registry_version": "signals-v1",
        "decision_fingerprint": f"decision-{date}",
        "eligible_sample_count": 2,
        "metrics_available": [],
        "regime_novelty": 0.0,
        "sector_novelty": 0.0,
        "decision_uncertainty": 0.0,
    }
    payload.update(overrides)
    return payload


def test_ranking_is_deterministic_and_prefers_evidence_gain():
    low = _candidate("2026-01-02")
    high = _candidate(
        "2026-01-03",
        eligible_sample_count=5,
        metrics_available=["walk_forward_ok"],
        regime_novelty=1.0,
        decision_uncertainty=1.0,
    )
    first = rank_historical_sessions(_request(), [low, high])
    second = rank_historical_sessions(_request(), [high, low])
    assert first == second
    assert first[0]["session_date"] == "2026-01-03"
    assert first[0]["evidence_origin"] == "HISTORICAL_REPLAY"
    assert first[0]["acquisition_fingerprint"].startswith("acq_")


def test_forward_request_cannot_be_satisfied_by_historical_ranking():
    request = _request(allowed_lanes=["FORWARD_PAPER"])
    assert rank_historical_sessions(request, [_candidate()]) == []


def test_candidate_without_immutable_identity_is_rejected():
    candidate = _candidate()
    candidate["universe_snapshot_id"] = ""
    assert rank_historical_sessions(_request(), [candidate]) == []


def test_thesis_mismatch_is_rejected():
    assert rank_historical_sessions(
        _request(), [_candidate(thesis_hash="different-production-thesis")]
    ) == []


def test_closed_request_produces_no_acquisition_work():
    assert rank_historical_sessions(
        _request(status="SATISFIED"), [_candidate()]
    ) == []
