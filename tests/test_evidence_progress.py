from __future__ import annotations

from research.autonomy import evidence_acquisition as EA
from research.autonomy import evidence_progress as EP


def _request(**over):
    kwargs = dict(
        session_date="2026-09-21",
        strategy_id="MOM",
        gap_kind="insufficient_sample",
        diagnosis="Need more historical evidence.",
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=10,
        target_samples=20,
        thesis_hash="thesis-a",
    )
    kwargs.update(over)
    return EA.build_request(**kwargs)


def test_batch_progress_is_idempotent_and_updates_request_file(tmp_path):
    req = _request()
    request_path = tmp_path / "request.json"
    progress_path = tmp_path / "progress.json"
    EA.save_request(req, path=request_path)
    batch = {"batch_id": "b1", "historical_paper_trades": 4}
    research = {"decision": "EVIDENCE_ACQUISITION"}

    first = EP.record_historical_batch(
        req.as_dict(), batch, research, path=progress_path, request_path=request_path
    )
    second = EP.record_historical_batch(
        req.as_dict(), batch, research, path=progress_path, request_path=request_path
    )

    assert first["sample_count"] == 14
    assert first["samples_acquired"] == 4
    assert first["batches_completed"] == 1
    assert second["sample_count"] == 14
    assert second["batches_completed"] == 1
    persisted = EA.load_request(request_path)
    assert persisted["status"] == EA.OPEN
    assert persisted["progress"]["sample_count"] == 14


def test_request_closes_when_sample_target_is_satisfied(tmp_path):
    req = _request(current_samples=18, target_samples=20)
    request_path = tmp_path / "request.json"
    EA.save_request(req, path=request_path)
    out = EP.record_historical_batch(
        req.as_dict(),
        {"batch_id": "b1", "historical_paper_trades": 2},
        {"decision": "EVIDENCE_ACQUISITION"},
        path=tmp_path / "progress.json",
        request_path=request_path,
    )
    assert out["status"] == EA.SATISFIED
    assert out["sample_deficit"] == 0
    assert out["next_action"] == "CLOSE_REQUEST"
    assert EA.load_request(request_path)["status"] == EA.SATISFIED


def test_three_zero_progress_batches_plateau_unresolved_metric(tmp_path):
    req = _request(
        current_samples=20,
        target_samples=20,
        missing_metrics=("deflated_sharpe",),
    )
    request_path = tmp_path / "request.json"
    progress_path = tmp_path / "progress.json"
    EA.save_request(req, path=request_path)
    out = {}
    for i in range(1, 4):
        out = EP.record_historical_batch(
            req.as_dict(),
            {"batch_id": f"b{i}", "historical_paper_trades": 0},
            {"decision": "EVIDENCE_ACQUISITION"},
            path=progress_path,
            request_path=request_path,
            plateau_batches=3,
        )
    assert out["status"] == EA.PLATEAUED
    assert out["stagnant_batches"] == 3
    assert out["unresolved_metrics"] == ["deflated_sharpe"]
    assert out["next_action"] == "REPLAN_RESEARCH_QUESTION"
    assert EA.load_request(request_path)["status"] == EA.PLATEAUED


def test_metric_resolution_counts_as_progress_and_can_satisfy(tmp_path):
    req = _request(
        current_samples=20,
        target_samples=20,
        missing_metrics=("deflated_sharpe",),
    )
    out = EP.record_historical_batch(
        req.as_dict(),
        {"batch_id": "b1", "historical_paper_trades": 0},
        {"decision": "EVIDENCE_ACQUISITION", "resolved_metrics": ["deflated_sharpe"]},
        path=tmp_path / "progress.json",
        request_path=tmp_path / "request.json",
    )
    assert out["status"] == EA.SATISFIED
    assert out["resolved_metrics"] == ["deflated_sharpe"]


def test_terminal_research_decision_closes_request_even_before_sample_target(tmp_path):
    req = _request(current_samples=10, target_samples=30)
    out = EP.record_historical_batch(
        req.as_dict(),
        {"batch_id": "b1", "historical_paper_trades": 1},
        {"decision": "REJECT"},
        path=tmp_path / "progress.json",
        request_path=tmp_path / "request.json",
    )
    assert out["status"] == EA.SATISFIED
    assert "terminal decision REJECT" in out["reason"]


def test_forward_only_request_is_blocked_from_historical_progress(tmp_path):
    req = EA.build_request(
        session_date="2026-09-21",
        strategy_id="MOM",
        gap_kind="insufficient_sample",
        diagnosis="Need real forward outcomes.",
        evidence_origin="FORWARD_PAPER",
        current_samples=5,
        target_samples=30,
    )
    out = EP.record_historical_batch(
        req.as_dict(),
        {"batch_id": "b1", "historical_paper_trades": 50},
        {"decision": "EVIDENCE_ACQUISITION"},
        path=tmp_path / "progress.json",
        request_path=tmp_path / "request.json",
    )
    assert out["status"] == EA.BLOCKED
    assert out["reason"] == "HISTORICAL_REPLAY_NOT_ALLOWED_FOR_REQUEST"


def test_exhausted_historical_source_plateaus_request_and_requires_replan(tmp_path):
    req = _request(current_samples=12, target_samples=30)
    request_path = tmp_path / "request.json"
    progress_path = tmp_path / "progress.json"
    EA.save_request(req, path=request_path)

    out = EP.mark_historical_source_exhausted(
        req.as_dict(),
        eligible_sessions=120,
        processed_sessions=120,
        path=progress_path,
        request_path=request_path,
    )

    assert out["status"] == EA.PLATEAUED
    assert out["source_exhausted"] is True
    assert out["sample_count"] == 12
    assert out["sample_deficit"] == 18
    assert out["next_action"] == "REPLAN_RESEARCH_QUESTION"
    assert out["eligible_sessions"] == 120
    assert out["processed_sessions"] == 120
    persisted = EA.load_request(request_path)
    assert persisted["status"] == EA.PLATEAUED
    assert persisted["progress"]["source_exhausted"] is True


def test_information_gain_plateau_is_not_reported_as_source_exhaustion(tmp_path):
    req = _request(current_samples=12, target_samples=30)
    request_path = tmp_path / "request.json"
    progress_path = tmp_path / "progress.json"
    EA.save_request(req, path=request_path)

    out = EP.mark_historical_source_exhausted(
        req.as_dict(),
        reason="realized_information_gain_plateau",
        eligible_sessions=120,
        processed_sessions=40,
        path=progress_path,
        request_path=request_path,
    )

    assert out["status"] == EA.PLATEAUED
    assert out["source_exhausted"] is False
    assert out["source_exhaustion_reason"] == ""
    assert out["stopping_reason"] == "realized_information_gain_plateau"
    assert out["next_action"] == "REPLAN_RESEARCH_QUESTION"
    assert "cannot justify another evidence acquisition" in out["reason"]
