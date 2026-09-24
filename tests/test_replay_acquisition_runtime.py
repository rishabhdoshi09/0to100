import json

import pytest

from research.autonomy.replay_acquisition_runtime import (
    persist_runtime_realized_gain,
    plan_runtime_acquisitions,
)


def _request(**overrides):
    row = {
        "status": "OPEN",
        "allowed_lanes": ["HISTORICAL_REPLAY"],
        "request_id": "req-runtime-1",
        "strategy_id": "prod-selection",
        "thesis_hash": "thesis-v1",
        "sample_deficit": 3,
        "missing_metrics": ["walk_forward_ok"],
    }
    row.update(overrides)
    return row


def _snapshot():
    return {
        "snapshot_id": "cal-1",
        "identities": {
            "feature_version": "feat-1",
            "model_version": "model-1",
            "signal_registry_version": "sig-1",
        },
    }


def test_runtime_plan_records_selection_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scan.signal_registry_versions.persist_canonical_version",
        lambda: {"registry_version": "sig-1"},
    )
    journal = tmp_path / "acquisition.jsonl"
    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05", "2026-01-06"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        curriculum={
            "coverage_before": {"BULL_TREND": 2},
            "session_states": {
                "2026-01-05": {"available": True, "regime": "BULL_TREND"},
                "2026-01-06": {"available": True, "regime": "BEAR"},
            },
        },
        batch_size=2,
        journal_path=journal,
        calibration_snapshot=_snapshot(),
        data_version="data-1",
        policy_versions={"decision_engine_version": "v1"},
    )

    assert planned["selection_policy"] == "INFORMATION_GAIN"
    assert planned["sessions"] == ["2026-01-06", "2026-01-05"]
    assert planned["selected_count"] == 2
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["SELECTED", "SELECTED"]
    assert all(row["evidence_origin"] == "HISTORICAL_REPLAY" for row in rows)


def test_runtime_realization_uses_settled_trade_yield_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scan.signal_registry_versions.persist_canonical_version",
        lambda: {"registry_version": "sig-1"},
    )
    journal = tmp_path / "acquisition.jsonl"
    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05", "2026-01-06"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        batch_size=2,
        journal_path=journal,
        calibration_snapshot=_snapshot(),
        data_version="data-1",
        policy_versions={"decision_engine_version": "v1"},
    )
    report = {"status": "SUCCEEDED", "provenance": "HISTORICAL_REPLAY"}
    trades = [
        {"trade_id": "t1", "entry_date": "2026-01-05"},
        {"trade_id": "t2", "entry_date": "2026-01-05"},
    ]

    first = persist_runtime_realized_gain(
        planned["acquisitions"],
        replay_report=report,
        trades=trades,
        journal_path=journal,
    )
    second = persist_runtime_realized_gain(
        planned["acquisitions"],
        replay_report=report,
        trades=trades,
        journal_path=journal,
    )

    assert first == second
    realized = [
        json.loads(line)
        for line in journal.read_text().splitlines()
        if json.loads(line)["event"] == "REALIZED"
    ]
    assert len(realized) == 2
    by_day = {row["session_date"]: row["eligible_samples"] for row in realized}
    assert by_day == {"2026-01-05": 2, "2026-01-06": 0}


def test_degraded_replay_does_not_write_realized_gain(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scan.signal_registry_versions.persist_canonical_version",
        lambda: {"registry_version": "sig-1"},
    )
    journal = tmp_path / "acquisition.jsonl"
    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        batch_size=1,
        journal_path=journal,
        calibration_snapshot=_snapshot(),
        data_version="data-1",
        policy_versions={"decision_engine_version": "v1"},
    )
    result = persist_runtime_realized_gain(
        planned["acquisitions"],
        replay_report={"status": "DEGRADED", "provenance": "HISTORICAL_REPLAY"},
        trades=[{"entry_date": "2026-01-05"}],
        journal_path=journal,
    )
    assert result["records"] == []
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["SELECTED"]


def test_realized_plateau_stops_before_another_acquisition(tmp_path):
    journal = tmp_path / "acquisition.jsonl"
    rows = []
    for index in range(8):
        fingerprint = f"acq-{index}"
        rows.extend([
            {
                "event": "SELECTED",
                "evidence_origin": "HISTORICAL_REPLAY",
                "request_id": "req-runtime-1",
                "acquisition_fingerprint": fingerprint,
                "strategy_id": "prod-selection",
                "thesis_hash": "thesis-v1",
                "record_fingerprint": f"selected-{index}",
            },
            {
                "event": "REALIZED",
                "evidence_origin": "HISTORICAL_REPLAY",
                "request_id": "req-runtime-1",
                "acquisition_fingerprint": fingerprint,
                "record_fingerprint": f"real-{index}",
                "eligible_samples": 0,
            },
        ])
    journal.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        journal_path=journal,
    )
    assert planned["stop"] is True
    assert planned["reason"] == "realized_information_gain_plateau"
    assert planned["stopping_reason"] == "SUSTAINED_ZERO_INFORMATION_GAIN"


def test_old_thesis_realized_gain_cannot_stop_current_thesis(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scan.signal_registry_versions.persist_canonical_version",
        lambda: {"registry_version": "sig-1"},
    )
    journal = tmp_path / "acquisition.jsonl"
    rows = []
    for index in range(8):
        fingerprint = f"old-acq-{index}"
        rows.extend([
            {
                "event": "SELECTED",
                "evidence_origin": "HISTORICAL_REPLAY",
                "request_id": "req-runtime-1",
                "acquisition_fingerprint": fingerprint,
                "strategy_id": "prod-selection",
                "thesis_hash": "old-thesis",
                "record_fingerprint": f"old-selected-{index}",
            },
            {
                "event": "REALIZED",
                "evidence_origin": "HISTORICAL_REPLAY",
                "request_id": "req-runtime-1",
                "acquisition_fingerprint": fingerprint,
                "record_fingerprint": f"old-real-{index}",
                "eligible_samples": 0,
            },
        ])
    journal.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        batch_size=1,
        journal_path=journal,
        calibration_snapshot=_snapshot(),
        data_version="data-1",
        policy_versions={"decision_engine_version": "v1"},
    )
    assert planned["stop"] is False
    assert planned["sessions"] == ["2026-01-05"]


def test_runtime_realization_refuses_forward_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scan.signal_registry_versions.persist_canonical_version",
        lambda: {"registry_version": "sig-1"},
    )
    journal = tmp_path / "acquisition.jsonl"
    planned = plan_runtime_acquisitions(
        _request(),
        ["2026-01-05"],
        thesis_hash="thesis-v1",
        universe_limit=40,
        batch_size=1,
        journal_path=journal,
        calibration_snapshot=_snapshot(),
        data_version="data-1",
        policy_versions={"decision_engine_version": "v1"},
    )
    with pytest.raises(ValueError, match="HISTORICAL_REPLAY"):
        persist_runtime_realized_gain(
            planned["acquisitions"],
            replay_report={"status": "SUCCEEDED", "provenance": "FORWARD_PAPER"},
            trades=[],
            journal_path=journal,
        )
