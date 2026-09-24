import json

import pytest

from research.autonomy.acquisition_journal import (
    record_realized_gain,
    record_selection,
    records_for_request,
)


def _acquisition(**overrides):
    row = {
        "evidence_origin": "HISTORICAL_REPLAY",
        "request_id": "evreq_1",
        "acquisition_fingerprint": "acq_abc",
        "session_date": "2025-01-10",
        "strategy_id": "prod-selection",
        "thesis_hash": "thesis-v1",
        "universe_snapshot_id": "universe-v1",
        "data_version": "data-v1",
        "feature_version": "features-v1",
        "model_version": "model-v1",
        "signal_registry_version": "signals-v1",
        "decision_fingerprint": "decision-v1",
        "score": 1.25,
        "rationale": ["sample_deficit:4", "regime_novelty"],
    }
    row.update(overrides)
    return row


def test_selection_is_idempotent_and_keeps_rationale(tmp_path):
    path = tmp_path / "journal.jsonl"
    first = record_selection(_acquisition(), path=path)
    second = record_selection(_acquisition(), path=path)
    assert first == second
    assert first["event"] == "SELECTED"
    assert first["evidence_origin"] == "HISTORICAL_REPLAY"
    assert first["rationale"] == ["sample_deficit:4", "regime_novelty"]
    assert len(path.read_text().splitlines()) == 1


def test_forward_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="HISTORICAL_REPLAY"):
        record_selection(_acquisition(evidence_origin="FORWARD_PAPER"), path=tmp_path / "j.jsonl")


def test_missing_version_identity_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="feature_version"):
        record_selection(_acquisition(feature_version=""), path=tmp_path / "j.jsonl")


def test_realized_gain_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "journal.jsonl"
    acquisition = _acquisition()
    selected = record_selection(acquisition, path=path)
    realized = record_realized_gain(
        acquisition,
        eligible_samples=3,
        metrics_realized=["walk_forward_ok", "deflated_sharpe", "walk_forward_ok"],
        path=path,
    )
    again = record_realized_gain(
        acquisition,
        eligible_samples=3,
        metrics_realized=["deflated_sharpe", "walk_forward_ok"],
        path=path,
    )
    assert realized == again
    assert realized["event"] == "REALIZED"
    assert realized["metrics_realized"] == ["deflated_sharpe", "walk_forward_ok"]
    rows = records_for_request("evreq_1", path=path)
    assert [row["event"] for row in rows] == ["SELECTED", "REALIZED"]
    assert rows[0]["record_fingerprint"] == selected["record_fingerprint"]
    assert len(path.read_text().splitlines()) == 2


def test_journal_lines_are_valid_json(tmp_path):
    path = tmp_path / "journal.jsonl"
    record_selection(_acquisition(), path=path)
    for line in path.read_text().splitlines():
        assert isinstance(json.loads(line), dict)
