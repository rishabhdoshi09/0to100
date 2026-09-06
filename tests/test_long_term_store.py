"""Durable long-term artifact invariant: len(records) == summary.candidates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from product.long_term_store import (
    load_long_term_scan,
    reconcile_long_term_payload,
    save_long_term_scan,
    summary_from_records,
)


def _record(symbol: str, classification: str = "LONG_TERM_WATCH") -> dict:
    return {
        "symbol": symbol,
        "classification": classification,
        "fundamental_coverage": 0.8,
        "combined_score": 70,
    }


def _payload(records, candidates=None) -> dict:
    body = {
        "schema_version": 1,
        "records": list(records),
        "summary": {"candidates": len(records) if candidates is None else candidates},
    }
    return body


def test_normal_consistent_artifact_roundtrip(tmp_path: Path):
    path = tmp_path / "latest_long_term_scan.json"
    records = [_record("AAA", "QUALITY_COMPOUNDER"), _record("BBB", "GARP_CANDIDATE")]
    save_long_term_scan(_payload(records), path)
    loaded = load_long_term_scan(path)
    assert loaded is not None
    assert len(loaded["records"]) == loaded["summary"]["candidates"] == 2
    assert loaded["summary"]["quality_compounder"] == 1
    assert loaded["summary"]["garp_candidate"] == 1


def test_records_dropped_before_persistence_recompute_count(tmp_path: Path):
    path = tmp_path / "latest_long_term_scan.json"
    records = [_record("AAA"), "not-a-record", None, 12, _record("BBB")]
    save_long_term_scan(_payload(records, candidates=5), path)
    loaded = load_long_term_scan(path)
    assert loaded is not None
    assert all(isinstance(row, dict) for row in loaded["records"])
    assert len(loaded["records"]) == loaded["summary"]["candidates"] == 2


def test_dedup_and_filter_do_not_leave_bad_count(tmp_path: Path):
    path = tmp_path / "latest_long_term_scan.json"
    records = [_record("AAA", "QUALITY_COMPOUNDER"), _record("AAA", "AVOID_REVIEW"), _record("CCC")]
    save_long_term_scan(_payload(records, candidates=3), path)
    loaded = load_long_term_scan(path)
    assert loaded is not None
    symbols = [row["symbol"] for row in loaded["records"]]
    assert symbols == ["AAA", "CCC"]
    assert len(loaded["records"]) == loaded["summary"]["candidates"] == 2
    # First occurrence is kept, so the compounder classification survives.
    assert loaded["summary"]["quality_compounder"] == 1
    assert loaded["summary"]["avoid_review"] == 0


def test_deliberately_mismatched_summary_is_recomputed(tmp_path: Path):
    path = tmp_path / "latest_long_term_scan.json"
    records = [_record("AAA"), _record("BBB"), _record("CCC")]
    save_long_term_scan(_payload(records, candidates=99), path)
    loaded = load_long_term_scan(path)
    assert loaded is not None
    assert loaded["summary"]["candidates"] != 99
    assert len(loaded["records"]) == loaded["summary"]["candidates"] == 3


def test_empty_artifact_is_consistent(tmp_path: Path):
    path = tmp_path / "latest_long_term_scan.json"
    save_long_term_scan(_payload([], candidates=7), path)
    loaded = load_long_term_scan(path)
    assert loaded is not None
    assert loaded["records"] == []
    assert loaded["summary"]["candidates"] == 0
    assert loaded["summary"]["coverage_pct"] == 0.0


def test_failed_serialization_does_not_corrupt_existing_artifact(tmp_path, monkeypatch):
    path = tmp_path / "latest_long_term_scan.json"
    save_long_term_scan(_payload([_record("KEEP")], candidates=1), path)
    original = path.read_text(encoding="utf-8")

    def boom(*_a, **_k):
        raise TypeError("cannot serialize")

    monkeypatch.setattr("product.long_term_store.json.dumps", boom)
    with pytest.raises(TypeError):
        save_long_term_scan(_payload([_record("NEW"), _record("OTHER")], candidates=2), path)
    assert path.read_text(encoding="utf-8") == original
    reloaded = load_long_term_scan(path)
    assert reloaded["records"][0]["symbol"] == "KEEP"
    assert len(reloaded["records"]) == reloaded["summary"]["candidates"] == 1
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_invariant_helper_rejects_internal_disagreement():
    payload = reconcile_long_term_payload(_payload([_record("AAA")], candidates=50))
    assert payload["summary"]["candidates"] == 1
    # Direct summary_from_records is the source of the persisted count.
    summary = summary_from_records([_record("AAA"), _record("BBB")])
    assert summary["candidates"] == 2


def test_nifty500_filter_persists_matching_counts(tmp_path, monkeypatch):
    from scan.long_term_consistency import postprocess_report, _bind_payload
    from scan.long_term_service import LongTermScanReport

    path = tmp_path / "latest_long_term_scan.json"
    monkeypatch.setattr("product.long_term_store.DEFAULT_LONG_TERM_PATH", path)
    payload = {
        "schema_version": 1,
        "records": [
            {"symbol": "AAA", "classification": "QUALITY_COMPOUNDER", "technical_score": 80,
             "fundamental_score": 80, "fundamental_coverage": 0.9, "combined_score": 80, "timing": "NOW"},
            {"symbol": "OUTSIDER", "classification": "GARP_CANDIDATE", "technical_score": 70,
             "fundamental_score": 70, "fundamental_coverage": 0.9, "combined_score": 70, "timing": "NOW"},
        ],
        "summary": {"candidates": 99},
    }
    report = postprocess_report(LongTermScanReport("SUCCEEDED", payload))
    filtered = dict(report.payload)
    filtered["records"] = [row for row in filtered["records"] if row["symbol"] == "AAA"]
    report = _bind_payload(report, filtered)
    save_long_term_scan(report.payload, path)
    loaded = load_long_term_scan(path)
    assert len(loaded["records"]) == loaded["summary"]["candidates"] == 1
    assert loaded["records"][0]["symbol"] == "AAA"
