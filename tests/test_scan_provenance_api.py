from __future__ import annotations

from product import data_api


def test_scan_provenance_workspace_projects_persisted_metadata_only(monkeypatch):
    import product.scan_store as scan_store

    saved = {
        "schema_version": 2,
        "scan_id": "scan:2026-09-13T12:50:00+00:00",
        "scanned_at": "2026-09-13T12:50:00+00:00",
        "scan_started_at": "2026-09-13T12:49:30+00:00",
        "scan_completed_at": "2026-09-13T12:50:00+00:00",
        "scan_duration_s": 30.0,
        "scan_duration_status": "AVAILABLE",
        "scan_duration_reason": None,
        "market_session_date": "2026-09-11",
        "price_data_as_of": "2026-09-11",
        "expected_session_date": "2026-09-11",
        "freshness_state": "CURRENT",
        "provenance_reason": "CURRENT",
        "requested_universe": 701,
        "universe_requested": 701,
        "universe_loaded": 701,
        "universe_scanned": 697,
        "universe_failed": 4,
        "candidate_count": 17,
        "source_snapshot_id": "snap-1",
        "coverage_state": "DEGRADED",
        "scan_status": "SUCCEEDED",
        "records": [{"symbol": "ABC", "price": 123.0}],
    }
    monkeypatch.setattr(scan_store, "load_scan", lambda: saved)

    out = data_api.scan_provenance_workspace()

    assert out["available"] is True
    assert out["source"] == "product.scan_store.load_scan"
    assert out["scan_completed_at"] == "2026-09-13T12:50:00+00:00"
    assert out["market_session_date"] == "2026-09-11"
    assert out["price_data_as_of"] == "2026-09-11"
    assert out["universe_failed"] == 4
    assert out["candidate_count"] == 17
    assert "records" not in out


def test_scan_provenance_workspace_is_explicit_when_no_saved_scan(monkeypatch):
    import product.scan_store as scan_store

    monkeypatch.setattr(scan_store, "load_scan", lambda: None)

    assert data_api.scan_provenance_workspace() == {
        "available": False,
        "reason": "NO_SAVED_SCAN",
        "source": "product.scan_store.load_scan",
    }
