from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy.jobs import run_discovery_refresh


def test_discovery_refresh_reprojects_saved_scan_without_execution(monkeypatch):
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-09-22T19:02:49+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_recommendations_and_discovery",
        lambda _scan: {
            "recommendations": "saved",
            "decision_discovery": "saved",
            "decision_discovery_actionable": 7,
            "decision_discovery_thesis_hash": "thesis-b",
        },
    )

    result = run_discovery_refresh(SimpleNamespace())

    assert result.status == JS.SUCCEEDED
    assert result.new_entries_allowed is False
    assert result.metadata["scan_scanned_at"] == "2026-09-22T19:02:49+00:00"
    assert result.metadata["thesis_hash"] == "thesis-b"
    assert result.metadata["live_money_unchanged"] is True


def test_discovery_refresh_fails_closed_when_projection_is_not_durable(monkeypatch):
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-09-22T19:02:49+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_recommendations_and_discovery",
        lambda _scan: {
            "recommendations": "saved",
            "decision_discovery": "error",
            "decision_discovery_error": {
                "error_code": "DESK_PERSIST_FAILED",
                "error_message": "synthetic projection write failure",
            },
        },
    )

    result = run_discovery_refresh(SimpleNamespace())

    assert result.status == JS.RETRYABLE_FAILED
    assert result.new_entries_allowed is False
    assert result.error_code == "DESK_PERSIST_FAILED"
    assert "projection" in result.summary.lower()
