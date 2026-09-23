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
    calls = []

    def project(_scan, *, persist_ledger=True):
        calls.append(bool(persist_ledger))
        return {
            "recommendations": "saved",
            "decision_discovery": "saved",
            "decision_discovery_actionable": 7,
            "decision_discovery_thesis_hash": "thesis-b",
        }

    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_recommendations_and_discovery",
        project,
    )

    result = run_discovery_refresh(SimpleNamespace())

    assert result.status == JS.SUCCEEDED
    assert result.new_entries_allowed is False
    assert result.metadata["scan_scanned_at"] == "2026-09-22T19:02:49+00:00"
    assert result.metadata["thesis_hash"] == "thesis-b"
    assert result.metadata["live_money_unchanged"] is True
    assert calls == [False]


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
        lambda _scan, *, persist_ledger=True: {
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


def test_discovery_refresh_retires_superseded_scan_without_writing(monkeypatch):
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-09-22T19:05:00+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("superseded discovery job must not publish anything")

    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_recommendations_and_discovery",
        forbidden,
    )
    ctx = SimpleNamespace(
        job=SimpleNamespace(
            input_snapshot_id="2026-09-22T19:02:49+00:00",
            idempotency_key=(
                "discovery_refresh:2026-09-22T19:02:49+00:00:"
                "none:thesis-b"
            ),
        )
    )

    result = run_discovery_refresh(ctx)

    assert result.status == JS.SKIPPED_IDEMPOTENT
    assert result.new_entries_allowed is False
    assert result.metadata["requested_scan_scanned_at"] == "2026-09-22T19:02:49+00:00"
    assert result.metadata["current_scan_scanned_at"] == "2026-09-22T19:05:00+00:00"
    assert result.metadata["live_money_unchanged"] is True
