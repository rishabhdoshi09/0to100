"""Regression tests for fail-fast desk read admission on overloaded Macs.

An abandoned HTTP request must not multiply slow SQLite/JSON work. Admission
belongs to the synchronous handler and never alters market execution logic.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_duplicate_dashboard_read_is_rejected_before_heavy_work():
    import terminal_api as core

    with core._bounded_status_read("/api/dashboard"):
        with pytest.raises(HTTPException) as err:
            with core._bounded_status_read("/api/dashboard"):
                raise AssertionError("duplicate status read was admitted")
        assert err.value.status_code == 503
        assert err.value.detail["code"] == "API_STATUS_BUSY"
        assert err.value.headers["Retry-After"] == "10"

    # A completed handler always returns its capacity.
    with core._bounded_status_read("/api/dashboard"):
        pass


def test_two_heavy_reads_cannot_admit_a_third_concurrent_request():
    import terminal_api as core

    with core._bounded_status_read("/api/dashboard"):
        with core._bounded_status_read("/api/decision-simulation-gate"):
            with pytest.raises(HTTPException) as err:
                with core._bounded_status_read("/api/recommendations-workspace"):
                    raise AssertionError("third expensive read was admitted")
            assert err.value.status_code == 503

    with core._bounded_status_read("/api/recommendations-workspace"):
        pass


def test_status_read_admission_released_after_handler_error():
    import terminal_api as core

    with pytest.raises(RuntimeError):
        with core._bounded_status_read("/api/dashboard"):
            raise RuntimeError("simulated slow handler fault")
    with core._bounded_status_read("/api/dashboard"):
        pass


def test_data_readiness_does_not_start_heavy_history_warmer(monkeypatch):
    import terminal_api as core

    from data import bhavcopy_runtime

    monkeypatch.setattr(
        bhavcopy_runtime, "status",
        lambda *, load_cache=False: {
            "ready": False, "cache_exists": True, "sessions": 0,
            "symbols": 0, "latest_date": "", "csv_files": 0,
            "minimum_sessions": 60,
        },
    )
    monkeypatch.setattr(
        bhavcopy_runtime, "official_history_freshness",
        lambda *args, **kwargs: {"current": False, "reason_code": "HISTORY_NOT_READY"},
    )
    monkeypatch.setattr(core, "_snapshot_payload", lambda: {"ready": False})
    monkeypatch.setattr(
        core, "_schedule_warm",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Home polling started heavy OHLCV cache load")
        ),
    )
    result = core._data_payload(
        {"available": False, "records": []},
        {"available": False, "records": []},
        {"running": True},
        {"available": False},
        {"available": False},
    )
    assert result["store_loaded"] is False
    assert result["ready"] is False
    assert any("loading" in str(x).lower() or "history" in str(x).lower() for x in result["blockers"])
