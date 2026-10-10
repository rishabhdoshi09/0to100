"""Regression tests for API read-only status price-history access.

A browser GET must not unpickle the multi-year official OHLCV history.
Worker-side data readiness independently retains strict loaded-store gates.
"""
from __future__ import annotations


def test_scan_artifact_freshness_probes_disk_without_unpickling(monkeypatch):
    from product import desk_pipeline
    from product import scan_store
    from data import bhavcopy_runtime

    calls = []

    def official_freshness(*, load_cache=True, require_store=True):
        calls.append((load_cache, require_store))
        return {
            "usable_for_scan": True,
            "current": True,
            "available_session": "2026-10-09",
        }

    monkeypatch.setattr(bhavcopy_runtime, "official_history_freshness", official_freshness)
    monkeypatch.setattr(
        scan_store,
        "load_scan",
        lambda _path: {
            "scanned_at": "2026-10-10T16:32:00+00:00",
            "provenance": {"market_session_date": "2026-10-09"},
            "records": [{"symbol": "INFY"}],
        },
    )
    assert desk_pipeline.scan_is_fresh() is True
    assert calls == [(False, False)]


def test_read_only_freshness_rejects_stale_official_session(monkeypatch):
    from product import desk_pipeline
    from product import scan_store
    from data import bhavcopy_runtime

    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_kwargs: {
            "usable_for_scan": True,
            "current": True,
            "available_session": "2026-10-09",
        },
    )
    monkeypatch.setattr(
        scan_store,
        "load_scan",
        lambda _path: {
            "scanned_at": "2026-10-09T09:00:00+00:00",
            "provenance": {"market_session_date": "2026-10-08"},
            "records": [{"symbol": "INFY"}],
        },
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_read_only_freshness_rejects_unavailable_history(monkeypatch):
    from product import desk_pipeline
    from data import bhavcopy_runtime

    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_kwargs: {
            "usable_for_scan": False,
            "current": False,
            "available_session": "2026-10-08",
        },
    )
    assert desk_pipeline.scan_is_fresh() is False
