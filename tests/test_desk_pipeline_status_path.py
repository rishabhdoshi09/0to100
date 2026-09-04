"""GET /api/desk-pipeline is a cheap persisted-status read."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from operations.market_ops import DATA_PREPARE
from product.desk_pipeline import (
    SNAPSHOT_STALE,
    SNAPSHOT_UNKNOWN,
    advance_desk_pipeline,
    load_desk_pipeline_status,
    persist_desk_pipeline_snapshot,
)
from operations.store import OperationStore


def _fresh(monkeypatch, **flags):
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: flags.get("prices", None))
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: flags.get("scan", True))
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: flags.get("long_term", True))
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: flags.get("news", True))
    research_calls: list[int] = []

    def research():
        research_calls.append(1)
        return {
            "fresh": bool(flags.get("investigate", True)),
            "retry_due": not bool(flags.get("investigate", True)),
            "state": "CURRENT" if flags.get("investigate", True) else "RETRY_DUE",
            "unresolved_symbols": [] if flags.get("investigate", True) else ["AAA"],
        }

    monkeypatch.setattr("product.desk_pipeline.acquire_freshness", research)
    return research_calls


def test_get_desk_pipeline_does_not_inspect_coverage(tmp_path: Path, monkeypatch):
    import terminal_product_api as tpa

    persist_desk_pipeline_snapshot({
        "sequential": True,
        "queued_kind": None,
        "steps": [{"id": "prices", "state": "ready"}],
        "message": "Desk data is current.",
        "research_freshness": {"fresh": True, "state": "CURRENT"},
    })
    hits: list[str] = []
    monkeypatch.setattr(
        "product.due_diligence.freshness.research_freshness",
        lambda **_k: hits.append("research_freshness") or {"fresh": True},
    )
    monkeypatch.setattr(
        "product.desk_pipeline.acquire_freshness",
        lambda: hits.append("acquire_freshness") or {"fresh": True},
    )
    monkeypatch.setattr(
        "product.due_diligence.acquire.inspect_symbol_coverage",
        lambda *_a, **_k: hits.append("inspect") or {},
    )
    monkeypatch.setattr(
        "product.due_diligence.acquire.shortlist_symbols",
        lambda **_k: hits.append("shortlist") or ["RELIANCE"],
    )
    client = TestClient(tpa.app)
    response = client.get("/api/desk-pipeline")
    assert response.status_code == 200
    body = response.json()
    assert body["status_source"] == "persisted"
    assert hits == []


def test_get_desk_pipeline_is_bounded(tmp_path: Path, monkeypatch):
    import terminal_product_api as tpa

    persist_desk_pipeline_snapshot({
        "sequential": True,
        "queued_kind": None,
        "steps": [{"id": "prices", "state": "ready"}],
        "message": "ok",
    })
    client = TestClient(tpa.app)
    durations: list[float] = []
    for _ in range(40):
        started = time.monotonic()
        response = client.get("/api/desk-pipeline")
        durations.append(time.monotonic() - started)
        assert response.status_code == 200
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    assert p95 < 0.25, p95
    assert max(durations) < 1.0, max(durations)


def test_research_freshness_once_per_worker_cycle(tmp_path: Path, monkeypatch):
    calls = _fresh(monkeypatch, prices=DATA_PREPARE, investigate=False)
    store = OperationStore(tmp_path / "jobs.db")
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] == DATA_PREPARE
    assert calls == [1]


def test_missing_snapshot_is_unknown_not_recomputed(monkeypatch):
    monkeypatch.setenv("QT_DESK_PIPELINE_SNAPSHOT", str(Path("/tmp/does-not-exist-desk-pipeline.json")))
    hits: list[str] = []
    monkeypatch.setattr("product.desk_pipeline.acquire_freshness", lambda: hits.append("x") or {})
    payload = load_desk_pipeline_status()
    assert payload["freshness"] == SNAPSHOT_UNKNOWN
    assert hits == []


def test_stale_snapshot_is_labelled_not_recomputed(tmp_path: Path, monkeypatch):
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "sequential": True,
        "message": "was current",
        "steps": [{"id": "prices", "state": "ready"}],
        "persisted_at": "2020-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    hits: list[str] = []
    monkeypatch.setattr("product.desk_pipeline.acquire_freshness", lambda: hits.append("x") or {})
    payload = load_desk_pipeline_status(path)
    assert payload["freshness"] == SNAPSHOT_STALE
    assert "stale" in payload["message"].lower()
    assert hits == []
