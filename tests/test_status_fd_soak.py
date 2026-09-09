"""Repeated status reads must not grow process FDs linearly."""
from __future__ import annotations

from fastapi.testclient import TestClient

from product.desk_pipeline import persist_desk_pipeline_snapshot
from product.process_resources import count_open_fds


def test_repeated_status_requests_do_not_grow_fds():
    import terminal_product_api as tpa

    persist_desk_pipeline_snapshot({
        "sequential": True,
        "queued_kind": None,
        "steps": [{"id": "prices", "state": "ready"}],
        "message": "ok",
    })
    client = TestClient(tpa.app)
    for _ in range(8):
        assert client.get("/api/desk-pipeline").status_code == 200
    warm = count_open_fds()
    assert warm is not None
    for _ in range(600):
        assert client.get("/api/desk-pipeline").status_code == 200
        assert client.get("/api/health").status_code == 200
    after = count_open_fds()
    assert after is not None
    assert after - warm <= 8, (warm, after)
