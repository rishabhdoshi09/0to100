"""Watchlist store and API contract."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import terminal_api as api
from product.watchlist_store import add_item, list_items, remove_item


def test_watchlist_crud(tmp_path: Path):
    db = tmp_path / "watchlist.db"
    item = add_item("RELIANCE", notes="test", path=db)
    assert item["symbol"] == "RELIANCE"
    rows = list_items(path=db)
    assert len(rows) == 1
    remove_item(rows[0]["id"], path=db)
    assert list_items(path=db) == []


def test_watchlist_api_get_and_post(tmp_path: Path, monkeypatch):
    db = tmp_path / "watchlist.db"
    monkeypatch.setattr("product.watchlist_store.DEFAULT_DB", db)
    client = TestClient(api.app)
    from product.observer_api import install
    install(api.app)
    post = client.post("/api/watchlist", json={"symbol": "TCS", "notes": "radar"})
    assert post.status_code == 200
    get = client.get("/api/watchlist")
    assert get.status_code == 200
    assert get.json()["count"] >= 1
