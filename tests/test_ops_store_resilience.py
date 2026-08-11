"""Market-ops SQLite store must survive missing parent dirs / lease failures."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from operations.store import OperationStore


def test_store_recreates_missing_parent_on_connect(tmp_path: Path):
    db = tmp_path / "market_ops" / "jobs.db"
    store = OperationStore(db)
    assert db.exists()
    # Simulate a cleanup wiping the directory while the process is alive.
    for child in db.parent.iterdir():
        child.unlink()
    db.parent.rmdir()
    assert not db.parent.exists()
    with store._connect() as con:
        con.execute("SELECT 1")
    assert db.parent.is_dir()


def test_store_uses_absolute_path(tmp_path: Path, monkeypatch):
    db = tmp_path / "rel" / "jobs.db"
    db.parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    store = OperationStore(Path("rel/jobs.db"))
    assert store.path.is_absolute()
    monkeypatch.chdir(tmp_path / "rel")
    # Still opens the original absolute DB after CWD change.
    store.enqueue("MARKET_SCAN", lane="market_scan", requested_by="test")
    assert store.path.exists()


def test_lane_loop_survives_lease_operational_error(tmp_path: Path, monkeypatch):
    import operations.market_ops as mo

    class BoomStore:
        def __init__(self):
            self.calls = 0

        def lease_next(self, lane, *, worker_pid):
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("unable to open database file")
            return None

        def _ensure_parent(self):
            return None

    worker = mo.MarketOperationsWorker.__new__(mo.MarketOperationsWorker)
    worker.store = BoomStore()
    worker.stop_event = __import__("threading").Event()
    worker._active = {}
    worker._set_active = lambda *a, **k: None

    def stop_soon(*_a, **_k):
        worker.stop_event.set()

    monkeypatch.setattr(worker.stop_event, "wait", stop_soon)
    # Must return (not raise) after the first lease failure.
    worker._lane_loop("data")
    assert worker.store.calls >= 1
