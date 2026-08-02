"""OperationStore must close SQLite FDs — with sqlite3.connect() does not."""
from __future__ import annotations

import gc
import os
import sqlite3
from pathlib import Path

import pytest

from operations.store import OperationStore


def _open_fd_count() -> int:
    fd_dir = Path("/proc/self/fd")
    if fd_dir.is_dir():
        return len(list(fd_dir.iterdir()))
    # macOS / other: best-effort via resource is not a count; skip if unavailable.
    return -1


def test_sqlite_with_does_not_close_connection(tmp_path: Path):
    """Document the CPython gotcha this store must defend against."""
    db = tmp_path / "demo.db"
    with sqlite3.connect(str(db)) as con:
        con.execute("CREATE TABLE t(x)")
    # Connection remains usable after the with-block on modern CPython.
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()


def test_operation_store_progress_does_not_leak_fds(tmp_path: Path):
    before = _open_fd_count()
    if before < 0:
        pytest.skip("fd counting unavailable on this platform")

    store = OperationStore(tmp_path / "jobs.db")
    record, created = store.enqueue("MARKET_SCAN", lane="market_scan")
    assert created is True
    leased = store.lease_next("market_scan", worker_pid=os.getpid())
    assert leased is not None
    op_id = leased["operation_id"]

    gc.disable()
    try:
        for i in range(200):
            store.progress(
                op_id,
                stage="SCANNING",
                message=f"tick {i}",
                current=i,
                total=200,
            )
            store.lease_next("idle_lane", worker_pid=os.getpid())
        after = _open_fd_count()
    finally:
        gc.enable()
        gc.collect()

    # Allow a small amount of noise from the test harness / sqlite WAL sidecars.
    assert after - before < 40, f"FD leak suspected: before={before} after={after}"


def test_fundamentals_cache_closes_connections(tmp_path: Path, monkeypatch):
    import fundamentals.cache as cache_mod

    db = tmp_path / "fundamentals_cache.db"
    monkeypatch.setattr(cache_mod, "_DB_PATH", db)
    before = _open_fd_count()
    if before < 0:
        pytest.skip("fd counting unavailable on this platform")

    cache = cache_mod.FundamentalsCache()
    gc.disable()
    try:
        for i in range(120):
            cache.set(f"SYM{i}", {"pe": float(i)})
            cache.get(f"SYM{i}")
            cache.get_any(f"SYM{i}")
            cache.has(f"SYM{i}")
        after = _open_fd_count()
    finally:
        gc.enable()
        gc.collect()

    assert after - before < 40, f"FD leak suspected: before={before} after={after}"
