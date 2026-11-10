"""QT-PRA-086: the daily operating report detects corrupt SQLite stores.

On the owner's Mac (2026-11-10) feature_store.db had carried a corrupt UNIQUE
index ("row 10505 missing from index") for over a month and nothing noticed.
quick_check does not compare index entries with rows, and checking live files
in place on the USB-backed runtime took minutes per store while slowing the
desk, so the check copies each store to local scratch and runs the full
integrity_check on the copy.
"""
from __future__ import annotations

import sqlite3

import pytest

from core.runtime_paths import logs_dir
from product import silent_wrongness_checks as W


@pytest.fixture(autouse=True)
def _own_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    (tmp_path / "runtime" / "logs").mkdir(parents=True)


def _make_store(path, rows=400):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE obs (id TEXT PRIMARY KEY, payload TEXT)")
    conn.executemany("INSERT INTO obs VALUES (?, ?)", [(f"id-{i:05d}", "x" * 200) for i in range(rows)])
    conn.commit()
    conn.close()


def _corrupt(path):
    """Damage b-tree structure (a page header), not just payload bytes."""
    data = bytearray(path.read_bytes())
    page = 4096
    for page_no in (2, 3, 4):
        start = page_no * page
        data[start + 3:start + 8] = b"\xff\xff\xff\xff\xff"  # cell count / content offset
    path.write_bytes(bytes(data))


def test_healthy_stores_pass_without_touching_the_live_files():
    root = logs_dir()
    _make_store(root / "trades.db")
    _make_store(root / "product" / "fo_paper.sqlite3")
    before = sorted(p.name for p in root.rglob("*"))
    assert W.check_sqlite_store_integrity() is None
    assert sorted(p.name for p in root.rglob("*")) == before  # no -wal/-shm/-journal created


def test_a_corrupt_store_is_critical_with_its_name():
    root = logs_dir()
    _make_store(root / "trades.db")
    _make_store(root / "feature_store.db")
    _corrupt(root / "feature_store.db")
    finding = W.check_sqlite_store_integrity()
    assert finding is not None and finding.severity == W.CRITICAL
    assert [c["store"] for c in finding.evidence["corrupt"]] == ["feature_store.db"]
    assert finding.evidence["checked"] == 2


def test_a_transient_failure_is_rechecked_before_it_counts(monkeypatch):
    _make_store(logs_dir() / "orders.db")
    calls = {"n": 0}

    def flaky(path, scratch):
        calls["n"] += 1
        return ["*** in database main *** Page 3: torn read"] if calls["n"] == 1 else ["ok"]

    monkeypatch.setattr(W, "_integrity_check_copy", flaky)
    assert W.check_sqlite_store_integrity() is None
    assert calls["n"] == 2


def test_backups_and_deep_archives_are_not_live_stores():
    root = logs_dir()
    _make_store(root / "backup" / "2026-07-30" / "trades.db")
    _make_store(root / "kite_history" / "TCS" / "deep" / "x.db")
    _make_store(root / "oms" / "orders.db")
    assert [p.relative_to(root).as_posix() for p in W._live_sqlite_stores()] == ["oms/orders.db"]


def test_the_check_runs_in_the_daily_silent_wrongness_battery():
    assert W.check_sqlite_store_integrity in W.CHECKS
