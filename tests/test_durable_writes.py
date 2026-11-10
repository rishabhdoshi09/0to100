"""QT-PRA-084: state whose loss matters reaches stable storage on macOS.

On macOS ``fsync`` only hands data to the drive; the owner's runtime sits on a
USB disk that disconnected on 2026-11-09, and a research SQLite index was found
corrupt. ``F_FULLFSYNC`` makes the drive flush its cache, but costs ~0.4 s per
call there (measured), so it is opt-in: the paper book and the owner's
pause/halt use it; heartbeats and caches do not.
"""
from __future__ import annotations

import fcntl
import json
import os

import pytest

from core import atomic_io as A


@pytest.fixture
def full_syncs(monkeypatch):
    calls = []
    real = fcntl.fcntl

    def fake(fd, cmd, *args):
        if cmd == getattr(fcntl, "F_FULLFSYNC", -12345):
            calls.append(fd)
            return 0
        return real(fd, cmd, *args)

    monkeypatch.setattr(A.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", getattr(fcntl, "F_FULLFSYNC", 51), raising=False)
    monkeypatch.setattr(fcntl, "fcntl", fake)
    return calls


def test_durable_write_full_syncs_file_and_directory(tmp_path, full_syncs):
    target = tmp_path / "owner_state.json"
    A.atomic_write_text(target, '{"new_entries_paused": true}', durable=True)
    assert json.loads(target.read_text()) == {"new_entries_paused": True}
    assert len(full_syncs) == 2  # the data and the directory entry (rename)


def test_default_write_never_pays_for_a_full_flush(tmp_path, full_syncs):
    A.atomic_write_text(tmp_path / "heartbeat.json", "{}")
    A.atomic_write_json(tmp_path / "status.json", {"ok": True})
    assert full_syncs == []


def test_refused_full_flush_falls_back_to_fsync(tmp_path, monkeypatch):
    monkeypatch.setattr(A.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", getattr(fcntl, "F_FULLFSYNC", 51), raising=False)

    def refuse(fd, cmd, *args):
        raise OSError(45, "Operation not supported")

    monkeypatch.setattr(fcntl, "fcntl", refuse)
    fsyncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(A.os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1])
    A.atomic_write_text(tmp_path / "book.json", "{}", durable=True)
    assert len(fsyncs) >= 1


def test_paper_book_persist_uses_the_durable_path(tmp_path, monkeypatch):
    from research.auto_research.scheduler import AutoResearchBrain

    book = tmp_path / "intel_book.json"
    brain = AutoResearchBrain(paper_config_path=tmp_path / "paper_config.json", intel_book_path=book)
    synced = []
    monkeypatch.setattr(A, "durable_sync", lambda fd: synced.append(fd))
    monkeypatch.setattr(A, "_sync_directory", lambda d: synced.append(str(d)))

    assert brain._save_intel_book() is True
    assert book.exists() and "_runtime_commit" in json.loads(book.read_text())
    assert len(synced) == 2 and synced[-1] == str(tmp_path)


def test_owner_pause_state_is_saved_durably(tmp_path, monkeypatch):
    from research.autonomy import supervisor as SUP

    seen = []
    real = A.atomic_write_text

    def spy(path, text, **kwargs):
        seen.append((str(path), kwargs.get("durable")))
        return real(path, text, **kwargs)

    monkeypatch.setattr(A, "atomic_write_text", spy)
    holder = SUP.Supervisor.__new__(SUP.Supervisor)
    holder._owner_path = tmp_path / "owner_state.json"
    holder.owner_state = {"new_entries_paused": True, "halted": False}
    holder._save_owner_state()
    assert seen == [(str(tmp_path / "owner_state.json"), True)]


def _pragma(conn, name):
    return conn.execute(f"PRAGMA {name}").fetchone()[0]


def test_money_path_sqlite_stores_request_full_sync_on_macos(tmp_path, monkeypatch):
    from execution.oms.store import OmsStore
    from execution.protection.store import ProtectionStore
    from product.fo_paper_store import FoPaperStore

    monkeypatch.setattr(A.sys, "platform", "darwin")
    oms = OmsStore(tmp_path / "orders.db")._connect()
    protection = ProtectionStore(tmp_path / "plans.db")._connect()
    fo = FoPaperStore(tmp_path / "fo_paper.sqlite3").conn
    for conn in (oms, protection, fo):
        assert _pragma(conn, "fullfsync") == 1
        assert _pragma(conn, "checkpoint_fullfsync") == 1


def test_sqlite_durability_is_left_alone_off_macos(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setattr(A.sys, "platform", "linux")
    conn = sqlite3.connect(tmp_path / "x.db")
    A.make_sqlite_durable(conn)
    assert _pragma(conn, "fullfsync") == 0
