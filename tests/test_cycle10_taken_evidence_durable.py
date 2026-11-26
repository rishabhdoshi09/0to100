"""Cycle 10: taken-trade evidence is appended durably and never fused onto a torn line."""
from __future__ import annotations

import json

from product import _paper_learning_loop_core as core


def test_an_append_is_synced_before_it_returns(tmp_path, monkeypatch):
    from core import atomic_io

    synced = []
    real = atomic_io.durable_sync
    monkeypatch.setattr(atomic_io, "durable_sync", lambda fd: synced.append(fd) or real(fd))
    core._append_jsonl(tmp_path / "taken.jsonl", {"symbol": "TCS"})
    assert len(synced) == 1


def test_a_record_after_a_torn_append_is_kept_whole(tmp_path):
    path = tmp_path / "taken.jsonl"
    path.write_bytes(b'{"symbol": "INFY"}\n{"symbol": "WIP')         # a crash cut the last append
    core._append_jsonl(path, {"symbol": "TCS"})
    assert [r["symbol"] for r in core._read_jsonl(path)] == ["INFY", "TCS"]
    assert path.read_bytes().endswith(b'\n{"symbol": "TCS"}\n')


def test_appends_to_a_clean_file_add_no_blank_lines(tmp_path):
    path = tmp_path / "taken.jsonl"
    for s in ("A", "B"):
        core._append_jsonl(path, {"symbol": s})
    assert path.read_text().splitlines() == [json.dumps({"symbol": "A"}), json.dumps({"symbol": "B"})]


def test_C10_02_a_short_write_never_corrupts_a_concurrent_append(tmp_path, monkeypatch):
    """GPT C10-02: writer A's record is cut by a short write; writer B appends meanwhile. Without
    a lock from the tail check through the sync, B lands inside A's record and both are lost."""
    import os
    import threading

    path = tmp_path / "taken.jsonl"
    path.write_bytes(b'{"symbol": "INFY"}\n')
    real_write = os.write
    a_partial, release_a = threading.Event(), threading.Event()

    def write(fd, data):
        if threading.current_thread().name == "writer-a" and not a_partial.is_set():
            n = real_write(fd, bytes(data)[:8])            # a short write: the rest comes later
            a_partial.set()
            assert release_a.wait(10)
            return n
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", write)
    a = threading.Thread(target=core._append_jsonl, args=(path, {"symbol": "TCS"}), name="writer-a")
    b = threading.Thread(target=core._append_jsonl, args=(path, {"symbol": "SBIN"}), name="writer-b")
    a.start()
    assert a_partial.wait(10)
    b.start()
    b.join(0.5)
    blocked = b.is_alive()                                  # B waits for A's lock
    release_a.set()
    a.join(10)
    b.join(10)
    assert blocked
    assert {r["symbol"] for r in core._read_jsonl(path)} == {"INFY", "TCS", "SBIN"}
