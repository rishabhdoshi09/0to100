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
