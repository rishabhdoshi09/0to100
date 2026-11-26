"""Lifecycle audit F8: a counterfactual freeze no longer reads the whole ledger.

Every freeze read, decoded, json-parsed and hashed the WHOLE ledger to ask whether its
counterfactual_id was already there: about 1 s per freeze on a 32 MB ledger (measured on a
disposable copy), O(n²) over a cycle. The freeze path now keeps an index that it trusts only
while the ledger's identity (device, inode, size, mtime_ns, ctime_ns) is exactly what this
process last read or wrote under the writer lock; any other change is read and validated
in full again, exactly as before. These pin both halves: the reuse, and every way the file
can change underneath it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cf(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(tmp_path / "counterfactuals.jsonl"))
    from product import counterfactual_learning as CF

    getattr(CF, "_INDEX", {}).clear()
    builds = []                                   # every whole-ledger read the freeze path makes
    for name in ("_build_index", "_read_ledger"):
        real = getattr(CF, name, None)
        if real is not None:
            monkeypatch.setattr(CF, name, lambda path, _real=real: builds.append(path) or _real(path))

    def freeze(symbol, decision_id=None):
        return CF.freeze_decision(symbol=symbol, reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                                  entry=100, stop=94, target=115, as_of="2026-09-10",
                                  evidence={"decision_id": decision_id or f"d-{symbol}"})

    yield CF, tmp_path / "counterfactuals.jsonl", freeze, builds
    getattr(CF, "_INDEX", {}).clear()
    from product import sqlite_runtime
    sqlite_runtime.close_keepers()


def _rows(led: Path) -> list[dict]:
    return [json.loads(line) for line in led.read_text().splitlines() if line.strip()]


def test_a_cycle_of_freezes_reads_the_ledger_once(cf):
    CF, led, freeze, builds = cf
    for i in range(30):
        freeze(f"S{i}")
    for i in range(30):
        freeze(f"S{i}")                                   # every retry dedupes from the index
    assert len(builds) == 1                               # the first freeze only (an empty ledger)
    rows = _rows(led)
    assert [r["symbol"] for r in rows] == [f"S{i}" for i in range(30)]
    assert len({r["counterfactual_id"] for r in rows}) == 30


def test_a_duplicate_freeze_returns_the_row_already_on_disk(cf):
    CF, led, freeze, _ = cf
    first = freeze("TCS")
    again = freeze("TCS")
    assert again == _rows(led)[0] and again["counterfactual_id"] == first["counterfactual_id"]
    assert again["recorded_at"] == first["recorded_at"]   # the stored row, not this call's


def test_a_freeze_by_another_process_between_two_of_ours_is_seen(cf):
    CF, led, freeze, builds = cf
    freeze("TCS")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from product import counterfactual_learning as CF;"
            "CF.freeze_decision(symbol='INFY', reason_code='WAIT_FOR_PULLBACK', decision='WAIT', entry=100,"
            " stop=94, target=115, as_of='2026-09-10', evidence={'decision_id': 'd-INFY'})")
    env = {**os.environ, "QT_RUNTIME_ROOT": str(led.parent / "rt"), "QT_COUNTERFACTUALS": str(led)}
    (led.parent / "rt").mkdir()
    subprocess.run([sys.executable, "-c", code, str(REPO)], env=env, check=True, timeout=120)
    again = freeze("INFY")                                # the other process's row, never a second one
    assert [r["symbol"] for r in _rows(led)] == ["TCS", "INFY"] and again == _rows(led)[1]
    assert len(builds) == 2                               # its write changed the identity: read again


def test_an_in_place_edit_that_keeps_the_size_is_never_trusted(cf):
    CF, led, freeze, _ = cf
    freeze("TCS")
    freeze("INFY")
    raw = led.read_bytes()
    at = raw.index(b'"symbol"')
    fd = os.open(led, os.O_WRONLY)
    try:
        os.pwrite(fd, b"#", at)                           # same size, same inode: now invalid JSON
    finally:
        os.close(fd)
    with pytest.raises(ValueError, match="invalid JSON"):
        freeze("WIPRO")
    with pytest.raises(ValueError, match="invalid JSON"):
        freeze("TCS")                                     # not even a duplicate is answered from memory


def test_a_settlement_rewrite_is_read_again(cf):
    CF, led, freeze, _ = cf
    freeze("TCS")
    freeze("INFY")
    from core.atomic_io import atomic_write_text
    with CF.locked_ledger() as (path, rows, _):
        rows[0]["classification"] = "GOOD_WAIT"
        atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows), durable=True)
    assert freeze("TCS")["classification"] == "GOOD_WAIT"        # the rewritten row
    assert len(_rows(led)) == 2


def test_an_append_cut_after_its_intent_is_finished_from_the_index_prefix(cf, monkeypatch):
    """The intent's prefix sha256 comes from the index's running digest: a wrong digest would
    make the recovery refuse the ledger as changed outside its writers."""
    CF, led, freeze, _ = cf
    freeze("TCS")
    freeze("INFY")
    with monkeypatch.context() as m:
        m.setattr(CF, "_finish_append", lambda path, missing: (_ for _ in ()).throw(RuntimeError("power cut")))
        with pytest.raises(RuntimeError):
            freeze("WIPRO")
    assert CF._append_intent_path(led).exists()
    freeze("HDFC")                                        # the recovery finishes WIPRO first
    assert [r["symbol"] for r in _rows(led)] == ["TCS", "INFY", "WIPRO", "HDFC"]
    assert not CF._append_intent_path(led).exists()
    assert freeze("WIPRO")["symbol"] == "WIPRO" and len(_rows(led)) == 4


def test_a_ledger_without_a_final_newline_gets_its_separator(cf):
    CF, led, freeze, _ = cf
    freeze("TCS")
    led.write_bytes(led.read_bytes().rstrip(b"\n"))      # a complete last row, no newline
    freeze("INFY")
    assert [r["symbol"] for r in _rows(led)] == ["TCS", "INFY"]
    assert freeze("INFY") == _rows(led)[1]               # read back from the index's span, past the separator
    freeze("WIPRO")
    assert [r["symbol"] for r in _rows(led)] == ["TCS", "INFY", "WIPRO"]


def test_the_index_matches_a_full_read_after_every_freeze(cf):
    CF, led, freeze, _ = cf
    for i in range(12):
        freeze(f"S{i % 5}", decision_id=f"d{i}")
        index = CF._INDEX[str(led)]
        fresh = CF._build_index(led)
        assert (index["size"], index["ids"], index["digest"].hexdigest(), index["key"]) == (
            fresh["size"], fresh["ids"], fresh["digest"].hexdigest(), fresh["key"])


# ── the torn-tail check reads only the tail ──

def test_a_torn_tail_longer_than_one_read_is_quarantined_whole(tmp_path):
    from core.atomic_io import quarantine_torn_jsonl_tail

    led = tmp_path / "l.jsonl"
    head = b'{"a": 1}\n' * 20000
    tail = b'{"torn": "' + b"x" * 200_000
    led.write_bytes(head + tail)
    side = quarantine_torn_jsonl_tail(led)
    assert side is not None and side.read_bytes() == tail and led.read_bytes() == head


def test_a_torn_file_with_no_newline_at_all_is_quarantined_whole(tmp_path):
    from core.atomic_io import quarantine_torn_jsonl_tail

    led = tmp_path / "l.jsonl"
    led.write_bytes(b'{"torn": ' + b"y" * 70_000)
    side = quarantine_torn_jsonl_tail(led)
    assert side is not None and side.read_bytes().startswith(b'{"torn": ') and led.read_bytes() == b""


def test_the_torn_tail_check_never_reads_the_whole_file(tmp_path, monkeypatch):
    from core import atomic_io

    led = tmp_path / "l.jsonl"
    led.write_bytes(b'{"a": 1}\n' * 100_000 + b'{"b": 2}')
    reads = []
    real = os.pread
    monkeypatch.setattr(atomic_io.os, "pread", lambda fd, n, off: reads.append(n) or real(fd, n, off))
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("a whole-file read")))
    assert atomic_io.quarantine_torn_jsonl_tail(led) is None
    assert sum(reads) < 70_000 < led.stat().st_size
