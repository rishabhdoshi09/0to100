"""QT-PRA-073: the autonomy dialogue tail must not parse the whole log.

On the owner's Mac (2026-11-10) dialogue.jsonl was 13 MB / ~17k records and
Home called ``DialogueLog.recent(8)`` seven times per request through the
autonomy status read: 119,601 ``json.loads`` and ~5.7 s of an 8.3 s warm Home
request. ``recent`` must return exactly ``all()[-limit:]`` while parsing only
the tail.
"""
from __future__ import annotations

import json

import pytest

from research.autonomy import dialogue as D


def _write(path, lines):
    path.write_bytes(b"".join(lines))


def _record(i, pad=0):
    return (json.dumps({"record_type": D.OBSERVATION, "i": i, "claim": "é✓" + "x" * pad}) + "\n").encode()


@pytest.mark.parametrize("limit", [1, 2, 8, 20, 500])
def test_recent_matches_all_tail_across_block_boundaries_and_bad_lines(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(D, "_TAIL_BLOCK", 97)  # force lines to straddle many blocks
    lines = []
    for i in range(120):
        lines.append(_record(i, pad=i % 37))
        if i % 11 == 0:
            lines.append(b"\n")              # blank line
        if i % 17 == 0:
            lines.append(b"{not json\n")     # torn / garbage line
    lines.append(b'{"record_type": "Observa')  # partial append without newline
    path = tmp_path / "dialogue.jsonl"
    _write(path, lines)
    log = D.DialogueLog(path)

    assert log.recent(limit) == log.all()[-limit:]


def test_recent_on_missing_empty_and_short_logs(tmp_path):
    log = D.DialogueLog(tmp_path / "absent.jsonl")
    assert log.recent(8) == []
    path = tmp_path / "dialogue.jsonl"
    path.write_bytes(b"")
    assert D.DialogueLog(path).recent(8) == []
    _write(path, [_record(1), _record(2)])
    assert [r["i"] for r in D.DialogueLog(path).recent(8)] == [1, 2]
    assert D.DialogueLog(path).recent(0) == D.DialogueLog(path).all()


def test_recent_parses_only_the_tail_of_a_large_log(tmp_path, monkeypatch):
    path = tmp_path / "dialogue.jsonl"
    _write(path, [_record(i, pad=200) for i in range(20_000)])
    parsed = {"n": 0}
    real = D.json.loads

    def counting(text, *a, **kw):
        parsed["n"] += 1
        return real(text, *a, **kw)

    monkeypatch.setattr(D.json, "loads", counting)
    rows = D.DialogueLog(path).recent(8)

    assert [r["i"] for r in rows] == list(range(19_992, 20_000))
    assert parsed["n"] == 8
