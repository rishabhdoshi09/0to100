"""Lifecycle audit of 24 Nov (F3, F4, F5): one bad byte must not stop learning for good.

Each was a permanent wedge: every freeze, fill credit and settlement refused, while every
producer swallowed the error. Only what an interrupted write can legitimately leave is
recovered (a torn LAST line, a stale append intent); anything else is still refused.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from core.atomic_io import quarantine_torn_jsonl_tail


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── the helper ──

def test_a_torn_last_line_is_moved_to_a_sidecar_and_cut(tmp_path):
    led = tmp_path / "l.jsonl"
    led.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": 3, "d"')
    side = quarantine_torn_jsonl_tail(led)
    assert side is not None and side.read_bytes() == b'{"c": 3, "d"'
    assert led.read_bytes() == b'{"a": 1}\n{"b": 2}\n'


@pytest.mark.parametrize("raw", [b'{"a": 1}\n{"b": 2}', b'{"a": 1}\n', b""])
def test_a_complete_or_empty_ledger_is_left_alone(tmp_path, raw):
    led = tmp_path / "l.jsonl"
    led.write_bytes(raw)
    assert quarantine_torn_jsonl_tail(led) is None and led.read_bytes() == raw
    assert not list(tmp_path.glob("*.torn-*"))


def test_an_invalid_middle_line_is_corruption_not_touched(tmp_path):
    led = tmp_path / "l.jsonl"
    raw = b'{"a": 1}\nGARBAGE\n{"c": 3}\n'
    led.write_bytes(raw)
    assert quarantine_torn_jsonl_tail(led) is None and led.read_bytes() == raw


# ── F3: the counterfactual ledger ──

@pytest.fixture
def cf(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(tmp_path / "counterfactuals.jsonl"))
    from product import counterfactual_learning as CF

    def freeze(symbol):
        return CF.freeze_decision(symbol=symbol, reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                                  entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": symbol})

    yield CF, tmp_path / "counterfactuals.jsonl", freeze
    from product import sqlite_runtime
    sqlite_runtime.close_keepers()


def test_a_torn_ledger_tail_no_longer_stops_freezes(cf):
    CF, led, freeze = cf
    freeze("TCS")
    freeze("INFY")
    with led.open("ab") as fh:
        fh.write(b'{"counterfactual_id": "torn", "sym')                 # a pre-intent append cut short
    freeze("WIPRO")
    assert [r["symbol"] for r in _lines(led)] == ["TCS", "INFY", "WIPRO"]
    sides = list(led.parent.glob(led.name + ".torn-*"))
    assert len(sides) == 1 and sides[0].read_bytes() == b'{"counterfactual_id": "torn", "sym'


def test_an_invalid_middle_line_still_refuses(cf):
    CF, led, freeze = cf
    freeze("TCS")
    led.write_bytes(b"GARBAGE\n" + led.read_bytes())
    with pytest.raises(ValueError, match="invalid JSON"):
        freeze("INFY")


# ── F4: a stale append intent ──
# Its trigger, a non-durable rewrite lost to a power cut, is gone at the root: every rewrite is
# durable before the next write (tests/test_learning_durability.py). A ledger changed outside
# its writers keeps Codex's refusal (tests/test_decision_freeze_integrity.py), now explained.

def test_a_changed_ledger_under_an_intent_refuses_with_a_remedy(cf, monkeypatch):
    CF, led, freeze = cf
    freeze("TCS")

    def cut(path, missing):
        raise RuntimeError("power cut after the intent")

    with monkeypatch.context() as patch:
        patch.setattr(CF, "_finish_append", cut)
        with pytest.raises(RuntimeError):
            freeze("WIPRO")
    led.write_bytes(led.read_bytes().replace(b'"TCS"', b'"HAL"'))
    with pytest.raises(ValueError, match="changed outside its writers") as exc:
        freeze("HDFC")
    assert "restore the ledger that matches the intent" in str(exc.value)
    assert CF._append_intent_path(led).exists()


# ── F5: the learning audit file ──

def test_a_torn_learning_audit_file_no_longer_wedges_the_replay(tmp_path, monkeypatch):
    from product import autonomous_loop as loop

    path = tmp_path / "learning_observations.jsonl"
    monkeypatch.setattr(loop, "LEARNING_PATH", path)
    path.write_bytes(b'{"settlement_event_id": "e1"}\n{"observation": "memory_refr')
    loop._publish_settlement_audit([{"settlement_event_id": "e2", "symbol": "TCS"}])
    assert [r.get("settlement_event_id") for r in _lines(path)] == ["e1", "e2"]
    assert len(list(tmp_path.glob(path.name + ".torn-*"))) == 1


def test_the_memory_append_is_synced(tmp_path, monkeypatch):
    from core import atomic_io
    from product import autonomous_loop as loop

    synced = []
    real = atomic_io.durable_sync
    monkeypatch.setattr(atomic_io, "durable_sync", lambda fd: (synced.append(fd), real(fd)))
    path = tmp_path / "learning_observations.jsonl"
    loop._append_jsonl_durable(path, {"observation": "memory_refresh"})
    assert synced and _lines(path) == [{"observation": "memory_refresh"}]


# ── F3: producers no longer swallow a refused ledger ──

def test_a_recording_failure_is_logged(capsys):
    from product.counterfactual_learning import note_recording_failure

    note_recording_failure("counterfactual freeze", "TCS", ValueError("invalid JSON in counterfactual ledger"))
    assert "[COUNTERFACTUAL] counterfactual freeze failed for TCS: ValueError" in capsys.readouterr().err


def test_every_producer_reports_instead_of_passing():
    from product import autonomous_loop, paper_autopilot

    assert inspect.getsource(paper_autopilot).count("note_recording_failure(") == 3
    assert inspect.getsource(autonomous_loop).count("note_recording_failure(") == 1
