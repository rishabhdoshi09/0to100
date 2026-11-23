import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from product import counterfactual_learning as CF
from product import decision_freeze as DF


def _record(*, entry=100.0):
    return {
        "decision_id": "d-1",
        "symbol": "TCS",
        "as_of": "2026-09-21",
        "entry": entry,
        "versions": {"decision_engine_version": "v1"},
    }


@pytest.mark.parametrize("competing_entry", [100.0, 101.0])
def test_freeze_handles_a_decision_committed_by_a_concurrent_writer(tmp_path, monkeypatch, competing_entry):
    path = tmp_path / "freezes.db"
    connect = DF._connect
    winner = None

    class InterleavedConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def __enter__(self):
            return self.connection.__enter__()

        def __exit__(self, *exc):
            return self.connection.__exit__(*exc)

        def execute(self, sql, *args):
            nonlocal winner
            if sql.startswith("INSERT INTO freezes"):
                with monkeypatch.context() as patch:
                    patch.setattr(DF, "_connect", connect)
                    winner = DF.freeze(_record(entry=competing_entry), path=path)
            return self.connection.execute(sql, *args)

    with monkeypatch.context() as patch:
        patch.setattr(DF, "_connect", lambda path: InterleavedConnection(connect(path)))
        if competing_entry == 100.0:
            assert DF.freeze(_record(), path=path) == winner
        else:
            with pytest.raises(DF.DecisionIdentityCollision):
                DF.freeze(_record(), path=path)

    assert DF.get_freeze("d-1", path=path) == winner


def _freeze_rejection(ledger, decision_id="rejected-1"):
    return CF.freeze_decision(
        symbol="TCS", reason_code="ENTRY_TOO_EXTENDED", decision="WAIT",
        entry=100.0, stop=94.0, target=115.0, as_of="2026-09-21",
        evidence={"decision_id": decision_id}, path=ledger,
    )


@pytest.mark.parametrize("tail", [b'{"torn":', b'[]\n', b'\xff\n'])
def test_counterfactual_freeze_refuses_to_append_to_corrupt_ledger(tmp_path, tail):
    ledger = tmp_path / "counterfactuals.jsonl"
    first = _freeze_rejection(ledger)
    original = ledger.read_bytes() + tail
    ledger.write_bytes(original)
    with pytest.raises(ValueError, match="counterfactual ledger"):
        _freeze_rejection(ledger)
    assert ledger.read_bytes() == original
    assert DF.get_freeze("rejected-1", path=CF._counterfactual_freeze_path(ledger))["fingerprint"] == first["decision_fingerprint"]


def test_counterfactual_freeze_appends_after_valid_unterminated_row(tmp_path):
    ledger = tmp_path / "counterfactuals.jsonl"
    first = _freeze_rejection(ledger)
    ledger.write_bytes(ledger.read_bytes().rstrip(b"\n"))
    second = _freeze_rejection(ledger, "rejected-2")
    assert [json.loads(line) for line in ledger.read_bytes().splitlines()] == [first, second]


def test_counterfactual_concurrent_retries_return_the_same_projection(tmp_path, monkeypatch):
    ledger = tmp_path / "counterfactuals.jsonl"
    read = CF._read_ledger
    first_read = threading.Event()
    second_read = threading.Event()

    def interleaved_read(path):
        rows = read(path)
        if first_read.is_set():
            second_read.set()
        else:
            first_read.set()
            # Without the writer lock the second call reads the same old
            # ledger while the first call is still between read and append.
            second_read.wait(timeout=1)
        return rows

    monkeypatch.setattr(CF, "_read_ledger", interleaved_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_freeze_rejection, ledger)
        assert first_read.wait(timeout=10)
        second = pool.submit(_freeze_rejection, ledger)
        assert first.result(timeout=10) == second.result(timeout=10)

    assert [json.loads(line) for line in ledger.read_bytes().splitlines()] == [first.result()]
