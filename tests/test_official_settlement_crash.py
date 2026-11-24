"""Crash recovery for the official counterfactual settlement projections."""
import json
import os
import subprocess
import sys

import pytest

from product import candidate_lifecycle as candidates
from product import counterfactual_learning as counterfactual

_RUN = r"""
import os
import sys
from pathlib import Path

from tests import network_policy
network_policy.install()
from core import atomic_io, outcome_resolver
from product import autonomous_loop as loop, candidate_lifecycle as candidates, forward_soak
from tests.evidence_fixtures import rulebook_outcome

root, checkpoint = Path(sys.argv[1]), sys.argv[2]
loop.LEARNING_PATH = root / "learning.jsonl"
candidates.DB_PATH = root / "candidates.db"
outcome_resolver.counterfactual_outcome = lambda *a: rulebook_outcome(1.0, "WIN", 10.0)
forward_soak.settle_pending_from_market = lambda: {}
write = atomic_io.atomic_write_text
close_candidate = candidates.close_counterfactual
ledger_writes = 0

def crash_after_write(path, *args, **kwargs):
    global ledger_writes
    if Path(path) == root / "counterfactuals.jsonl":
        ledger_writes += 1
        if checkpoint == "before-commit":
            os._exit(77)
    result = write(path, *args, **kwargs)
    if Path(path) == root / "counterfactuals.jsonl":
        if (ledger_writes == 1 and checkpoint == "after-commit"
                or ledger_writes == 2 and checkpoint == "after-ack"):
            os._exit(77)
    if Path(path) == loop.LEARNING_PATH and checkpoint == "after-audit":
        os._exit(77)
    return result

def crash_after_transition(*args):
    close_candidate(*args)
    if checkpoint == "after-transition":
        os._exit(77)

atomic_io.atomic_write_text = crash_after_write
candidates.close_counterfactual = crash_after_transition
result = loop.settle_official_outcomes("2026-08-03")
assert not result["failed"], result
"""


@pytest.mark.parametrize(
    "checkpoint", ["before-commit", "after-commit", "after-audit", "after-transition", "after-ack"],
)
def test_official_settlement_recovers_after_process_exit(tmp_path, monkeypatch, checkpoint):
    ledger = tmp_path / "counterfactuals.jsonl"
    audit = tmp_path / "learning.jsonl"
    monkeypatch.setattr(candidates, "DB_PATH", tmp_path / "candidates.db")
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(ledger))
    counterfactual.freeze_decision(symbol="HAL", reason_code="WAIT_FOR_ENTRY", decision="WAIT",
                                  entry=100, stop=90, target=120, as_of="2026-08-03")
    cid = candidates.upsert(symbol="HAL", session_date="2026-08-03", state=candidates.WAIT)["candidate_id"]

    def run(point):
        return subprocess.run(
            [sys.executable, "-c", _RUN, str(tmp_path), point],
            env=os.environ.copy(), capture_output=True, text=True, timeout=60,
        )

    interrupted = run(checkpoint)
    assert interrupted.returncode == 77, interrupted.stderr
    (row,) = map(json.loads, ledger.read_text().splitlines())
    if checkpoint == "before-commit":
        assert row["outcome"] is None
        assert not audit.exists()
        assert candidates.get(cid)["state"] == candidates.WAIT
    else:
        assert row["classification"] == "MISSED_WINNER"

    for _ in range(2):
        replay = run("no-crash")
        assert replay.returncode == 0, replay.stderr

    (row,) = map(json.loads, ledger.read_text().splitlines())
    (event,) = map(json.loads, audit.read_text().splitlines())
    assert row["classification"] == event["classification"] == "MISSED_WINNER"
    assert "pending_settlement_audit" not in row
    assert [t["to_state"] for t in candidates.transitions_for(cid)] == [candidates.WAIT, candidates.CLOSED]
