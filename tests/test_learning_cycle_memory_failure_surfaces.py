"""A learning-memory consolidation failure must not be reported as a clean
"learning complete" success.

run_learning_cycle (research/autonomy/jobs.py) runs the primary learning
cycle, then folds settled outcomes into calibrated memory via
consume_learning_memory(). That second step used to have its exception
caught and stuffed into result["settled_memory"]["error"] with NO other
trace: the job still returned JS.SUCCEEDED with a summary literally saying
"learning complete", and did not set the existing H.LEARNING_FAILED flag.
That buries a real learning-ingestion failure -- the operator, the
supervisor's active-failures set, and the job's own error_code/error_message
all saw a clean success. This violates the product invariant that no system
failure may look like a valid decision.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import supervisor_state as ST
from research.autonomy.jobs import _Ctx, run_learning_cycle


class _Deps:
    def __init__(self, learning_result=None):
        self._learning_result = learning_result or {"diagnostics": 3, "paper_closed": 2}

    def now_ist(self):
        return datetime(2026, 9, 18, 20, 0)

    def holidays(self):
        return set()

    def run_learning(self, session_date, dialogue):
        return dict(self._learning_result)


def _ctx(deps):
    return _Ctx(deps, active_failures=set())


def test_memory_consolidation_failure_sets_learning_failed_and_is_not_silent(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    def _boom(session_date):
        raise RuntimeError("simulated ledger corruption")

    monkeypatch.setattr(
        "product.autonomous_loop.consume_learning_memory", _boom
    )

    result = run_learning_cycle(_ctx(_Deps()))

    assert result.status == JS.SUCCEEDED  # the primary learning cycle itself did succeed
    assert H.LEARNING_FAILED in result.failures
    assert result.error_code == "LEARNING_MEMORY_ERROR"
    assert "simulated ledger corruption" in result.error_message
    assert "complete" not in result.summary.lower() or "failed" in result.summary.lower()
    assert result.metadata["settled_memory"]["error"] == "simulated ledger corruption"


def test_successful_memory_consolidation_clears_learning_failed(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        "product.autonomous_loop.consume_learning_memory",
        lambda session_date: {"observations": 5},
    )

    result = run_learning_cycle(_ctx(_Deps()))

    assert result.status == JS.SUCCEEDED
    assert H.LEARNING_FAILED in result.clears
    assert not result.failures
    assert result.error_code == ""
    assert "learning complete" in result.summary.lower()
