"""An outcome-resolution failure must fail the job, never look like success.

run_outcome_resolution's entire purpose is to resolve paper-book outcomes
for the session. Its exception used to be caught and stuffed into
result["paper_book_error"], with the job still returning JS.SUCCEEDED and a
summary literally saying "outcomes resolved · 0 book closes · 0 decoded ·
0 official" -- indistinguishable from a genuine day with nothing to
resolve. That is exactly the "system failure disguised as a valid
decision" pattern the product explicitly forbids, for a job section 15 of
the audit names by name (OUTCOME_RESOLUTION).
"""
from __future__ import annotations

from datetime import datetime

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy.jobs import _Ctx, run_outcome_resolution


class _Deps:
    def __init__(self, resolve_error=None):
        self._resolve_error = resolve_error

    def now_ist(self):
        return datetime(2026, 9, 18, 20, 0)

    def holidays(self):
        return set()

    def official_history(self):
        return {"current": True, "available_session": "2026-09-18"}

    def resolve_outcomes(self, session_date, active_failures):
        if self._resolve_error:
            raise self._resolve_error
        return {"positions_closed": ["TCS"], "outcomes_recorded": ["TCS"]}


def _ctx(deps):
    return _Ctx(deps, active_failures=set())


def test_resolve_outcomes_exception_fails_the_job_not_a_fake_success(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        "product.autonomous_loop.settle_official_outcomes",
        lambda session: {"n_settled": 0},
    )
    monkeypatch.setattr(
        "product.paper_self_feed.ingest_paper_cycle", lambda *a, **k: None
    )

    deps = _Deps(resolve_error=RuntimeError("simulated paper book corruption"))
    result = run_outcome_resolution(_ctx(deps))

    assert result.status == JS.RETRYABLE_FAILED
    assert "simulated paper book corruption" in result.error_message
    assert result.error_code == "OUTCOME_RESOLUTION_ERROR"
    assert H.UNRECONCILED in result.failures
    assert "resolved" not in result.summary.lower()


def test_successful_resolution_clears_unreconciled_and_reports_real_counts(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        "product.autonomous_loop.settle_official_outcomes",
        lambda session: {"n_settled": 1},
    )
    monkeypatch.setattr(
        "product.paper_self_feed.ingest_paper_cycle", lambda *a, **k: None
    )

    result = run_outcome_resolution(_ctx(_Deps()))

    assert result.status == JS.SUCCEEDED
    assert H.UNRECONCILED in result.clears
    assert not result.failures
    assert "outcomes resolved · 1 book closes · 1 decoded · 1 official" == result.summary
