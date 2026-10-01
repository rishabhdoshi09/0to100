"""Evolution Engine scheduler wiring: SCH.TOURNAMENT_CYCLE job type.

Off-hours only (enqueued alongside LEARNING_CYCLE/RESEARCH_CYCLE in
research/autonomy/supervisor.py's settlement path, see
_advance_forward_settlement), never critical, never allowed to fail the
job ledger even when its internals break.
"""
from __future__ import annotations

from unittest import mock

import research.autonomy.jobs as JOBS
import research.autonomy.job_store as JS
import research.autonomy.schedules as SCH


def test_tournament_cycle_is_a_known_non_critical_job_type_with_a_handler():
    assert SCH.TOURNAMENT_CYCLE in SCH.ALL_JOB_TYPES
    assert SCH.TOURNAMENT_CYCLE not in SCH.CRITICAL_JOBS
    assert SCH.TOURNAMENT_CYCLE in JOBS.HANDLERS
    assert JOBS.HANDLERS[SCH.TOURNAMENT_CYCLE] is JOBS.run_tournament_cycle


def test_tournament_cycle_key_is_idempotent_per_session_date():
    a = SCH.tournament_cycle_key("2026-09-30")
    b = SCH.tournament_cycle_key("2026-09-30")
    c = SCH.tournament_cycle_key("2026-10-01")
    assert a == b
    assert a != c


def test_run_tournament_cycle_grades_and_evaluates_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    with mock.patch("product.evolution.grading.grade_pending_decisions", return_value=[{"shadow_id": "x"}]) as graded, \
         mock.patch("product.evolution.promotion.evaluate_promotion_batch", return_value=[{"status": "PROMOTION_ELIGIBLE"}]) as promo:
        result = JOBS.run_tournament_cycle(ctx=None)

    assert result.status == JS.SUCCEEDED
    assert "graded=1" in result.summary
    assert "promotion_eligible=" in result.summary
    graded.assert_called_once()
    assert promo.call_count == 2  # once per domain (EQUITY, FNO_UNDERLYING)


def test_run_tournament_cycle_never_fails_the_job_even_if_grading_breaks(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    with mock.patch("product.evolution.grading.grade_pending_decisions", side_effect=RuntimeError("disk error")), \
         mock.patch("product.evolution.promotion.evaluate_promotion_batch", return_value=[]):
        result = JOBS.run_tournament_cycle(ctx=None)
    assert result.status == JS.SUCCEEDED
    assert "grading_error" in result.summary


def test_run_tournament_cycle_never_fails_the_job_even_if_promotion_breaks(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    with mock.patch("product.evolution.grading.grade_pending_decisions", return_value=[]), \
         mock.patch("product.evolution.promotion.evaluate_promotion_batch", side_effect=RuntimeError("boom")):
        result = JOBS.run_tournament_cycle(ctx=None)
    assert result.status == JS.SUCCEEDED
    assert "promotion_error" in result.summary


def test_supervisor_enqueues_tournament_cycle_after_learning_succeeds():
    """research/autonomy/supervisor.py's _advance_forward_settlement wiring:
    the exact call this test protects against regressing."""
    import inspect

    import research.autonomy.supervisor as SUP

    source = inspect.getsource(SUP)
    assert "SCH.TOURNAMENT_CYCLE" in source
    assert "SCH.tournament_cycle_key(session_date)" in source
