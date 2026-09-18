from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from product import historical_paper_loop as HPL
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS


def test_obsolete_batch_retires_running_state_without_advancing_cursor(tmp_path):
    state = tmp_path / "historical_state.json"
    HPL._save_state(
        {
            "phase": HPL.PHASE_RUNNING,
            "current_batch_id": "hist-old",
            "current_sessions": ["2026-01-05", "2026-01-06"],
            "last_completed_session": "2026-01-04",
            "thesis_hash": "thesis-old",
        },
        state,
    )

    result = HPL._retire_obsolete_batch(
        batch_id="hist-old",
        expected_thesis_hash="thesis-old",
        current_thesis_hash="thesis-new",
        message="changed",
        state_path=state,
    )

    assert result["status"] == "OBSOLETE_THESIS"
    saved = HPL.load_state(state)
    assert saved["phase"] == HPL.PHASE_IDLE
    assert saved["current_batch_id"] == ""
    assert saved["current_sessions"] == []
    # The new thesis starts its evidence history from the beginning; the old
    # thesis cursor cannot be inherited.
    assert saved["last_completed_session"] == ""
    assert saved["thesis_hash"] == "thesis-new"
    assert saved["last_result"]["status"] == "OBSOLETE_THESIS"


def test_stale_old_daemon_cannot_overwrite_already_reset_new_thesis_state(tmp_path):
    state = tmp_path / "historical_state.json"
    HPL._save_state(
        {
            "phase": HPL.PHASE_IDLE,
            "current_batch_id": "",
            "current_sessions": [],
            "last_completed_session": "",
            "thesis_hash": "thesis-new",
        },
        state,
    )

    HPL._retire_obsolete_batch(
        batch_id="hist-old",
        expected_thesis_hash="thesis-old",
        current_thesis_hash="thesis-new",
        message="changed",
        state_path=state,
    )

    saved = HPL.load_state(state)
    assert saved["phase"] == HPL.PHASE_IDLE
    assert saved["thesis_hash"] == "thesis-new"
    assert saved["current_batch_id"] == ""


def test_obsolete_historical_job_is_terminal_and_never_enters_learning(monkeypatch):
    monkeypatch.setattr(JOBS.SCH, "market_is_open", lambda *_a, **_k: False)
    monkeypatch.setattr(
        HPL,
        "ensure_next_batch_started",
        lambda **_kwargs: {
            "status": "OBSOLETE_THESIS",
            "batch_id": "hist-old",
            "thesis_hash": "thesis-old",
            "current_thesis_hash": "thesis-new",
        },
    )

    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            now_ist=lambda: datetime(2026, 9, 18, 21, 0),
            holidays=lambda: set(),
        ),
        job=SimpleNamespace(input_snapshot_id="hist-old"),
    )

    result = JOBS.run_historical_paper_cycle(ctx)

    assert result.status == JS.SKIPPED_IDEMPOTENT
    assert result.metadata["status"] == "OBSOLETE_THESIS"
    assert "retired" in result.summary.lower()
