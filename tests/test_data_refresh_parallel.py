from __future__ import annotations

import json
import threading
import time

from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy import data_refresh_parallel as DR
from research.autonomy.data_refresh_parallel import IN_PROGRESS, make_parallel_data_refresh_handler


class _Ctx:
    required_session_date = "2026-08-28"


def test_data_refresh_returns_control_while_canonical_refresh_runs():
    entered = threading.Event()
    release = threading.Event()

    def original(_ctx):
        entered.set()
        assert release.wait(timeout=3.0)
        return JOBS.JobResult(
            JS.SUCCEEDED,
            "genuine snapshot active",
            metadata={"latest_date": "2026-08-28"},
        )

    handler = make_parallel_data_refresh_handler(original)
    started = time.monotonic()
    first = handler(_Ctx())
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert first.status == JS.RETRYABLE_FAILED
    assert first.error_code == IN_PROGRESS
    assert entered.wait(timeout=1.0)

    second = handler(_Ctx())
    assert second.error_code == IN_PROGRESS

    release.set()
    deadline = time.time() + 3.0
    while time.time() < deadline and handler.runtime_state["running"]:
        time.sleep(0.01)

    final = handler(_Ctx())
    assert final.status == JS.SUCCEEDED
    assert final.metadata["latest_date"] == "2026-08-28"


def test_completed_refresh_is_not_reused_for_newer_required_session():
    calls = []
    release = threading.Event()
    release.set()

    def original(ctx):
        calls.append(str(getattr(ctx, "required_session_date", "")))
        return JOBS.JobResult(
            JS.SUCCEEDED,
            "snapshot active",
            metadata={"latest_date": str(getattr(ctx, "required_session_date", ""))},
        )

    handler = make_parallel_data_refresh_handler(original)
    first_ctx = _Ctx()
    assert handler(first_ctx).error_code == IN_PROGRESS
    deadline = time.time() + 2.0
    while time.time() < deadline and handler.runtime_state["running"]:
        time.sleep(0.01)
    assert handler(first_ctx).status == JS.SUCCEEDED

    class _NextCtx:
        required_session_date = "2026-08-31"

    next_result = handler(_NextCtx())
    assert next_result.error_code == IN_PROGRESS
    deadline = time.time() + 2.0
    while time.time() < deadline and handler.runtime_state["running"]:
        time.sleep(0.01)
    assert handler(_NextCtx()).status == JS.SUCCEEDED


def test_dead_worker_is_reported_instead_of_polling_forever():
    class _Dead:
        def is_alive(self):
            return False

    handler = make_parallel_data_refresh_handler(lambda _ctx: None, clock=lambda: 100.0)
    handler.runtime_state.update({
        "running": True,
        "started_at": 1.0,
        "finished_at": 0.0,
        "required": "2026-08-28",
        "result": None,
        "thread": _Dead(),
    })
    result = handler(_Ctx())
    assert result.error_code == "DATA_REFRESH_WORKER_STUCK"
    assert result.status == JS.RETRYABLE_FAILED


def test_poll_reschedule_does_not_inflate_durable_attempt_count(tmp_path):
    store = JS.JobStore(tmp_path / "jobs.db")
    job = store.enqueue("polling-test", idempotency_key="polling-test")

    first = store.lease_due("owner")
    assert first is not None
    assert first.attempt == 1

    store.reschedule_poll(
        first.job_id,
        when=store.clock(),
        error_code=IN_PROGRESS,
        error_message="still running",
        result_summary="data refresh running in background · historical_sync · 45s",
    )
    after_first_poll = store.get(job.job_id)
    assert after_first_poll is not None
    assert after_first_poll.status == JS.PENDING
    assert after_first_poll.attempt == 0
    assert after_first_poll.result_summary == "data refresh running in background · historical_sync · 45s"

    second = store.lease_due("owner")
    assert second is not None
    assert second.attempt == 1
    store.reschedule_poll(
        second.job_id,
        when=store.clock(),
        error_code=IN_PROGRESS,
        error_message="still running",
    )
    assert store.get(job.job_id).attempt == 0



def test_long_refresh_is_not_called_stalled_when_progress_is_recent(tmp_path, monkeypatch):
    class _Alive:
        def is_alive(self):
            return True

    now = {"value": 1000.0}
    telemetry = tmp_path / "runtime_progress.json"
    monkeypatch.setattr(DR, "PROGRESS_PATH", telemetry)
    telemetry.write_text(json.dumps({
        "schema_version": 1,
        "stage": "historical_sync",
        "progress_current": 500,
        "progress_total": 1000,
        "percent_complete": 50.0,
        "symbols_per_sec": 2.5,
        "started_epoch": 100.0,
        "last_progress_epoch": 995.0,
    }))

    handler = make_parallel_data_refresh_handler(
        lambda _ctx: None,
        clock=lambda: now["value"],
    )
    handler.runtime_state.update({
        "running": True,
        "started_at": 100.0,
        "finished_at": 0.0,
        "required": "2026-08-28",
        "result": None,
        "thread": _Alive(),
    })

    result = handler(_Ctx())
    assert result.error_code == IN_PROGRESS
    assert result.metadata["stall_warning"] is False
    assert result.metadata["progress_current"] == 500
    assert result.metadata["progress_total"] == 1000
    assert "500/1000" in result.summary
    assert "stall warning" not in result.summary


def test_stall_warning_requires_no_measurable_progress(tmp_path, monkeypatch):
    class _Alive:
        def is_alive(self):
            return True

    telemetry = tmp_path / "runtime_progress.json"
    monkeypatch.setattr(DR, "PROGRESS_PATH", telemetry)
    telemetry.write_text(json.dumps({
        "schema_version": 1,
        "stage": "historical_sync",
        "progress_current": 500,
        "progress_total": 1000,
        "percent_complete": 50.0,
        "symbols_per_sec": 2.5,
        "started_epoch": 100.0,
        "last_progress_epoch": 300.0,
    }))

    handler = make_parallel_data_refresh_handler(lambda _ctx: None, clock=lambda: 1000.0)
    handler.runtime_state.update({
        "running": True,
        "started_at": 100.0,
        "finished_at": 0.0,
        "required": "2026-08-28",
        "result": None,
        "thread": _Alive(),
    })

    result = handler(_Ctx())
    assert result.metadata["stall_warning"] is True
    assert result.metadata["progress_age_s"] == 700.0
    assert "stall warning: no measurable progress for 700s" in result.summary


def test_previous_worker_telemetry_cannot_trigger_false_stall(tmp_path, monkeypatch):
    class _Alive:
        def is_alive(self):
            return True

    telemetry = tmp_path / "runtime_progress.json"
    monkeypatch.setattr(DR, "PROGRESS_PATH", telemetry)
    telemetry.write_text(json.dumps({
        "stage": "historical_sync",
        "progress_current": 999,
        "progress_total": 1000,
        "started_epoch": 50.0,
        "last_progress_epoch": 60.0,
    }))

    handler = make_parallel_data_refresh_handler(lambda _ctx: None, clock=lambda: 1000.0)
    handler.runtime_state.update({
        "running": True,
        "started_at": 900.0,
        "finished_at": 0.0,
        "required": "2026-08-28",
        "result": None,
        "thread": _Alive(),
    })

    result = handler(_Ctx())
    assert result.metadata["stall_warning"] is False
    assert result.metadata["progress"] == {}
    assert result.metadata["progress_total"] == 0
    assert "progress telemetry starting" in result.summary
