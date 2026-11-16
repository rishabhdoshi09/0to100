"""Historical replay never deadlocks on a batch an earlier run already finished.

Owner's Mac: replay IDLE from 2026-11-06. peek_next_batch kept proposing
hist_8ca54fec48762d28 (2020-02-26..03-06), which had fully completed on
2026-09-21; its deterministic job key handed back the old SKIPPED_IDEMPOTENT
job, nothing ran, nothing was recorded, and the same batch came back forever.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from product import historical_paper_loop as HL
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor

SESSIONS = [f"2020-01-{d:02d}" for d in range(1, 31)]


def _state(tmp_path, processed=()):
    path = tmp_path / "loop.json"
    HL._save_state({"phase": HL.PHASE_IDLE, "processed_sessions": list(processed),
                    "thesis_hash": "t"}, path)
    return path


def _peek(path):
    def peek():
        done = set(HL.load_state(path)["processed_sessions"])
        remaining = [d for d in SESSIONS if d not in done]
        if not remaining:
            return {"available": False, "reason": "historical_backlog_caught_up"}
        batch = remaining[:3]
        return {"available": True, "batch_id": HL._batch_id(batch, 40, "t"), "sessions": batch}
    return peek


def _stub(tmp_path):
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: 1000.0)
    incidents = []
    stub = SimpleNamespace(
        jobs=store,
        deps=SimpleNamespace(now_ist=lambda: datetime(2026, 11, 16, 1, 0)),
        REPLAY_RECONCILE_MAX=Supervisor.REPLAY_RECONCILE_MAX,
        _FINISHED_OK=Supervisor._FINISHED_OK,
        _FINISHED_BAD=Supervisor._FINISHED_BAD,
        _incident=lambda code, msg, job=None: incidents.append((code, msg)),
    )
    for name in ("_replay_batch_outcome", "_retry_if_failed", "_finished_earlier"):
        setattr(stub, name, getattr(Supervisor, name).__get__(stub))
    return stub, store, incidents


def _finish(store, job_type, key, status, batch):
    job = store.enqueue(job_type, idempotency_key=key, input_snapshot_id=batch)
    store.complete(job.job_id, status, result_summary="earlier run")


def test_batches_finished_in_an_earlier_run_are_recorded_and_the_cursor_moves(tmp_path, monkeypatch):
    path = _state(tmp_path)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    stub, store, incidents = _stub(tmp_path)
    peek = _peek(path)
    for _ in range(2):                                     # two batches done back in September
        nxt = peek()
        bid = nxt["batch_id"]
        _finish(store, SCH.HISTORICAL_PAPER_CYCLE, SCH.historical_paper_key(bid), JS.SKIPPED_IDEMPOTENT, bid)
        _finish(store, SCH.LEARNING_CYCLE, SCH.historical_learning_key(bid), JS.SUCCEEDED, bid)
        HL._save_state({"processed_sessions": HL.load_state(path)["processed_sessions"] + nxt["sessions"]}, path)
    HL._save_state({"processed_sessions": []}, path)        # ... then the record lost them
    handled, nxt = Supervisor._schedule_replay_batch(stub, peek)
    assert handled is True
    assert HL.load_state(path)["processed_sessions"] == SESSIONS[:6]
    pending = [j for j in store.list() if j.status == JS.PENDING]
    assert [j.idempotency_key for j in pending] == [SCH.historical_paper_key(nxt["batch_id"])]
    assert nxt["sessions"] == SESSIONS[6:9]
    assert incidents and "2 replay batch" in incidents[0][1]


def test_a_batch_that_never_completed_is_retried_once_a_day(tmp_path, monkeypatch):
    path = _state(tmp_path)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    stub, store, _ = _stub(tmp_path)
    peek = _peek(path)
    bid = peek()["batch_id"]
    _finish(store, SCH.HISTORICAL_PAPER_CYCLE, SCH.historical_paper_key(bid), JS.PERMANENT_FAILED, bid)
    for _ in range(3):
        assert Supervisor._schedule_replay_batch(stub, peek)[0] is True
    retries = [j for j in store.list() if ":retry:" in str(j.idempotency_key)]
    assert len(retries) == 1 and retries[0].input_snapshot_id == bid
    assert retries[0].idempotency_key.startswith("hist_paper:")
    assert HL.load_state(path)["processed_sessions"] == []    # never marked done without running


def test_finished_paper_without_learning_is_rerun_not_skipped(tmp_path, monkeypatch):
    path = _state(tmp_path)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    stub, store, _ = _stub(tmp_path)
    peek = _peek(path)
    bid = peek()["batch_id"]
    _finish(store, SCH.HISTORICAL_PAPER_CYCLE, SCH.historical_paper_key(bid), JS.SKIPPED_IDEMPOTENT, bid)
    Supervisor._schedule_replay_batch(stub, peek)
    assert HL.load_state(path)["processed_sessions"] == []
    assert any(":retry:" in str(j.idempotency_key) for j in store.list())


def test_new_batches_are_queued_normally(tmp_path, monkeypatch):
    path = _state(tmp_path)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    stub, store, incidents = _stub(tmp_path)
    handled, nxt = Supervisor._schedule_replay_batch(stub, _peek(path))
    assert handled is True and incidents == []
    assert [j.status for j in store.list()] == [JS.PENDING]


def test_caught_up_falls_through_to_the_research_replan(tmp_path, monkeypatch):
    path = _state(tmp_path, processed=SESSIONS)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    stub, store, _ = _stub(tmp_path)
    handled, nxt = Supervisor._schedule_replay_batch(stub, _peek(path))
    assert handled is False and nxt["reason"] == "historical_backlog_caught_up" and store.list() == []


def test_reconcile_only_touches_an_idle_loop(tmp_path):
    path = _state(tmp_path)
    HL._save_state({"phase": HL.PHASE_RUNNING, "current_batch_id": "b"}, path)
    assert HL.reconcile_completed_batch("x", ["2020-01-01"], state_path=path)["processed_sessions"] == []
    HL._save_state({"phase": HL.PHASE_IDLE}, path)
    out = HL.reconcile_completed_batch("x", ["2020-01-02", "2020-01-01"], state_path=path)
    assert out["processed_sessions"] == ["2020-01-01", "2020-01-02"]
    assert out["last_completed_session"] == "2020-01-02"
    assert out["selection_details"]["reconciled_batch_id"] == "x"


def test_learning_already_done_in_an_earlier_run_advances_the_phase(tmp_path, monkeypatch):
    path = _state(tmp_path)
    monkeypatch.setattr(HL, "DEFAULT_STATE", path)
    HL._save_state({"phase": HL.PHASE_AWAITING_LEARNING, "current_batch_id": "b1",
                    "current_sessions": ["2020-01-01"]}, path)
    stub, store, _ = _stub(tmp_path)
    _finish(store, SCH.LEARNING_CYCLE, SCH.historical_learning_key("b1"), JS.SUCCEEDED, "b1")
    job = store.enqueue(SCH.LEARNING_CYCLE, idempotency_key=SCH.historical_learning_key("b1"), input_snapshot_id="b1")
    assert stub._finished_earlier(job)
    HL.mark_learning_complete("b1")
    assert HL.load_state(path)["phase"] == HL.PHASE_AWAITING_RESEARCH
