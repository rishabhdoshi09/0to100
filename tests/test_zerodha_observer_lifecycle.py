"""QT-PRA-081: the broker observer must not outlive the API that started it.

On the owner's Mac (2026-11-10) an `operations.zerodha_observer` started on
2026-11-09 10:49 by an API that later died without stopping it was re-parented
to PID 1 and kept running yesterday's code across three deployments: it held
the observer lock (the deployed observer never started) and wrote heartbeats
into the runtime while the host was stopped for a backup.
"""
from __future__ import annotations

import json

import operations.zerodha_observer as Z


def _worker(monkeypatch, state):
    worker = Z.ZerodhaObserverWorker()
    monkeypatch.setattr(worker.lock, "acquire", lambda: True)
    monkeypatch.setattr(worker.lock, "release", lambda: None)
    monkeypatch.setattr(worker, "_run_slot", lambda *a: True)
    monkeypatch.setattr(Z, "observation_slot", lambda *a: None)
    monkeypatch.setattr(Z.os, "getppid", lambda: state["ppid"])

    def fake_wait(timeout):
        state["waits"] += 1
        if state["waits"] == state.get("orphan_at"):
            state["ppid"] = 1
        return state["waits"] >= state.get("stop_at", 10_000)

    monkeypatch.setattr(worker.stop_event, "wait", fake_wait)
    return worker


def test_observer_exits_when_its_parent_api_is_gone(monkeypatch):
    state = {"ppid": 4242, "waits": 0, "orphan_at": 3, "stop_at": 50}
    worker = _worker(monkeypatch, state)

    assert worker.run() == 0
    assert state["waits"] == 3
    assert worker._last_error == "PARENT_EXITED"
    runtime = json.loads(Z.RUNTIME_PATH.read_text())
    assert runtime["running"] is False and runtime["phase"] == "STOPPED"


def test_observer_started_without_a_parent_keeps_running_until_stopped(monkeypatch):
    state = {"ppid": 1, "waits": 0, "stop_at": 5}
    worker = _worker(monkeypatch, state)

    assert worker.run() == 0
    assert state["waits"] == 5
    assert worker._last_error != "PARENT_EXITED"


def test_observer_heartbeat_records_its_parent_and_build(monkeypatch):
    monkeypatch.setenv("QT_BUILD_SHA", "abc123")
    runtime = Z.ZerodhaObserverWorker()._runtime(running=True)
    assert runtime["build_sha"] == "abc123"
    assert isinstance(runtime["parent_pid"], int)
