from __future__ import annotations

import json

from product import host_supervisor as HS
from research import autonomy
from research.autonomy import health as AH


def test_autonomy_health_uses_runtime_heartbeat_only(tmp_path, monkeypatch):
    runtime = tmp_path / "autonomy"
    runtime.mkdir()
    (runtime / "runtime.json").write_text(
        json.dumps({
            "heartbeat_ist": "2026-09-25T18:00:00+05:30",
            "process_running": True,
            "scheduler_owner_pid": 4242,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomy, "default_root", lambda: runtime)
    monkeypatch.setattr(AH, "_fresh", lambda payload: True)
    seen = []
    monkeypatch.setattr(HS.os, "kill", lambda pid, sig: seen.append((pid, sig)))

    assert HS._autonomy_health() is True
    assert seen == [(4242, 0)]


def test_autonomy_health_fails_closed_for_stale_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "autonomy"
    runtime.mkdir()
    (runtime / "runtime.json").write_text(
        json.dumps({
            "heartbeat_ist": "2026-09-25T17:00:00+05:30",
            "process_running": True,
            "scheduler_owner_pid": 4242,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomy, "default_root", lambda: runtime)
    monkeypatch.setattr(AH, "_fresh", lambda payload: False)
    monkeypatch.setattr(
        HS.os,
        "kill",
        lambda pid, sig: (_ for _ in ()).throw(AssertionError("stale heartbeat must not probe PID")),
    )

    assert HS._autonomy_health() is False
