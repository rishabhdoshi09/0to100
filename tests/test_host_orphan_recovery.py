from __future__ import annotations

import json
from pathlib import Path

import pytest

from product import host_orphan_recovery as recovery


START = "2026-09-16T02:00:00+00:00"
HEARTBEAT = "2026-09-16T02:10:00+00:00"


def _fixtures(tmp_path: Path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    state = runtime / "state"
    state.mkdir(parents=True)
    repo.mkdir()
    owner_path = tmp_path / "quantterm.supervisor.owner.json"
    status = {
        "pid": 900,
        "started_at": START,
        "heartbeat_at": HEARTBEAT,
        "production_sha": "abc",
        "runtime_root": str(runtime),
        "children": {
            "market_ops": {"pid": 111, "alive": True},
        },
    }
    owner = {
        "pid": 900,
        "started_at": START,
        "root": str(repo),
        "runtime_root": str(runtime),
        "sha": "abc",
    }
    (state / "host_supervisor.json").write_text(json.dumps(status), encoding="utf-8")
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    return repo, runtime, owner_path, status, owner


def _patch_paths(monkeypatch, repo: Path, runtime: Path, owner_path: Path):
    monkeypatch.setattr(recovery, "REPO_ROOT", repo)
    monkeypatch.setattr(recovery, "runtime_root", lambda: runtime)
    monkeypatch.setattr(recovery, "runtime_path", lambda *parts: runtime.joinpath(*parts))
    monkeypatch.setattr(recovery, "machine_owner_path", lambda: owner_path)


def _good_row(repo: Path, *, started_epoch: float) -> dict:
    return {
        "pid": 111,
        "ppid": 1,
        "pgid": 111,
        "started": "Wed Sep 16 07:31:00 2026",
        "started_epoch": started_epoch,
        "command": f"/usr/bin/python3 -u -m operations.market_ops --repo {repo}",
    }


def test_verified_orphan_generation_is_reaped_and_persisted(tmp_path, monkeypatch):
    repo, runtime, owner_path, status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    start_epoch = recovery._iso_epoch(START)
    assert start_epoch is not None

    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 111)
    monkeypatch.setattr(
        recovery, "_process_row", lambda pid: _good_row(repo, started_epoch=start_epoch + 60),
    )
    seen = []
    monkeypatch.setattr(
        recovery, "_terminate_groups",
        lambda candidates: seen.extend(candidates) or [{**candidates[0], "signal": "SIGTERM"}],
    )

    result = recovery.reconcile_previous_children()
    assert result["state"] == "RECOVERED"
    assert seen and seen[0]["pid"] == 111
    persisted = json.loads((runtime / "state" / "host_orphan_recovery.json").read_text())
    assert persisted["previous_supervisor_pid"] == 900
    assert not owner_path.exists()


def test_owner_mismatch_fails_before_any_signal(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, owner = _fixtures(tmp_path)
    owner["runtime_root"] = str(tmp_path / "other")
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: False)
    called = {"terminate": False}
    monkeypatch.setattr(
        recovery, "_terminate_groups",
        lambda candidates: called.__setitem__("terminate", True) or [],
    )

    with pytest.raises(recovery.OrphanRecoveryError, match="runtime"):
        recovery.reconcile_previous_children()
    assert called["terminate"] is False


def test_command_mismatch_fails_before_any_signal(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    start_epoch = recovery._iso_epoch(START)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 111)
    bad = _good_row(repo, started_epoch=float(start_epoch) + 10)
    bad["command"] = "/usr/bin/python3 unrelated.py"
    monkeypatch.setattr(recovery, "_process_row", lambda pid: bad)
    called = {"terminate": False}
    monkeypatch.setattr(
        recovery, "_terminate_groups",
        lambda candidates: called.__setitem__("terminate", True) or [],
    )

    with pytest.raises(recovery.OrphanRecoveryError, match="command"):
        recovery.reconcile_previous_children()
    assert called["terminate"] is False


def test_pid_reuse_time_guard_fails_before_signal(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    heartbeat_epoch = recovery._iso_epoch(HEARTBEAT)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 111)
    monkeypatch.setattr(
        recovery,
        "_process_row",
        lambda pid: _good_row(
            repo,
            started_epoch=float(heartbeat_epoch) + recovery.START_SLOP_AFTER_S + 1,
        ),
    )
    called = {"terminate": False}
    monkeypatch.setattr(
        recovery, "_terminate_groups",
        lambda candidates: called.__setitem__("terminate", True) or [],
    )

    with pytest.raises(recovery.OrphanRecoveryError, match="started after"):
        recovery.reconcile_previous_children()
    assert called["terminate"] is False


def test_no_live_recorded_children_is_safe_noop(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        recovery, "_terminate_groups", lambda candidates: pytest.fail("must not terminate"),
    )

    result = recovery.reconcile_previous_children()
    assert result["state"] == "CLEAR"
    assert result["terminated"] == []


def test_missing_owner_blocks_credible_recorded_orphan(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, _owner = _fixtures(tmp_path)
    owner_path.unlink()
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    start_epoch = recovery._iso_epoch(START)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 111)
    monkeypatch.setattr(
        recovery, "_process_row", lambda pid: _good_row(repo, started_epoch=float(start_epoch) + 30),
    )
    with pytest.raises(recovery.OrphanRecoveryError, match="owner evidence is missing"):
        recovery.reconcile_previous_children()


def test_quiesce_verified_live_supervisor_then_reconciles_children(tmp_path, monkeypatch):
    repo, runtime, owner_path, status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    status["children"] = {}
    (runtime / "state" / "host_supervisor.json").write_text(json.dumps(status), encoding="utf-8")

    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 900)
    monkeypatch.setattr(
        recovery,
        "_process_row",
        lambda pid: {
            "pid": 900,
            "ppid": 1,
            "pgid": 900,
            "started": "Wed Sep 16 07:30:00 2026",
            "started_epoch": float(recovery._iso_epoch(START) or 0) - 30,
            "command": "/usr/bin/python3 -u -m product.host_launchd_entrypoint",
        },
    )
    signals = []
    monkeypatch.setattr(recovery.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(recovery, "_wait_pid_gone", lambda pid, timeout_s: True)
    monkeypatch.setattr(
        recovery,
        "reconcile_previous_children",
        lambda: {"state": "CLEAR", "terminated": []},
    )

    result = recovery.quiesce_live_previous_supervisor()

    assert result["state"] == "QUIESCED"
    assert result["previous_supervisor_pid"] == 900
    assert result["signal"] == "SIGTERM"
    assert signals == [(900, recovery.signal.SIGTERM)]


def test_quiesce_refuses_noncanonical_live_supervisor_before_signal(tmp_path, monkeypatch):
    repo, runtime, owner_path, _status, _owner = _fixtures(tmp_path)
    _patch_paths(monkeypatch, repo, runtime, owner_path)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 900)
    monkeypatch.setattr(
        recovery,
        "_process_row",
        lambda pid: {
            "pid": 900,
            "ppid": 1,
            "pgid": 900,
            "started": "Wed Sep 16 07:30:00 2026",
            "started_epoch": float(recovery._iso_epoch(START) or 0),
            "command": "/usr/bin/python3 unrelated_service.py",
        },
    )
    signals = []
    monkeypatch.setattr(recovery.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(recovery.OrphanRecoveryError, match="not canonical QuantTerm host"):
        recovery.quiesce_live_previous_supervisor()

    assert signals == []
