"""Stale-lease recovery must depend on worker identity, not on a bare PID.

The audited defect: ``recover_stale_running`` skipped any row whose recorded
worker PID was alive. ``pid_is_alive`` is ``os.kill(pid, 0)``, which

  * succeeds for any reused PID, and
  * succeeds as root for every process on the host, including PID 1,

so an operation could stay RUNNING forever. These tests pin the corrected
contract across the environments in the mandate: root, unprivileged, a dead
PID, a reused PID, an unrelated live PID, and a hung worker.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from operations import store as store_module
from operations.store import (
    ABSOLUTE_MAX_RUNNING_S,
    FAILED,
    PENDING,
    RUNNING,
    OperationStore,
    worker_pid_is_live,
)


def _leased(tmp_path: Path, *, worker_pid: int, kind: str = "DUE_DILIGENCE_ACQUIRE"):
    store = OperationStore(tmp_path / "jobs.db")
    item, _ = store.enqueue(kind, lane="due_diligence", requested_by="test")
    leased = store.lease_next("due_diligence", worker_pid=worker_pid)
    assert leased is not None and leased["status"] == RUNNING
    return store, item, leased


# --------------------------------------------------------------------------
# worker_pid_is_live
# --------------------------------------------------------------------------

def test_pid_one_is_never_our_worker_even_when_signalable():
    """PID 1 exists in every container and is signalable as root.

    This is the exact condition that made the canonical suite pass on CI's
    unprivileged runner and fail as root.
    """
    assert worker_pid_is_live(1) is False


def test_dead_pid_is_not_live():
    assert worker_pid_is_live(999_999_000) is False
    assert worker_pid_is_live(0) is False
    assert worker_pid_is_live(None) is False


def test_the_current_process_always_recognises_itself():
    """Whoever leased the row is asking in-process whether it is still running.

    A process is the authority on its own liveness, whatever its command line
    looks like — that also covers embedded and differently-named entrypoints.
    """
    assert worker_pid_is_live(os.getpid()) is True


def test_unrelated_live_process_is_not_our_worker():
    """A live process running something else must not hold a lease open."""
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert proc.pid != os.getpid()
        assert worker_pid_is_live(proc.pid) is False
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_reused_pid_running_another_program_is_not_our_worker(monkeypatch):
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(store_module, "process_command", lambda _pid: "/usr/bin/postgres -D /var/lib/pg")
    assert worker_pid_is_live(4242) is False


def test_real_worker_command_is_recognised(monkeypatch):
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        store_module, "process_command", lambda _pid: "python -u -m operations.market_ops"
    )
    assert worker_pid_is_live(4242) is True


def test_unreadable_command_defers_recovery(monkeypatch):
    """When ps gives us nothing, prefer deferring over cancelling live work."""
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(store_module, "process_command", lambda _pid: "")
    assert worker_pid_is_live(4242) is True


# --------------------------------------------------------------------------
# recover_stale_running
# --------------------------------------------------------------------------

def test_stale_lease_under_pid_one_is_recovered(tmp_path: Path):
    """The regression for the failing canonical gate."""
    store, item, leased = _leased(tmp_path, worker_pid=1)
    started = float(leased["started_at"])

    assert store.recover_stale_running(now=started + 16 * 60) == 1

    row = store.get(item["operation_id"])
    assert row["status"] == FAILED
    assert row["error_code"] == "DEADLINE_EXCEEDED"


def test_fresh_running_is_never_failed(tmp_path: Path):
    store, _item, leased = _leased(tmp_path, worker_pid=1)
    started = float(leased["started_at"])

    assert store.recover_stale_running(now=started + 30) == 0
    assert store.get(leased["operation_id"])["status"] == RUNNING


def test_live_worker_defers_recovery(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        store_module, "process_command", lambda _pid: "python -u -m operations.market_ops"
    )
    store, _item, leased = _leased(tmp_path, worker_pid=4242)
    started = float(leased["started_at"])

    assert store.recover_stale_running(now=started + 16 * 60) == 0
    assert store.get(leased["operation_id"])["status"] == RUNNING


def test_hung_worker_is_recovered_at_the_absolute_ceiling(tmp_path: Path, monkeypatch):
    """A live-but-wedged worker cannot hold an operation RUNNING forever."""
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        store_module, "process_command", lambda _pid: "python -u -m operations.market_ops"
    )
    store, item, leased = _leased(tmp_path, worker_pid=4242)
    started = float(leased["started_at"])

    # Still under the ceiling: the live worker keeps its lease.
    assert store.recover_stale_running(now=started + ABSOLUTE_MAX_RUNNING_S - 60) == 0
    assert store.get(item["operation_id"])["status"] == RUNNING

    # Past the ceiling: recovered regardless of liveness.
    assert store.recover_stale_running(now=started + ABSOLUTE_MAX_RUNNING_S + 1) == 1
    row = store.get(item["operation_id"])
    assert row["status"] == FAILED
    assert row["error_code"] == "DEADLINE_EXCEEDED"
    assert "ceiling" in str(row["error_message"]).lower()


def test_no_operation_can_remain_running_forever(tmp_path: Path, monkeypatch):
    """The invariant, stated directly."""
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        store_module, "process_command", lambda _pid: "python -u -m operations.market_ops"
    )
    store, item, leased = _leased(tmp_path, worker_pid=4242, kind="MARKET_SCAN")
    started = float(leased["started_at"])

    store.recover_stale_running(now=started + 10 * ABSOLUTE_MAX_RUNNING_S)
    assert store.get(item["operation_id"])["status"] != RUNNING


# --------------------------------------------------------------------------
# recover_dead_running shares the same identity rule
# --------------------------------------------------------------------------

def test_dead_running_recovery_also_verifies_identity(tmp_path: Path):
    store, item, _leased_row = _leased(tmp_path, worker_pid=1)

    assert store.recover_dead_running() == 1
    assert store.get(item["operation_id"])["status"] == PENDING


def test_dead_running_recovery_keeps_a_real_worker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        store_module, "process_command", lambda _pid: "python -u -m operations.market_ops"
    )
    store, item, _row = _leased(tmp_path, worker_pid=4242)

    assert store.recover_dead_running() == 0
    assert store.get(item["operation_id"])["status"] == RUNNING


@pytest.mark.parametrize("euid", [0, 1000])
def test_recovery_outcome_does_not_depend_on_privilege(tmp_path: Path, monkeypatch, euid):
    """Same verdict as root and as an unprivileged user.

    ``os.kill(1, 0)`` raises PermissionError unprivileged and succeeds as root.
    Identity verification makes both paths agree.
    """
    monkeypatch.setattr(store_module, "pid_is_alive", lambda _pid: euid == 0)
    monkeypatch.setattr(store_module, "process_command", lambda _pid: "/sbin/init")

    store, item, leased = _leased(tmp_path, worker_pid=1)
    started = float(leased["started_at"])

    assert store.recover_stale_running(now=started + 16 * 60) == 1
    assert store.get(item["operation_id"])["status"] == FAILED
