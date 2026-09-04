"""Stuck RUNNING operations leave RUNNING after their declared deadline."""
from __future__ import annotations

import os
from pathlib import Path

from operations.store import FAILED, RUNNING, OperationStore


def test_stale_due_diligence_leaves_running(tmp_path: Path):
    store = OperationStore(tmp_path / "jobs.db")
    item, _ = store.enqueue("DUE_DILIGENCE_ACQUIRE", lane="due_diligence", requested_by="test")
    leased = store.lease_next("due_diligence", worker_pid=9_999_999)
    assert leased is not None
    assert leased["status"] == RUNNING
    started = float(leased["started_at"])
    recovered = store.recover_stale_running(now=started + 16 * 60)
    assert recovered == 1
    row = store.get(item["operation_id"])
    assert row["status"] == FAILED
    assert row["error_code"] == "DEADLINE_EXCEEDED"
    assert row["status"] != RUNNING


def test_fresh_running_is_not_failed(tmp_path: Path):
    store = OperationStore(tmp_path / "jobs.db")
    store.enqueue("DUE_DILIGENCE_ACQUIRE", lane="due_diligence", requested_by="test")
    leased = store.lease_next("due_diligence", worker_pid=1)
    started = float(leased["started_at"])
    assert store.recover_stale_running(now=started + 30) == 0
    assert store.get(leased["operation_id"])["status"] == RUNNING


def test_live_pid_is_not_marked_failed_by_store_sweep(tmp_path: Path):
    store = OperationStore(tmp_path / "jobs.db")
    store.enqueue("DUE_DILIGENCE_ACQUIRE", lane="due_diligence", requested_by="test")
    leased = store.lease_next("due_diligence", worker_pid=os.getpid())
    started = float(leased["attempt_started_at"] or leased["started_at"])
    assert store.recover_stale_running(now=started + 16 * 60, keep_pid=os.getpid()) == 0
    assert store.get(leased["operation_id"])["status"] == RUNNING
    overdue = store.overdue_running(now=started + 16 * 60)
    assert [row["operation_id"] for row in overdue] == [leased["operation_id"]]
