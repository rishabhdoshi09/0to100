from __future__ import annotations

import multiprocessing as mp
import os

from research.autonomy.safe_singleton_lock import StableSingleInstanceLock


def _hold_supervisor_lock(path: str, ready, release) -> None:
    lock = StableSingleInstanceLock(path)
    acquired = lock.acquire()
    ready.put((acquired, os.getpid()))
    if not acquired:
        return
    try:
        release.wait(10)
    finally:
        lock.release()


def test_losing_contender_does_not_truncate_owner_pid(tmp_path):
    path = tmp_path / "supervisor.lock"
    first = StableSingleInstanceLock(path)
    second = StableSingleInstanceLock(path)

    assert first.acquire() is True
    assert path.read_text(encoding="utf-8") == str(os.getpid())
    assert second.acquire() is False
    assert path.read_text(encoding="utf-8") == str(os.getpid())

    first.release()
    assert path.exists(), "flock path must remain stable across ownership handoff"
    assert second.acquire() is True
    second.release()


def test_independent_process_contender_cannot_truncate_or_split_lock_path(tmp_path):
    path = tmp_path / "supervisor.lock"
    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Event()
    owner = ctx.Process(target=_hold_supervisor_lock, args=(str(path), ready, release))
    owner.start()
    acquired, owner_pid = ready.get(timeout=10)
    assert acquired is True
    assert path.read_text(encoding="utf-8") == str(owner_pid)

    contender = StableSingleInstanceLock(path)
    try:
        assert contender.acquire() is False
        assert path.read_text(encoding="utf-8") == str(owner_pid)
        assert path.exists()
    finally:
        release.set()
        owner.join(timeout=10)
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=5)
    assert owner.exitcode == 0
    assert path.exists(), "stable flock path must survive ownership release"

    next_owner = StableSingleInstanceLock(path)
    assert next_owner.acquire() is True
    next_owner.release()


def test_release_keeps_lock_inode_path_for_future_contenders(tmp_path):
    path = tmp_path / "supervisor.lock"
    lock = StableSingleInstanceLock(path)
    assert lock.acquire() is True
    lock.release()
    assert path.exists()

    next_owner = StableSingleInstanceLock(path)
    assert next_owner.acquire() is True
    next_owner.release()
