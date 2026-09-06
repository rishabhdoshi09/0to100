from __future__ import annotations

import os

from research.autonomy.safe_singleton_lock import StableSingleInstanceLock


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


def test_release_keeps_lock_inode_path_for_future_contenders(tmp_path):
    path = tmp_path / "supervisor.lock"
    lock = StableSingleInstanceLock(path)
    assert lock.acquire() is True
    lock.release()
    assert path.exists()

    next_owner = StableSingleInstanceLock(path)
    assert next_owner.acquire() is True
    next_owner.release()
