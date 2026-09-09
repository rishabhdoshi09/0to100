from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from data import bhavcopy_runtime


@pytest.fixture(autouse=True)
def _reset_cache_reload_guard(monkeypatch):
    monkeypatch.setattr(bhavcopy_runtime, "_LAST_CACHE_RELOAD_ATTEMPT", None)


class _FakeStore:
    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def __init__(self, *, latest: date, disk_latest: date):
        self._lock = self._Lock()
        self._store = {"TCS": object()}
        self._store_sessions = 100
        self._store_last_day = latest
        self._PKL = _FakePath()
        self._BHAV_DIR = _FakePath()
        self._MIN_DAYS = 60
        self.disk_latest = disk_latest
        self.loads = 0

    def _dates_on_disk(self):
        return [self.disk_latest]

    def _load_pkl(self):
        self.loads += 1
        self._store_last_day = self.disk_latest
        self._store_sessions = 101
        return True


class _StalePersistedStore(_FakeStore):
    def _load_pkl(self):
        self.loads += 1
        return True


class _FakePath:
    def __init__(self):
        self.version = 1

    def exists(self):
        return True

    def stat(self):
        return SimpleNamespace(
            st_mtime_ns=self.version,
            st_size=100,
            st_ino=self.version,
        )

    def __str__(self):
        return "/tmp/fake"


def test_status_reloads_persisted_cache_when_disk_session_is_newer(monkeypatch):
    store = _FakeStore(latest=date(2026, 9, 4), disk_latest=date(2026, 9, 7))
    monkeypatch.setattr(bhavcopy_runtime, "_store_module", lambda: store)

    payload = bhavcopy_runtime.status(load_cache=True)

    assert store.loads == 1
    assert payload["latest_date"] == "2026-09-07"
    assert payload["sessions"] == 101


def test_status_does_not_reload_when_memory_is_already_current(monkeypatch):
    store = _FakeStore(latest=date(2026, 9, 7), disk_latest=date(2026, 9, 7))
    monkeypatch.setattr(bhavcopy_runtime, "_store_module", lambda: store)

    payload = bhavcopy_runtime.status(load_cache=True)

    assert store.loads == 0
    assert payload["latest_date"] == "2026-09-07"


def test_status_does_not_reload_same_stale_pickle_on_every_probe(monkeypatch):
    store = _StalePersistedStore(latest=date(2026, 9, 7), disk_latest=date(2026, 9, 8))
    monkeypatch.setattr(bhavcopy_runtime, "_store_module", lambda: store)

    first = bhavcopy_runtime.status(load_cache=True)
    second = bhavcopy_runtime.status(load_cache=True)

    assert first["latest_date"] == "2026-09-07"
    assert second["latest_date"] == "2026-09-07"
    assert store.loads == 1


def test_status_retries_after_persisted_pickle_changes(monkeypatch):
    store = _StalePersistedStore(latest=date(2026, 9, 7), disk_latest=date(2026, 9, 8))
    monkeypatch.setattr(bhavcopy_runtime, "_store_module", lambda: store)

    bhavcopy_runtime.status(load_cache=True)
    bhavcopy_runtime.status(load_cache=True)
    assert store.loads == 1

    store._PKL.version += 1
    bhavcopy_runtime.status(load_cache=True)

    assert store.loads == 2
