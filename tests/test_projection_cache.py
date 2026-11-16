"""Compute-on-change projection cache (desk screens took 5-35 s per request on
the owner's Mac, 2026-11-16, because each GET rebuilt everything)."""
from __future__ import annotations

import threading
import time

import pytest

from product.projection_cache import ProjectionCache, file_identity


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _wait_idle(cache, key, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        entry = cache._entries.get(key)
        if entry is not None and not entry.building:
            return
        time.sleep(0.01)
    raise AssertionError("build did not finish")


def test_repeat_reads_are_served_from_cache_and_labelled():
    calls = []
    cache = ProjectionCache("t", fresh_s=60)
    build = lambda: calls.append(1) or {"rows": [1, 2]}
    first = cache.get("k", build)
    second = cache.get("k", build)
    assert calls == [1]
    assert first["projection_meta"]["status"] == "MISS"
    assert second["projection_meta"]["status"] == "HIT" and second["rows"] == [1, 2]


def test_changed_inputs_serve_the_last_build_and_refresh_once_in_background():
    clock = Clock()
    ident = {"v": 1}
    calls = []
    gate = threading.Event()

    def build():
        calls.append(ident["v"])
        if len(calls) > 1:
            gate.wait(2)
        return {"v": ident["v"]}

    cache = ProjectionCache("t", fresh_s=60, clock=clock)
    assert cache.get("k", build, identity=lambda: ident["v"])["v"] == 1
    ident["v"] = 2
    stale = cache.get("k", build, identity=lambda: ident["v"])
    again = cache.get("k", build, identity=lambda: ident["v"])
    assert stale["v"] == 1 and stale["projection_meta"]["stale"] is True
    assert stale["projection_meta"]["inputs_changed"] is True
    assert again["v"] == 1                                     # still one refresh in flight
    gate.set()
    _wait_idle(cache, "k")
    assert calls == [1, 2]
    fresh = cache.get("k", build, identity=lambda: ident["v"])
    assert fresh["v"] == 2 and fresh["projection_meta"]["status"] == "HIT"


def test_age_alone_triggers_a_refresh():
    clock = Clock()
    calls = []
    cache = ProjectionCache("t", fresh_s=60, clock=clock)
    cache.get("k", lambda: calls.append(1) or {"n": len(calls)})
    clock.t += 61
    assert cache.get("k", lambda: calls.append(1) or {"n": len(calls)})["projection_meta"]["stale"] is True
    _wait_idle(cache, "k")
    assert len(calls) == 2


def test_first_build_is_single_flight_across_threads():
    calls = []
    start = threading.Event()

    def build():
        calls.append(1)
        time.sleep(0.2)
        return {"ok": True}

    cache = ProjectionCache("t", fresh_s=60)
    results = []

    def reader():
        start.wait()
        results.append(cache.get("k", build)["ok"])

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(5)
    assert calls == [1] and results == [True] * 6


def test_a_failed_refresh_keeps_the_last_good_build_and_says_so():
    clock = Clock()
    cache = ProjectionCache("t", fresh_s=60, clock=clock)
    cache.get("k", lambda: {"good": True})
    clock.t += 61

    def boom():
        raise RuntimeError("store locked")

    cache.get("k", boom)
    _wait_idle(cache, "k")
    served = cache.get("k", boom)
    assert served["good"] is True and "store locked" in served["projection_meta"]["refresh_error"]


def test_a_failed_first_build_raises_and_is_retried_next_time():
    cache = ProjectionCache("t", fresh_s=60)
    with pytest.raises(RuntimeError):
        cache.get("k", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cache.get("k", lambda: {"ok": 1})["ok"] == 1


def test_served_copies_cannot_corrupt_the_cache():
    cache = ProjectionCache("t", fresh_s=60)
    first = cache.get("k", lambda: {"rows": [1]})
    first["rows"].append(99)
    assert cache.get("k", lambda: {"rows": [1]})["rows"] == [1]


def test_invalidate_forces_a_synchronous_rebuild_and_entries_are_bounded():
    cache = ProjectionCache("t", fresh_s=60, max_entries=2)
    calls = []
    cache.get("a", lambda: calls.append("a") or {"x": 1})
    cache.invalidate("a")
    assert cache.get("a", lambda: calls.append("a2") or {"x": 2})["x"] == 2
    cache.get("b", lambda: {"x": 3})
    cache.get("c", lambda: {"x": 4})
    assert len(cache._entries) == 2


def test_file_identity_tracks_changes(tmp_path):
    path = tmp_path / "f.json"
    path.write_text("1")
    before = file_identity(path, tmp_path / "missing")
    path.write_text("22")
    assert file_identity(path, tmp_path / "missing") != before


def test_watch_precomputes_and_follows_input_changes():
    ident = {"v": 1}
    cache = ProjectionCache("t", fresh_s=600)
    stop = threading.Event()
    cache.watch("k", lambda: {"v": ident["v"]}, identity=lambda: ident["v"], interval_s=0.05, stop=stop)
    try:
        _wait_idle(cache, "k")
        assert cache.get("k", lambda: {"v": -1}, identity=lambda: ident["v"])["v"] == 1
        ident["v"] = 2
        deadline = time.time() + 5
        while time.time() < deadline:
            if cache._entries["k"].value == {"v": 2}:
                break
            time.sleep(0.05)
        assert cache._entries["k"].value == {"v": 2}
    finally:
        stop.set()
