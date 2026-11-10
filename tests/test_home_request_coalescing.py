"""QT-PRA-087: concurrent Home reads share one build; sequential reads never reuse one.

Measured on the owner's Mac: 4 concurrent /api/radar-home requests took ~5 s
each (p50) because each rebuilt the same projection in one API process.
"""
from __future__ import annotations

import threading
import time

import pytest

import product.observer_api as O


def _slow_builder(calls, delay=0.3, fail=False):
    def build():
        calls.append(time.monotonic())
        time.sleep(delay)
        if fail:
            raise RuntimeError("build failed")
        return {"home_os": {"state": "PAUSED", "n": len(calls)}}
    return build


def test_concurrent_reads_share_one_build(monkeypatch):
    calls = []
    monkeypatch.setattr(O, "_build_radar_home_workspace", _slow_builder(calls))
    results = [None] * 4
    start = threading.Barrier(4)

    def worker(i):
        start.wait()
        results[i] = O.radar_home_workspace()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert all(r == {"home_os": {"state": "PAUSED", "n": 1}} for r in results)
    # followers get copies: mutating one response cannot alter another
    results[1]["home_os"]["state"] = "X"
    assert results[2]["home_os"]["state"] == "PAUSED"


def test_sequential_reads_always_build_fresh(monkeypatch):
    calls = []
    monkeypatch.setattr(O, "_build_radar_home_workspace", _slow_builder(calls, delay=0.0))
    assert O.radar_home_workspace()["home_os"]["n"] == 1
    assert O.radar_home_workspace()["home_os"]["n"] == 2


def test_a_failed_build_fails_its_followers_and_does_not_stick(monkeypatch):
    calls = []
    monkeypatch.setattr(O, "_build_radar_home_workspace", _slow_builder(calls, fail=True))
    errors = []
    start = threading.Barrier(3)

    def worker():
        start.wait()
        try:
            O.radar_home_workspace()
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == ["build failed"] * 3 and len(calls) == 1
    monkeypatch.setattr(O, "_build_radar_home_workspace", lambda: {"ok": True})
    assert O.radar_home_workspace() == {"ok": True}
