"""Regression coverage for the loaded 8 GB Mac read-only dashboard path."""
import threading
import time

from product.desk_read_snapshot import DeskReadSnapshot


def _payload():
    return {
        "generated_at": "2026-11-09T10:00:00+00:00",
        "data": {"ready": True, "history_current": True, "blockers": []},
        "scan": {"available": True, "records": [{"symbol": "TCS"}],
                 "provenance": {"data_current": True}},
        "market": {"available": True, "trade_stance": "Assess verified entries"},
        "conviction": [{"symbol": "TCS", "decision": "BUY"}],
        "fno": {"available": True},
    }


def _eventually(test, deadline_s=2.0):
    until = time.monotonic() + deadline_s
    while time.monotonic() < until:
        if test():
            return
        time.sleep(0.01)
    assert test(), "Background projection did not settle"


def test_first_load_is_nonblocking_and_singleflight():
    started, release = threading.Event(), threading.Event()
    calls = []

    def build():
        calls.append(1)
        started.set()
        assert release.wait(2)
        return _payload()

    cache = DeskReadSnapshot()
    fallback = lambda: {"data": {"ready": False}, "conviction": []}
    first = cache.read(build, fallback)
    assert first["response_meta"]["status"] == "WARMING"
    assert first["data"]["ready"] is False
    assert started.wait(1)
    for _ in range(10):
        assert cache.read(build, fallback)["response_meta"]["status"] == "WARMING"
    assert len(calls) == 1
    release.set()
    _eventually(lambda: cache.read(build, fallback)["response_meta"]["status"] == "CURRENT")
    assert len(calls) == 1
    a = cache.read(build, fallback)
    a["scan"]["records"].clear()
    assert cache.read(build, fallback)["scan"]["records"] == [{"symbol": "TCS"}]


def test_overdue_cache_returns_research_but_never_buy_authority():
    clock = [10.0]
    started, release = threading.Event(), threading.Event()
    builds = [0]

    def build():
        builds[0] += 1
        if builds[0] > 1:
            started.set()
            assert release.wait(2)
        return _payload()

    cache = DeskReadSnapshot(fresh_s=5, stale_cap_s=40, clock=lambda: clock[0])
    fallback = lambda: {"data": {"ready": False}, "conviction": []}
    assert cache.read(build, fallback)["response_meta"]["status"] == "WARMING"
    _eventually(lambda: cache.read(build, fallback)["response_meta"]["status"] == "CURRENT")
    clock[0] = 20.0
    stale = cache.read(build, fallback)
    assert started.wait(1)
    assert stale["response_meta"] == {
        "source": "READ_ONLY_DESK_PROJECTION", "status": "REFRESHING",
        "stale": True, "age_s": 10.0,
    }
    assert stale["generated_at"] == "2026-11-09T10:00:00+00:00"
    assert stale["data"]["ready"] is False
    assert stale["data"]["history_current"] is False
    assert stale["scan"]["records"] == [{"symbol": "TCS"}]
    assert stale["scan"]["provenance"]["data_current"] is False
    assert stale["conviction"] == []
    assert stale["market"]["available"] is False
    assert stale["fno"]["available"] is False
    assert len(cache.read(build, fallback)["scan"]["records"]) == 1
    assert builds[0] == 2
    clock[0] = 60.0
    expired = cache.read(build, fallback)
    assert expired["response_meta"]["status"] == "WARMING"
    assert expired["conviction"] == []
    release.set()


def test_failed_refresh_stays_fail_closed_then_retries():
    cache = DeskReadSnapshot()
    tried = threading.Event()
    def failing():
        tried.set()
        raise RuntimeError("test only")
    fallback = lambda: {"data": {"ready": False}, "conviction": []}
    assert cache.read(failing, fallback)["response_meta"]["status"] == "WARMING"
    assert tried.wait(1)
    # Retry is permitted but no failed build may be represented as CURRENT.
    assert cache.read(lambda: _payload(), fallback)["response_meta"]["status"] in {"WARMING", "CURRENT"}
    _eventually(lambda: cache.read(lambda: _payload(), fallback)["response_meta"]["status"] == "CURRENT")
