"""Tests for the TTL cache."""
import time

from sq_ai.backend import cache as kv


def test_cache_set_and_get(monkeypatch, tmp_path):
    monkeypatch.setenv("SQ_DB_PATH", str(tmp_path / "kv.db"))
    kv.cache_set("foo", {"a": 1}, ttl_seconds=60)
    assert kv.cache_get("foo") == {"a": 1}


def test_cache_expires(monkeypatch, tmp_path):
    monkeypatch.setenv("SQ_DB_PATH", str(tmp_path / "kv.db"))
    kv.cache_set("foo", "bar", ttl_seconds=1)
    time.sleep(1.1)
    assert kv.cache_get("foo") is None


def test_cache_forever(monkeypatch, tmp_path):
    monkeypatch.setenv("SQ_DB_PATH", str(tmp_path / "kv.db"))
    kv.cache_set("perm", "value", ttl_seconds=0)
    assert kv.cache_get("perm") == "value"


def test_cached_decorator_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SQ_DB_PATH", str(tmp_path / "kv.db"))
    counter = {"n": 0}

    @kv.cached("test", ttl_seconds=60)
    def expensive(x):
        counter["n"] += 1
        return {"echo": x, "calls": counter["n"]}

    a = expensive("hi")
    b = expensive("hi")
    assert a == b
    assert counter["n"] == 1
