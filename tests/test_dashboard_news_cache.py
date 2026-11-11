"""Dashboard news stage reuses its payload while the news DB is unchanged.

Owner's Mac 2026-11-11: [API SLOW] dashboard builds spent 2-7 s in the news
stage on every rebuild although news changes only on NEWS_REFRESH.
"""
from __future__ import annotations

import os
import time


def _setup(monkeypatch, tmp_path):
    import terminal_api

    monkeypatch.setattr(terminal_api, "logs_dir", lambda: tmp_path)
    terminal_api._news_cache.update({"key": None, "at": 0.0, "payload": None})
    calls = []

    def build():
        calls.append(1)
        return {"available": True, "articles": [{"n": len(calls)}]}

    monkeypatch.setattr(terminal_api, "_build_news_payload", build)
    return terminal_api, calls


def test_unchanged_db_is_served_from_cache(monkeypatch, tmp_path):
    api, calls = _setup(monkeypatch, tmp_path)
    (tmp_path / "news_curator.sqlite3").write_bytes(b"x")
    first = api._news_payload()
    second = api._news_payload()
    assert calls == [1] and first == second


def test_db_change_rebuilds(monkeypatch, tmp_path):
    api, calls = _setup(monkeypatch, tmp_path)
    db = tmp_path / "news_curator.sqlite3"
    db.write_bytes(b"x")
    api._news_payload()
    db.write_bytes(b"xy")
    later = time.time() + 5
    os.utime(db, (later, later))
    assert api._news_payload()["articles"] == [{"n": 2}]


def test_ttl_expiry_rebuilds(monkeypatch, tmp_path):
    api, calls = _setup(monkeypatch, tmp_path)
    (tmp_path / "news_curator.sqlite3").write_bytes(b"x")
    api._news_payload()
    api._news_cache["at"] -= api._NEWS_CACHE_TTL_S + 1
    api._news_payload()
    assert calls == [1, 1]


def test_missing_db_or_error_is_never_cached(monkeypatch, tmp_path):
    api, calls = _setup(monkeypatch, tmp_path)
    api._news_payload()
    api._news_payload()
    assert calls == [1, 1]                       # no DB file: no cache key
    (tmp_path / "news_curator.sqlite3").write_bytes(b"x")
    monkeypatch.setattr(api, "_build_news_payload", lambda: {"available": False, "error": "locked"})
    api._news_payload()
    assert api._news_cache["payload"] is None


def test_callers_cannot_mutate_the_cached_payload(monkeypatch, tmp_path):
    api, _ = _setup(monkeypatch, tmp_path)
    (tmp_path / "news_curator.sqlite3").write_bytes(b"x")
    api._news_payload()["available"] = False
    assert api._news_payload()["available"] is True
