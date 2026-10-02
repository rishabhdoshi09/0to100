from __future__ import annotations

import json
import time

import fundamentals.cache as FC


class _Cursor:
    def __init__(self, *, row=None, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, *, row=None, rowcount=0):
        self.row = row
        self.rowcount = rowcount
        self.closed = False
        self.commits = 0
        self.execute_calls = []

    def execute(self, sql, params=()):
        self.execute_calls.append((sql, params))
        return _Cursor(row=self.row, rowcount=self.rowcount)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def test_get_closes_connection(monkeypatch):
    conn = _FakeConnection(row=(json.dumps({"roe": 18.0}), time.time()))
    monkeypatch.setattr(FC, "_connect", lambda: conn)

    row = FC.FundamentalsCache().get("infy")

    assert row["roe"] == 18.0
    assert conn.closed is True


def test_get_miss_closes_connection(monkeypatch):
    conn = _FakeConnection(row=None)
    monkeypatch.setattr(FC, "_connect", lambda: conn)

    assert FC.FundamentalsCache().get("infy") is None
    assert conn.closed is True


def test_set_closes_connection(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(FC, "_connect", lambda: conn)

    FC.FundamentalsCache().set("infy", {"roe": 18.0})

    assert conn.closed is True
    assert conn.commits == 1


def test_clear_old_closes_connection(monkeypatch):
    conn = _FakeConnection(rowcount=3)
    monkeypatch.setattr(FC, "_connect", lambda: conn)

    assert FC.FundamentalsCache().clear_old() == 3
    assert conn.closed is True


def test_invalidate_closes_connection(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(FC, "_connect", lambda: conn)

    FC.FundamentalsCache().invalidate("infy")

    assert conn.closed is True
    assert conn.commits == 1
