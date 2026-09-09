"""NSE/Moneycontrol sessions and responses are deterministically closed."""
from __future__ import annotations

from pathlib import Path

from product.due_diligence import acquire as AQ


class FakeResponse:
    def __init__(self, status_code=200, content=b"{}", headers=None, text="{}"):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"Content-Type": "application/json"}
        self.text = text
        self.closed = False

    def close(self):
        self.closed = True

    def json(self):
        import json
        return json.loads(self.text)


class FakeSession:
    opened = 0
    closed = 0
    responses: list[FakeResponse] = []

    def __init__(self):
        type(self).opened += 1
        self.headers = {}

    def get(self, url, timeout=None, allow_redirects=True):
        response = FakeResponse()
        type(self).responses.append(response)
        return response

    def close(self):
        type(self).closed += 1


def _reset():
    FakeSession.opened = 0
    FakeSession.closed = 0
    FakeSession.responses = []


def test_nse_session_closes_warmup_response(monkeypatch):
    _reset()
    import sys
    import types

    fake_requests = types.ModuleType("requests")
    fake_requests.Session = FakeSession
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    session = AQ._nse_session()
    assert FakeSession.opened == 1
    assert FakeSession.responses[0].closed is True
    session.close()
    assert FakeSession.closed == 1


def test_download_closes_response_on_success_and_error(tmp_path, monkeypatch):
    _reset()
    monkeypatch.setattr(AQ, "ROOT", tmp_path)
    monkeypatch.setattr(AQ, "_symbol_dir", lambda symbol: tmp_path / symbol)
    (tmp_path / "AAA").mkdir()
    session = FakeSession()
    ok = AQ._download(session, "https://www.nseindia.com/a.pdf", symbol="AAA", name="a.pdf")
    assert ok["ok"] is True
    assert FakeSession.responses[-1].closed is True
    session.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout"))
    failed = AQ._download(session, "https://www.nseindia.com/b.pdf", symbol="AAA", name="b.pdf")
    assert failed["ok"] is False
    assert "timeout" in failed["error"]


def test_acquire_symbol_closes_session_on_success_and_timeout(tmp_path, monkeypatch):
    _reset()
    monkeypatch.setattr(AQ, "facts_path", lambda symbol: tmp_path / f"{symbol}.json")
    monkeypatch.setattr(AQ, "load_autonomy_facts", lambda symbol: {})
    monkeypatch.setattr(AQ, "save_autonomy_facts", lambda symbol, payload: tmp_path / f"{symbol}.json")
    monkeypatch.setattr(AQ, "plan_acquire", lambda symbol, **k: {
        "to_fetch": ["exchange_filings"],
        "lanes": {"nse_filings": True, "nse_annual": False, "option_chain": False, "screener": False},
        "coverage": {},
        "force": False,
    })
    monkeypatch.setattr(AQ, "_nse_session", FakeSession)
    monkeypatch.setattr(AQ, "_fetch_nse", lambda symbol, session: {
        "step": {"id": "nse_filings", "ok": True},
        "downloads": [],
        "texts": [],
        "headlines": [],
    })
    monkeypatch.setattr(AQ, "_framework_id_for", lambda symbol, raw: "generic")
    monkeypatch.setattr(AQ, "_empty_nse", AQ._empty_nse)
    out = AQ.acquire_symbol("AAA")
    assert out["symbol"] == "AAA"
    assert FakeSession.opened == 1
    assert FakeSession.closed == 1

    _reset()
    monkeypatch.setattr(AQ, "_nse_session", FakeSession)
    monkeypatch.setattr(
        AQ,
        "_fetch_nse",
        lambda symbol, session: (_ for _ in ()).throw(AQ.AcquireTimeout("deadline")),
    )
    try:
        AQ.acquire_symbol("BBB", deadline_monotonic=0)
    except AQ.AcquireTimeout:
        pass
    # deadline at start raises before session
    assert FakeSession.opened == 0


def test_acquire_symbol_closes_session_when_provider_raises(tmp_path, monkeypatch):
    _reset()
    monkeypatch.setattr(AQ, "load_autonomy_facts", lambda symbol: {})
    monkeypatch.setattr(AQ, "save_autonomy_facts", lambda symbol, payload: tmp_path / f"{symbol}.json")
    monkeypatch.setattr(AQ, "plan_acquire", lambda symbol, **k: {
        "to_fetch": ["exchange_filings"],
        "lanes": {"nse_filings": True, "nse_annual": False, "option_chain": False, "screener": False},
        "coverage": {},
        "force": False,
    })
    monkeypatch.setattr(AQ, "_nse_session", FakeSession)
    monkeypatch.setattr(AQ, "_fetch_nse", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("provider down")))
    monkeypatch.setattr(AQ, "_framework_id_for", lambda symbol, raw: "generic")
    out = AQ.acquire_symbol("CCC")
    assert FakeSession.closed == FakeSession.opened == 1
    assert out["steps"]


def test_repeated_acquire_does_not_grow_open_sessions(tmp_path, monkeypatch):
    _reset()
    monkeypatch.setattr(AQ, "load_autonomy_facts", lambda symbol: {})
    monkeypatch.setattr(AQ, "save_autonomy_facts", lambda symbol, payload: tmp_path / f"{symbol}.json")
    monkeypatch.setattr(AQ, "plan_acquire", lambda symbol, **k: {
        "to_fetch": ["exchange_filings"],
        "lanes": {"nse_filings": True, "nse_annual": False, "option_chain": False, "screener": False},
        "coverage": {},
        "force": False,
    })
    monkeypatch.setattr(AQ, "_nse_session", FakeSession)
    monkeypatch.setattr(AQ, "_fetch_nse", lambda symbol, session: {
        "step": {"id": "nse_filings", "ok": True}, "downloads": [], "texts": [], "headlines": [],
    })
    monkeypatch.setattr(AQ, "_framework_id_for", lambda symbol, raw: "generic")
    for _ in range(12):
        AQ.acquire_symbol("REL")
    assert FakeSession.opened == 12
    assert FakeSession.closed == 12
    assert FakeSession.opened - FakeSession.closed == 0
