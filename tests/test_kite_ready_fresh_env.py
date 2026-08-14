"""kite_ready must see a fresh login without restarting the process."""
from __future__ import annotations


def test_kite_ready_rereads_env_token(monkeypatch):
    import execution.trade_executor as te

    state = {"token": "", "key": ""}

    def _fresh(name, default=""):
        if name == "KITE_ACCESS_TOKEN":
            return state["token"]
        if name == "KITE_API_KEY":
            return state["key"]
        return default

    class _Fake:
        def is_connected(self):
            return bool(state["token"] and state["key"])

    monkeypatch.setattr("data.kite_client._fresh_env", _fresh)
    monkeypatch.setattr("data.kite_client.KiteClient", _Fake)

    assert te.kite_ready() is False
    state["key"] = "key123"
    state["token"] = "tok_live"
    assert te.kite_ready() is True
