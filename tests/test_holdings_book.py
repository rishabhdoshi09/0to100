"""Demat holdings book — empty until the user syncs/imports their own rows."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_be_series_is_valid_and_searchable():
    from data.nse_universe import _is_valid_symbol, reset_universe_cache, search_nse_symbols
    import data.nse_universe as U

    assert _is_valid_symbol("ACMECO-BE")
    reset_universe_cache()
    U._cached_universe = ["ACMECO-BE", "BETAIND", "GAMMACO"]
    U._cached_names = {"ACMECO-BE": "Acme", "BETAIND": "Beta", "GAMMACO": "Gamma"}
    U._universe_loaded = True
    rows = search_nse_symbols("ACME", limit=5)
    assert any(r["symbol"] == "ACMECO-BE" for r in rows)


def test_holdings_import_keeps_user_book(tmp_path, monkeypatch):
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    rows = [
        {"tradingsymbol": "AAA", "quantity": 10, "average_price": 100.0, "last_price": 110.0},
        {"symbol": "BBB", "quantity": 5, "average_price": 200.0, "last_price": 180.0},
        {"tradingsymbol": "ACMECO-BE", "quantity": 8, "average_price": 50.0, "last_price": 60.0},
    ]
    book = HB.save_holdings(rows, source="paste", path=path)
    assert book["available"] is True
    assert book["summary"]["count"] == 3
    assert book["summary"]["invested"] == 10 * 100 + 5 * 200 + 8 * 50
    symbols = {h["tradingsymbol"] for h in book["holdings"]}
    assert symbols == {"AAA", "BBB", "ACMECO-BE"}
    be = next(h for h in book["holdings"] if h["tradingsymbol"] == "ACMECO-BE")
    assert be["research_symbol"] == "ACMECO"
    assert be["pnl"] > 0


def test_holdings_start_empty(tmp_path, monkeypatch):
    from product import holdings_book as HB

    path = tmp_path / "missing.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    book = HB.build_holdings_payload(path)
    assert book["available"] is False
    assert book["holdings"] == []
    assert "never invents" in book["message"].lower() or "no broker holdings" in book["message"].lower()


def test_holdings_api_import_and_get(tmp_path, monkeypatch):
    import terminal_product_api as api
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    monkeypatch.setattr(HB, "DEFAULT_PATH", path)

    client = TestClient(api.app)
    payload = {
        "holdings": [
            {"tradingsymbol": "AAA", "quantity": 2, "average_price": 100, "last_price": 105},
            {"tradingsymbol": "ACMECO-BE", "quantity": 3, "average_price": 50, "last_price": 55},
        ]
    }
    r = client.post("/api/holdings/import", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["summary"]["count"] == 2

    got = client.get("/api/holdings").json()
    assert got["summary"]["count"] == 2
    assert {h["tradingsymbol"] for h in got["holdings"]} == {"AAA", "ACMECO-BE"}


def test_connection_status_reports_missing_token(tmp_path, monkeypatch):
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    status = HB.connection_status(path)
    assert status["available"] is False
    assert status["kite_connected"] is False
    assert status["telegram_configured"] is False
    assert "KITE_API_KEY" in status["message"] or "token" in status["message"].lower()


def test_notify_holdings_telegram_sends_when_configured(tmp_path, monkeypatch):
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    book = HB.save_holdings(
        [{"tradingsymbol": "INFY", "quantity": 10, "average_price": 100, "last_price": 110}],
        source="paste",
        path=path,
    )

    class _FakeEngine:
        def is_configured(self):
            return True

        def send(self, message: str):
            assert "INFY" in message
            assert "My Holdings" in message
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _FakeEngine)
    result = HB.notify_holdings_telegram(book)
    assert result["sent"] is True
    assert result["count"] == 1


def test_holdings_status_and_notify_api(tmp_path, monkeypatch):
    import terminal_product_api as api
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    monkeypatch.setattr(HB, "DEFAULT_PATH", path)
    HB.save_holdings(
        [{"tradingsymbol": "TCS", "quantity": 1, "average_price": 3000, "last_price": 3100}],
        source="paste",
        path=path,
    )

    sent = {"n": 0}

    class _FakeEngine:
        def is_configured(self):
            return True

        def send(self, message: str):
            sent["n"] += 1
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _FakeEngine)
    client = TestClient(api.app)
    status = client.get("/api/holdings/status").json()
    assert status["available"] is True
    assert status["count"] == 1
    assert status["telegram_configured"] is True

    got = client.get("/api/holdings").json()
    assert "connection" in got
    assert got["connection"]["count"] == 1

    notify = client.post("/api/holdings/notify").json()
    assert notify["accepted"] is True
    assert notify["telegram"]["sent"] is True
    assert sent["n"] == 1


def test_symbol_directory_pins_holdings(tmp_path, monkeypatch):
    import data.nse_universe as U
    from product import holdings_book as HB
    from product.symbol_directory import build_symbol_directory

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    monkeypatch.setattr(HB, "DEFAULT_PATH", path)
    HB.save_holdings(
        [{"tradingsymbol": "ACMECO-BE", "quantity": 10, "average_price": 100, "last_price": 110}],
        source="paste",
        path=path,
    )
    U.reset_universe_cache()
    U._cached_universe = ["RELIANCE"]
    U._cached_names = {"RELIANCE": "Reliance"}
    U._universe_loaded = True

    payload = build_symbol_directory(query="ACME", limit=10)
    assert any(row["symbol"] == "ACMECO-BE" for row in payload["symbols"])
    assert payload["holdings_pinned"] >= 1
