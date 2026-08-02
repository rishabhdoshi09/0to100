"""Demat holdings book — broker shares must show even when scan filters drop them."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_be_series_is_valid_and_searchable():
    from data.nse_universe import _is_valid_symbol, reset_universe_cache, search_nse_symbols
    import data.nse_universe as U

    assert _is_valid_symbol("GAUDIUMIVF-BE")
    reset_universe_cache()
    # Force a tiny universe that includes the -BE holding.
    U._cached_universe = ["GAUDIUMIVF-BE", "NACLIND", "ARSSBL"]
    U._cached_names = {"GAUDIUMIVF-BE": "Gaudium", "NACLIND": "NACL", "ARSSBL": "ARSS"}
    U._universe_loaded = True
    rows = search_nse_symbols("GAUDIUM", limit=5)
    assert any(r["symbol"] == "GAUDIUMIVF-BE" for r in rows)


def test_holdings_import_keeps_user_book(tmp_path, monkeypatch):
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    rows = [
        {"tradingsymbol": "ARSSBL", "quantity": 25, "average_price": 609.06, "last_price": 510.80},
        {"symbol": "FILATFASH", "quantity": 229, "average_price": 0.25, "last_price": 0.19},
        {"tradingsymbol": "GAUDIUMIVF-BE", "quantity": 118, "average_price": 112.34, "last_price": 137.50},
        {"tradingsymbol": "NACLIND", "quantity": 103, "average_price": 235.51, "last_price": 188.93},
        {"tradingsymbol": "SKMEGGPROD", "quantity": 154, "average_price": 205.0, "last_price": 249.85},
    ]
    book = HB.save_holdings(rows, source="paste", path=path)
    assert book["available"] is True
    assert book["summary"]["count"] == 5
    assert book["summary"]["invested"] > 80_000
    symbols = {h["tradingsymbol"] for h in book["holdings"]}
    assert symbols == {"ARSSBL", "FILATFASH", "GAUDIUMIVF-BE", "NACLIND", "SKMEGGPROD"}
    gaud = next(h for h in book["holdings"] if h["tradingsymbol"] == "GAUDIUMIVF-BE")
    assert gaud["research_symbol"] == "GAUDIUMIVF"
    assert gaud["pnl"] > 0


def test_holdings_api_import_and_get(tmp_path, monkeypatch):
    import terminal_product_api as api
    from product import holdings_book as HB

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    # Ensure module reads env on each call via holdings_path()
    monkeypatch.setattr(HB, "DEFAULT_PATH", path)

    client = TestClient(api.app)
    payload = {
        "holdings": [
            {"tradingsymbol": "SKMEGGPROD", "quantity": 154, "average_price": 205, "last_price": 249.85},
            {"tradingsymbol": "GAUDIUMIVF-BE", "quantity": 118, "average_price": 112.34, "last_price": 137.5},
        ]
    }
    r = client.post("/api/holdings/import", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["summary"]["count"] == 2

    got = client.get("/api/holdings").json()
    assert got["summary"]["count"] == 2
    assert {h["tradingsymbol"] for h in got["holdings"]} == {"SKMEGGPROD", "GAUDIUMIVF-BE"}


def test_symbol_directory_pins_holdings(tmp_path, monkeypatch):
    import data.nse_universe as U
    from product import holdings_book as HB
    from product.symbol_directory import build_symbol_directory

    path = tmp_path / "holdings.json"
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(path))
    monkeypatch.setattr(HB, "DEFAULT_PATH", path)
    HB.save_holdings(
        [{"tradingsymbol": "GAUDIUMIVF-BE", "quantity": 10, "average_price": 100, "last_price": 110}],
        source="paste",
        path=path,
    )
    U.reset_universe_cache()
    U._cached_universe = ["RELIANCE"]
    U._cached_names = {"RELIANCE": "Reliance"}
    U._universe_loaded = True

    payload = build_symbol_directory(query="GAUDIUM", limit=10)
    assert any(row["symbol"] == "GAUDIUMIVF-BE" for row in payload["symbols"])
    assert payload["holdings_pinned"] >= 1
