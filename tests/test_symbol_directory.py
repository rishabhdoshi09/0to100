"""Full NSE universe directory — search must not be limited to scan setups."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_universe_names_cover_every_symbol(monkeypatch):
    import data.nse_universe as U

    U.reset_universe_cache()
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: (["AAA", "BBB"], {"AAA": "Alpha"}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: (["BBB", "CCC"], {"CCC": "Charlie"}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: ["DDD", "AAA"])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    syms = U.get_nse_universe()
    names = U.get_nse_universe_with_names()
    assert set(syms) == {"AAA", "BBB", "CCC", "DDD"}
    assert set(names) == set(syms)
    assert names["AAA"] == "Alpha"
    assert names["CCC"] == "Charlie"
    assert names["DDD"] == ""


def test_universe_nifty500_fallback_still_has_name_keys(monkeypatch):
    import data.nse_universe as U

    U.reset_universe_cache()
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: ([], {}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    names = U.get_nse_universe_with_names()
    assert "RELIANCE" in names
    assert len(names) == len(U.get_nse_universe()) >= 50


def test_search_nse_symbols_prefix(monkeypatch):
    import data.nse_universe as U

    U.reset_universe_cache()
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: (["RELIANCE", "RELINFRA", "TCS"], {"RELIANCE": "Reliance"}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    rows = U.search_nse_symbols("REL", limit=10)
    assert [r["symbol"] for r in rows] == ["RELIANCE", "RELINFRA"]


def test_market_scan_default_universe_not_empty_on_fallback(monkeypatch):
    import data.nse_universe as U
    from scan.market_scan_service import _default_universe

    U.reset_universe_cache()
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: ([], {}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    names = _default_universe()
    assert "RELIANCE" in names
    assert len(names) >= 50


def test_symbols_api_returns_full_directory(monkeypatch):
    import data.nse_universe as U
    import terminal_product_api as api

    U.reset_universe_cache()
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: (["AAA", "AARTIIND", "TCS"], {"TCS": "Tata"}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    client = TestClient(api.app)
    body = client.get("/api/symbols?q=AA&limit=10").json()
    assert body["universe_size"] == 3
    assert body["count"] >= 1
    assert all(row["symbol"].startswith("AA") or "AA" in row["symbol"] for row in body["symbols"])
