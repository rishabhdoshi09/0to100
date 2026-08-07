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
    # Build a universe that spans A→Z so truncation past M would be visible.
    fake = [f"A{i:04d}" for i in range(40)] + [f"M{i:04d}" for i in range(40)] + [
        f"N{i:04d}" for i in range(40)
    ] + [f"Z{i:04d}" for i in range(40)]
    names = {s: s for s in fake}
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: (fake, names))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    client = TestClient(api.app)
    body = client.get("/api/symbols?limit=0").json()
    assert body["truncated"] is False
    assert body["count"] == len(fake)
    letters = set(body["letter_coverage"])
    assert {"A", "M", "N", "Z"}.issubset(letters)

    n_body = client.get("/api/symbols?q=N&limit=20").json()
    assert n_body["count"] >= 1
    assert all(row["symbol"].startswith("N") or "N" in row["symbol"] for row in n_body["symbols"][:10])


def test_empty_directory_always_covers_n_to_z(monkeypatch):
    from product.symbol_directory import build_symbol_directory
    import data.nse_universe as U

    U.reset_universe_cache()
    fake = [f"{letter}{i:03d}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for i in range(10)]
    monkeypatch.setattr(U, "_load_from_kite_cache", lambda: (fake, {s: "" for s in fake}))
    monkeypatch.setattr(U, "_load_from_fallback_csv", lambda: ([], {}))
    monkeypatch.setattr(U, "_bhav_symbols", lambda: [])
    monkeypatch.setattr(U, "_load_from_nse_website", lambda: ([], {}))
    monkeypatch.setattr(U, "_filter_to_instruments", lambda symbols, token_map: symbols)

    # Even a thin requested limit must expand on empty query so N…Z stay visible.
    full = build_symbol_directory(query="", limit=100)
    assert full["truncated"] is False
    assert {"N", "O", "P", "Z"}.issubset(set(full["letter_coverage"]))
    assert full["count"] == len(fake)
