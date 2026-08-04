"""F&O desk chain payload honesty + history route shape."""
from __future__ import annotations

import pandas as pd


def test_chain_workspace_exposes_oi_iv_pcr_not_greeks(monkeypatch):
    from options import chain_fetch as CF

    df = pd.DataFrame(
        [
            {"strike": 24000, "ce_oi": 100, "pe_oi": 200, "ce_iv": 12.0, "pe_iv": 14.0, "ce_coi": 0, "pe_coi": 0},
            {"strike": 24500, "ce_oi": 80, "pe_oi": 90, "ce_iv": 11.0, "pe_iv": 13.0, "ce_coi": 0, "pe_coi": 0},
            {"strike": 25000, "ce_oi": 120, "pe_oi": 60, "ce_iv": 10.0, "pe_iv": 12.0, "ce_coi": 0, "pe_coi": 0},
        ]
    )
    monkeypatch.setattr(CF, "fetch_option_chain", lambda symbol: (df, "2026-08-07", 24500.0))
    payload = CF.chain_workspace("NIFTY", spot=24500)
    assert payload["available"] is True
    assert payload["pcr"] == round((200 + 90 + 60) / (100 + 80 + 120), 2)
    assert payload["atm_iv"] > 0
    assert payload["total_ce_oi"] == 300
    assert payload["total_pe_oi"] == 350
    assert payload["greeks_available"] is False
    assert payload["signal_desk"] is False
    assert "Greeks" in payload["honesty"]


def test_rows_from_records_accepts_v3_shape():
    from options.chain_fetch import _rows_from_records

    data = {
        "records": {
            "expiryDates": ["04-Aug-2026"],
            "data": [
                {
                    "expiryDates": "04-Aug-2026",
                    "strikePrice": 24500,
                    "CE": {
                        "openInterest": 10,
                        "changeinOpenInterest": 1,
                        "impliedVolatility": 12.5,
                        "lastPrice": 100,
                        "totalTradedVolume": 5,
                    },
                    "PE": {
                        "openInterest": 20,
                        "changeinOpenInterest": 2,
                        "impliedVolatility": 13.5,
                        "lastPrice": 90,
                        "totalTradedVolume": 7,
                    },
                }
            ],
        }
    }
    rows, expiry, underlying = _rows_from_records(data, "04-Aug-2026")
    assert expiry == "04-Aug-2026"
    assert underlying is None
    assert len(rows) == 1
    assert rows[0]["ce_oi"] == 10
    assert rows[0]["pe_oi"] == 20


def test_fetch_nse_prefers_v3(monkeypatch):
    from options import chain_fetch as CF
    import types

    calls: list[str] = []

    def fake_v3(session, symbol):
        calls.append("v3")
        df = pd.DataFrame(
            [{
                "strike": 100, "ce_oi": 1, "pe_oi": 2, "ce_iv": 1, "pe_iv": 1,
                "ce_coi": 0, "pe_coi": 0, "ce_ltp": 1, "pe_ltp": 1, "ce_volume": 0, "pe_volume": 0,
            }]
        )
        return df, "04-Aug-2026", 24500.0

    def boom_legacy(session, symbol):
        calls.append("legacy")
        raise AssertionError("legacy must not run when v3 succeeds")

    class FakeSession:
        pass

    fake_requests = types.SimpleNamespace(Session=lambda: FakeSession())
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    monkeypatch.setattr(CF, "_prime_nse_session", lambda session: None)
    monkeypatch.setattr(CF, "_fetch_nse_v3", fake_v3)
    monkeypatch.setattr(CF, "_fetch_nse_legacy", boom_legacy)

    df, expiry, underlying = CF._fetch_nse("NIFTY")
    assert expiry == "04-Aug-2026"
    assert underlying == 24500.0
    assert df is not None and len(df) == 1
    assert calls == ["v3"]


def test_options_history_workspace_empty(tmp_path, monkeypatch):
    from product import market_api as MA

    monkeypatch.setattr("options.eod_store._DEFAULT_DB", tmp_path / "eod.sqlite3")
    payload = MA.options_history_workspace("NIFTY", days=14)
    assert payload["symbol"] == "NIFTY"
    assert payload["available"] is False
    assert payload["rows"] == []
