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
    monkeypatch.setattr(CF, "fetch_option_chain", lambda symbol: (df, "2026-08-07"))
    payload = CF.chain_workspace("NIFTY", spot=24500)
    assert payload["available"] is True
    assert payload["pcr"] == round((200 + 90 + 60) / (100 + 80 + 120), 2)
    assert payload["atm_iv"] > 0
    assert payload["total_ce_oi"] == 300
    assert payload["total_pe_oi"] == 350
    assert payload["greeks_available"] is False
    assert payload["signal_desk"] is False
    assert "Greeks" in payload["honesty"]


def test_options_history_workspace_empty(tmp_path, monkeypatch):
    from product import market_api as MA

    monkeypatch.setattr("options.eod_store._DEFAULT_DB", tmp_path / "eod.sqlite3")
    payload = MA.options_history_workspace("NIFTY", days=14)
    assert payload["symbol"] == "NIFTY"
    assert payload["available"] is False
    assert payload["rows"] == []
