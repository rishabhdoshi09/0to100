"""On-file fundamentals for Top Stocks — no invented ratios, no live scrape."""
from __future__ import annotations

from product.top_stocks import fund_rank, is_financial_sector, pack_fundamentals, sector_metric_specs
from product.recommendations_workspace import build_recommendations_workspace


def test_bank_sector_skips_generic_leverage():
    keys = {row[0] for row in sector_metric_specs("Private Bank")}
    assert "pe" in keys and "roe" in keys
    assert "debt_to_equity" not in keys
    assert is_financial_sector("NBFC") is True
    assert is_financial_sector("IT - Software") is False


def test_pack_fundamentals_does_not_invent_blank_pe():
    pack = pack_fundamentals({
        "sector": "Capital Goods",
        "classification": "QUALITY_COMPOUNDER",
        "fundamental_coverage": 0.8,
        "fundamentals": {"roe": 22.1, "roce": 18.0},
    })
    keys = {m["key"] for m in pack["metrics"]}
    assert "roe" in keys and "roce" in keys
    assert "pe" not in keys
    assert pack["available"] is True
    assert pack["source"] == "long_term_pack"


def test_empty_pack_when_nothing_on_file():
    pack = pack_fundamentals({"symbol": "X", "sector": "IT"})
    assert pack["available"] is False
    assert pack["metrics"] == []
    assert "on file" in pack["note"].lower()


def test_fund_rank_is_tie_break_only():
    assert fund_rank({"classification": "QUALITY_COMPOUNDER", "fundamental_coverage": 0.8}) == 2
    assert fund_rank({"classification": "NEEDS_FUNDAMENTALS", "fundamental_coverage": 0.1}) == 0


def test_best_setups_attach_onfile_funds_from_long_term(monkeypatch):
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=280, freq="B")
    close = pd.Series([80 + i * 0.6 for i in range(280)], index=index)
    frame = pd.DataFrame(
        {"open": close - 0.4, "high": close + 1.2, "low": close - 1.0, "close": close, "volume": [1] * 280},
        index=index,
    )
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-19T10:00:00+00:00",
            "records": [{
                "symbol": "LEADER", "score": 88, "signals": ["MOMENTUM"],
                "verdict": "BUY", "status": "Ready to trade", "chase_risk": False,
                "price": float(close.iloc[-1]), "rsi": 55, "volume_ratio": 1.4, "avg_vol20": 1e6,
            }],
        },
        long_term_payload={
            "records": [{
                "symbol": "LEADER", "sector": "Capital Goods",
                "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.8,
                "fundamentals": {"pe": 24.5, "roe": 19.0, "roce": 16.0},
            }],
        },
        refresh_technicals=False,
        compute_sepa=True,
        sepa_load_frame=lambda symbol: frame,
    )
    best = next(c for c in payload["categories"] if c["id"] == "best_setups")
    card = best["cards"][0]
    assert card["fundamentals"]["available"] is True
    assert {m["key"] for m in card["fundamentals"]["metrics"]} >= {"pe", "roe"}
    assert payload["tape"]["fundamental"]
    assert "Google" in payload["tape"]["price"] or "google" in payload["tape"]["price"].lower()
    assert "indices" in payload
    assert "stage" in payload["tape"]["technical"].lower() or "Nifty" in payload["tape"]["technical"]
