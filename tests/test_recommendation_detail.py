"""Tests for Reco-style recommendation detail (Performance / Thesis / KPIs)."""
from __future__ import annotations

from product.recommendation_detail import build_recommendation_detail


def test_detail_exposes_kpi_groups_without_inventing_numbers():
    scan = {
        "scanned_at": "2026-08-15T10:00:00+00:00",
        "records_status": "CURRENT_DAY",
        "same_ist_day": True,
        "records": [
            {
                "symbol": "RECOV", "company": "Recovery Inc", "score": 60,
                "signals": ["DOUBLE_BOTTOM"], "verdict": "WATCH", "status": "Watch",
                "chase_risk": False, "price": 50, "entry": 48, "target": 60, "stop": 42,
                "rsi": 45, "volume_ratio": 1.2, "avg_vol20": 5e5,
            },
        ],
    }
    detail = build_recommendation_detail(
        "RECOV",
        category_id="recovery_setups",
        scan_payload=scan,
        long_term_payload={"records": []},
    )
    assert detail["symbol"] == "RECOV"
    assert detail["performance"]["entry"] == 48
    assert detail["performance"]["target"] == 60
    assert detail["performance"]["stop"] == 42
    assert detail["performance"]["downside_from_cmp_pct"] is not None
    assert set(detail["kpis"]) == {"profitability", "valuation", "margins"}
    for group in detail["kpis"].values():
        assert isinstance(group, list)
        for row in group:
            # Missing fundamentals must stay unavailable — never fabricated.
            if row["value"] is None:
                assert row["available"] is False
                assert row["display"] == "—"
    assert "our_take" in detail["thesis"]
    assert detail["fundamentals_note"]


def test_detail_maps_fundamentals_into_kpi_tabs(monkeypatch):
    import product.recommendation_detail as rd

    def fake_workspace(symbol: str):
        return {
            "company": "Quality Ltd",
            "sector": "Chemicals",
            "fundamentals": {
                "available": True,
                "coverage_pct": 80,
                "quality_factors": ["ROCE 18%"],
                "risk_flags": ["Rich valuation"],
                "metrics": [
                    {"key": "roe", "label": "ROE", "value": 14.5, "unit": "%"},
                    {"key": "roce", "label": "ROCE", "value": 18.0, "unit": "%"},
                    {"key": "pe", "label": "P/E", "value": 22.0, "unit": "x"},
                    {"key": "market_cap", "label": "Market cap", "value": 12000, "unit": "INR Cr"},
                ],
                "raw_values": {"roe": 14.5, "roce": 18.0, "pe": 22.0, "market_cap": 12000},
            },
        }

    monkeypatch.setattr("product.stock_workspace.build_stock_workspace", fake_workspace)
    monkeypatch.setattr(rd, "_ratio_map", lambda symbol, raw: {"operating_margin": 21.5, "net_margin": 12.0})
    monkeypatch.setattr(
        rd,
        "_find_card",
        lambda symbol, category_id="", scan_payload=None, long_term_payload=None: {
            "symbol": "QUAL",
            "company": "Quality Ltd",
            "category_id": "wealth_builders",
            "category_label": "Wealth Builders",
            "action_badge": "Hold / Research",
            "risk_tier": "Low",
            "entry": 100,
            "cmp": 110,
            "target": 140,
            "stop": 90,
            "upside_from_entry_pct": 10.0,
            "upside_to_target_pct": 27.3,
            "qualify_reason": "Quality compounder",
        },
    )

    detail = build_recommendation_detail("QUAL", category_id="wealth_builders")
    by_p = {m["key"]: m for m in detail["kpis"]["profitability"]}
    by_v = {m["key"]: m for m in detail["kpis"]["valuation"]}
    by_m = {m["key"]: m for m in detail["kpis"]["margins"]}
    assert by_p["roe"]["value"] == 14.5
    assert by_p["roce"]["value"] == 18.0
    assert by_v["pe"]["value"] == 22.0
    assert by_m["operating_margin"]["value"] == 21.5
    assert by_m["net_margin"]["value"] == 12.0
    assert detail["fundamentals_ready"] is True
    assert "ROCE" in detail["thesis"]["our_take"] or detail["thesis"]["quality_factors"]
