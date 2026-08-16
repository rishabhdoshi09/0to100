"""Buy thesis composition — why a name is on the desk."""
from __future__ import annotations

from product.buy_thesis import (
    _earnings_block,
    _plan,
    _why_chosen,
    build_buy_thesis,
    build_sector_wave,
    build_smart_money,
    classify_client,
    resolve_sector,
)


def test_why_chosen_uses_scan_evidence_not_vibes():
    why = _why_chosen(
        {
            "signals": ["BREAKOUT_52W"],
            "reasons": ["Broke 52-week high on volume"],
            "breakout_grade": "A",
            "volume_ratio": 2.4,
            "rsi": 61,
        },
        {"classification": "QUALITY_COMPOUNDER", "fundamental_coverage": 0.8,
         "quality_factors": ["ROCE 22%"]},
        {"trend_explanation": "Price holds above the 200-day average."},
    )
    joined = " ".join(why)
    assert "52-week" in joined
    assert "grade a" in joined.lower()
    assert "ROCE" in joined
    assert "not an order" in joined.lower()
    assert "QUALITY" in joined or "compounder" in joined.lower()


def test_plan_fills_stop_target_from_price_when_scan_blank():
    plan = _plan({}, {"price": 1000, "vol_pct": 2.0}, {"close": 1000})
    assert plan["buy"] == 1000
    assert plan["stop"] < plan["buy"] < plan["target"]
    assert plan["upside_from_buy_pct"] > 0


def test_build_buy_thesis_does_not_invent_a_blank_symbol_book():
    from unittest.mock import patch
    fake_book = {
        "available": False,
        "status": "unavailable",
        "note": "NSE book is empty (closed session or no quotes).",
        "source": "nse",
        "bids": [],
        "asks": [],
    }
    with patch("product.buy_thesis._order_book", return_value=fake_book), \
         patch("product.buy_thesis._load_flows", return_value={}), \
         patch("product.buy_thesis._sector_avg_move", return_value={"chg_1d": None, "chg_5d": None, "members": 0}), \
         patch("product.buy_thesis._index_return_pct", return_value=None):
        payload = build_buy_thesis("BSE", fetch_missing=False)
    assert payload["symbol"] == "BSE"
    assert payload["why"]
    assert payload["order_book"]["status"] == "unavailable"
    assert payload["order_book"].get("bids") == []
    assert "sector_wave" in payload
    assert "smart_money" in payload
    assert "earnings" in payload


def test_resolve_sector_prefers_universe_map():
    ident = resolve_sector("RELIANCE", workspace_sector="Unclassified")
    # Reliance is in the NSE map; if the map loaded, it must not stay unclassified.
    if ident["mapped"]:
        assert ident["identified"]
        assert ident["source"] == "nse_universe_map"
        assert ident["sector"] != "Unclassified"


def test_sector_wave_no_claim_when_sector_unknown():
    from unittest.mock import patch
    with patch("product.buy_thesis.resolve_sector", return_value={
        "sector": "", "source": "", "mapped": "", "workspace": "", "identified": False,
        "nse_sector": "", "nse_industry": "",
    }):
        wave = build_sector_wave("ZZZNOTASECTOR", workspace_sector="", scan_records=[], flows={})
    assert wave["wave"] == "NO_CLAIM"
    assert wave["identified"] is False
    assert "inflow" not in wave["headline"].lower() or "not identified" in wave["headline"].lower()


def test_nse_industry_aligns_to_mapped_sector():
    from unittest.mock import patch
    with patch("scan.sector_heat.sector_of", return_value=""), \
         patch("data.nse_live.fetch_equity_industry", return_value={
             "macro": "Industrials",
             "sector": "Capital Goods",
             "industry": "Industrial Manufacturing",
             "basic_industry": "Mining Machinery",
             "source": "nse_quote_equity",
             "error": "",
         }):
        ident = resolve_sector("EIMCOELECO", workspace_sector="Unclassified")
    assert ident["identified"]
    assert ident["source"] == "nse_industry"
    assert "Capital Goods" in ident["sector"] or ident["sector"] == "Engineering"


def test_sector_wave_inflow_from_basket_and_pack():
    from unittest.mock import patch
    with patch("product.buy_thesis.resolve_sector", return_value={
        "sector": "Capital Goods", "source": "nse_universe_map",
        "mapped": "Capital Goods", "workspace": "", "identified": True,
    }), patch("product.buy_thesis._sector_avg_move", return_value={
        "chg_1d": 1.2, "chg_5d": 3.4, "members": 12,
    }), patch("product.buy_thesis._index_return_pct", side_effect=lambda p: 0.1 if p == 1 else 0.5):
        wave = build_sector_wave(
            "BSE",
            "Capital Goods",
            scan_records=[{"symbol": "BSE", "sector": "Capital Goods"},
                          {"symbol": "HAL", "sector": "Capital Goods"},
                          {"symbol": "BEL", "sector": "Capital Goods"}],
            flows={},
        )
    assert wave["wave"] == "INFLOW"
    assert "Capital Goods" in wave["headline"]
    assert any("Nifty" in b for b in wave["bullets"])


def test_classify_client_does_not_call_a_desk_influential():
    assert classify_client("HDFC MUTUAL FUND") == "institution"
    assert classify_client("HRTI PRIVATE LIMITED") == "desk"
    assert classify_client("RAMACHANDRAN RADHAKRISHNAN") == "named_person"


def test_smart_money_fii_bought_from_shareholding_not_vibes():
    share = [
        {"": "FIIs", "Dec 2025": 5.0, "Mar 2026": 6.8},
        {"": "DIIs", "Dec 2025": 8.0, "Mar 2026": 7.1},
        {"": "Promoters", "Dec 2025": 48.0, "Mar 2026": 48.0},
        {"": "Promoter holding pledged", "Dec 2025": 0, "Mar 2026": 0},
    ]
    flows = {
        "bulk_deals": [{
            "symbol": "BSE", "client": "SBI MUTUAL FUND",
            "side": "BUY", "qty": 100000, "price": 2500.0,
        }],
        "fii_dii": {"date": "14-Aug-2026", "fii_net_cr": 508, "dii_net_cr": 356, "bias": "SUPPORTIVE"},
    }
    money = build_smart_money("BSE", share, flows)
    assert money["fii"]["action"] == "bought"
    assert money["dii"]["action"] == "sold"
    assert money["stance"] == "BOUGHT"
    assert money["promoter"]["latest"] == 48.0
    joined = " ".join(money["bullets"])
    assert "SBI MUTUAL FUND" in joined
    assert "market-wide, not this stock" in joined.lower() or "All-India" in joined


def test_smart_money_does_not_invent_a_print():
    money = build_smart_money("NOSUCH", [], {})
    assert money["stance"] == "NO_CLAIM"
    assert money["deals"] == []
    assert money["influencers"] == []


def test_earnings_block_reads_quarters_margins_valuations():
    raw = {
        "quarterly_results": [
            {"": "Sales", "Jun 2025": 100, "Sep 2025": 110, "Dec 2025": 120, "Mar 2026": 140},
            {"": "Net Profit", "Jun 2025": 10, "Sep 2025": 12, "Dec 2025": 14, "Mar 2026": 18},
        ],
        "profit_loss": [
            {"": "Sales", "Mar 2024": 400, "Mar 2025": 480, "Mar 2026": 560},
            {"": "Operating Profit", "Mar 2024": 80, "Mar 2025": 100, "Mar 2026": 126},
            {"": "Net Profit", "Mar 2024": 40, "Mar 2025": 50, "Mar 2026": 70},
            {"": "OPM %", "Mar 2024": 20, "Mar 2025": 20.8, "Mar 2026": 22.5},
        ],
        "key_ratios": [{"name": "P/B", "value": "4.2"}],
    }
    metrics = [
        {"key": "pe", "label": "Price / earnings", "value": 30.2, "unit": "x"},
        {"key": "roe", "label": "Return on equity", "value": 10.4, "unit": "%"},
        {"key": "sales_growth_3y", "label": "Sales CAGR (3Y)", "value": 12.0, "unit": "%"},
    ]
    block = _earnings_block(raw, metrics)
    assert block["available"]
    assert block["sales_qoq"]["latest"] == 140
    assert block["opm"][-1]["value"] == 22.5
    assert any(v["key"] == "pe" for v in block["valuations"])
    assert any(v["key"] == "pb" for v in block["valuations"])
    joined = " ".join(block["bullets"])
    assert "P/E" in joined
    assert "Operating margin" in joined

    from_book = _earnings_block(
        {
            "key_ratios": [
                {"name": "Current Price", "value": "2,006"},
                {"name": "Book Value", "value": "669"},
            ]
        },
        [{"key": "pe", "label": "Price / earnings", "value": 30.2, "unit": "x"}],
    )
    pb = next(v for v in from_book["valuations"] if v["key"] == "pb")
    assert 2.5 < pb["value"] < 4.0
