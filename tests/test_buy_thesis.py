"""Buy thesis composition — why a name is on the desk."""
from __future__ import annotations

from product.buy_thesis import _plan, _why_chosen, build_buy_thesis


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
    with patch("product.buy_thesis._order_book", return_value=fake_book):
        payload = build_buy_thesis("BSE", fetch_missing=False)
    assert payload["symbol"] == "BSE"
    assert payload["why"]
    assert payload["order_book"]["status"] == "unavailable"
    assert payload["order_book"].get("bids") == []
