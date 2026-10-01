"""screener/engine.py::_extract_fundamentals feeds Stock Intelligence's
Financials/Ownership tabs with current-snapshot fundamentals scraped from
screener.in. Before this fix, nothing checked whether a parsed number was
even POSSIBLE (a promoter holding of 450%, a negative market cap, a
debt/equity ratio off by a lakh/crore unit mismatch) -- it would be shown to
the user as a confident, plausible-looking fact. An out-of-bounds value must
be dropped to None (honestly "unavailable"), never clamped or passed through.
"""
from __future__ import annotations

from screener.engine import _extract_fundamentals, _sane


def _snapshot(**key_ratio_overrides):
    key_ratios = [
        {"name": "Stock P/E", "value": "25"},
        {"name": "ROE", "value": "22%"},
        {"name": "ROCE", "value": "25%"},
        {"name": "Debt to equity", "value": "0.20"},
        {"name": "Market Cap", "value": "25000"},
        {"name": "Dividend Yield", "value": "1.5%"},
    ]
    for row in key_ratios:
        name = row["name"].lower()
        if name in key_ratio_overrides:
            row["value"] = key_ratio_overrides[name]
    return {
        "key_ratios": key_ratios,
        "profit_loss": [], "cash_flow": [], "shareholding": [],
    }


def test_promoter_holding_above_100_percent_is_rejected():
    data = _snapshot()
    data["shareholding"] = [{"": "Promoters", "2025": 58, "2026": 450}]
    fund = _extract_fundamentals(data)
    assert fund["promoter_holding"] is None
    assert "promoter_holding" not in fund["available_fields"]


def test_promoter_holding_in_bounds_is_kept():
    data = _snapshot()
    data["shareholding"] = [{"": "Promoters", "2025": 58, "2026": 59}]
    fund = _extract_fundamentals(data)
    assert fund["promoter_holding"] == 59


def test_negative_market_cap_is_rejected():
    fund = _extract_fundamentals(_snapshot(**{"market cap": "-5000"}))
    assert fund["market_cap_cr"] is None


def test_zero_market_cap_is_rejected():
    fund = _extract_fundamentals(_snapshot(**{"market cap": "0"}))
    assert fund["market_cap_cr"] is None


def test_debt_to_equity_unit_conversion_blowout_is_rejected():
    # A lakh/crore mixup could easily produce a 4-digit "ratio".
    fund = _extract_fundamentals(_snapshot(**{"debt to equity": "15000"}))
    assert fund["debt_to_equity"] is None


def test_dividend_yield_above_100_percent_is_rejected():
    fund = _extract_fundamentals(_snapshot(**{"dividend yield": "1050%"}))
    assert fund["dividend_yield"] is None


def test_dividend_yield_negative_is_rejected():
    fund = _extract_fundamentals(_snapshot(**{"dividend yield": "-2%"}))
    assert fund["dividend_yield"] is None


def test_pledge_above_100_percent_is_rejected():
    data = _snapshot()
    data["shareholding"] = [{"": "Pledged percentage", "2025": 10, "2026": 230}]
    fund = _extract_fundamentals(data)
    assert fund["promoter_pledge"] is None


def test_legitimately_negative_roe_is_kept_not_treated_as_impossible():
    # A loss-making company's ROE is real evidence, not an error.
    fund = _extract_fundamentals(_snapshot(**{"roe": "-18%"}))
    assert fund["roe"] == -18.0


def test_sane_helper_passes_none_through():
    assert _sane("promoter_holding", None) is None


def test_sane_helper_has_no_bound_for_unknown_fields():
    assert _sane("some_unbounded_field", 1e12) == 1e12
