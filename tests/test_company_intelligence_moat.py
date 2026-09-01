"""Phase 5 — Company Intelligence moat layer (extends existing DD)."""

from __future__ import annotations

from product.due_diligence.moat_layer import (
    company_intelligence_moat,
    metric_applicable,
    promise_vs_actual,
)
from product.due_diligence.provenance import resolve_fact


def _q(label, **periods):
    return {"row_label": label, **periods}


def _industrial_raw(**over):
    data = {
        "quarterly_results": [],
        "profit_loss": [
            _q("Sales", **{"Mar 2025": 100, "Mar 2026": 90}),
            _q("Net profit", **{"Mar 2025": 20, "Mar 2026": 22}),
            _q("Operating profit", **{"Mar 2025": 30, "Mar 2026": 28}),
            _q("Interest", **{"Mar 2025": 4, "Mar 2026": 5}),
        ],
        "cash_flow": [
            _q("Cash from operating", **{"Mar 2025": 18, "Mar 2026": 6}),
            _q("Cash from investing", **{"Mar 2025": -8, "Mar 2026": -20}),
        ],
        "balance_sheet": [
            _q("Trade receivables", **{"Mar 2025": 10, "Mar 2026": 18}),
            _q("Inventories", **{"Mar 2025": 8, "Mar 2026": 9}),
            _q("Borrowings", **{"Mar 2025": 40, "Mar 2026": 55}),
            _q("Equity capital", **{"Mar 2025": 50, "Mar 2026": 50}),
        ],
        "shareholding": [
            _q("Promoters", **{"Mar 2025": 52, "Mar 2026": 47}),
            _q("Pledge", **{"Mar 2025": 2, "Mar 2026": 2}),
        ],
    }
    data.update(over)
    return data


def test_sector_specific_applicability():
    assert metric_applicable("cfo_conversion", "industrials") is True
    assert metric_applicable("cfo_conversion", "bank") is False
    assert metric_applicable("cfo_conversion", "nbfc") is False
    assert metric_applicable("inventory_stress", "it") is False
    assert metric_applicable("roic", "metals") is False
    bank = company_intelligence_moat(_industrial_raw(), framework_id="bank")
    assert bank["by_id"]["cfo_conversion"]["applicable"] is False
    assert bank["by_id"]["cfo_conversion"]["value"] is None
    assert bank["invents_buy"] is False


def test_source_fallback_official_then_secondary_then_last_good():
    fact = resolve_fact(
        official=None,
        secondary={"value": 12.0, "source": "screener.in"},
        last_good={"value": 11.0, "source": "cache", "stale": True},
    )
    assert fact["value"] == 12.0
    assert fact["tier"] == "reputable_public_secondary"
    last = resolve_fact(official=None, secondary=None, last_good={"value": 11.0, "source": "cache"})
    assert last["value"] == 11.0
    assert last["stale"] is True
    assert last["source_label"] == "last_good_snapshot"
    missing = resolve_fact(official=None, secondary=None, last_good=None)
    assert missing["value"] is None


def test_source_conflict_keeps_official():
    moat = company_intelligence_moat(
        _industrial_raw(),
        framework_id="industrials",
        official={"pat": {"value": 22.0, "source": "nseindia.com"}},
        secondary={"pat": {"value": 30.0, "source": "screener.in"}},
    )
    assert moat["source_conflicts"]
    pref = moat["source_conflicts"][0]["preferred"]
    assert pref["value"] == 22.0


def test_stale_last_good_labelled():
    moat = company_intelligence_moat(
        {"profit_loss": [], "cash_flow": [], "balance_sheet": [], "shareholding": [], "quarterly_results": []},
        framework_id="industrials",
        last_good={"revenue": {"value": 100.0, "source": "cache", "stale": True}},
    )
    rev = moat["by_id"]["revenue"]
    assert rev["value"] == 100.0
    assert rev["provenance"]["stale"] is True


def test_management_promise_vs_actual():
    unknown = promise_vs_actual([{"metric": "sales", "promised": 120}], {"sales": None})
    assert unknown["status"] == "UNKNOWN"
    missed = promise_vs_actual([{"metric": "sales", "promised": 120}], {"sales": 80})
    assert missed["missed"] is True
    delivered = promise_vs_actual([{"metric": "sales", "promised": 100}], {"sales": 110})
    assert delivered["delivered"] is True


def test_cash_conversion_deterioration():
    moat = company_intelligence_moat(_industrial_raw(), framework_id="industrials")
    assert any(f["id"] == "cash_conversion_deterioration" for f in moat["flags"])
    assert moat["dd_effect"] in {"PENALIZE", "BLOCK"}


def test_capital_allocation_deterioration():
    moat = company_intelligence_moat(_industrial_raw(), framework_id="industrials")
    assert any(f["id"] == "capital_allocation_deterioration" for f in moat["flags"])


def test_missing_evidence_stays_neutral_unknown():
    moat = company_intelligence_moat({}, framework_id="industrials")
    assert moat["dd_effect"] == "NEUTRAL"
    debt = moat["by_id"]["debt"]
    assert debt["available"] is False
    assert debt["value"] is None  # not 0
    assert debt["status"] == "UNKNOWN"


def test_dd_can_downgrade_block():
    raw = _industrial_raw(shareholding=[
        _q("Promoters", **{"Mar 2025": 52, "Mar 2026": 47}),
        _q("Pledge", **{"Mar 2025": 5, "Mar 2026": 25}),
    ])
    moat = company_intelligence_moat(raw, framework_id="industrials")
    assert moat["dd_effect"] == "BLOCK"
    assert any(f["id"] == "pledging" for f in moat["flags"])


def test_dd_cannot_create_buy():
    moat = company_intelligence_moat(_industrial_raw(), framework_id="industrials")
    assert moat["invents_buy"] is False
    assert moat["cannot_create_buy"] is True
    empty = company_intelligence_moat({}, framework_id="industrials")
    assert empty["invents_buy"] is False
    assert empty["dd_effect"] != "BUY"
