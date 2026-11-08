"""Safety and source-lineage tests for read-only quarterly briefs."""
from __future__ import annotations

import math

from product.due_diligence.quarterly_brief import check_price_track, compose_quarterly_brief


def _report() -> dict:
    return {
        "symbol": "TEST",
        "point_in_time": False,
        "as_of": {"generated_at": "2026-11-08T12:00:00+05:30"},
        "decision_coverage": {"coverage_pct": 80},
        "kpis": [
            {
                "id": "revenue", "label": "Revenue", "available": True,
                "trend": "improving", "source": "Quarterly results",
                "source_url": "https://example.org/result", "source_date": "2026-11-04",
                "snapshot": {"current": 1211, "current_period": "Sep 2026"},
                "fact": "Revenue increased",
            },
            {
                "id": "margin", "label": "EBITDA margin", "available": True,
                "trend": "deteriorating", "source": "Company filing",
                "source_date": "2026-11-05",
                "snapshot": {"current": 10.3, "current_period": "Sep 2026"},
            },
        ],
        "red_flags": [
            {
                "id": "leverage", "title": "Rising leverage",
                "severity": "critical", "source_date": "2026-11-04",
                "source": "Quarterly filing", "evidence": "Debt rose",
            },
        ],
        "missing_evidence": [{"id": "cash", "label": "Cash flow"}],
        "watch_next": ["Check next quarter cash conversion"],
    }


def test_price_track_recomputes_not_copies_reported_upside():
    # Prices and displayed percentage disagree; the engine must flag it.
    result = check_price_track(
        buy=1085, cmp=970, target=1251,
        reported_vs_buy_pct=-10.6, reported_to_target_pct=25.9,
        price_as_of="2026-11-07",
    )
    assert result["computed_vs_buy_pct"] == -10.6
    assert result["computed_to_target_pct"] == 28.97
    assert not result["arithmetic_valid"]
    assert [item["code"] for item in result["issues"]] == ["RETURN_MISMATCH"]
    assert result["source_verified"] is False
    assert result["not_a_trade_signal"] is True


def test_missing_and_invalid_prices_do_not_invent_returns():
    missing = check_price_track(buy=100, cmp=None, target=110)
    assert missing["computed_to_target_pct"] is None
    assert not missing["arithmetic_valid"]
    bad = check_price_track(buy=0, cmp=math.inf, target=-10,
                            price_as_of="2026-11-07")
    assert bad["computed_to_target_pct"] is None
    assert not bad["arithmetic_valid"]


def test_brief_provides_three_sides_and_never_trading_authority():
    brief = compose_quarterly_brief(_report())
    assert brief["review_status"] == "RISK_REVIEW"
    assert [x["id"] for x in brief["thesis"]] == ["revenue"]
    assert any(x["id"] == "margin" for x in brief["anti_thesis"])
    assert any(x["id"] == "leverage" for x in brief["anti_thesis"])
    assert len(brief["quarterly_evidence"]) == 2
    assert brief["quarterly_evidence"][0]["source_date"] == "2026-11-04"
    assert brief["watch_next"] == ["Check next quarter cash conversion"]
    assert brief["missing_evidence"] == ["Cash flow"]
    assert brief["execution_authority"] is False
    assert brief["not_a_buy_sell_signal"] is True


def test_no_facts_is_not_a_favorable_thesis():
    base = _report()
    base["kpis"] = []
    base["red_flags"] = []
    base["decision_coverage"] = {"coverage_pct": 0}
    brief = compose_quarterly_brief(base)
    assert brief["review_status"] == "EVIDENCE_INCOMPLETE"
    assert brief["thesis"] == []
    assert brief["anti_thesis"] == []
    assert brief["quarterly_evidence"] == []


def test_future_dated_kpis_and_risk_flags_are_withheld():
    base = _report()
    base["point_in_time"] = True
    base["pit_as_of"] = "2026-10-31"
    brief = compose_quarterly_brief(base)
    assert brief["evidence_mode"] == "POINT_IN_TIME_INPUT"
    assert brief["quarterly_evidence"] == []
    assert brief["risk_flags"] == []
    assert len(brief["quality_warnings"]) == 3
    assert brief["review_status"] == "EVIDENCE_INCOMPLETE"


def test_missing_source_is_labelled_and_zero_is_not_missing():
    base = _report()
    base["kpis"][0]["source"] = ""
    base["kpis"][0]["snapshot"]["current"] = 0
    base["kpis"][0]["source_date"] = ""
    brief = compose_quarterly_brief(base)
    fact = brief["quarterly_evidence"][0]
    assert fact["value"] == 0
    assert fact["source_status"] == "SOURCE_MISSING"
    assert fact["source"] == "Source unavailable"


def test_report_input_is_not_modified_and_document_prices_are_opt_in():
    base = _report()
    before = base["kpis"][0]["snapshot"].copy()
    brief = compose_quarterly_brief(base, price_track={
        "buy": 199, "cmp": 144, "target": 223,
        "reported_to_target_pct": 39.5,
        "price_as_of": "2026-11-07",
    })
    assert before == base["kpis"][0]["snapshot"]
    assert brief["price_arithmetic"]["computed_to_target_pct"] == 54.86
    assert any(x["code"] == "RETURN_MISMATCH" for x in brief["price_arithmetic"]["issues"])


def test_canonical_investigator_includes_brief_without_execution():
    """The user's existing Investigate action must return the new projection."""
    from datetime import datetime, timezone
    from product.due_diligence.engine import build_due_diligence

    raw = {
        "available": True,
        "fetched_at": "2026-11-05T08:00:00+00:00",
        "freshness": "FRESH",
        "data": {
            "url": "https://www.screener.in/company/TESTIT/",
            "about": "Test IT Ltd provides financial software services.",
            "quarterly_results": [
                {"row_label": "Sales+", "Sep 2025": 100, "Sep 2026": 132},
                {"row_label": "Net Profit+", "Sep 2025": 10, "Sep 2026": 16},
                {"row_label": "OPM %", "Sep 2025": 20, "Sep 2026": 24},
            ],
            "shareholding": [],
            "cash_flow": [],
        },
    }
    report = build_due_diligence(
        "TESTIT", raw_fundamentals=raw, news=[],
        scan_payload={"records": []},
        long_term_payload={"records": []},
        now=datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc),
    )
    assert report["engine"] == "StockResearchEngine"
    assert report["places_orders"] is False
    assert report["quarterly_brief"]["symbol"] == "TESTIT"
    assert report["quarterly_brief"]["execution_authority"] is False
    assert report["quarterly_brief"]["not_a_buy_sell_signal"] is True
