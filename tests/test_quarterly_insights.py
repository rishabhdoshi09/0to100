"""Regression gates for the read-only Q2-style company review contract."""
from __future__ import annotations

import pytest

from product.due_diligence.quarterly_insights import compose_quarterly_insights


def row(
    period: str,
    published: str,
    facts: dict,
    *,
    symbol: str = "ABC",
    consolidated: bool = True,
    available: str | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "period_end": period,
        "publication_date": published,
        "available_from": available or published,
        "evidence_id": f"test-{period}-{published}",
        "source": "NSE company filing",
        "source_url": "https://example.test/filing",
        "parser_version": "test",
        "consolidated": consolidated,
        "facts": facts,
    }


def test_missing_quarters_never_become_a_positive_view():
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=[])
    assert out["status"] == "NO_VERIFIABLE_QUARTER"
    assert out["thesis_evidence"] == []
    assert out["anti_thesis_evidence"] == []
    assert out["decision"] == "RESEARCH_ONLY"
    assert out["places_orders"] is False
    assert out["conviction"]["win_probability"] is None
    assert out["price_track"]["status"] == "UNAVAILABLE"


def test_reject_future_publication_even_when_period_is_past():
    out = compose_quarterly_insights(
        "ABC",
        as_of="2026-11-08",
        rows=[row("2026-09-30", "2026-11-09", {"revenue": 200, "pat": 30})],
    )
    assert out["status"] == "NO_VERIFIABLE_QUARTER"
    assert out["source_documents"] == []


def test_reject_future_available_from_even_if_publication_is_past():
    out = compose_quarterly_insights(
        "ABC",
        as_of="2026-11-08",
        rows=[row("2026-09-30", "2026-11-02", {"revenue": 200}, available="2026-11-10")],
    )
    assert out["status"] == "NO_VERIFIABLE_QUARTER"


def test_only_same_quarter_last_year_counts_as_yoy():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 150, "pat": 30}),
        row("2026-06-30", "2026-08-01", {"revenue": 90, "pat": 10}),
        row("2025-09-30", "2025-11-02", {"revenue": 100, "pat": 20}),
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert out["status"] == "QUARTER_COMPARABLE"
    assert out["comparison_period_end"] == "2025-09-30"
    assert out["metrics"]["revenue"]["yoy_pct"] == 50
    assert out["metrics"]["pat"]["yoy_pct"] == 50
    assert len(out["thesis_evidence"]) == 2
    assert all(note["sources"][0]["evidence_id"] for note in out["thesis_evidence"])


def test_one_previous_quarter_is_not_mislabeled_yoy():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 150, "pat": 30}),
        row("2026-06-30", "2026-08-01", {"revenue": 100, "pat": 20}),
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert out["status"] == "PARTIAL_QUARTER_EVIDENCE"
    assert out["metrics"]["revenue"]["yoy_pct"] is None
    assert out["thesis_evidence"] == []
    assert "SAME_QUARTER_PRIOR_YEAR" in out["unknowns"]


def test_debt_increase_is_counter_case_for_industrials_not_nbfc():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 100, "pat": 15, "bs_total_debt": 200}),
        row("2025-09-30", "2025-11-01", {"revenue": 110, "pat": 18, "bs_total_debt": 100}),
    ]
    industrial = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data, framework_id="industrials")
    lending = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data, framework_id="nbfc")
    assert any(x["id"] == "debt_growth" for x in industrial["anti_thesis_evidence"])
    assert not any(x["id"] == "debt_growth" for x in lending["anti_thesis_evidence"])


def test_does_not_mix_standalone_with_consolidated_in_yoy():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 150, "pat": 30}, consolidated=True),
        row("2025-09-30", "2025-11-01", {"revenue": 100, "pat": 20}, consolidated=False),
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert out["status"] == "PARTIAL_QUARTER_EVIDENCE"
    assert out["metrics"]["revenue"]["yoy_pct"] is None


def test_malformed_and_non_finite_values_remain_unknown():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": float("nan"), "pat": float("inf")}),
        row("2025-09-30", "2025-11-01", {"revenue": 100, "pat": 20}),
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert out["thesis_evidence"] == []
    assert out["metrics"]["pat"]["current"] is None
    assert "PAT_YOY" in out["unknowns"]


def test_requires_provenance_and_matching_symbol():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 150}, symbol="XYZ"),
        {**row("2026-09-30", "2026-11-01", {"revenue": 100}), "evidence_id": ""},
        {**row("2026-09-30", "2026-11-01", {"revenue": 100}), "publication_date": ""},
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert out["status"] == "NO_VERIFIABLE_QUARTER"


@pytest.mark.parametrize("symbol,as_of", [
    ("", "2026-11-08"), ("BAD/SYMBOL", "2026-11-08"), ("ABC", "2026-11-08junk"),
    ("ABC", "not-a-date"),
])
def test_invalid_inputs_fail_explicitly(symbol, as_of):
    with pytest.raises(ValueError):
        compose_quarterly_insights(symbol, as_of=as_of, rows=[])


def test_declining_profit_and_negative_cash_flow_create_counter_case():
    data = [
        row("2026-09-30", "2026-11-01", {"revenue": 150, "pat": -5, "cfo": -20}),
        row("2025-09-30", "2025-11-01", {"revenue": 100, "pat": 10}),
    ]
    out = compose_quarterly_insights("ABC", as_of="2026-11-08", rows=data)
    assert {x["id"] for x in out["anti_thesis_evidence"]} >= {
        "pat_yoy", "negative_operating_cash_flow"
    }
    assert {x["id"] for x in out["thesis_evidence"]} == {"revenue_yoy"}
    assert out["places_orders"] is False


def test_real_pit_warehouse_wiring_is_temporal(tmp_path):
    """Exercise the production adapter, not only synthetic composer rows."""
    from product.pit_warehouse import DOC_QUARTERLY_RESULT, persist
    from product.due_diligence.quarterly_insights import quarterly_insights

    db = tmp_path / "financials.db"
    for period, pub, revenue, pat in (
        ("2025-09-30", "2025-11-01", 100, 20),
        ("2026-09-30", "2026-11-01", 150, 30),
    ):
        persist({
            "symbol": "ABC",
            "evidence_type": DOC_QUARTERLY_RESULT,
            "period_end": period,
            "publication_date": pub,
            "available_from": pub,
            "source": "NSE quarterly results",
            "source_url": "https://example.test/filing",
            "source_identity": "nse:" + period,
            "extracted": {
                "numbers_parsed": True,
                "nature": "Consolidated",
                "facts": {"revenue": revenue, "pat": pat},
            },
        }, path=db)

    early = quarterly_insights("ABC", as_of="2026-10-31", path=db)
    assert early["status"] == "PARTIAL_QUARTER_EVIDENCE"
    assert early["latest_period_end"] == "2025-09-30"
    assert early["metrics"]["revenue"]["current"] == 100

    current = quarterly_insights("ABC", as_of="2026-11-08", path=db)
    assert current["status"] == "QUARTER_COMPARABLE"
    assert current["metrics"]["revenue"]["yoy_pct"] == 50
    assert len(current["source_documents"]) == 2
    assert current["decision"] == "RESEARCH_ONLY"
