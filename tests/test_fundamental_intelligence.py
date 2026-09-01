"""Fundamental Intelligence is a projection of measured due-diligence evidence only."""
from __future__ import annotations

from product.due_diligence.dashboard import attach_fundamental_intelligence
from product.due_diligence.fundamental_intelligence import build_fundamental_intelligence


def _finding(
    kpi_id: str,
    pillar: str,
    *,
    points: float | None,
    weight: float = 1.0,
    fact: str = "measured fact",
    importance: str = "decision",
):
    return {
        "id": kpi_id,
        "label": kpi_id.upper(),
        "pillar": pillar,
        "points": points,
        "weight": weight,
        "available": points is not None,
        "fact": fact if points is not None else "Data unavailable",
        "importance": importance,
    }


def _base_report() -> dict:
    return {
        "framework": {"id": "it", "label": "Software / IT Services"},
        "kpis": [
            _finding("sales", "growth", points=88, weight=2, fact="Sales grew YoY"),
            _finding("opm", "profitability", points=82, weight=2, fact="OPM improved"),
            _finding("attrition", "sector", points=72, weight=1, fact="Attrition stable"),
            _finding("promoter", "governance", points=62, weight=1, fact="Promoter holding stable"),
        ],
        "fundamental_quality": {
            "score": 79,
            "coverage_pct": 80.0,
            "score_coverage_pct": 80.0,
            "explain": "Measured sector KPIs",
            "breakdown": {
                "pillars": [
                    {
                        "id": "growth",
                        "label": "Growth",
                        "awarded": 16.0,
                        "max": 20.0,
                        "display": "16/20",
                        "explain": "Existing sector score bucket",
                    }
                ]
            },
        },
        "cash_flow_quality": {
            "applicable": True,
            "label": "Strong",
            "detail": "CFO conversion measured",
            "metrics": [
                {"id": "cfo_to_pat", "available": True, "fact": "CFO/PAT = 1.1x"},
                {"id": "fcf", "available": True, "fact": "FCF proxy positive"},
            ],
        },
        "balance_sheet_rules": {
            "label": "Adequate",
            "metrics": [
                {"id": "debt_equity", "available": True, "fact": "Debt/equity = 0.2x"},
                {"id": "interest_coverage", "available": True, "fact": "Interest coverage = 8x"},
            ],
        },
        "growth_quality": {
            "label": "Improving",
            "n_growth": 1,
            "n_profit": 1,
            "notes": ["Sales: accelerating versus year-ago."],
        },
        "red_flags": [],
        "extracted_guidance": [
            {"tone": "positive", "excerpt": "Management expects growth", "source": "concall"}
        ],
        # A current PE print is useful context but must not be converted into a
        # cheap/expensive score without a validated historical/peer rule.
        "valuation": [
            {"id": "pe", "available": True, "fact": "P/E: 42x", "interpretation": "Current snapshot"}
        ],
    }


def test_current_valuation_snapshot_is_not_invented_into_a_score():
    intelligence = build_fundamental_intelligence(_base_report())
    valuation = next(c for c in intelligence["components"] if c["id"] == "intelligence_valuation")
    assert valuation["score"] is None
    assert valuation["awarded"] is None
    assert "Valuation" in intelligence["missing_components"]
    assert intelligence["valuation_without_context_is_scored"] is False
    assert intelligence["score"] is not None  # enough other measured evidence remains


def test_commentary_alone_does_not_create_management_execution_score():
    report = _base_report()
    report["kpis"] = [f for f in report["kpis"] if f["pillar"] != "governance"]
    report["red_flags"] = []
    intelligence = build_fundamental_intelligence(report)
    governance = next(
        c for c in intelligence["components"]
        if c["id"] == "intelligence_management_governance"
    )
    assert governance["score"] is None
    assert intelligence["management_commentary_is_execution_score"] is False
    assert any("not yet scored" in line for line in governance["evidence"])


def test_insufficient_coverage_suppresses_overall_score_instead_of_guessing():
    report = {
        "framework": {"id": "generic", "label": "Generic"},
        "kpis": [],
        "fundamental_quality": {
            "score": 85,
            "coverage_pct": 20.0,
            "score_coverage_pct": 20.0,
            "explain": "Thin evidence",
        },
        "cash_flow_quality": {"applicable": True, "label": "Unmeasured", "metrics": []},
        "balance_sheet_rules": {"label": "Unmeasured", "metrics": []},
        "growth_quality": {"label": "Unmeasured", "n_growth": 0, "n_profit": 0},
        "red_flags": [],
        "valuation": [],
    }
    intelligence = build_fundamental_intelligence(report)
    assert intelligence["score"] is None
    assert intelligence["label"] == "INSUFFICIENT EVIDENCE"
    assert intelligence["coverage_pct"] < 50


def test_dashboard_projection_is_visible_and_idempotent():
    report = _base_report()
    first = attach_fundamental_intelligence(report)
    second = attach_fundamental_intelligence(report)
    assert first["score"] == second["score"]
    assert report["fundamental_intelligence"]["score"] == first["score"]

    breakdown = report["fundamental_quality"]["breakdown"]
    ids = [row["id"] for row in breakdown["pillars"]]
    assert ids[0] == "fundamental_intelligence_total"
    assert ids.count("fundamental_intelligence_total") == 1
    assert ids.count("intelligence_business_quality") == 1
    assert ids.count("growth") == 1  # original sector bucket preserved once
    assert breakdown["sector_pillars"][0]["id"] == "growth"
