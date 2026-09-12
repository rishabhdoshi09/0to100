"""The screen that answers "why this decision?" must be checkable.

The failure mode being designed out: a fluent paragraph that reads equally
confident whether the system knew a great deal or nothing at all. Every section
here is either filled from the decision record or says plainly that it is not,
and nothing in the payload comes from a language model.
"""
from __future__ import annotations

import pytest

from product.decision import (
    AVOID,
    BUY,
    CONFLICTING,
    Decision,
    EvidenceItem,
    MISSING,
    SUPPORTING,
)
from product.decision_explanation import (
    AUTHORITY,
    UNAVAILABLE,
    explain,
    missing_sections,
)

EXPECTED_SECTIONS = [
    "Supporting evidence",
    "Conflicting evidence",
    "Missing evidence",
    "Market context",
    "Sector context",
    "Setup",
    "Risk",
    "Entry",
    "Stop",
    "Target",
    "Invalidation",
    "Expected value",
    "Portfolio effect",
    "Data freshness",
    "Data sources",
]


def _rich() -> Decision:
    return Decision(
        symbol="INFY", state=BUY, setup="VCP", score=82.0,
        calibrated_confidence=0.66, expected_value=0.31,
        market_state="HEALTHY", sector_state="LEADING",
        entry=100.0, stop=95.0, target=115.0,
        chase_risk="AT_PIVOT", liquidity_state="LIQUID",
        portfolio_effect={"sector_exposure_after_pct": 18.0},
        position_size={"qty": 40},
        invalidation_conditions=("closes below 95", "volume dries up"),
        source_scan_id="scan-1", evidence_snapshot_id="snap-1",
        evidence_class="PAPER_FORWARD",
        decision_engine_version="e1", feature_schema_version="f1",
        strategy_version="s1",
    ).with_evidence(
        supporting=[
            EvidenceItem(id="vol", label="Volume expansion", source="NSE bhavcopy",
                         as_of="2026-09-11", evidence_class="PAPER_FORWARD"),
            EvidenceItem(id="rs", label="Relative strength", source="computed",
                         as_of="2026-09-09"),
        ],
        conflicting=[
            EvidenceItem(id="rsi", label="RSI 74", direction=CONFLICTING,
                         source="computed", as_of="2026-09-11"),
        ],
        missing=[
            EvidenceItem(id="funds", label="No fundamentals on file",
                         direction=MISSING),
        ],
    )


def test_every_section_is_present_and_in_order():
    payload = explain(_rich())
    assert [s["title"] for s in payload["sections"]] == EXPECTED_SECTIONS


def test_an_empty_decision_still_renders_every_section():
    """A blank screen hides the gaps; a screen full of 'not available' shows them."""
    payload = explain(Decision(symbol="AAA", state=AVOID))
    assert [s["title"] for s in payload["sections"]] == EXPECTED_SECTIONS
    gaps = missing_sections(payload)
    assert "Setup" in gaps
    assert "Entry" in gaps
    assert "Supporting evidence" in gaps
    for section in payload["sections"]:
        if not section["available"]:
            assert section["text"] == UNAVAILABLE


def test_conflicting_evidence_is_surfaced_not_buried():
    payload = explain(_rich())
    conflicting = next(s for s in payload["sections"]
                       if s["title"] == "Conflicting evidence")
    assert conflicting["available"]
    assert conflicting["items"][0]["label"] == "RSI 74"


def test_missing_evidence_is_a_visible_answer():
    payload = explain(_rich())
    missing = next(s for s in payload["sections"] if s["title"] == "Missing evidence")
    assert missing["available"]
    assert missing["items"][0]["label"] == "No fundamentals on file"


def test_freshness_reports_the_stalest_input_not_the_newest():
    """A decision is exactly as fresh as the oldest thing it stands on."""
    payload = explain(_rich())
    freshness = next(s for s in payload["sections"]
                     if s["title"] == "Data freshness")["fields"]
    assert freshness["oldest_input_as_of"] == "2026-09-09"
    assert freshness["oldest_input_label"] == "Relative strength"


def test_sources_are_attributed_per_fact():
    payload = explain(_rich())
    sources = next(s for s in payload["sections"] if s["title"] == "Data sources")
    names = [item["label"] for item in sources["items"]]
    assert "NSE bhavcopy" in names
    assert "computed" in names
    unattributed = next(i for i in sources["items"] if i["label"] == "unattributed")
    assert "No fundamentals on file" in unattributed["detail"]


def test_risk_shows_the_R_implied_by_the_levels():
    """Published R and R implied by entry/stop/target are shown side by side."""
    risk = next(s for s in explain(_rich())["sections"]
                if s["title"] == "Risk")["fields"]
    assert risk["risk_per_share"] == pytest.approx(5.0)
    assert risk["expected_R_from_levels"] == pytest.approx(3.0)


def test_the_headline_is_assembled_from_counts_not_adjectives():
    headline = explain(_rich())["headline"]
    assert "INFY: BUY on VCP" in headline
    assert "2 supporting, 1 conflicting, 1 missing" in headline
    for word in ("strong", "excellent", "great", "compelling", "high-quality"):
        assert word not in headline.lower()


def test_the_payload_declares_itself_deterministic():
    payload = explain(_rich())
    assert payload["authority"] == AUTHORITY
    assert "never in place of it" in payload["note"]


def test_the_explanation_is_a_pure_function_of_the_decision():
    decision = _rich()
    assert explain(decision) == explain(decision)


def test_versions_travel_with_the_explanation():
    versions = explain(_rich())["versions"]
    assert versions["decision_engine_version"] == "e1"
    assert versions["feature_schema_version"] == "f1"
    assert versions["strategy_version"] == "s1"
