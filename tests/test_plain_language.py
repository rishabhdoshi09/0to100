"""Common-man presentation layer — labels stay friendly; schemas stay technical."""
from product.plain_language import (
    FIELD_LABELS,
    NAV_JOBS,
    PlainCard,
    beginner_ok,
    bulk_explain,
    explain_decision,
    explain_metric,
    explain_pit_state,
    explain_research_verdict,
    explain_trust_class,
    label_for,
    render_layers,
    research_report_blurb,
)
from product.projection import TERMINOLOGY


def test_internal_fields_are_not_renamed_in_map():
    # Presentation map must keep technical keys as keys (schemas unchanged).
    assert "network_concentration_score" in FIELD_LABELS
    assert FIELD_LABELS["network_concentration_score"] == "Portfolio overlap risk"
    assert FIELD_LABELS["evidence_level"] == "Research confidence"
    assert FIELD_LABELS["pit_state"] == "Historical data quality"
    assert FIELD_LABELS["regime"] == "Market condition"


def test_trust_class_display_only_is_human():
    card = explain_trust_class("DISPLAY_ONLY")
    assert "charts and exploration" in card.explanation.lower()
    assert "not for proving" in card.explanation.lower()
    assert "DISPLAY_ONLY" in card.technical
    assert card.internal_value == "DISPLAY_ONLY"
    assert beginner_ok(card)


def test_not_pit_safe_and_research_grade():
    bad = explain_pit_state("NOT_PIT_SAFE")
    assert "not actually known" in bad.explanation.lower()
    good = explain_trust_class("RESEARCH_GRADE")
    assert "scientific historical testing" in good.explanation.lower()


def test_inconclusive_research_verdict_plain():
    card = explain_research_verdict("INCONCLUSIVE")
    assert "unproven" in card.explanation.lower()
    assert card.state == "UNPROVEN"
    blurb = research_report_blurb("INCONCLUSIVE", trust_class="DISPLAY_ONLY")
    assert "unproven" in blurb.lower()
    assert "charts and exploration" in blurb.lower()


def test_fail_verdict_says_no_real_trades():
    card = explain_research_verdict("FAIL")
    assert "will not use it for real trades" in card.explanation.lower()
    assert card.state == "FAILED"


def test_decision_watch_has_layers():
    card = explain_decision(
        "WATCH",
        why="The stock itself is strong, but portfolio overlap is high.",
        positives=["Strong price trend", "Strong sector"],
        negatives=["Portfolio overlap is high", "Market condition is uncertain"],
    )
    layers = render_layers(card)
    assert layers["layer1"]["label"].startswith("Decision:")
    assert "overlap" in layers["layer1"]["explanation"].lower()
    assert layers["layer2"]["technical_details"]
    assert beginner_ok(card)


def test_metric_never_naked_number():
    card = explain_metric(
        "betweenness_centrality",
        0.71,
        higher_is_better=False,
        good_at=0.2,
        caution_at=0.4,
    )
    assert card.label == "Portfolio dependency"
    assert "0.71" in card.technical
    assert card.explanation  # meaning, not just the number
    assert card.state in {"GOOD", "CAUTION", "RISKY", "NOT_ENOUGH_DATA"}


def test_bulk_explain_two_layers():
    out = bulk_explain({
        "trust_class": "DISPLAY_ONLY",
        "verdict": "INCONCLUSIVE",
        "network_concentration_score": 0.62,
    })
    assert out["trust_class"]["layer1"]["state"]
    assert "DISPLAY_ONLY" in out["trust_class"]["layer2"]["technical_details"]
    assert out["verdict"]["layer1"]["explanation"]
    assert out["network_concentration_score"]["layer1"]["label"] == "Portfolio overlap risk"


def test_nav_jobs_are_plain():
    assert NAV_JOBS["find_stocks"] == "Find Stocks"
    assert "Feature Store" not in NAV_JOBS.values()
    assert "Gauntlet" not in NAV_JOBS.values()


def test_projection_terminology_includes_common_man_map():
    assert TERMINOLOGY["regime"] == "Market condition"
    assert TERMINOLOGY["snapshot"] == "Saved market data"
    assert label_for("expected_value") == "How much can we expect?"


def test_plain_card_roundtrip_dict():
    card = PlainCard(label="x", explanation="y", technical="z")
    d = card.as_dict()
    assert d["label"] == "x" and d["technical"] == "z"
