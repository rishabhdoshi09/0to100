"""Decision Memory honesty: empty is empty; n<30 never saved-money / well-calibrated."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_journal(tmp_path, monkeypatch):
    import core.decision_journal as dj
    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "dec.db"))


def test_setup_quality_is_explicitly_not_a_probability():
    from product.decision_memory import setup_quality

    got = setup_quality({"conviction_score": 87, "score": 80})
    assert got["score"] == 87
    assert got["label"] == "Setup Quality"
    assert got["not_probability"] is True
    assert "not the chance" in got["blurb"].lower()


def test_empty_shadow_and_trust_do_not_invent_sample_size(monkeypatch):
    from product import decision_memory as dm

    monkeypatch.setattr(dm, "shadow_book", dm.shadow_book)
    monkeypatch.setattr(
        "research.calibration.calibration_report",
        lambda: {"n": 0},
    )
    monkeypatch.setattr(
        "core.decision_journal.decision_report",
        lambda min_n=1: {
            "taken": {"n": 0, "avg_outcome_pct": 0, "win_rate": 0},
            "rejected": {"n": 0, "avg_outcome_pct": 0, "win_rate": 0},
            "wait": {"n": 0},
        },
    )
    monkeypatch.setattr("research.counterfactual._load_decisions", lambda: ({}, []))
    shadow = dm.shadow_book()
    trust = dm.trust_score()
    assert shadow["proven"] is False
    assert shadow["taken"]["avg_r"] is None
    assert shadow["rejected"]["avg_r"] is None
    assert shadow["taken"]["n"] == 0
    assert "18" not in shadow["line"]
    assert trust["n"] == 0
    assert trust["status"] == "unmeasured"
    assert "94" not in trust["line"]
    strip = dm.morning_strip()
    assert strip["places_orders"] is False
    assert strip["shadow"]["proven"] is False


def test_unproven_why_not_does_not_claim_the_rule_earned_its_keep(monkeypatch):
    from product.decision_memory import why_not

    monkeypatch.setattr(
        "research.explainability.explain_rejection",
        lambda symbol: {
            "found": True,
            "reason": "EXTENSION",
            "label": "Extension Guard",
            "n_observations": 18,
            "avg_fwd_pct": -2.3,
            "verdict": "EARNING",
            "summary": "Extension Guard earning its keep.",
        },
    )
    got = why_not("TATAELXSI")
    assert got["found"] is True
    assert got["n_observations"] == 18
    assert got["avg_fwd_pct"] is None
    assert got["verdict"] == "unproven"
    assert "not proven" in got["line"].lower()
    assert "earning its keep" not in got["line"].lower()


def test_trust_below_thirty_is_unproven_not_well_calibrated(monkeypatch):
    from product.decision_memory import trust_score

    monkeypatch.setattr(
        "research.calibration.calibration_report",
        lambda: {"n": 18, "mean_pred": 0.61, "mean_obs": 0.59, "ece": 0.02},
    )
    got = trust_score()
    assert got["n"] == 18
    assert got["status"] == "unproven"
    assert got["status"] != "well_calibrated"
    assert "no accuracy headline" in got["line"].lower()


def test_stance_and_attach_do_not_place_orders():
    from product.decision_memory import attach_to_case, stance_for_row

    assert stance_for_row({"verdict": "BUY"}) == "YES"
    assert stance_for_row({"chase_risk": True, "verdict": "WATCH"}) == "NO"
    assert stance_for_row({"verdict": "WATCH"}) == "WAIT"
    case = attach_to_case(
        {"symbol": "HAL", "setup": "MOMENTUM", "n_similar": 0},
        row={"verdict": "BUY", "score": 80},
    )
    assert case["stance"] == "YES"
    assert case["setup_quality"]["not_probability"] is True
    assert case["places_orders"] is False
