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


def test_live_no_explains_current_gate_without_inventing_history(monkeypatch):
    from product.decision_memory import why_not

    monkeypatch.setattr(
        "research.explainability.explain_rejection",
        lambda symbol: {"found": False, "summary": f"No recorded rejection for {symbol}."},
    )
    got = why_not("BLUSPRING", row={"chase_risk": True, "verdict": "WATCH"})
    assert got["found"] is True
    assert got["reason"] == "EXTENSION"
    assert got["n_observations"] == 0
    assert got["avg_fwd_pct"] is None
    assert "will not" in got["line"].lower()
    assert "extension" in got["line"].lower()


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


def test_two_rejection_reasons_count_once_in_shadow_book(tmp_path, monkeypatch):
    import core.decision_journal as dj
    from product.decision_memory import shadow_book

    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "dec.db"))
    for reason in ("EXTENSION", "WEAK_CLOSE"):
        dj.log_decision("BLUSPRING", "REJECTED", reason, "scanner", 100, 96, 40,
                        p_win=55.0)
    c = dj._conn()
    c.execute("UPDATE decisions SET decided_at='2026-08-01T10:00:00', "
              "outcome_pct=-2.0, outcome_price=98.0, outcome_checked_at='2026-08-10'")
    c.commit(); c.close()
    rep = dj.decision_report(min_n=1)
    assert rep["rejected"]["n"] == 1
    assert rep["rejected"]["unit"] == "opportunity"
    assert rep["by_reason"]["EXTENSION"]["n"] == 1
    assert rep["by_reason"]["WEAK_CLOSE"]["n"] == 1
    cal = dj.calibration_report(min_n=1)
    scored = [b for b in cal["buckets"] if b["predicted"] is not None]
    assert scored and scored[0]["n"] == 1
    shadow = shadow_book(min_n=30)
    assert shadow["rejected"]["n"] == 1
    assert shadow["proven"] is False


def test_wait_is_timing_not_a_second_no(tmp_path, monkeypatch):
    import pandas as pd
    import core.decision_journal as dj
    from research.counterfactual import _load_decisions

    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "dec.db"))
    dj.log_decision("TRENT", "WAIT", "", "scanner", 100, 96, 55)
    c = dj._conn()
    c.execute("UPDATE decisions SET decided_at='2026-08-01T10:00:00'")
    c.commit(); c.close()
    idx = pd.bdate_range("2026-08-01", periods=16)
    highs = [101] + [103] * 15
    lows = [99] + [100] * 15
    closes = [100] + [102] * 15
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: df)
    assert dj.update_outcomes(lookback_days=400) == 1
    c = dj._conn()
    row = c.execute("SELECT wait_result, decision FROM decisions").fetchone()
    c.close()
    assert row["decision"] == "WAIT"
    assert row["wait_result"] == "RAN_AWAY"
    rejected, taken = _load_decisions()
    assert taken == []
    assert rejected == {}
    rep = dj.decision_report(min_n=1)
    assert rep["wait"]["n"] == 1
    assert rep["wait"]["timing"]["RAN_AWAY"] == 1
    assert rep["rejected"]["n"] == 0
