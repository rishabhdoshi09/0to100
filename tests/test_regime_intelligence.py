"""Phase 4 — Regime Intelligence 2.0 (shadow)."""

from __future__ import annotations

from product.paper_autopilot import INVALID_STOP, evaluate_candidate
from product.regime_intelligence import (
    BROAD_RISK_ON,
    NARROW_LEADERSHIP,
    RISK_OFF,
    UNKNOWN,
    VOLATILE_RISK_ON,
    RegimeIntelligenceEngine,
    shadow_classify,
)
from research.auto_research.paper_book import PaperBook


def _evidence(**over):
    base = {
        "breadth": {"value": 72, "source": "advance_decline_scan", "confidence": "high"},
        "pct_above_ma": {"value": 68, "source": "index_members", "confidence": "high"},
        "index_trend": {"value": 0.8, "source": "nifty_slope", "confidence": "high"},
        "volatility": {"value": 0.3, "source": "india_vix", "confidence": "high"},
        "leadership_concentration": {"value": 0.3, "source": "sector_weights", "confidence": "medium"},
        "breakout_failure_rate": {"value": 0.2, "source": "setup_stats", "confidence": "medium"},
    }
    base.update(over)
    return base


def test_same_setup_behaves_differently_by_regime_policy():
    eng = RegimeIntelligenceEngine(shadow_mode=True)
    on = eng.classify(_evidence())
    off = eng.classify(_evidence(
        index_trend={"value": -0.8, "source": "nifty_slope"},
        breadth={"value": 25, "source": "ad"},
        advance_decline={"value": -0.4, "source": "ad"},
    ))
    assert on["state"] == BROAD_RISK_ON
    assert off["state"] == RISK_OFF
    assert on["policy"]["max_new"] != off["policy"]["max_new"]
    assert on["policy"]["enter_bias"] != off["policy"]["enter_bias"]
    assert on["policy"]["hard_gates_intact"] is True
    assert off["policy"]["hard_gates_intact"] is True


def test_insufficient_regime_evidence_stays_unknown():
    out = shadow_classify({"volatility": {"value": 0.4, "source": "vix"}})
    assert out["state"] == UNKNOWN
    assert out["n_measured"] < 3
    assert out["affects_production"] is False


def test_one_bad_week_cannot_change_production():
    eng = RegimeIntelligenceEngine(shadow_mode=False, promoted=True)
    out = eng.classify(
        _evidence(
            index_trend={"value": -0.9, "source": "nifty"},
            breadth={"value": 20, "source": "ad"},
            advance_decline={"value": -0.5, "source": "ad"},
        ),
        production_regime="RISK_ON",
        weeks_of_agreement=1,
        last_week_only=True,
    )
    assert out["state"] == RISK_OFF
    assert out["affects_production"] is False
    assert out["production_regime"] == "RISK_ON"


def test_hard_gates_remain_intact():
    book = PaperBook(capital=100_000)
    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry": 100,
        "stop": 101,
        "target": 120,
        "entry_state": "ready",
        "volume_ratio": 1.4,
    }
    d = evaluate_candidate(card, book=book, regime="RISK_ON")
    assert d.reason_code == INVALID_STOP
    shadow = shadow_classify(_evidence(), production_regime="RISK_ON")
    assert shadow["hard_gates_intact"] is True
    assert shadow["affects_production"] is False


def test_shadow_regime_cannot_affect_production_before_promotion():
    out = shadow_classify(_evidence(
        leadership_concentration={"value": 0.8, "source": "w"},
        breadth={"value": 30, "source": "b"},
        index_trend={"value": 0.5, "source": "t"},
        volatility={"value": 0.2, "source": "v"},
    ), production_regime="RISK_ON")
    assert out["state"] == NARROW_LEADERSHIP
    assert out["shadow_mode"] is True
    assert out["promoted"] is False
    assert out["affects_production"] is False
    assert out["policy"]["applies_to_production"] is False


def test_volatile_risk_on_labelled():
    out = shadow_classify(_evidence(
        volatility={"value": 0.85, "source": "vix"},
        index_trend={"value": 0.6, "source": "t"},
        breadth={"value": 55, "source": "b"},
    ))
    assert out["state"] == VOLATILE_RISK_ON
