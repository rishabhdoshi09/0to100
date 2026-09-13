"""WHY THIS DECISION must answer for any symbol, not only shortlisted ones.

The page used to return a single dead sentence — "not in the last saved scan"
— for every name outside the recommendations shortlist. That collapsed three
materially different answers into one: evaluated-and-passed-over,
never-evaluated, and not-a-real-symbol.
"""
from __future__ import annotations

import product.decision_service as DS

SCAN = {
    "records": [{
        "symbol": "EVALD", "status": "Watch", "verdict": "WATCH", "score": 61.0,
        "signals": ["PRE_BREAKOUT"], "reasons": ["volume dry-up"],
        "why": "coiling under resistance",
        "price": 250.0, "entry": 256.0, "stop": 240.0, "target": 300.0,
        "reward_risk": 2.75, "upside_pct": 17.19, "downside_pct": 6.25,
        "plan_complete": True, "plan_missing": [],
    }],
    "provenance": {"market_session_date": "2026-09-11", "data_current": True},
}


def _patch(monkeypatch, scan, universe):
    monkeypatch.setattr(DS, "_scan_membership",
                        lambda sym: (next((r for r in scan.get("records", [])
                                           if r["symbol"] == sym), None),
                                     scan.get("provenance", {})))
    monkeypatch.setattr(DS, "_is_known_equity", lambda sym: sym in universe)


def test_evaluated_but_unshortlisted_reports_real_evidence(monkeypatch):
    _patch(monkeypatch, SCAN, {"EVALD"})
    out = DS._unshortlisted_view("EVALD")
    assert out["stance"] == "NOT_SHORTLISTED"
    assert out["in_latest_scan"] is True
    assert out["score"] == 61.0
    assert out["signals"] == ["PRE_BREAKOUT"]
    assert out["levels"]["entry"] == 256.0
    assert out["levels"]["reward_risk"] == 2.75
    assert out["provenance"]["market_session_date"] == "2026-09-11"


def test_known_symbol_absent_from_scan_is_not_evaluated(monkeypatch):
    _patch(monkeypatch, SCAN, {"EVALD", "QUIET"})
    out = DS._unshortlisted_view("QUIET")
    assert out["stance"] == "NOT_EVALUATED"
    assert out["in_latest_scan"] is False
    assert out["known_equity"] is True


def test_unknown_symbol_is_named_as_untradable(monkeypatch):
    _patch(monkeypatch, SCAN, {"EVALD"})
    out = DS._unshortlisted_view("NOTREAL")
    assert out["stance"] == "NOT_A_TRADABLE_SYMBOL"
    assert out["known_equity"] is False


def test_unreadable_universe_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(DS, "_scan_membership", lambda sym: (None, {}))
    monkeypatch.setattr(DS, "_is_known_equity", lambda sym: None)
    out = DS._unshortlisted_view("ANY")
    assert out["known_equity"] is None
    assert "could not be read" in out["reason"]


def test_every_branch_carries_a_stance_and_reason(monkeypatch):
    _patch(monkeypatch, SCAN, {"EVALD", "QUIET"})
    for sym in ("EVALD", "QUIET", "NOTREAL"):
        out = DS._unshortlisted_view(sym)
        assert out["stance"], sym
        assert out["reason"], sym
        assert out["symbol"] == sym
