"""Phase 4 — decision integrity, scoped census, opportunity memory."""
from __future__ import annotations

import json
from pathlib import Path

from product import decision_taxonomy as T
from product.counterfactual_learning import MISSED_WINNER
from product.decision_committee import evaluate_committee
from product.judgment_census import build_census
from product.missed_winner import analyze_decision_quality
from product.operator_metrics import _classify
from product.paper_autopilot import ENTER_NOW, WAIT
from product.opportunity_memory import events_for, get as mem_get, next_session_set, remember
from research.autonomy import health as H
from research.autonomy import supervisor_state as ST


class _Paper:
    def __init__(self, decision, reason_code, detail=""):
        self.decision = decision
        self.reason_code = reason_code
        self.detail = detail

    def as_dict(self):
        return {"decision": self.decision, "reason_code": self.reason_code}


def _card(**extra):
    row = {
        "symbol": "GUFICBIO",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 442,
        "stop": 403.6,
        "target": 518.8,
        "families": [
            {"id": "structure", "status": "pass"},
            {"id": "price_leadership", "status": "pass"},
            {"id": "business_quality", "status": "pass"},
        ],
        "methods": [
            {"id": "sepa", "label": "SEPA", "status": "pass"},
            {"id": "trend", "label": "Trend", "status": "pass"},
            {"id": "conviction", "label": "Conviction", "status": "pass"},
        ],
    }
    row.update(extra)
    return row


def test_ready_requires_research_not_just_enter_now(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    rec = evaluate_committee(_card(), broker_ok=False, entry_window=False, load_research=False)
    assert rec.decision == T.WAIT_DECISION
    assert rec.candidate_state == T.WAIT_EVIDENCE
    assert rec.reason_code == T.INSUFFICIENT_EVIDENCE
    assert rec.execution_state == T.NOT_APPLICABLE
    assert rec.information_value == "HIGH"


def test_committee_buy_keeps_broker_on_execution(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    monkeypatch.setattr(
        "product.decision_committee._research_snapshot",
        lambda *a, **k: {
            "available": True,
            "acquired_at": "2026-09-03T00:00:00+00:00",
            "coverage_pct": 86.4,
            "missing_critical": [],
            "framework_id": "pharma_formulations",
            "quality_label": "healthy",
        },
    )
    rec = evaluate_committee(_card(), broker_ok=False, entry_window=False)
    assert rec.decision == T.BUY
    assert rec.candidate_state == T.READY
    assert rec.entry_state == T.ENTER_NOW
    assert rec.execution_state == T.BLOCKED_BROKER_AUTH
    assert rec.reason_code == "COMMITTEE_BUY"
    assert rec.reason_code != T.BROKER_AUTH_REQUIRED


def test_extended_wait_has_machine_trigger(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(WAIT, T.ENTRY_TOO_EXTENDED, "11.4% above risk point"),
    )
    rec = evaluate_committee(
        _card(entry_state="extended", chase_risk=True, buy_zone_high=400),
        broker_ok=False, load_research=False,
    )
    assert rec.decision == T.WAIT_DECISION
    assert rec.candidate_state == T.WAIT
    assert rec.wait_trigger.get("kind") == "PRICE_LTE"
    assert rec.wait_trigger.get("price") == 400
    assert "price <=" in rec.wait_trigger.get("reconsider_when", "")


def test_census_is_monotonic_and_scoped():
    census = build_census(
        scan={
            "coverage": {
                "requested": 10232,
                "checked": 2419,
                "qualified": 713,
                "reason_counts": {"NO_OHLCV": 7770, "LOW_LIQUIDITY": 573, "NO_SETUP": 863},
            },
            "summary": {"extended": 279, "qualified": 713},
        },
        reco={
            "ensemble": {
                "high_conviction_count": 8,
                "good_setup_count": 42,
                "watch_count": 49,
            },
            "categories": [
                {"id": "high_conviction", "cards": [{"symbol": "AAA"}, {"symbol": "BBB"}]},
                {"id": "good_setup", "cards": [{"symbol": "CCC"}, {"symbol": "AAA"}]},
            ],
        },
        committee=[
            {"symbol": "AAA", "decision": "BUY", "candidate_state": "READY", "tier": "high_conviction", "vetoes": [], "disagreement": False},
            {"symbol": "BBB", "decision": "WAIT", "candidate_state": "WAIT", "tier": "high_conviction", "vetoes": [{"code": "ENTRY_TOO_EXTENDED"}], "disagreement": True},
            {"symbol": "CCC", "decision": "AVOID", "candidate_state": "REJECTED", "tier": "good_setup", "vetoes": [], "disagreement": False},
            {"symbol": "OLD", "decision": "WAIT", "candidate_state": "WAIT", "tier": "good_setup", "vetoes": [], "disagreement": False},
        ],
        session="2026-09-02",
        scan_run_id="scan-1",
        generated_at="2026-09-03T00:00:00+00:00",
        researched_symbols=["AAA"],
    )
    assert census["scope"]["kind"] == "CURRENT_SCAN"
    assert census["monotonic_funnel"] is True
    ids = [row["id"] for row in census["funnel"]]
    assert ids == [
        "RAW_INSTRUMENTS", "ELIGIBLE", "SETUP_CANDIDATES", "RECOMMENDATION_SHORTLIST",
        "COMMITTEE", "SERIOUS_CANDIDATES", "BUY", "READY",
    ]
    assert census["funnel"][3]["n"] == 3  # unique AAA,BBB,CCC
    assert census["funnel"][4]["n"] <= 3  # remembered OLD does not inflate funnel
    assert census["overlapping_diagnostics"]["reason_counts"]["NO_OHLCV"] == 7770
    assert census["side_paths"]["deep_research"]["n"] == 1
    ns = [row["n"] for row in census["funnel"]]
    assert ns == sorted(ns, reverse=True)


def test_opportunity_memory_same_symbol_across_sessions(tmp_path):
    db = tmp_path / "mem.db"
    for symbol, session, state, reason in (
        ("AAA", "2026-08-28", "WAIT", "ENTRY_TOO_EXTENDED"),
        ("BBB", "2026-08-28", "WAIT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
        ("CCC", "2026-08-29", "WATCH", "NEAR_SETUP"),
    ):
        remember(
            symbol=symbol, session_date=session, state=state, reason=reason,
            scan_run_id="s1", decision="WAIT", entry_state="EXTENDED",
            wait_trigger={"kind": "PRICE_LTE", "price": 100, "reconsider_when": "price <= 100"},
            path=db,
        )
    remember(
        symbol="AAA", session_date="2026-09-02", state="READY", reason="COMMITTEE_BUY",
        scan_run_id="s2", decision="BUY", entry_state="ENTER_NOW",
        execution_state="BLOCKED_BROKER_AUTH", path=db,
    )
    row = mem_get("AAA", path=db)
    assert row["opportunity_id"] == "AAA"
    assert row["first_seen_at"]
    assert row["last_session"] == "2026-09-02"
    assert row["last_state"] == "READY"
    hist = events_for("AAA", path=db)
    assert hist[0]["event"] == "DISCOVERED"
    assert any(e["old_state"] == "WAIT" and e["new_state"] == "READY" for e in hist)
    assert mem_get("BBB", path=db)["opportunity_id"] == "BBB"
    assert mem_get("CCC", path=db)["opportunity_id"] == "CCC"
    nxt = next_session_set(path=db)
    assert any(x["symbol"] == "AAA" for x in nxt["READY"])
    assert any(x["symbol"] == "BBB" for x in nxt["RESEARCH_PENDING"])


def test_missed_winner_extended_was_rational():
    quality = analyze_decision_quality(
        {"symbol": "AAA", "decision": "WAIT", "reason_code": T.ENTRY_TOO_EXTENDED},
        classification=MISSED_WINNER,
        forward_return_pct=14.2,
    )
    assert quality["original_decision_rational"] is True
    assert "never offered acceptable risk entry" in quality["note"]
    assert quality["updates_policy"] is False


def test_operator_classes_do_not_mix_terminal_into_current_run():
    assert _classify("pipeline", "MARKET_SCAN") == "AUTOMATED_JOB"
    assert _classify("terminal", "MARKET_SCAN") == "AVOIDABLE_HUMAN_ACTION"
    assert _classify("user", "KITE_LOGIN") == "NECESSARY_HUMAN_ACTION"


def test_scan_fresh_when_session_identity_current(monkeypatch, tmp_path):
    from product import desk_pipeline as DP

    product = tmp_path / "logs" / "product"
    product.mkdir(parents=True)
    (product / "latest_momentum_scan.json").write_text(json.dumps({
        "schema_version": 1,
        "records": [],
        "scanned_at": "2020-01-01T00:00:00+00:00",
        "as_of_session": "2026-09-02",
    }), encoding="utf-8")
    monkeypatch.setattr(DP, "_root", lambda: tmp_path)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda load_cache=True: {
            "current": True,
            "expected_latest_completed_session": "2026-09-02",
        },
    )
    assert DP.scan_is_fresh() is True
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda load_cache=True: {
            "current": True,
            "expected_latest_completed_session": "2026-09-03",
        },
    )
    assert DP.scan_is_fresh() is False


def test_auth_hint_does_not_own_supervisor(tmp_path):
    from research.autonomy.supervisor import Supervisor
    from tests.test_autonomy import FakeDeps

    sup = Supervisor(tmp_path / "auto", deps=FakeDeps(authed=False))
    assert sup.start() is True
    assert sup._gated_state(ST.AUTH_REQUIRED) != ST.AUTH_REQUIRED
    assert sup._gated_state(ST.OBSERVING) == ST.OBSERVING
    sup.failures.add(H.AUTH_MISSING)
    assert sup._gated_state(ST.RESEARCHING) == ST.RESEARCHING
    assert sup._gated_state(ST.DATA_READY) == ST.DATA_READY
    sup.shutdown()
