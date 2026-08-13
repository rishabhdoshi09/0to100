"""Forward trading evidence system — allowlist, snapshots, ledger, paper≠live."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.forward_evidence import sources as SRC
from research.forward_evidence.policy_allowlist import (
    PaperPolicy,
    PaperPolicyAllowlist,
    seed_default_allowlist,
)
from research.forward_evidence.decision_snapshot import freeze_decision, get_snapshot, load_snapshots
from research.forward_evidence.outcome_ledger import (
    load_outcomes,
    record_from_closed_trade,
)
from research.forward_evidence.paper_vs_live import compare_policy
from research.forward_evidence.hooks import may_open_paper, on_paper_intent_opened, on_paper_trade_closed
from research.forward_evidence.service import ensure_armed, system_status
from research.forward_evidence.reporting import policy_report
from research.intelligence.runtime import modes as MODES
from execution.paper_pipeline import BROKER_MUTATIONS_ENABLED


def test_paper_not_live_authorization(tmp_path):
    al = seed_default_allowlist(tmp_path / "al.json", force=True)
    assert al.may_paper_trade("cross_sectional_momentum")
    assert not al.may_live_trade("cross_sectional_momentum")
    assert not al.may_paper_trade("EXP-FUND-03")
    assert not al.may_paper_trade("earnings_growth")
    mom = al.get("cross_sectional_momentum")
    assert mom.paper_enabled is True
    assert mom.live_enabled is False
    fund = al.get("EXP-FUND-03")
    assert fund.paper_observation_status == "DENIED"
    assert fund.scientific_status == "INCONCLUSIVE_FOLLOWUP"


def test_confirmed_status_does_not_auto_enable_fund03(tmp_path):
    al = PaperPolicyAllowlist(tmp_path / "al2.json")
    # Attempt to upsert with paper from "CONFIRMED" still denied for deny list
    al.upsert(PaperPolicy(
        research_policy_id="EXP-FUND-03",
        paper_enabled=True,
        live_enabled=True,
        scientific_status="CONFIRMED",
        paper_observation_status="ACTIVE",
        approval_reason="should be stripped",
    ))
    p = al.get("EXP-FUND-03")
    assert p.paper_enabled is False
    assert p.live_enabled is False
    assert p.paper_observation_status == "DENIED"


def test_decision_snapshot_freeze_idempotent(tmp_path):
    path = tmp_path / "snaps.jsonl"
    a = freeze_decision(
        policy_id="MOM", symbol="RELIANCE", timestamp="2024-01-02",
        entry=100, stop=95, target=110, intended_quantity=10,
        cycle_id="cyc-1", intent_id="int-1", path=path,
    )
    b = freeze_decision(
        policy_id="MOM", symbol="RELIANCE", timestamp="2024-01-02",
        entry=100, stop=95, target=110, intended_quantity=10,
        cycle_id="cyc-1", intent_id="int-1", path=path,
    )
    assert a.decision_id == b.decision_id
    rows = load_snapshots(path)
    assert len(rows) == 1
    got = get_snapshot(a.decision_id, path)
    assert got["entry"] == 100
    assert got["evidence_source"] == SRC.PAPER_FORWARD


def test_forward_outcome_source_separation(tmp_path):
    path = tmp_path / "outs.jsonl"
    trade = {
        "strategy_id": "MOM", "symbol": "RELIANCE",
        "entry_price": 100, "exit_price": 105, "qty": 10,
        "entry_date": "2024-01-02", "exit_date": "2024-01-10",
        "exit_reason": "TARGET", "realized_R": 1.0, "pnl": 50,
    }
    o1 = record_from_closed_trade(trade, decision_id="dec-a", evidence_source=SRC.PAPER_FORWARD, path=path)
    o2 = record_from_closed_trade(trade, decision_id="dec-a", evidence_source=SRC.LIVE, path=path)
    assert o1.outcome_id != o2.outcome_id
    paper = load_outcomes(path=path, evidence_source=SRC.PAPER_FORWARD)
    live = load_outcomes(path=path, evidence_source=SRC.LIVE)
    assert len(paper) == 1 and len(live) == 1
    # Idempotent re-append
    record_from_closed_trade(trade, decision_id="dec-a", evidence_source=SRC.PAPER_FORWARD, path=path)
    assert len(load_outcomes(path=path, evidence_source=SRC.PAPER_FORWARD)) == 1


def test_paper_vs_live_shows_no_live_yet():
    cmp_ = compare_policy(policy_id="MOM", paper_outcomes=[
        {"evidence_source": SRC.PAPER_FORWARD, "r_outcome": 0.1, "net_pnl": 10, "slippage": 0.08}
    ], live_outcomes=[])
    assert cmp_.live_n == 0
    assert cmp_.plain_language == "NO LIVE EVIDENCE YET"


def test_paper_mode_alias_and_live_blocked():
    assert MODES.opens_new_entries(MODES.PAPER_AUTO)
    assert MODES.opens_new_entries(MODES.PAPER_FORWARD_EVIDENCE)
    with pytest.raises(MODES.LiveModeDisabled):
        MODES.assert_no_live(MODES.LIMITED_LIVE)
    with pytest.raises(MODES.LiveModeDisabled):
        MODES.assert_no_live(MODES.FULL_AUTO)
    assert BROKER_MUTATIONS_ENABLED is False


def test_hooks_open_close_and_family_allowlist(tmp_path, monkeypatch):
    al_path = tmp_path / "al.json"
    seed_default_allowlist(al_path, force=True)
    monkeypatch.setattr(
        "research.forward_evidence.hooks.seed_default_allowlist",
        lambda path=None, force=False: seed_default_allowlist(al_path),
    )
    assert may_open_paper("MOM", family="cross_sectional_momentum")
    assert not may_open_paper("EXP-FUND-03", family="earnings_growth")

    snap_path = tmp_path / "snaps.jsonl"
    out_path = tmp_path / "outs.jsonl"
    monkeypatch.setattr("research.forward_evidence.decision_snapshot.DEFAULT_PATH", snap_path)
    monkeypatch.setattr("research.forward_evidence.outcome_ledger.DEFAULT_PATH", out_path)

    snap = on_paper_intent_opened(
        policy_id="MOM", symbol="INFY", timestamp="2024-02-01",
        entry=1500, stop=1450, target=1600, qty=5,
        cycle_id="cyc-x", intent_id="int-x",
    )
    closed = on_paper_trade_closed({
        "strategy_id": "MOM", "symbol": "INFY",
        "entry_price": 1500, "exit_price": 1480, "qty": 5,
        "entry_date": "2024-02-01", "exit_date": "2024-02-05",
        "exit_reason": "STOP", "realized_R": -0.4, "pnl": -100,
    }, update_memory=False)
    assert closed["evidence_source"] == SRC.PAPER_FORWARD
    assert closed["decision_id"] == snap["decision_id"]
    assert closed["outcome_status"] == "STOPPED"


def test_ensure_armed_does_not_enable_live(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research.forward_evidence.service.REPO", tmp_path,
    )
    monkeypatch.setattr(
        "research.forward_evidence.policy_allowlist.DEFAULT_PATH",
        tmp_path / "logs" / "forward_evidence" / "paper_policy_allowlist.json",
    )
    monkeypatch.setattr(
        "research.forward_evidence.service.STATUS_PATH",
        tmp_path / "logs" / "forward_evidence" / "system_status.json",
    )
    st = ensure_armed(enable_paper_auto=True)
    assert st["live_trading_enabled"] is False
    assert st["broker_mutations_enabled"] is False
    assert st["paper_auto_trading_ready"] is True
    assert "cross_sectional_momentum" in st["current_paper_policies"]
    assert "EXP-FUND-03" in st["denied_policies"]
    st2 = system_status()
    assert st2["live_trading_enabled"] is False


def test_policy_report_does_not_average_sources(tmp_path, monkeypatch):
    path = tmp_path / "outs.jsonl"
    monkeypatch.setattr("research.forward_evidence.outcome_ledger.DEFAULT_PATH", path)
    monkeypatch.setattr("research.forward_evidence.reporting.load_outcomes",
                        lambda **kw: load_outcomes(path=path, **kw))
    for i, r in enumerate([0.2, 0.1, -0.05]):
        record_from_closed_trade({
            "strategy_id": "breakout", "symbol": f"S{i}",
            "entry_price": 100, "exit_price": 100 + r * 5, "qty": 1,
            "entry_date": f"2024-01-{i+1:02d}", "exit_date": f"2024-01-{i+10:02d}",
            "exit_reason": "TARGET", "realized_R": r, "pnl": r * 100,
        }, decision_id=f"d{i}", evidence_source=SRC.PAPER_FORWARD, path=path)
    rep = policy_report("breakout")
    assert rep["live"]["n"] == 0
    assert rep["paper_forward"]["n"] == 3
    assert "separately" in rep["combined_methodology"].lower() or "not averaged" in rep["combined_methodology"].lower()
    assert rep["live_authorized"] is False
