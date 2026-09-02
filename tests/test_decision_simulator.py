"""Take-vs-skip simulator stays BACKTEST and never books reject P&L."""
from __future__ import annotations

from product.counterfactual_learning import CORRECT_REJECTION, MISSED_WINNER, classify_forward
from product.decision_simulator import run_decision_simulator
from product.forward_evidence import BACKTEST


def test_classify_reject_that_later_stopped_is_correct_rejection():
    assert classify_forward(entry=100, stop=95, target=110, forward_return_pct=-6) == CORRECT_REJECTION


def test_classify_reject_that_later_ripped_is_missed_winner():
    assert classify_forward(entry=100, stop=95, target=110, forward_return_pct=9) == MISSED_WINNER


def test_simulator_uses_journal_and_stays_backtest(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_AUTOPILOT_JOURNAL", str(tmp_path / "journal.json"))
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(tmp_path / "cf.jsonl"))
    monkeypatch.setenv("QT_DECISION_SIMULATOR", str(tmp_path / "sim.json"))
    from product.autopilot_journal import record_cycle
    record_cycle({
        "as_of": "2026-08-20",
        "taken": [{"symbol": "TCS", "entry": 100, "stop": 94, "target": 115}],
        "rejections": [{"symbol": "RELIANCE", "reason_code": "ENTRY_TOO_EXTENDED", "decision": "REJECTED", "entry": 1400, "stop": 1350, "target": 1500}],
        "waits": [{"symbol": "INFY", "reason_code": "WAIT_FOR_ENTRY", "decision": "WAITED"}],
    })
    first = run_decision_simulator(force=True)
    assert first["provenance"] == BACKTEST
    assert first["not_promotion_evidence"] is True
    assert first["live_locked"] is True
    assert first["not_promotion_evidence"] is True
    assert first["decisions_tested"] >= 3
    assert first["would_take"] >= 1
    assert first["rejected"] >= 1
    assert first["provenance"] != "REAL_FORWARD_MARKET"
    assert "does not change REAL_FORWARD_MARKET" in first["note"]
    second = run_decision_simulator(force=False)
    assert second["cache_hit"] is True
    assert second["version"] == first["version"]
