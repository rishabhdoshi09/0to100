"""Home Best Trades integration with the Evolution Engine consensus board
(sections 17, 42, 45).

Best Trades eligibility must come from the Champion's real production
decision alone; the consensus board can only ADD explanatory context to an
already-eligible row, never push an otherwise ineligible one onto Home.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from product.evolution.consensus_board import save_latest_consensus
from product.home_os import build_home_os

IST = ZoneInfo("Asia/Kolkata")


def _open() -> datetime:
    return datetime(2026, 9, 1, 10, 45, tzinfo=IST)


def _dashboard():
    return {
        "autonomy": {"state": "RUNNING", "running": True, "broker": {"ready": True}},
        "data": {
            "ready": True,
            "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True, "reason_code": "HISTORY_CURRENT"},
        },
    }


def _gate_with(best_trades):
    return {"scan_fresh": True, "discovery_ready": True, "best_trades": best_trades}


def test_eligible_row_gets_enriched_with_consensus_when_board_has_data(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import product.decision_simulation_gate as gate

    monkeypatch.setattr(gate, "status", lambda: _gate_with([
        {"symbol": "RELIANCE", "setup_label": "Ready to trade", "decision": "BUY",
         "discovery_decision": "ENTER_NOW", "reason_code": "ELIGIBLE"},
    ]))

    save_latest_consensus({
        "champion_policy_id": "EQUITY_CHAMPION_V1",
        "as_of": "2026-09-01",
        "results": [{
            "market_snapshot_id": "snap-1", "symbol": "RELIANCE",
            "champion": {"decision": "ENTER_NOW"},
            "consensus": {"qualified_count": 10, "selecting_count": 8, "consensus_pct": 80.0, "main_dissent_reason": "ENTRY_TOO_EXTENDED"},
        }],
    })

    os = build_home_os(
        dashboard=_dashboard(), paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "rejections": [], "waits": []},
        reco={"schema_version": 4, "categories": []}, now=_open(),
    )
    rows = os["opportunities"]
    assert len(rows) == 1
    consensus = rows[0]["evolution_consensus"]
    assert consensus["consensus_pct"] == 80.0
    assert consensus["champion_policy_id"] == "EQUITY_CHAMPION_V1"
    assert consensus["main_dissent_reason"] == "ENTRY_TOO_EXTENDED"


def test_ineligible_row_never_reaches_home_even_with_strong_consensus(monkeypatch, tmp_path):
    """A high-consensus symbol the REAL Champion rejected must still be
    absent from Best Trades -- Market Twin data explains, it never selects."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import product.decision_simulation_gate as gate

    monkeypatch.setattr(gate, "status", lambda: _gate_with([
        {"symbol": "TCS", "setup_label": "Ready to trade", "decision": "WAIT",
         "discovery_decision": "ENTER_NOW", "reason_code": "ENTRY_TOO_EXTENDED"},
    ]))

    save_latest_consensus({
        "champion_policy_id": "EQUITY_CHAMPION_V1", "as_of": "2026-09-01",
        "results": [{
            "market_snapshot_id": "snap-2", "symbol": "TCS",
            "champion": {"decision": "REJECT"},
            "consensus": {"qualified_count": 10, "selecting_count": 9, "consensus_pct": 90.0, "main_dissent_reason": None},
        }],
    })

    os = build_home_os(
        dashboard=_dashboard(), paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "rejections": [], "waits": []},
        reco={"schema_version": 4, "categories": []}, now=_open(),
    )
    assert os["opportunities"] == []


def test_missing_board_entry_is_silently_absent_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import product.decision_simulation_gate as gate

    monkeypatch.setattr(gate, "status", lambda: _gate_with([
        {"symbol": "INFY", "setup_label": "Ready to trade", "decision": "BUY",
         "discovery_decision": "ENTER_NOW", "reason_code": "ELIGIBLE"},
    ]))
    # No save_latest_consensus call at all -- fresh install / before first cycle.
    os = build_home_os(
        dashboard=_dashboard(), paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "rejections": [], "waits": []},
        reco={"schema_version": 4, "categories": []}, now=_open(),
    )
    assert len(os["opportunities"]) == 1
    assert "evolution_consensus" not in os["opportunities"][0]
