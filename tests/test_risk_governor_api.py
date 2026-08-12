from risk.governor import GovernorDecision
from risk.governor_store import RiskDecisionStore

import terminal_product_api


def test_risk_governor_endpoint_projects_shadow_decisions(tmp_path, monkeypatch):
    path = tmp_path / "risk.db"
    store = RiskDecisionStore(path)
    decision = GovernorDecision(
        decision_id="risk-1",
        action="APPROVE",
        approved_quantity=10,
        requested_quantity=10,
        reasons=("ALL_LIMITS_PASS",),
        order_id="oms-1",
        symbol="AAA",
        state_snapshot_id="state-1",
        metrics={"open_risk": 100.0},
    )
    store.record(decision)
    monkeypatch.setattr(terminal_product_api, "RISK_DB", path)

    payload = terminal_product_api.risk_governor_status()

    assert payload["available"] is True
    assert payload["mode"] == "SHADOW"
    assert payload["authoritative_state_connected"] is False
    assert payload["certified_for_live"] is False
    assert payload["summary"]["decisions"] == 1
    assert payload["summary"]["by_action"] == {"APPROVE": 1}


def test_risk_governor_endpoint_does_not_create_state(tmp_path, monkeypatch):
    path = tmp_path / "missing-risk.db"
    monkeypatch.setattr(terminal_product_api, "RISK_DB", path)

    payload = terminal_product_api.risk_governor_status()

    assert payload["available"] is False
    assert payload["summary"] == {"decisions": 0, "by_action": {}}
    assert payload["authoritative_state_connected"] is False
    assert path.exists() is False
