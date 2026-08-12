from research.intelligence import schemas as SC
from research.intelligence.event_store import EventStore

import terminal_product_api


def test_target_portfolio_endpoint_projects_only_persisted_state(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    store = EventStore(path)
    position = SC.TargetPosition(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle",
        symbol="AAA",
        desired_quantity=50,
        required_quantity=50,
        target_risk_pct=0.5,
        target_risk_amount=500,
        incremental_risk_amount=500,
        capital_required=5_000,
        status="TARGETED",
    )
    portfolio = SC.TargetPortfolio(
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle",
        mode="PAPER_AUTO",
        capital=100_000,
        available_cash=95_000,
        current_open_risk_pct=0,
        pending_open_risk_pct=0,
        target_open_risk_pct=0.5,
        max_total_risk_pct=5.0,
        current_position_count=0,
        target_position_count=1,
        position_ids=(position.record_id,),
        executable_position_ids=(position.record_id,),
    )
    store.append(position)
    store.append(portfolio)
    monkeypatch.setattr(terminal_product_api, "TARGET_EVENT_STORE", path)

    payload = terminal_product_api.target_portfolio()

    assert payload["available"] is True
    assert payload["portfolio"]["record_id"] == portfolio.record_id
    assert payload["positions"][0]["record_id"] == position.record_id
    assert payload["summary"]["executable_changes"] == 1
    assert payload["summary"]["target_open_risk_pct"] == 0.5


def test_target_portfolio_endpoint_does_not_fabricate_state(tmp_path, monkeypatch):
    path = tmp_path / "missing-events.jsonl"
    monkeypatch.setattr(terminal_product_api, "TARGET_EVENT_STORE", path)

    payload = terminal_product_api.target_portfolio()

    assert payload["available"] is False
    assert payload["portfolio"] == {}
    assert payload["positions"] == []
