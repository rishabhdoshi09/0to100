from execution.oms.store import OmsStore
from research.intelligence import schemas as SC

import terminal_product_api


def _intent():
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle",
        symbol="AAA",
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=10,
        required_quantity=10,
    )


def test_oms_endpoint_projects_durable_state_without_submission(tmp_path, monkeypatch):
    path = tmp_path / "orders.db"
    store = OmsStore(path)
    order = store.ingest_intent(_intent())
    monkeypatch.setattr(terminal_product_api, "OMS_DB", path)

    payload = terminal_product_api.oms_status()

    assert payload["available"] is True
    assert payload["broker_connected"] is False
    assert payload["submission_enabled"] is False
    assert payload["summary"]["orders"] == 1
    assert payload["orders"][0]["order_id"] == order.order_id
    assert payload["orders"][0]["status"] == "PROPOSED"


def test_oms_endpoint_does_not_create_state(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(terminal_product_api, "OMS_DB", path)

    payload = terminal_product_api.oms_status()

    assert payload["available"] is False
    assert payload["orders"] == []
    assert path.exists() is False
