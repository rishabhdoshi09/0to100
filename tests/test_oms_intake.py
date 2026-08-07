from execution.oms.intake import ingest_event_store_intents
from execution.oms.store import OmsStore
from research.intelligence import schemas as SC
from research.intelligence.event_store import EventStore


def _linked_intent(symbol="AAA"):
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle-1",
        symbol=symbol,
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio-1",
        target_position_id=f"position-{symbol}",
        desired_quantity=10,
        required_quantity=10,
    )


def test_shadow_intake_is_idempotent(tmp_path):
    events = EventStore()
    intent = _linked_intent()
    events.append(intent)
    oms = OmsStore(tmp_path / "oms.db")

    first = ingest_event_store_intents(events, oms)
    second = ingest_event_store_intents(events, oms)

    assert first["accepted_count"] == 1
    assert first["existing_count"] == 0
    assert second["accepted_count"] == 0
    assert second["existing_count"] == 1
    assert len(oms.list_orders()) == 1
    assert oms.list_orders()[0].trade_intent_id == intent.record_id


def test_shadow_intake_skips_legacy_unlinked_intents(tmp_path):
    events = EventStore()
    legacy = SC.TradeIntent(
        strategy_id="legacy",
        symbol="AAA",
        intended_entry=100,
        stop_price=90,
        target_price=120,
        required_quantity=10,
    )
    events.append(legacy)
    oms = OmsStore(tmp_path / "oms.db")

    result = ingest_event_store_intents(events, oms)

    assert result["accepted_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["reason"] == "LEGACY_UNLINKED_INTENT"
    assert oms.list_orders() == []


def test_shadow_intake_can_scope_one_cycle(tmp_path):
    events = EventStore()
    first = _linked_intent("AAA")
    second = SC.TradeIntent(
        **{
            **_linked_intent("BBB").as_dict(),
            "record_id": "",
            "cycle_id": "cycle-2",
            "target_portfolio_id": "portfolio-2",
        }
    )
    events.extend([first, second])
    oms = OmsStore(tmp_path / "oms.db")

    result = ingest_event_store_intents(events, oms, cycle_id="cycle-2")

    assert result["accepted_count"] == 1
    assert oms.list_orders()[0].symbol == "BBB"
