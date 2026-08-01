from types import SimpleNamespace

from research.auto_research.paper_book import PaperBook
from research.intelligence import schemas as SC
from research.intelligence.allocation_brain import AllocationConfig
from research.intelligence.event_store import EventStore
from research.intelligence.runtime.runtime_state import RuntimeState
from research.intelligence.runtime.target_portfolio import (
    BLOCKED,
    HOLD,
    TARGETED,
    build_target_portfolio,
    trade_intent_from_target,
)


def _ctx(**overrides):
    values = {
        "as_of_date": "2026-08-01",
        "mode": "PAPER_AUTO",
        "data_snapshot_id": "snapshot-1",
        "data_ok": True,
        "market_regime": "RISK_ON",
        "clusters": {},
        "strategies": (),
        "fresh_live_symbols": frozenset(),
        "live_confirmation_required": False,
        "pending_quantities": {},
        "pending_risk_amounts": {},
        "pending_capital_amounts": {},
        "cycle_id": lambda: "cycle-1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _decision(strategy_id="s1", symbol="AAA", risk=0.5, score=1.0, family="momentum"):
    return SimpleNamespace(
        strategy_id=strategy_id,
        strategy_version=1,
        rules_hash=f"rules-{strategy_id}",
        family=family,
        action="DEPLOY",
        target_risk_pct=risk,
        score=score,
        record_id=f"allocation-{strategy_id}",
        reasons=("evidence-qualified",),
        symbol=symbol,
    )


def _card(strategy_id="s1"):
    return SimpleNamespace(strategy_id=strategy_id, record_id=f"card-{strategy_id}")


def _signals(*items):
    return {
        strategy_id: [
            {"symbol": symbol, "entry": 100, "stop": 90, "target": 120, "max_hold": 20}
        ]
        for strategy_id, symbol in items
    }


def test_target_portfolio_is_the_source_of_exact_trade_quantity():
    book = PaperBook(capital=100_000)
    build = build_target_portfolio(
        _ctx(),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals=_signals(("s1", "AAA")),
        cards=[_card()],
        cfg=AllocationConfig(),
    )

    assert build.portfolio.current_position_count == 0
    assert build.portfolio.target_position_count == 1
    assert len(build.executable) == 1
    target = build.executable[0]
    assert target.status == TARGETED
    assert target.current_quantity == 0
    assert target.pending_quantity == 0
    assert target.desired_quantity == 50
    assert target.required_quantity == 50

    intent = trade_intent_from_target(target, build.portfolio)
    assert intent.target_portfolio_id == build.portfolio.record_id
    assert intent.target_position_id == target.record_id
    assert intent.required_quantity == 50
    assert intent.schema_version == 2

    position = book.open_intent(intent, date="2026-08-01")
    assert position is not None
    assert position.qty == target.required_quantity
    assert position.risk_amount == target.incremental_risk_amount


def test_pending_quantity_is_subtracted_before_creating_an_intent():
    book = PaperBook(capital=100_000)
    build = build_target_portfolio(
        _ctx(
            pending_quantities={"AAA": 20},
            pending_risk_amounts={"AAA": 200},
            pending_capital_amounts={"AAA": 2_000},
        ),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals=_signals(("s1", "AAA")),
        cards=[_card()],
        cfg=AllocationConfig(),
    )

    target = build.executable[0]
    assert target.desired_quantity == 50
    assert target.pending_quantity == 20
    assert target.required_quantity == 30
    assert build.portfolio.pending_open_risk_pct == 0.2


def test_pending_quantity_can_fully_satisfy_the_target():
    book = PaperBook(capital=100_000)
    build = build_target_portfolio(
        _ctx(pending_quantities={"AAA": 50}),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals=_signals(("s1", "AAA")),
        cards=[_card()],
        cfg=AllocationConfig(),
    )

    assert len(build.executable) == 0
    assert build.positions[0].status == HOLD
    assert build.positions[0].required_quantity == 0


def test_duplicate_symbol_proposals_do_not_double_count_exposure():
    book = PaperBook(capital=100_000)
    high = _decision("s1", "AAA", risk=0.5, score=1.0)
    low = _decision("s2", "AAA", risk=0.5, score=0.5)
    build = build_target_portfolio(
        _ctx(),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[low, high],
        today_signals=_signals(("s1", "AAA"), ("s2", "AAA")),
        cards=[_card("s1"), _card("s2")],
        cfg=AllocationConfig(),
    )

    assert len(build.executable) == 1
    assert build.executable[0].strategy_id == "s1"
    blocked = next(position for position in build.positions if position.strategy_id == "s2")
    assert blocked.status == BLOCKED
    assert "DUPLICATE_SYMBOL_PROPOSAL" in blocked.blocked_by


def test_total_risk_cap_is_applied_to_the_whole_target_portfolio():
    book = PaperBook(capital=100_000, max_total_risk_pct=0.005, max_positions=5)
    first = _decision("s1", "AAA", risk=0.5, score=1.0, family="f1")
    second = _decision("s2", "BBB", risk=0.5, score=0.9, family="f2")
    build = build_target_portfolio(
        _ctx(),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[first, second],
        today_signals=_signals(("s1", "AAA"), ("s2", "BBB")),
        cards=[_card("s1"), _card("s2")],
        cfg=AllocationConfig(max_family_risk_pct=2.0, max_cluster_risk_pct=2.0),
    )

    assert len(build.executable) == 1
    blocked = next(position for position in build.positions if position.status == BLOCKED)
    assert "TOTAL_OPEN_RISK_CAP" in blocked.blocked_by
    assert build.portfolio.target_open_risk_pct <= build.portfolio.max_total_risk_pct


def test_target_portfolio_records_round_trip_through_event_store(tmp_path):
    book = PaperBook(capital=100_000)
    build = build_target_portfolio(
        _ctx(),
        book=book,
        runtime_state=RuntimeState(),
        decisions=[_decision()],
        today_signals=_signals(("s1", "AAA")),
        cards=[_card()],
        cfg=AllocationConfig(),
    )
    path = tmp_path / "events.jsonl"
    store = EventStore(path)
    store.extend(build.positions)
    store.append(build.portfolio)
    store.append(trade_intent_from_target(build.executable[0], build.portfolio))

    restored = EventStore(path)
    assert len(restored.of_type("TargetPosition")) == 1
    assert len(restored.of_type("TargetPortfolio")) == 1
    assert len(restored.of_type("TradeIntent")) == 1
    restored_intent = restored.of_type("TradeIntent")[0]
    assert restored_intent.target_portfolio_id == build.portfolio.record_id
    assert restored_intent.required_quantity == build.executable[0].required_quantity


def test_trade_intent_cannot_be_created_from_blocked_target():
    target = SC.TargetPosition(status=BLOCKED, required_quantity=50)
    portfolio = SC.TargetPortfolio()

    try:
        trade_intent_from_target(target, portfolio)
    except ValueError as exc:
        assert "not executable" in str(exc)
    else:
        raise AssertionError("blocked target unexpectedly created an intent")
