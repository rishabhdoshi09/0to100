from types import SimpleNamespace

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.paper_pipeline import BROKER_MUTATIONS_ENABLED, PaperExecutionPipeline
from execution.protection.models import VERIFIED
from execution.protection.store import ProtectionStore
from execution.tca.store import TcaStore
from research.auto_research.paper_book import PaperBook
from research.intelligence.event_store import EventStore
from research.intelligence.schemas import TradeIntent
from risk.governor_store import RiskDecisionStore
from risk.oms_gate import evaluate_oms_order


def _intent(*, symbol="AAA", quantity=10, risk_pct=0.5):
    return TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        cycle_id="cycle",
        symbol=symbol,
        intended_entry=100,
        intended_risk_pct=risk_pct,
        stop_price=90,
        target_price=120,
        holding_horizon_days=20,
        target_portfolio_id="portfolio",
        target_position_id=f"position-{symbol}",
        desired_quantity=quantity,
        required_quantity=quantity,
    )


def _pipeline(tmp_path, events):
    oms = OmsStore(tmp_path / "oms.db")
    risk = RiskDecisionStore(tmp_path / "risk.db")
    protection = ProtectionStore(tmp_path / "protection.db")
    tca = TcaStore(tmp_path / "tca.db")
    return (
        PaperExecutionPipeline(
            oms_store=oms,
            risk_store=risk,
            protection_store=protection,
            tca_store=tca,
            event_store=events,
        ),
        oms,
        risk,
        protection,
        tca,
    )


def test_full_paper_execution_reaches_protected_state(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    pipeline, oms, risk, protection, tca = _pipeline(tmp_path, events)
    book = PaperBook(capital=100_000)

    result = pipeline.execute(
        intent,
        book=book,
        date="2026-08-01T09:30:00+05:30",
        runtime_state=SimpleNamespace(reconciled=True),
    )

    assert result.opened is True
    assert result.status == OM.PROTECTED
    assert result.position.qty == 10
    order = oms.get(result.order_id)
    assert order.status == OM.PROTECTED
    assert order.approved_quantity == 10
    assert order.filled_quantity == 10
    assert order.broker_order_id.startswith("paper-order-")
    assert len(oms.fills(order.order_id)) == 1
    assert risk.summary()["decisions"] == 1
    plan = protection.get_by_order(order.order_id)
    assert plan is not None and plan.status == VERIFIED and plan.fully_protected
    assert tca.summary()["assessments"] == 1
    assert BROKER_MUTATIONS_ENABLED is False


def test_repeated_execution_is_idempotent(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    pipeline, oms, risk, protection, tca = _pipeline(tmp_path, events)
    book = PaperBook(capital=100_000)

    first = pipeline.execute(intent, book=book, date="2026-08-01", runtime_state=SimpleNamespace(reconciled=True))
    second = pipeline.execute(intent, book=book, date="2026-08-01", runtime_state=SimpleNamespace(reconciled=True))

    assert first.order_id == second.order_id
    assert second.opened is True
    assert second.resumed is True
    assert len(book.open) == 1
    assert len(oms.list_orders()) == 1
    assert len(oms.fills(first.order_id)) == 1
    assert risk.summary()["decisions"] == 1
    assert protection.summary()["plans"] == 1
    assert tca.summary()["assessments"] == 1


def test_pipeline_recovers_crash_after_book_open_before_fill_record(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    pipeline, oms, _risk, _protection, _tca = _pipeline(tmp_path, events)
    book = PaperBook(capital=100_000)

    order = oms.ingest_intent(intent)
    state = pipeline._risk_state(book, "2026-08-01", SimpleNamespace(reconciled=True), exclude_order_id=order.order_id)
    order = evaluate_oms_order(oms, order.order_id, state=state).order
    order = oms.prepare_submission(order.order_id, submission_token=f"paper-submit-{order.order_id}")
    order = oms.acknowledge(
        order.order_id,
        broker_order_id=f"paper-order-{order.order_id}",
        external_event_id=f"paper-ack-{order.order_id}",
    )
    position = book.open_position(
        order.strategy_id,
        order.symbol,
        order.intended_entry,
        order.stop_price,
        order.target_price,
        "2026-08-01",
        20,
        risk_pct_of_capital=order.intended_risk_pct,
        quantity=order.approved_quantity,
    )
    assert position is not None
    assert oms.get(order.order_id).status == OM.BROKER_ACKNOWLEDGED

    result = pipeline.execute(
        intent,
        book=book,
        date="2026-08-01",
        runtime_state=SimpleNamespace(reconciled=True),
    )

    assert result.opened is True
    assert result.resumed is True
    assert len(book.open) == 1
    assert len(oms.fills(order.order_id)) == 1
    assert oms.get(order.order_id).status == OM.PROTECTED


def test_risk_governor_reduces_oversized_paper_intent(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent(quantity=1_000, risk_pct=1.0)
    events.append(intent)
    pipeline, oms, _risk, _protection, _tca = _pipeline(tmp_path, events)
    book = PaperBook(capital=100_000)

    result = pipeline.execute(
        intent,
        book=book,
        date="2026-08-01",
        runtime_state=SimpleNamespace(reconciled=True),
    )

    order = oms.get(result.order_id)
    assert result.opened is True
    assert order.approved_quantity == 100
    assert result.position.qty == 100
    assert order.approved_quantity < order.requested_quantity


def test_unreconciled_runtime_rejects_before_paper_position(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    pipeline, oms, _risk, _protection, _tca = _pipeline(tmp_path, events)
    book = PaperBook(capital=100_000)

    result = pipeline.execute(
        intent,
        book=book,
        date="2026-08-01",
        runtime_state=SimpleNamespace(reconciled=False),
    )

    assert result.opened is False
    assert result.status == OM.REJECTED
    assert "STATE_UNRECONCILED" in result.reason
    assert len(book.open) == 0
    assert oms.get(result.order_id).status == OM.REJECTED
