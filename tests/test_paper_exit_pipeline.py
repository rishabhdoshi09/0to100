from types import SimpleNamespace

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.paper_book_adapter import InstitutionalPaperBookAdapter
from execution.paper_pipeline import PaperExecutionPipeline
from execution.protection.models import CANCELLED
from execution.protection.store import ProtectionStore
from execution.tca.store import TcaStore
from research.auto_research.paper_book import PaperBook
from research.intelligence.event_store import EventStore
from research.intelligence.schemas import TradeIntent
from risk.governor_store import RiskDecisionStore


def _intent():
    return TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        cycle_id="cycle",
        symbol="AAA",
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        holding_horizon_days=20,
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=10,
        required_quantity=10,
    )


def test_paper_stop_closes_oms_and_cancels_protection(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    oms = OmsStore(tmp_path / "oms.db")
    protection = ProtectionStore(tmp_path / "protection.db")
    pipeline = PaperExecutionPipeline(
        oms_store=oms,
        risk_store=RiskDecisionStore(tmp_path / "risk.db"),
        protection_store=protection,
        tca_store=TcaStore(tmp_path / "tca.db"),
        event_store=events,
    )
    raw_book = PaperBook(capital=100_000)
    book = InstitutionalPaperBookAdapter(
        raw_book,
        pipeline=pipeline,
        runtime_state=SimpleNamespace(reconciled=True),
    )

    position = book.open_intent(intent, date="2026-08-01")
    assert position is not None
    order = oms.list_orders()[0]
    assert order.status == OM.PROTECTED

    closed = book.mark({"AAA": (100, 101, 89, 90)}, "2026-08-02")

    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP"
    order = oms.get(order.order_id)
    assert order.status == OM.CLOSED
    plan = protection.get_by_order(order.order_id)
    assert plan is not None and plan.status == CANCELLED
    assert protection.summary()["entry_freeze_required"] is False


def test_repeated_mark_does_not_duplicate_exit_transitions(tmp_path):
    events = EventStore(tmp_path / "events.jsonl")
    intent = _intent()
    events.append(intent)
    oms = OmsStore(tmp_path / "oms.db")
    protection = ProtectionStore(tmp_path / "protection.db")
    pipeline = PaperExecutionPipeline(
        oms_store=oms,
        risk_store=RiskDecisionStore(tmp_path / "risk.db"),
        protection_store=protection,
        tca_store=TcaStore(tmp_path / "tca.db"),
        event_store=events,
    )
    book = InstitutionalPaperBookAdapter(
        PaperBook(capital=100_000),
        pipeline=pipeline,
        runtime_state=SimpleNamespace(reconciled=True),
    )
    book.open_intent(intent, date="2026-08-01")
    order_id = oms.list_orders()[0].order_id

    book.mark({"AAA": (100, 101, 89, 90)}, "2026-08-02")
    transition_count = len(oms.history(order_id))
    assert book.mark({}, "2026-08-03") == []

    assert len(oms.history(order_id)) == transition_count
    assert oms.get(order_id).status == OM.CLOSED
