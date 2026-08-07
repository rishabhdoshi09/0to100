from types import SimpleNamespace

from research.auto_research.paper_book import PaperBook
from research.intelligence.allocation_brain import AllocationConfig
from research.intelligence.runtime.autonomous_loop import _open_new_positions


class _Store:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)
        return True


class _State:
    reconciled = True

    def __init__(self):
        self.strategy = SimpleNamespace(
            allocation_pct=0.0,
            lifecycle="",
            latest_card_id="",
            latest_allocation_id="",
        )

    def get(self, strategy_id, family=""):
        return self.strategy


def test_runtime_executes_the_trade_intent_at_brain2_risk_size():
    ctx = SimpleNamespace(
        as_of_date="2026-08-01",
        clusters={},
        market_regime="RISK_ON",
        data_ok=True,
        strategies=(),
        fresh_live_symbols=frozenset(),
        live_confirmation_required=False,
        data_snapshot_id="snapshot",
        cycle_id=lambda: "cycle",
    )
    decision = SimpleNamespace(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        family="momentum",
        action="DEPLOY",
        target_risk_pct=0.25,
        score=1.0,
        record_id="allocation",
        reasons=("exploratory risk",),
    )
    card = SimpleNamespace(strategy_id="strategy", record_id="card")
    signals = {
        "strategy": [
            {"symbol": "AAA", "entry": 100, "stop": 90, "target": 120, "max_hold": 20}
        ]
    }
    result = SimpleNamespace(trade_intents=[], intents_blocked=[], positions_opened=[])
    book = PaperBook(capital=100_000)

    _open_new_positions(
        ctx,
        _Store(),
        book,
        _State(),
        result,
        [decision],
        signals,
        [card],
        AllocationConfig(),
    )

    position = book.open[("strategy", "AAA")]
    assert position.qty == 25
    assert position.risk_amount == 250
    assert position.requested_risk_pct == decision.target_risk_pct
    assert result.positions_opened == [("strategy", "AAA")]
    assert len(result.trade_intents) == 1
