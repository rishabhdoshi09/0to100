from research.auto_research.paper_book import PaperBook
from research.intelligence.schemas import TradeIntent


def test_paper_book_sizes_from_approved_percentage_points():
    exploratory = PaperBook(capital=100_000)
    established = PaperBook(capital=100_000)

    small = exploratory.open_position(
        "exploratory", "AAA", 100, 90, 120, "2026-08-01", 20,
        risk_pct_of_capital=0.25,
    )
    full = established.open_position(
        "established", "BBB", 100, 90, 120, "2026-08-01", 20,
        risk_pct_of_capital=1.0,
    )

    assert small is not None and full is not None
    assert small.qty == 25
    assert full.qty == 100
    assert small.risk_amount == 250
    assert full.risk_amount == 1_000
    assert small.requested_risk_pct == 0.25
    assert small.approved_risk_pct == 0.25


def test_book_level_maximum_caps_an_oversized_request():
    book = PaperBook(capital=100_000, risk_per_trade_pct=0.01)

    position = book.open_position(
        "strategy", "AAA", 100, 90, 120, "2026-08-01", 20,
        risk_pct_of_capital=5.0,
    )

    assert position is not None
    assert position.requested_risk_pct == 5.0
    assert position.approved_risk_pct == 1.0
    assert position.qty == 100


def test_trade_intent_is_the_source_of_paper_risk_size():
    book = PaperBook(capital=100_000)
    intent = TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="brain2",
        event_ts="2026-08-01",
        cycle_id="cycle",
        symbol="AAA",
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        holding_horizon_days=20,
    )

    position = book.open_intent(intent, date="2026-08-01")

    assert position is not None
    assert position.qty == 50
    assert position.risk_amount == 500
    assert position.requested_risk_pct == intent.intended_risk_pct


def test_non_positive_approved_risk_fails_closed():
    book = PaperBook(capital=100_000)

    position = book.open_position(
        "strategy", "AAA", 100, 90, 120, "2026-08-01", 20,
        risk_pct_of_capital=0,
    )

    assert position is None
    assert book.refusals[-1] == ("AAA", "approved risk percentage must be positive")


def test_old_book_snapshot_restores_with_new_audit_fields_defaulted():
    book = PaperBook(capital=100_000)
    book.restore(
        {
            "capital": 100_000,
            "realized_pnl": 0,
            "equity_curve": [100_000],
            "closed": [],
            "open": [
                {
                    "strategy_id": "legacy",
                    "symbol": "AAA",
                    "entry_price": 100,
                    "stop_price": 90,
                    "target_price": 120,
                    "qty": 25,
                    "entry_date": "2026-07-31",
                    "max_holding_days": 20,
                    "risk_amount": 250,
                    "bars_held": 0,
                }
            ],
        }
    )

    position = book.open[("legacy", "AAA")]
    assert position.requested_risk_pct == 0.0
    assert position.approved_risk_pct == 0.0
