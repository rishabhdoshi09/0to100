from research.auto_research.paper_book import PaperBook


def test_paper_position_sector_survives_snapshot_restore():
    book = PaperBook()
    pos = book.open_position(
        "QT_RECO",
        "INFY",
        100.0,
        95.0,
        115.0,
        "2026-09-18",
        20,
        risk_pct_of_capital=1.0,
    )
    assert pos is not None
    pos.sector = "IT"

    restored = PaperBook()
    restored.restore(book.snapshot())

    assert len(restored.open) == 1
    saved = next(iter(restored.open.values()))
    assert saved.symbol == "INFY"
    assert saved.sector == "IT"


def test_paper_risk_and_execution_contract_survives_snapshot_restore():
    from research.auto_research.costs import india_cash_costs

    book = PaperBook(
        capital=250_000,
        risk_per_trade_pct=0.0075,
        max_position_pct=0.08,
        max_total_risk_pct=0.03,
        max_positions=7,
        slippage_bps=4.5,
        cost_model=india_cash_costs,
    )
    snap = book.snapshot()

    restored = PaperBook(
        capital=10_000,
        risk_per_trade_pct=0.02,
        max_position_pct=0.20,
        max_total_risk_pct=0.10,
        max_positions=2,
        slippage_bps=0.0,
        cost_model=None,
    )
    restored.restore(snap)

    assert restored.capital == 250_000
    assert restored.risk_per_trade_pct == 0.0075
    assert restored.max_position_pct == 0.08
    assert restored.max_total_risk_pct == 0.03
    assert restored.max_positions == 7
    assert restored.slippage_bps == 4.5
    assert restored.cost_model is india_cash_costs
    assert restored.as_dict()["risk_config"] == {
        "risk_per_trade_pct": 0.0075,
        "max_position_pct": 0.08,
        "max_total_risk_pct": 0.03,
        "max_positions": 7,
        "slippage_bps": 4.5,
        "cost_model": "india_cash_costs",
    }


def test_legacy_snapshot_preserves_current_constructor_risk_contract():
    from research.auto_research.costs import india_cash_costs

    book = PaperBook(
        risk_per_trade_pct=0.005,
        max_position_pct=0.07,
        max_total_risk_pct=0.025,
        max_positions=4,
        slippage_bps=3.0,
        cost_model=india_cash_costs,
    )
    book.restore({
        "capital": 100_000,
        "realized_pnl": 0,
        "equity_curve": [100_000],
        "closed": [],
        "open": [],
    })

    assert book.risk_per_trade_pct == 0.005
    assert book.max_position_pct == 0.07
    assert book.max_total_risk_pct == 0.025
    assert book.max_positions == 4
    assert book.slippage_bps == 3.0
    assert book.cost_model is india_cash_costs


def test_corrupt_risk_contract_does_not_partially_restore_book():
    book = PaperBook(capital=100_000, max_positions=5)
    original = book.snapshot()
    corrupt = {
        **original,
        "capital": 999_999,
        "risk_config": {
            **original["risk_config"],
            "max_positions": 0,
        },
    }

    book.restore(corrupt)

    assert book.capital == 100_000
    assert book.max_positions == 5
