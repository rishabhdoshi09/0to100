"""Phase 3 — Portfolio Selection Authority."""

from __future__ import annotations

from datetime import datetime, timezone

from product.paper_autopilot import run_reco_paper_cycle
from product.portfolio_selection_authority import (
    ENTER_NOW,
    NO_TRADE,
    PORTFOLIO_BLOCK,
    allocate,
)
from research.auto_research.paper_book import PaperBook


def _name(symbol, score, sector, **over):
    row = {
        "symbol": symbol,
        "selection_score": score,
        "score": score,
        "sector": sector,
        "reco_tier": "high_conviction",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "volume_ratio": 1.2,
        "dd_verdict": "PASS",
    }
    row.update(over)
    return row


def test_highest_ranked_stock_is_not_automatically_chosen():
    existing = type("P", (), {"symbol": "INFY", "sector": "Technology"})()
    book = PaperBook(capital=100_000)
    book.open[("QT", "INFY")] = existing
    out = allocate(
        [
            _name("TCS", 92, "Technology"),
            _name("ONGC", 89, "Energy"),
        ],
        book=book,
        existing_sectors={"Technology": 1},
        max_new=1,
    )
    entered = [c for c in out if c.decision == ENTER_NOW]
    assert entered[0].symbol == "ONGC"
    assert entered[0].individual_rank == 2
    assert "independent" in entered[0].why_over.lower() or "duplicates" in entered[0].concentration_effect
    tcs = next(c for c in out if c.symbol == "TCS")
    assert tcs.decision != ENTER_NOW or tcs.portfolio_rank != 1


def test_lower_ranked_diversified_candidate_can_win():
    out = allocate(
        [_name("AAA", 92, "Banks"), _name("BBB", 89, "IT")],
        existing_sectors={"Banks": 1},
        max_new=1,
        capital=100_000,
    )
    assert [c.symbol for c in out if c.decision == ENTER_NOW] == ["BBB"]
    assert out[0].decision == ENTER_NOW or any(c.symbol == "BBB" and c.decision == ENTER_NOW for c in out)


def test_hard_cap_cannot_be_overridden():
    out = allocate(
        [_name("TCS", 99, "Technology", dd_verdict="FAIL")],
        max_new=3,
    )
    assert out[0].decision == PORTFOLIO_BLOCK
    assert out[0].reason_code == "DD_BLOCK"
    assert out[0].hard_cap_applied == "DD_BLOCK"

    dup = allocate(
        [_name("TCS", 99, "Technology")],
        book=type("B", (), {"open": {("s", "TCS"): type("P", (), {"symbol": "TCS"})()}, "capital": 100000, "max_positions": 5})(),
    )
    assert dup[0].reason_code == "DUPLICATE_POSITION"

    bad_stop = allocate([_name("TCS", 99, "Technology", stop=101)])
    assert bad_stop[0].reason_code == "INVALID_STOP"

    corr = allocate(
        [_name("WIPRO", 90, "Technology")],
        book=type("B", (), {"open": {("s", "TCS"): type("P", (), {"symbol": "TCS", "sector": "Technology"})()}, "capital": 1e5, "max_positions": 5})(),
        correlations={"TCS|WIPRO": 0.95},
        max_new=3,
    )
    assert corr[0].reason_code == "CORRELATION_CAP"


def test_no_trade_remains_possible():
    out = allocate([], max_new=3)
    assert out[0].decision == NO_TRADE


def test_portfolio_decision_does_not_create_buy_from_watch_or_avoid():
    out = allocate(
        [_name("TCS", 99, "Technology", reco_tier="watch")],
        max_new=3,
    )
    assert out[0].decision == NO_TRADE
    assert out[0].invents_buy is False
    avoid = allocate([_name("INFY", 98, "Technology", reco_tier="avoid")])
    assert avoid[0].decision == NO_TRADE


def test_deterministic_ranking_is_stable():
    names = [_name("TCS", 90, "IT"), _name("INFY", 90, "IT"), _name("WIPRO", 88, "IT")]
    a = [c.symbol for c in allocate(names, max_new=3)]
    b = [c.symbol for c in allocate(list(reversed(names)), max_new=3)]
    assert a == b


def test_single_eligible_name_still_trades_on_money_path():
    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "score": 82,
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }
    now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    book = PaperBook(capital=100_000)
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        now=now,
        as_of="2026-09-01",
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": 1, "cards": [card]}],
        },
    )
    assert out["taken"][0]["symbol"] == "TCS"
    assert next(iter(book.open.values())).stop_price == 94.0
