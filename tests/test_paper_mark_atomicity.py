"""Paper mark and closeout must be atomic under bad costs or poisoned bars."""
import math

import pytest

from research.auto_research.paper_book import PaperBook


def _open_two(book):
    assert book.open_position("s1", "AAA", 100, 90, 110, "2026-11-05", 3)
    assert book.open_position("s2", "BBB", 100, 90, 110, "2026-11-05", 3)


def _exit_bars():
    return {
        "AAA": (110, 112, 90, 110),
        "BBB": (110, 112, 90, 110),
    }


def test_second_cost_failure_cannot_partially_close_first_position():
    calls = []

    def costs(entry, exit_fill, quantity):
        calls.append(quantity)
        if len(calls) == 2:
            raise RuntimeError("failed cost feed")
        return 2.0

    book = PaperBook(capital=100_000, cost_model=costs)
    _open_two(book)
    original = book.snapshot()

    with pytest.raises(RuntimeError, match="cost"):
        book.mark(_exit_bars(), date="2026-11-06")

    assert book.snapshot() == original
    assert book.realized_pnl == 0
    assert len(book.open) == 2
    assert len(book.closed) == 0


@pytest.mark.parametrize("bad_cost", [-1.0, float("nan"), float("inf")])
def test_nonfinite_or_negative_cost_is_not_free_paper_fill(bad_cost):
    book = PaperBook(capital=100_000, cost_model=lambda *_: bad_cost)
    assert book.open_position("s1", "AAA", 100, 90, 110, "2026-11-05", 3)
    original = book.snapshot()

    with pytest.raises(ValueError, match="cost"):
        book.mark({"AAA": (110, 112, 90, 110)}, date="2026-11-06")

    assert book.snapshot() == original


def test_nonfinite_market_bar_cannot_poison_paper_equity_or_mark_session():
    book = PaperBook(capital=100_000)
    assert book.open_position("s1", "AAA", 100, 90, 110, "2026-11-05", 3)
    original = book.snapshot()

    with pytest.raises(ValueError, match="bar"):
        book.mark({"AAA": (100, math.nan, 90, 100)}, date="2026-11-06")

    assert book.snapshot() == original


def test_valid_closeouts_still_settle_with_real_transaction_costs():
    book = PaperBook(capital=100_000, cost_model=lambda *_: 2.0)
    _open_two(book)

    closed = book.mark(_exit_bars(), date="2026-11-06")

    assert len(closed) == 2
    assert len(book.open) == 0
    assert len(book.closed) == 2
    assert [t.pnl for t in closed] == [998.0, 998.0]
    assert book.realized_pnl == 1996.0
    assert book.equity_curve[-1] == 101996.0
