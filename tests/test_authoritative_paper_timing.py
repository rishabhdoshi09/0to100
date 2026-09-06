from __future__ import annotations

from types import SimpleNamespace

from research.auto_research.paper_book import PaperBook
from research.intelligence.runtime import modes as MODES
from research.intelligence.runtime.autonomous_loop import _manage_positions


class _Ctx:
    mode = MODES.PAPER_AUTO

    def __init__(self, day: str, bar):
        self.as_of_date = day
        self._bar = bar

    def today_bar(self, _symbol: str):
        return self._bar



def _result():
    return SimpleNamespace(positions_closed=[], outcomes_recorded=[])



def test_authoritative_runtime_never_uses_entry_session_full_daily_bar():
    book = PaperBook(capital=100_000.0)
    opened = book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "2026-09-04", 5,
    )
    assert opened is not None

    # The full daily bar crosses both stop and target. In the authoritative forward
    # runtime it is not valid evidence for an entry created during that same session.
    ctx = _Ctx("2026-09-04", (100.0, 120.0, 90.0, 105.0))
    result = _result()
    _manage_positions(ctx, [], book, None, result)

    assert ("S1", "AAA") in book.open
    assert book.open[("S1", "AAA")].bars_held == 0
    assert book.open[("S1", "AAA")].last_marked_session == ""
    assert result.positions_closed == []
    assert result.outcomes_recorded == []



def test_authoritative_runtime_can_mark_a_later_session():
    book = PaperBook(capital=100_000.0)
    opened = book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "2026-09-04", 5,
    )
    assert opened is not None

    # Once a later session is supplied the same authoritative path may settle it.
    ctx = _Ctx("2026-09-07", (100.0, 120.0, 90.0, 105.0))
    result = _result()
    _manage_positions(ctx, [], book, None, result)

    assert ("S1", "AAA") not in book.open
    assert len(book.closed) == 1
    assert book.closed[0].exit_reason == "STOP"
    assert result.positions_closed == [("S1", "AAA", "STOP")]
    assert result.outcomes_recorded == ["S1"]



def test_research_book_default_preserves_signal_before_bar_same_session_model():
    book = PaperBook(capital=100_000.0)
    opened = book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "d1", 5,
    )
    assert opened is not None

    # Research/historical simulations intentionally model the signal as known before
    # the supplied bar. The default remains True so those existing simulations keep
    # their signal-before-bar semantics; authoritative forward paths must opt out.
    closed = book.mark({"AAA": (100.0, 120.0, 90.0, 105.0)}, "d1")

    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP"
