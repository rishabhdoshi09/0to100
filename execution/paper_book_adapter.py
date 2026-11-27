"""Transparent PaperBook adapter that routes production through institutional state."""
from __future__ import annotations

from execution.oms.models import CLOSED
from execution.paper_exit import sync_paper_close


class InstitutionalPaperBookAdapter:
    """Delegate PaperBook while replacing entry and exit state transitions.

    The autonomous runtime continues to use the same canonical PaperBook positions and trade
    outcomes. New entries route through OMS/Risk/Protection/TCA, and closed trades synchronize
    the durable OMS and protection ledgers.
    """

    def __init__(self, book, *, pipeline, runtime_state) -> None:
        object.__setattr__(self, "_book", book)
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_runtime_state", runtime_state)

    def open_intent(self, intent, *, date: str):
        result = self._pipeline.execute(
            intent,
            book=self._book,
            date=date,
            runtime_state=self._runtime_state,
        )
        return result.position if result.opened else None

    def mark(self, bars: dict, date: str, *, allow_entry_session: bool = True):
        def sync_exit(trade):
            result = sync_paper_close(self._pipeline, trade)
            if result is not None and result.order_status != CLOSED:
                raise RuntimeError(f"paper OMS exit unresolved for {result.order_id}: {result.order_status}")

        # Preserve the mark cursor and P&L until the deterministic OMS exit can
        # be replayed safely. A completed OMS exit is a no-op on the next pass.
        return self._book.mark(bars, date, allow_entry_session=allow_entry_session,
                               before_settle=sync_exit)

    @property
    def institutional_execution_enabled(self) -> bool:
        return True

    @property
    def broker_mutations_enabled(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._book, name)

    def __setattr__(self, name, value) -> None:
        if name in {"_book", "_pipeline", "_runtime_state"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._book, name, value)
