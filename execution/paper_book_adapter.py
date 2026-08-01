"""Transparent PaperBook adapter that routes production intents through institutional state."""
from __future__ import annotations


class InstitutionalPaperBookAdapter:
    """Delegate the PaperBook API while replacing only ``open_intent``.

    The autonomous runtime and all position-management code continue to use the same canonical
    PaperBook object. Only new entries are routed through the durable PAPER execution pipeline.
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
