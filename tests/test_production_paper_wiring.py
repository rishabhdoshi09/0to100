from types import SimpleNamespace

from execution.paper_book_adapter import InstitutionalPaperBookAdapter
from research.auto_research import providers


class _Book:
    def __init__(self):
        self.value = 1

    def open_intent(self, intent, *, date):
        raise AssertionError("plain book shortcut must not be used")


class _Pipeline:
    def __init__(self, position):
        self.position = position
        self.calls = []

    def execute(self, intent, *, book, date, runtime_state):
        self.calls.append((intent, book, date, runtime_state))
        return SimpleNamespace(position=self.position, opened=True)


def test_adapter_replaces_only_new_entry_execution():
    book = _Book()
    pipeline = _Pipeline(position=SimpleNamespace(qty=10))
    runtime = SimpleNamespace(reconciled=True)
    adapter = InstitutionalPaperBookAdapter(
        book,
        pipeline=pipeline,
        runtime_state=runtime,
    )
    intent = SimpleNamespace(record_id="intent")

    position = adapter.open_intent(intent, date="2026-08-01")

    assert position.qty == 10
    assert pipeline.calls == [(intent, book, "2026-08-01", runtime)]
    assert adapter.value == 1
    adapter.value = 2
    assert book.value == 2
    assert adapter.institutional_execution_enabled is True
    assert adapter.broker_mutations_enabled is False


def test_missing_institutional_pipeline_forces_risk_off(monkeypatch):
    monkeypatch.setattr(providers, "ensure_production_paper_pipeline", lambda: False)
    assert providers.current_regime() == "RISK_OFF"
