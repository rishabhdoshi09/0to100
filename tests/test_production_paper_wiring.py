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


def test_empty_production_book_cannot_bypass_failed_protection_wiring(monkeypatch):
    from research.auto_research import scheduler
    from research.autonomy import jobs, paper_cycle_truth
    from product import paper_autopilot

    brain = SimpleNamespace(
        intel_book=SimpleNamespace(open={}),
        state=SimpleNamespace(),
        manage_intelligence_positions_day=lambda **_: {"status": "NO_ACTION"},
        is_paper_auto_enabled=lambda: True,
        _save_intel_book=lambda: True,
    )
    monkeypatch.setattr(scheduler, "_BRAIN", brain)
    monkeypatch.setattr(scheduler, "get_brain", lambda: brain)
    monkeypatch.setattr(providers, "ensure_production_paper_pipeline", lambda: False)
    monkeypatch.setattr(paper_cycle_truth, "_INSTALLED", False)
    monkeypatch.setattr(jobs.Deps, "run_paper_cycle", jobs.Deps.run_paper_cycle)

    def run(**kwargs):
        assert not kwargs["entries_allowed"]
        assert kwargs["entry_block_reason"] == "PAPER_PIPELINE_UNAVAILABLE"
        return {"eligibility": "BLOCKED_SAFETY", "entries_allowed": False,
                "entry_block_reason": kwargs["entry_block_reason"]}

    monkeypatch.setattr(paper_autopilot, "run_reco_paper_cycle", run)
    paper_cycle_truth.install_paper_cycle_truth()
    deps = SimpleNamespace(live_feed=None, telegram=SimpleNamespace(notify_paper_cycle=lambda *a, **k: None))
    result = jobs.Deps.run_paper_cycle(deps, True)
    assert result["entry_block_reason"] == "PAPER_PIPELINE_UNAVAILABLE"
