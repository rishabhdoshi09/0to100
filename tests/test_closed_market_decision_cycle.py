from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy.supervisor import Supervisor


class _CaptureJobs:
    def __init__(self):
        self.calls = []

    def enqueue(self, job_type, **kwargs):
        self.calls.append((job_type, dict(kwargs)))
        return SimpleNamespace(status=JS.PENDING)


class _ClosedMarketDeps:
    def __init__(self):
        self.decision_calls = []

    def now_ist(self):
        return datetime(2026, 9, 21, 23, 45)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snap-current"

    def run_decision_only_cycle(self, reason, phase):
        self.decision_calls.append((reason, phase))
        return {
            "eligibility": "BLOCKED_SAFETY",
            "decision_only": True,
            "not_forward_evidence": True,
            "taken": [],
            "rejections": [],
            "waits": [],
            "cycle_reasons": [reason],
        }


class _SupervisorHarness:
    def __init__(self):
        self.deps = _ClosedMarketDeps()
        self.jobs = _CaptureJobs()

    def _ensure_decision_simulation_authority(self):
        return True

    def _snapshot_token(self, *args, **kwargs):
        return "snap-current"


def test_closed_market_supervisor_enqueues_one_decision_only_identity(monkeypatch):
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": True,
            "scan_scanned_at": "2026-09-21T18:30:00+05:30",
            "thesis_hash": "thesis-abc",
        },
    )
    sup = _SupervisorHarness()

    Supervisor._ensure_closed_market_decision_cycle(sup)

    assert len(sup.jobs.calls) == 1
    job_type, kwargs = sup.jobs.calls[0]
    assert job_type == "paper_cycle"
    assert kwargs["idempotency_key"] == "snapshot_decision:snap-current:thesis-abc"
    assert kwargs["input_snapshot_id"] == "snap-current"
    assert kwargs["critical"] is False


def test_decision_only_job_uses_non_executing_dependency_lane():
    deps = _ClosedMarketDeps()
    job = SimpleNamespace(idempotency_key="snapshot_decision:snap-current:thesis-abc")
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps, job=job))

    assert result.status == JS.SUCCEEDED
    assert result.new_entries_allowed is False
    assert result.metadata["decision_only"] is True
    assert result.metadata["not_forward_evidence"] is True
    assert result.metadata["entry_block_reason"] == "ENTRY_WINDOW_CLOSED_DECISION_ONLY"
    assert deps.decision_calls == [
        ("ENTRY_WINDOW_CLOSED_DECISION_ONLY", "off_session")
    ]
    assert result.summary.startswith("decision-only cycle:")
