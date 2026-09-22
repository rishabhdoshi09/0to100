from __future__ import annotations

from types import SimpleNamespace

from research.autonomy.dialogue import OPERATIONAL_INCIDENT
from research.autonomy.incident_store import IncidentStore, STATUS_OPEN, STATUS_RECOVERED
from research.autonomy.supervisor import Supervisor


class _Deps:
    def now_ist(self):
        from datetime import datetime
        return datetime(2026, 9, 22, 8, 0)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snap-1"


def _job():
    return SimpleNamespace(
        job_id="job-1",
        job_type="research_cycle",
        idempotency_key="hist_research:b1",
        status="RUNNING",
        attempt=2,
        critical=False,
        scheduled_for=1.0,
        started_at=2.0,
        result_summary="research evaluation failed",
        error_code="HANDLER_EXCEPTION",
        error_message="boom",
        input_snapshot_id="b1",
        output_snapshot_id="",
    )


def test_identical_incident_updates_one_dossier_occurrence_count(tmp_path):
    store = IncidentStore(tmp_path / "incidents.json")
    first = store.upsert(code="HANDLER_EXCEPTION", message="boom", job=_job())
    second = store.upsert(code="HANDLER_EXCEPTION", message="boom", job=_job())

    assert first["incident_id"] == second["incident_id"]
    assert first["status"] == STATUS_OPEN
    assert second["occurrence_count"] == 2
    assert second["materially_changed"] is False
    assert len(store.recent(20)) == 1
    assert "durable backoff" in second["recovery_action"]


def test_material_incident_change_gets_same_identity_but_new_change_count(tmp_path):
    store = IncidentStore(tmp_path / "incidents.json")
    job = _job()
    first = store.upsert(code="HANDLER_EXCEPTION", message="boom", job=job)
    job.attempt = 3
    job.error_message = "different root cause"
    second = store.upsert(code="HANDLER_EXCEPTION", message="boom", job=job)

    assert first["incident_id"] == second["incident_id"]
    assert second["occurrence_count"] == 2
    assert second["materially_changed"] is True
    assert second["material_change_count"] == 2


def test_successful_job_recovery_closes_matching_open_incident(tmp_path):
    store = IncidentStore(tmp_path / "incidents.json")
    job = _job()
    opened = store.upsert(code="HANDLER_EXCEPTION", message="boom", job=job)
    recovered = store.recover_for_job(job, note="retry succeeded")

    assert len(recovered) == 1
    assert recovered[0]["incident_id"] == opened["incident_id"]
    assert recovered[0]["status"] == STATUS_RECOVERED
    assert recovered[0]["recovery_note"] == "retry succeeded"
    assert store.recent(10, open_only=True) == []


def test_supervisor_deduplicates_dialogue_but_preserves_occurrences(tmp_path):
    sup = Supervisor(tmp_path / "auto", deps=_Deps())
    assert sup.start() is True
    try:
        first = sup._incident("SUPERVISOR_TICK_EXCEPTION", "same tick failure")
        second = sup._incident("SUPERVISOR_TICK_EXCEPTION", "same tick failure")

        rows = sup.dialogue.by_type(OPERATIONAL_INCIDENT)
        assert len(rows) == 1
        assert first["incident_id"] == second["incident_id"]
        assert second["occurrence_count"] == 2
        assert second["materially_changed"] is False
        assert sup.incidents.recent(1)[0]["occurrence_count"] == 2
    finally:
        sup.shutdown()
