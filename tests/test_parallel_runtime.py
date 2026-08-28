from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import parallel_runtime as PR


class _FakeStore:
    def __init__(self, latest=None):
        self._latest = latest
        self.enqueued = []

    def latest(self, kind):
        return self._latest

    def enqueue(self, kind, **kwargs):
        self.enqueued.append((kind, kwargs))
        return ({
            "operation_id": "new-op",
            "kind": kind,
            "status": "PENDING",
            "requested_at": 1.0,
            **kwargs,
        }, True)

    def get(self, operation_id):
        return self._latest


def test_queue_reuses_running_market_operation(monkeypatch):
    latest = {
        "operation_id": "scan-1",
        "kind": "MARKET_SCAN",
        "status": "RUNNING",
        "requested_at": 1.0,
    }
    store = _FakeStore(latest)
    monkeypatch.setattr(PR, "_ops_store", lambda: store)
    monkeypatch.setattr(PR, "_ensure_ops_worker", lambda: None)

    operation = PR.ensure_market_scan_started()
    assert operation["operation_id"] == "scan-1"
    assert store.enqueued == []


def test_autonomy_market_scan_observes_market_ops_and_never_runs_duplicate(monkeypatch):
    operation = {
        "operation_id": "scan-ok",
        "status": "SUCCEEDED",
        "result": {"summary": {"with_any_setup": 7, "momentum": 2}},
    }
    monkeypatch.setattr(PR, "ensure_market_scan_started", lambda **_kw: operation)
    monkeypatch.setattr(PR, "_operation_result", lambda op: op)
    monkeypatch.setattr(PR, "_saved_scan_payload", lambda: {
        "summary": {"with_any_setup": 7, "momentum": 2},
        "records": [],
    })

    class Deps:
        def run_scan(self):
            raise AssertionError("autonomy must not execute a second full scan")

    result = PR._delegated_market_scan(SimpleNamespace(deps=Deps()))
    assert result.status == JS.SUCCEEDED
    assert result.metadata["execution_plane"] == "market_ops"
    assert result.metadata["with_any_setup"] == 7


def test_running_market_operation_is_poll_state_not_scan_failure(monkeypatch):
    operation = {
        "operation_id": "scan-live",
        "status": "RUNNING",
        "stage": "SCANNING",
        "progress_current": 500,
        "progress_total": 2300,
    }
    monkeypatch.setattr(PR, "ensure_market_scan_started", lambda **_kw: operation)
    monkeypatch.setattr(PR, "_operation_result", lambda op: op)
    result = PR._delegated_market_scan(SimpleNamespace(deps=object()))
    assert result.status == JS.RETRYABLE_FAILED
    assert result.error_code == "MARKET_OP_IN_PROGRESS"
    assert result.metadata["progress_current"] == 500


def test_corporate_action_handler_only_schedules_background_io(monkeypatch):
    launched = []
    monkeypatch.setattr(PR, "_ca_status", lambda: {
        "available": True,
        "coverage_complete": False,
        "refresh_due": True,
        "windows_complete": 7,
        "windows_total": 11,
        "n_events": 88,
    })
    monkeypatch.setattr(PR, "_ensure_ca_background", lambda **_kw: launched.append(True))
    monkeypatch.setattr(PR, "_ca_future", None)

    result = PR._background_corporate_actions(SimpleNamespace())
    assert launched == [True]
    assert result.status == JS.SUCCEEDED
    assert H.CA_INCOMPLETE in result.failures
    assert "7/11" in result.summary


def test_complete_ca_coverage_clears_capability_flag(monkeypatch):
    monkeypatch.setattr(PR, "_ca_status", lambda: {
        "available": True,
        "coverage_complete": True,
        "refresh_due": False,
        "windows_complete": 11,
        "windows_total": 11,
        "n_events": 120,
    })
    supervisor = SimpleNamespace(
        failures={H.CA_INCOMPLETE},
        _save_failures=lambda: None,
    )
    PR._reconcile_ca_failure(supervisor)
    assert H.CA_INCOMPLETE not in supervisor.failures
