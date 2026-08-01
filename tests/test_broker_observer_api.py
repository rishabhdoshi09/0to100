import json
import time

import terminal_product_api
from product import observer_api


def _routes(path: str):
    return [
        route
        for route in terminal_product_api.app.routes
        if getattr(route, "path", "") == path
    ]


def test_product_routes_are_installed_once_after_branch_reconciliation():
    assert len(_routes("/api/broker-observer")) == 1
    assert len(_routes("/api/command-center-workspace")) == 1
    assert len(_routes("/api/scanner-workspace/{mode}")) == 1


def test_missing_observer_state_is_read_only(tmp_path, monkeypatch):
    runtime = tmp_path / "observer_runtime.json"
    snapshots = tmp_path / "broker_snapshots.db"
    monkeypatch.setattr(observer_api, "RUNTIME_PATH", runtime)
    monkeypatch.setattr(observer_api, "SNAPSHOT_DB", snapshots)
    monkeypatch.setenv("QT_ENABLE_ZERODHA_OBSERVER", "1")

    payload = observer_api.observer_payload()

    assert payload["enabled"] is True
    assert payload["running"] is False
    assert payload["broker_mutations_enabled"] is False
    assert payload["snapshots"]["available"] is False
    assert runtime.exists() is False
    assert snapshots.exists() is False


def test_fresh_runtime_heartbeat_is_projected(tmp_path, monkeypatch):
    runtime = tmp_path / "observer_runtime.json"
    snapshots = tmp_path / "missing.db"
    runtime.write_text(
        json.dumps(
            {
                "process_running": True,
                "heartbeat_epoch": time.time(),
                "heartbeat": "2026-08-01T14:50:00+05:30",
                "phase": "IDLE",
                "last_result": {
                    "entries_allowed": False,
                    "blockers": ["PROTECTION_ENTRY_FREEZE"],
                },
                "last_error": "",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(observer_api, "RUNTIME_PATH", runtime)
    monkeypatch.setattr(observer_api, "SNAPSHOT_DB", snapshots)

    payload = observer_api.observer_payload()

    assert payload["running"] is True
    assert payload["phase"] == "IDLE"
    assert payload["last_result"]["entries_allowed"] is False
    assert payload["last_result"]["blockers"] == ["PROTECTION_ENTRY_FREEZE"]
