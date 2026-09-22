from __future__ import annotations

import json

from product.autonomy_status import read_autonomy_status
from research.autonomy.incident_store import IncidentStore


def test_autonomy_status_projects_open_and_recent_incident_dossiers(tmp_path, monkeypatch):
    root = tmp_path / "auto"
    root.mkdir(parents=True)
    (root / "status.json").write_text(
        json.dumps({
            "state": "OBSERVING",
            "process_running": False,
            "heartbeat_ist": "",
            "active_failures": [],
            "owner_state": {},
        }),
        encoding="utf-8",
    )
    store = IncidentStore(root / "incidents.json")
    store.upsert(
        code="HANDLER_EXCEPTION",
        message="research failed",
        job=None,
        activity_truth={"activity": "RESEARCH"},
        resource_governor={"decision": "ALLOW_RESEARCH"},
    )

    payload = read_autonomy_status(root=root)

    incidents = payload["operational_incidents"]
    assert incidents["open_count"] == 1
    assert incidents["open"][0]["code"] == "HANDLER_EXCEPTION"
    assert incidents["recent"][0]["occurrence_count"] == 1


def test_missing_incident_store_projects_truthful_empty_state(tmp_path):
    root = tmp_path / "auto"
    root.mkdir(parents=True)
    payload = read_autonomy_status(root=root)

    assert payload["operational_incidents"] == {
        "open_count": 0,
        "open": [],
        "recent": [],
    }
