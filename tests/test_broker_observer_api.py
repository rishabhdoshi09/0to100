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


def test_scanner_workspace_breakouts_reads_saved_scan(monkeypatch):
    scan = {
        "scanned_at": "2026-08-24T16:49:23+00:00",
        "universe_size": 3,
        "records": [
            {"symbol": "AAA", "signals": ["BREAKOUT_52W"], "score": 80, "chase_risk": False},
            {"symbol": "BBB", "signals": ["MOMENTUM"], "score": 70, "chase_risk": False},
        ],
    }
    monkeypatch.setattr(observer_api.core, "_scan_payload", lambda: scan)
    monkeypatch.setattr(observer_api.core, "_long_term_payload", lambda: {"records": [], "scanned_at": ""})
    monkeypatch.setattr(observer_api.core, "_market_payload", lambda: {})
    monkeypatch.setattr(observer_api.core, "_conviction", lambda _scan, _market: [])

    payload = observer_api.scanner_workspace("Breakouts")

    assert payload["mode"] == "Breakouts"
    assert payload["universe_size"] == 3
    assert payload["scanned_at"] == "2026-08-24T16:49:23+00:00"
    assert [row["symbol"] for row in payload["rows"]] == ["AAA"]


def test_scanner_workspace_rejects_unknown_mode():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        observer_api.scanner_workspace("Not A Mode")
    assert raised.value.status_code == 400


def test_command_center_workspace_imports_projector(monkeypatch):
    monkeypatch.setattr(observer_api.core, "_market_payload", lambda: {
        "health": "Healthy",
        "summary": "ok",
        "trade_stance": "Selective",
        "breadth": "Strong",
        "leaders": [],
        "laggards": [],
        "nifty_change_1d": 0,
        "vix": 12,
    })
    monkeypatch.setattr(observer_api.core, "_scan_payload", lambda: {
        "scanned_at": "2026-08-24T16:49:23+00:00",
        "universe_size": 1,
        "records": [{"symbol": "AAA", "signals": ["MOMENTUM"], "status": "Ready to trade", "score": 80}],
    })
    monkeypatch.setattr(observer_api.core, "_long_term_payload", lambda: {"records": [], "summary": {}})
    monkeypatch.setattr(observer_api.core, "_paper_payload", lambda: type("P", (), {
        "capital": 100000, "equity": 100000, "open_positions": (), "open_risk": 0, "enabled": False,
    })())
    monkeypatch.setattr(observer_api.core, "_autonomy_payload", lambda: {
        "running": False, "state": "IDLE", "plain_state": "Idle", "heartbeat_ist": "",
    })

    payload = observer_api.command_center_workspace()
    assert payload["ready_count"] == 1
    assert payload["scan_universe"] == 1


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
