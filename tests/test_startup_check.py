from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import data.bhavcopy_runtime as bhavcopy_runtime
import product.paper_status as paper_status
import product.startup_check as startup_check
from product.startup_check import (
    SCHEMA_VERSION,
    _history_readiness,
    _paper_readiness,
    _required_waiting,
    _scan_evidence_status,
    build_startup_check,
    maybe_open_home_browser,
    print_startup_summary,
)


def _current_scan(*, hours_ago: float = 0.5, records=None) -> dict:
    now = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "schema_version": 1,
        "scanned_at": now.isoformat(),
        "records": records if records is not None else [{"symbol": "AAA"}],
        "available": True,
    }


def _patch_operational_healthy(monkeypatch, *, autonomy=True, ops=True, paper=True, live_locked=True):
    monkeypatch.setattr(startup_check, "_port_open", lambda port: port in {5173, 8765})
    monkeypatch.setattr(startup_check, "_url_ok", lambda url: True)
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda **_k: {"running": autonomy},
        raising=False,
    )
    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=paper, enabled=True),
    )

    from pathlib import Path

    original_path_read_text = Path.read_text
    ops_payload = (
        '{"running": true, "process_running": true}'
        if ops
        else '{"running": false, "process_running": false}'
    )

    def patched_read_text(self, *a, **k):
        if "market_ops" in str(self) and str(self).endswith("runtime.json"):
            return ops_payload
        return original_path_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", patched_read_text)

    if live_locked:
        from product.execution_adapter import LiveMoneyLocked

        class _Adapter:
            def submit(self, _order):
                raise LiveMoneyLocked("locked")

        monkeypatch.setattr("product.execution_adapter.LiveExecutionAdapter", _Adapter)
    else:
        class _Adapter:
            def submit(self, _order):
                return {"ok": True}

        monkeypatch.setattr("product.execution_adapter.LiveExecutionAdapter", _Adapter)


def _patch_history(monkeypatch, *, current: bool, available="2026-09-04", expected="2026-09-05", reason="HISTORY_STALE"):
    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_kwargs: {
            "current": current,
            "reason_code": "HISTORY_CURRENT" if current else reason,
            "available_session": available if available is not None else "",
            "expected_latest_completed_session": expected,
        },
    )


def _patch_scan(monkeypatch, payload):
    monkeypatch.setattr("product.scan_store.load_scan", lambda *_a, **_k: payload)
    monkeypatch.setattr("product.scan_store.default_scan_path", lambda: "unused.json")


def _patch_soak(monkeypatch, status="COLLECTING"):
    monkeypatch.setattr("product.forward_soak.persist_soak_verification", lambda **_k: None)
    monkeypatch.setattr("product.forward_soak.soak_status", lambda: {"status": status})


def test_startup_check_live_money_locked_and_telegram_optional(monkeypatch):
    monkeypatch.setenv("QT_NONINTERACTIVE", "1")
    payload = build_startup_check(probe_network=False)
    names = [lane["name"] for lane in payload["lanes"]]
    assert "LIVE MONEY" in names
    assert "UI" in names
    assert "FORWARD EVIDENCE" in names
    live = next(lane for lane in payload["lanes"] if lane["name"] == "LIVE MONEY")
    assert live["status"] == "LOCKED"
    assert payload["live_locked"] is True
    assert "Telegram absence is not a product failure." in payload["note"]
    reports = next(lane for lane in payload["lanes"] if lane["name"] == "REPORTS")
    assert reports["required"] is False
    data = next(lane for lane in payload["lanes"] if lane["name"] == "DATA")
    assert data["domain"] == "evidence"
    api = next(lane for lane in payload["lanes"] if lane["name"] == "API")
    assert api["domain"] == "operational"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "operational" in payload and "evidence" in payload


def test_history_readiness_uses_canonical_freshness_not_old_scan_file(monkeypatch):
    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_kwargs: {
            "current": False,
            "reason_code": "HISTORY_STALE",
            "available_session": "2026-09-02",
            "expected_latest_completed_session": "2026-09-03",
        },
    )

    status, detail = _history_readiness()

    assert status == "STALE"
    assert "HISTORY_STALE" in detail
    assert "available 2026-09-02" in detail
    assert "expected 2026-09-03" in detail


def test_paper_readiness_requires_live_supervisor_not_truthy_default(monkeypatch):
    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=False, enabled=True),
    )
    status, detail = _paper_readiness()
    assert status == "WAITING"
    assert "not running" in detail.lower()

    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=True, enabled=False),
    )
    status, detail = _paper_readiness()
    assert status == "READY"
    assert "paused" in detail.lower()


def test_required_lanes_are_authoritative_for_ready_state():
    lanes = [
        {"name": "UI", "status": "READY", "required": True, "domain": "operational"},
        {"name": "API", "status": "READY", "required": True, "domain": "operational"},
        {"name": "DATA", "status": "WAITING", "required": True, "domain": "evidence"},
        {"name": "REPORTS", "status": "WAITING", "required": False, "domain": "capability"},
        {"name": "LIVE MONEY", "status": "LOCKED", "required": True, "domain": "operational"},
    ]
    waiting = _required_waiting(lanes, domain="evidence", ready_statuses={"READY", "HEALTHY", "CURRENT"})
    assert [row["name"] for row in waiting] == ["DATA"]
    operational = _required_waiting(lanes, domain="operational")
    assert operational == []


def test_browser_open_skipped_when_noninteractive(monkeypatch):
    monkeypatch.setenv("QT_NONINTERACTIVE", "1")
    assert maybe_open_home_browser() is False
    monkeypatch.delenv("QT_NONINTERACTIVE", raising=False)
    monkeypatch.setenv("QT_NO_BROWSER", "1")
    assert maybe_open_home_browser() is False


def test_operational_and_evidence_ready_together(monkeypatch):
    _patch_operational_healthy(monkeypatch)
    _patch_history(monkeypatch, current=True, available="2026-09-05", reason="HISTORY_CURRENT")
    _patch_scan(monkeypatch, _current_scan(hours_ago=0.2))
    _patch_soak(monkeypatch, "HEALTHY")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is True
    assert payload["ready"] is True
    assert payload["operational"]["status"] == "READY"
    assert payload["evidence"]["status"] == "READY"
    assert payload["operational"]["blockers"] == []
    assert payload["evidence"]["blockers"] == []


def test_operationally_healthy_with_evidence_not_ready(monkeypatch):
    _patch_operational_healthy(monkeypatch)
    _patch_history(monkeypatch, current=False, available="", expected="2026-09-05", reason="HISTORY_NOT_READY")
    _patch_scan(monkeypatch, None)
    _patch_soak(monkeypatch, "NOT_STARTED")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["ready"] is False
    assert payload["operational"]["status"] == "READY"
    assert payload["evidence"]["status"] in {"MISSING", "NOT_READY", "DEGRADED"}
    assert "DATA" in payload["evidence"]["blockers"]
    assert "SCAN PIPELINE" in payload["evidence"]["blockers"]
    assert "DATA" not in payload["operational"]["blockers"]
    assert "SCAN PIPELINE" not in payload["operational"]["blockers"]


def test_stale_evidence_does_not_masquerade_as_fully_ready(monkeypatch):
    _patch_operational_healthy(monkeypatch)
    _patch_history(monkeypatch, current=False, available="2026-09-02", expected="2026-09-05", reason="HISTORY_STALE")
    _patch_scan(monkeypatch, _current_scan(hours_ago=48.0, records=[{"symbol": "OLD"}]))
    _patch_soak(monkeypatch, "COLLECTING")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    data = next(lane for lane in payload["lanes"] if lane["name"] == "DATA")
    scan = next(lane for lane in payload["lanes"] if lane["name"] == "SCAN PIPELINE")
    assert data["status"] == "STALE"
    assert scan["status"] == "STALE"
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["ready"] is False
    assert payload["evidence"]["status"] == "STALE"


def test_missing_evidence_does_not_masquerade_as_fully_ready(monkeypatch):
    _patch_operational_healthy(monkeypatch)
    _patch_history(monkeypatch, current=True, available="2026-09-05", reason="HISTORY_CURRENT")
    _patch_scan(monkeypatch, {"schema_version": 1, "records": [], "available": True})
    _patch_soak(monkeypatch, "NOT_STARTED")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    scan = next(lane for lane in payload["lanes"] if lane["name"] == "SCAN PIPELINE")
    assert scan["status"] in {"MISSING", "INCOMPLETE"}
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["ready"] is False
    assert "SCAN PIPELINE" in payload["evidence"]["blockers"]


def test_operational_failure_is_independent_of_evidence(monkeypatch):
    _patch_operational_healthy(monkeypatch, autonomy=False, ops=False, paper=False)
    monkeypatch.setattr(startup_check, "_port_open", lambda port: False)
    _patch_history(monkeypatch, current=True, available="2026-09-05", reason="HISTORY_CURRENT")
    _patch_scan(monkeypatch, _current_scan(hours_ago=0.1))
    _patch_soak(monkeypatch, "HEALTHY")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    assert payload["operational_ready"] is False
    assert payload["operational"]["status"] in {"NOT_READY", "FAILED"}
    assert payload["ready"] is False
    assert "API" in payload["operational"]["blockers"] or "UI" in payload["operational"]["blockers"]
    # Evidence may still be ready: processes down does not invent missing artifacts.
    assert payload["evidence_ready"] is True
    assert payload["evidence"]["status"] == "READY"


def test_live_money_unlocked_is_operational_failure(monkeypatch):
    _patch_operational_healthy(monkeypatch, live_locked=False)
    _patch_history(monkeypatch, current=True, available="2026-09-05", reason="HISTORY_CURRENT")
    _patch_scan(monkeypatch, _current_scan())
    _patch_soak(monkeypatch, "HEALTHY")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    payload = build_startup_check(probe_network=False)
    assert payload["live_locked"] is False
    assert payload["operational_ready"] is False
    assert payload["operational"]["status"] == "FAILED"
    assert payload["ready"] is False


def test_print_startup_summary_separates_planes(monkeypatch):
    _patch_operational_healthy(monkeypatch)
    _patch_history(monkeypatch, current=False, available="2026-09-01", expected="2026-09-05", reason="HISTORY_STALE")
    _patch_scan(monkeypatch, None)
    _patch_soak(monkeypatch, "NOT_STARTED")
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *_a, **_k: "")

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    code = print_startup_summary(probe_network=False)
    text = buf.getvalue()
    assert code == 0
    assert "Operational runtime: READY" in text
    assert "Evidence:" in text
    assert "STALE" in text or "MISSING" in text
    assert "evidence-dependent claims are not fully ready" in text
    assert "QuantTerm is ready." not in text


def test_scan_evidence_status_invariants():
    assert _scan_evidence_status(None)[0] == "MISSING"
    assert _scan_evidence_status({})[0] == "MISSING"
    fresh = _current_scan(hours_ago=0.1, records=[{"symbol": "AAA"}])
    assert _scan_evidence_status(fresh)[0] == "READY"
    stale = _current_scan(hours_ago=30.0, records=[{"symbol": "AAA"}])
    assert _scan_evidence_status(stale)[0] == "STALE"
    empty = _current_scan(hours_ago=0.1, records=[])
    assert _scan_evidence_status(empty)[0] == "INCOMPLETE"
