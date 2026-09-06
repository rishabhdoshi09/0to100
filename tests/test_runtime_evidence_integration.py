"""Cross-contract tests for runtime-liveness + autonomous evidence evolution.

The two workstreams merged without textual conflicts. These tests prove they
remain correct *together*: autonomous bootstrap cannot fake startup evidence
READY, and runtime evidence READY cannot skip the history-first paper gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import data.bhavcopy_runtime as bhavcopy_runtime
import product.paper_status as paper_status
import product.startup_check as startup_check
from product.evidence_policy_engine import BLOCK, evaluate_policies
from product.execution_adapter import LiveExecutionAdapter, LiveMoneyLocked
from product.learning_policy_store import upsert_policy
from product.paper_autopilot import EVIDENCE_POLICY_BLOCK, INVALID_STOP, run_reco_paper_cycle
from product.runtime_lifecycle import DEGRADED, inspect_runtime
from product.startup_check import build_startup_check
from research.auto_research.paper_book import PaperBook


def _vcp_card(**over):
    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }
    card.update(over)
    return card


def _workspace(card, now):
    return {
        "schema_version": 4,
        "generated_at": now.isoformat(),
        "scan_scanned_at": now.isoformat(),
        "categories": [{"id": "w", "count": 1, "cards": [card]}],
    }


def _cycle(card, **kwargs):
    now = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    kwargs.setdefault("now", now)
    kwargs.setdefault("as_of", "2026-09-01")
    kwargs.setdefault("persist_journal", False)
    kwargs.setdefault("workspace", _workspace(card, now))
    return run_reco_paper_cycle(book=kwargs.pop("book", PaperBook(capital=100_000)), cards=[card], **kwargs)


def _patch_operational_healthy(monkeypatch, tmp_path):
    monkeypatch.setattr(startup_check, "_port_open", lambda port: port in {5173, 8765})
    monkeypatch.setattr(startup_check, "_url_ok", lambda url: True)
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda **_k: {"running": True},
        raising=False,
    )
    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=True, enabled=True),
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"running": true, "process_running": true}', encoding="utf-8")

    from pathlib import Path

    original = Path.read_text

    def patched(self, *a, **k):
        if "market_ops" in str(self) and str(self).endswith("runtime.json"):
            return runtime.read_text(encoding="utf-8")
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", patched)

    class _Adapter:
        def submit(self, _order):
            raise LiveMoneyLocked("locked")

    monkeypatch.setattr("product.execution_adapter.LiveExecutionAdapter", _Adapter)


def _write_hist_vcp(tmp_path, monkeypatch, *, fingerprint="gen-A", reproduced=True):
    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={
            "production_status": "ELIGIBLE" if reproduced else "EXPERIMENTAL",
            "historical_reproduced_positive": bool(reproduced),
            "historical_confidence_score": 72.0 if reproduced else 20.0,
            "generation_fingerprint": fingerprint,
            "affects_selection": bool(reproduced),
        },
    )
    return policy_path


def _ready_bootstrap(monkeypatch, *, fingerprint="gen-A"):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "SUCCEEDED",
            "analysis_complete": True,
            "paper_ready_setups": 1,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "SUCCEEDED"})
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {
            "fingerprint": fingerprint,
            "historical_replay_required": False,
            "changed": False,
        },
    )


def test_autonomous_bootstrap_does_not_make_startup_evidence_ready(monkeypatch, tmp_path):
    """Bootstrap SUCCEEDED is not a substitute for current official history + scan."""
    _patch_operational_healthy(monkeypatch, tmp_path)
    _ready_bootstrap(monkeypatch)
    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_k: {
            "current": False,
            "available_session": "2026-09-01",
            "expected_latest_completed_session": "2026-09-05",
            "reason_code": "HISTORY_STALE",
        },
    )
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda *_a, **_k: {},
        raising=False,
    )
    monkeypatch.setattr(
        "product.scan_store.default_scan_path",
        lambda: tmp_path / "missing_scan.json",
        raising=False,
    )

    payload = build_startup_check(probe_network=False)
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["ready"] is False
    assert payload["live_locked"] is True


def test_startup_evidence_ready_does_not_skip_history_first_paper_gate(monkeypatch, tmp_path):
    """Current scan/history does not authorize paper entry without reproduced HIST_SETUP."""
    _patch_operational_healthy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_k: {
            "current": True,
            "available_session": "2026-09-05",
            "expected_latest_completed_session": "2026-09-05",
            "reason_code": "",
        },
    )
    now = datetime.now(timezone.utc) - timedelta(hours=0.5)
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda *_a, **_k: {
            "schema_version": 1,
            "scanned_at": now.isoformat(),
            "records": [{"symbol": "AAA"}],
            "available": True,
        },
        raising=False,
    )
    payload = build_startup_check(probe_network=False)
    assert payload["evidence_ready"] is True

    import product.autonomous_evolution as evolution

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {"status": "NOT_STARTED", "analysis_complete": False, "paper_ready_setups": 0},
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "NOT_STARTED"})
    out = _cycle(_vcp_card())
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def test_reproduced_setup_still_paper_eligible_after_runtime_merge(monkeypatch, tmp_path):
    _ready_bootstrap(monkeypatch, fingerprint="gen-A")
    _write_hist_vcp(tmp_path, monkeypatch, fingerprint="gen-A")
    book = PaperBook(capital=100_000)
    out = _cycle(_vcp_card(), book=book)
    assert out["taken"]
    assert next(iter(book.open.values())).symbol == "TCS"


def test_generation_mismatch_still_blocks_after_runtime_merge(monkeypatch, tmp_path):
    _ready_bootstrap(monkeypatch, fingerprint="gen-A")
    _write_hist_vcp(tmp_path, monkeypatch, fingerprint="gen-OLD")
    out = _cycle(_vcp_card())
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def test_explicit_path_still_does_not_bypass_history_gate(monkeypatch, tmp_path):
    _ready_bootstrap(monkeypatch, fingerprint="gen-A")
    path = tmp_path / "empty.json"
    path.write_text('{"policies": []}', encoding="utf-8")
    effect = evaluate_policies({"setup_label": "VCP"}, path=path)
    assert effect["historical_forward_confidence"]["required"] is True
    assert effect["final_effect"] == BLOCK


def test_injected_policies_remain_the_explicit_seam(monkeypatch):
    import product.autonomous_evolution as evolution

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {"status": "NOT_STARTED", "analysis_complete": False, "paper_ready_setups": 0},
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "NOT_STARTED"})
    skipped = evaluate_policies({"setup_label": "VCP"}, policies=[])
    assert skipped["historical_forward_confidence"]["required"] is False
    assert skipped["final_effect"] != BLOCK


def test_history_gate_does_not_hide_invalid_stop_when_setup_reproduced(monkeypatch, tmp_path):
    _ready_bootstrap(monkeypatch, fingerprint="gen-A")
    _write_hist_vcp(tmp_path, monkeypatch, fingerprint="gen-A")
    out = _cycle(_vcp_card(stop=101.0))
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == INVALID_STOP


def test_operational_up_stale_evidence_is_degraded_not_starting(monkeypatch, tmp_path):
    ops = tmp_path / "ops.json"
    ops.write_text(
        '{"process_running": true, "worker_pid": 1, "heartbeat_epoch": 9999999999}',
        encoding="utf-8",
    )
    monkeypatch.setenv("QT_MARKET_OPS_RUNTIME", str(ops))
    monkeypatch.setattr("product.runtime_lifecycle._pid_alive", lambda pid: int(pid or 0) == 1)
    monkeypatch.setattr("product.runtime_lifecycle._port_open", lambda port: port in {5173, 8765, 8766})
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *_a, **_k: {"current": False, "ready": True, "available_session": "2026-09-01", "reason_code": "HISTORY_STALE"},
    )
    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        lambda **_k: {"current": False, "ready": True, "available_session": "2026-09-01"},
    )
    payload = inspect_runtime(api_serving=True)
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["lifecycle"] == DEGRADED
    assert payload["lifecycle"] != "STARTING"


def test_live_adapter_remains_locked_on_combined_head():
    try:
        LiveExecutionAdapter().submit(object())
    except LiveMoneyLocked:
        return
    raise AssertionError("live adapter must refuse")


def test_history_gate_source_has_no_pytest_bypass():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "product" / "evidence_policy_engine.py").read_text(
        encoding="utf-8"
    )
    assert "PYTEST_CURRENT_TEST" not in src
    assert "enforce_history = bool(store_backed and path is None)" not in src
