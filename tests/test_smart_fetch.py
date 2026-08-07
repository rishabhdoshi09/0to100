"""Load-safe symbol smart-fetch scheduler."""
from __future__ import annotations

import time
from pathlib import Path

import product.smart_fetch as SF
import reporting.evidence_intake as EI


def _reset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(SF, "STATUS_PATH", tmp_path / "smart_fetch_status.json")
    monkeypatch.setattr(SF, "_worker_started", False)
    monkeypatch.setattr(SF, "_queue", SF.deque())
    monkeypatch.setattr(SF, "_jobs", {})
    monkeypatch.setattr(EI, "ROOT", tmp_path)
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")


def test_schedule_is_single_flight_and_cooldowns(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    monkeypatch.setenv("QT_LOW_POWER", "1")

    calls = {"n": 0}

    def fake_run(symbol, job):
        calls["n"] += 1
        time.sleep(0.05)
        return {
            "accepted": True,
            "attached_count": 1,
            "failed_count": 0,
            "results": [{"kind": "financial_history", "ok": True}],
            "message": "ok",
        }

    monkeypatch.setattr(SF, "_run_job", fake_run)

    first = SF.schedule_symbol_fetch("TCS", requested_by="test")
    second = SF.schedule_symbol_fetch("TCS", requested_by="test")
    assert first["created"] is True
    assert second["created"] is False
    assert second["job"]["status"] in {"QUEUED", "RUNNING"}

    deadline = time.time() + 3
    while time.time() < deadline:
        status = SF.symbol_fetch_status("TCS")
        if status["job"] and status["job"].get("status") == "SUCCEEDED":
            break
        time.sleep(0.05)
    assert SF.symbol_fetch_status("TCS")["job"]["status"] == "SUCCEEDED"
    assert calls["n"] == 1

    cooled = SF.schedule_symbol_fetch("TCS", requested_by="test")
    assert cooled["created"] is False
    assert "cooldown" in (cooled.get("message") or "").lower()


def test_queue_depth_bounded(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    monkeypatch.setattr(SF, "_MAX_QUEUE", 2)

    # Never finish jobs — keep them queued by not starting worker consumption.
    # Start worker but block _run_job forever after first pop via a latch.
    block = {"go": False}

    def slow_run(symbol, job):
        while not block["go"]:
            time.sleep(0.01)
        return {"accepted": True, "attached_count": 0, "failed_count": 0, "results": [], "message": "done"}

    monkeypatch.setattr(SF, "_run_job", slow_run)

    SF.schedule_symbol_fetch("AAA")
    SF.schedule_symbol_fetch("BBB")
    SF.schedule_symbol_fetch("CCC")
    SF.schedule_symbol_fetch("DDD")

    snap = SF.scheduler_snapshot()
    # At most one running + bounded queued.
    assert snap["queue_depth"] <= SF._MAX_QUEUE
    statuses = {j.get("status") for j in SF._jobs.values()}
    assert "DROPPED" in statuses or snap["queue_depth"] <= 2
    block["go"] = True
    time.sleep(0.2)


def test_product_bootstrap_skips_heavy_lanes_in_low_power(monkeypatch):
    import os

    monkeypatch.setenv("QT_LOW_POWER", "1")
    from operations.market_ops import DATA_PREPARE, LONG_TERM_REFRESH, MARKET_SCAN, NEWS_REFRESH

    enqueued = []

    class FakeStore:
        def enqueue(self, kind, *, lane, requested_by="terminal", **kwargs):
            enqueued.append(kind)
            return {"operation_id": kind, "status": "PENDING"}, True

    import terminal_product_api as TPA

    monkeypatch.setattr(TPA.core, "_ensure_ops_worker", lambda: None)
    monkeypatch.setattr("operations.store.OperationStore", lambda *a, **k: FakeStore())
    monkeypatch.setattr(TPA, "product_readiness", lambda: {"ok": True})

    # Call the endpoint function directly.
    result = TPA.product_bootstrap()
    assert result["low_power"] is True
    assert DATA_PREPARE in enqueued
    assert NEWS_REFRESH in enqueued
    assert MARKET_SCAN not in enqueued
    assert LONG_TERM_REFRESH not in enqueued
    assert os.getenv("QT_LOW_POWER") == "1"


def test_autofetch_only_missing_skips_covered(tmp_path, monkeypatch):
    import reporting.evidence_autofetch as AF

    monkeypatch.setattr(EI, "ROOT", tmp_path)
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(AF, "save_upload", EI.save_upload)
    monkeypatch.setattr(AF, "evidence_requirements", EI.evidence_requirements)
    monkeypatch.setattr(AF, "load_raw_fundamentals", lambda symbol, auto_fetch=False: {"data": {}})

    EI.save_upload(
        "SKIPME",
        "annual_report",
        b"%PDF-1.4 already here",
        filename="ar.pdf",
        as_of="2026-01-01",
        source_url="https://example.com/ar.pdf",
    )
    report = AF.autofetch_evidence(
        "SKIPME",
        kinds=["annual_report"],
        refresh_screener=False,
        only_missing=True,
    )
    assert report.get("skipped") is True
    assert report["attached_count"] == 0
