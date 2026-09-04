from __future__ import annotations

from types import SimpleNamespace

import data.bhavcopy_runtime as bhavcopy_runtime
import product.paper_status as paper_status
from product.startup_check import (
    _history_readiness,
    _paper_readiness,
    _required_waiting,
    build_startup_check,
    maybe_open_home_browser,
)


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

    ready, detail = _history_readiness()

    assert ready is False
    assert "HISTORY_STALE" in detail
    assert "available 2026-09-02" in detail
    assert "expected 2026-09-03" in detail


def test_paper_readiness_requires_live_supervisor_not_truthy_default(monkeypatch):
    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=False, enabled=True),
    )
    ready, detail = _paper_readiness()
    assert ready is False
    assert "not running" in detail.lower()

    monkeypatch.setattr(
        paper_status,
        "read_paper_status",
        lambda: SimpleNamespace(supervisor_running=True, enabled=False),
    )
    ready, detail = _paper_readiness()
    assert ready is True
    assert "paused" in detail.lower()


def test_required_lanes_are_authoritative_for_ready_state():
    lanes = [
        {"name": "UI", "status": "READY", "required": True},
        {"name": "API", "status": "READY", "required": True},
        {"name": "DATA", "status": "WAITING", "required": True},
        {"name": "REPORTS", "status": "WAITING", "required": False},
        {"name": "LIVE MONEY", "status": "LOCKED", "required": True},
    ]
    waiting = _required_waiting(lanes)
    assert [row["name"] for row in waiting] == ["DATA"]


def test_browser_open_skipped_when_noninteractive(monkeypatch):
    monkeypatch.setenv("QT_NONINTERACTIVE", "1")
    assert maybe_open_home_browser() is False
    monkeypatch.delenv("QT_NONINTERACTIVE", raising=False)
    monkeypatch.setenv("QT_NO_BROWSER", "1")
    assert maybe_open_home_browser() is False
