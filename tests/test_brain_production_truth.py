"""Regression coverage for Brain readers using canonical production truth."""
from types import SimpleNamespace
import sys

from core import brain
import product.scan_store as scan_store
import product.paper_status as paper_status


def test_in_brain_prefers_canonical_saved_scan_over_legacy_auto_scan(monkeypatch):
    payload = {
        "schema_version": 2,
        "scanned_at": "2026-10-05T10:00:00+00:00",
        "records": [
            {"symbol": "CANON", "verdict": "BUY", "score": 91.0},
        ],
    }
    monkeypatch.setattr(scan_store, "resolved_scan_path", lambda: "canonical.json")
    monkeypatch.setattr(
        scan_store,
        "load_scan",
        lambda path: payload if str(path) == "canonical.json" else None,
    )

    class Legacy:
        @staticmethod
        def get_results():
            raise AssertionError("legacy auto_scan must not win when canonical scan exists")

    monkeypatch.setitem(sys.modules, "scan.auto_scan", Legacy)

    rows, stamp = brain._probe_setups("IN")
    assert [row["symbol"] for row in rows] == ["CANON"]
    assert stamp > 0


def test_in_brain_uses_modern_paper_status_before_legacy_autopilot(monkeypatch):
    modern = SimpleNamespace(
        enabled=True,
        open_positions=({"symbol": "AAA"}, {"symbol": "BBB"}),
        last_cycle={"positions_opened": [{"symbol": "BBB"}]},
    )
    monkeypatch.setattr(paper_status, "read_paper_status", lambda: modern)

    class Legacy:
        @staticmethod
        def get_status():
            raise AssertionError("legacy autopilot must not win when modern paper status is readable")

        @staticmethod
        def pnl_snapshot():
            raise AssertionError("legacy pnl must not be read")

    monkeypatch.setitem(sys.modules, "execution.autopilot", Legacy)

    state = brain._probe_autopilot("IN")
    assert state["armed"] is True
    assert state["paper_open_positions"] == 2
    assert state["trades_today"] == 1
    assert state["source"] == "product.paper_status"


def test_in_brain_falls_back_to_legacy_only_when_modern_status_fails(monkeypatch):
    def fail():
        raise RuntimeError("modern store unavailable")

    monkeypatch.setattr(paper_status, "read_paper_status", fail)

    legacy = SimpleNamespace(
        get_status=lambda: {"armed": True, "trades_today_count": 3},
        pnl_snapshot=lambda: {"day_pnl": 125.0},
    )
    monkeypatch.setitem(sys.modules, "execution.autopilot", legacy)

    state = brain._probe_autopilot("IN")
    assert state == {"armed": True, "trades_today": 3, "day_pnl": 125.0}
