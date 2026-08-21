"""Autopilot silence diagnosis + market-scan → on_setups feed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_sector_gate_fail_open_when_heat_unavailable(monkeypatch, tmp_path):
    # Stub heavy deps before loading autopilot
    sys.modules.setdefault("logger", type(sys)("logger"))
    sys.modules["logger"].get_logger = lambda *_a, **_k: type("L", (), {"warning": lambda *a, **k: None, "info": lambda *a, **k: None, "debug": lambda *a, **k: None})()
    # market_clock already used by autopilot — ensure importable
    ap = _load("ap_under_test", "execution/autopilot.py")

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_load", lambda: {
        **ap._DEFAULTS,
        "armed": True,
        "allocation": 100_000,
        "brain_gate": False,
        "regime_gate": False,
        "symbol_memory_gate": False,
        "min_score": 50,
    })
    monkeypatch.setattr(ap, "_in_window", lambda: True)
    monkeypatch.setattr(ap, "_open_autopilot_trades", lambda: [])
    monkeypatch.setattr(ap, "_top_sectors", lambda: ([], "unavailable"))
    assert ap._passes_gates("RELIANCE", 70.0, 0.2, "Energy", volume_ratio=1.5) is None


def test_sector_gate_blocks_when_all_sectors_red(monkeypatch, tmp_path):
    ap = sys.modules.get("ap_under_test") or _load("ap_under_test", "execution/autopilot.py")
    monkeypatch.setattr(ap, "_load", lambda: {
        **ap._DEFAULTS,
        "armed": True,
        "brain_gate": False,
        "regime_gate": False,
        "symbol_memory_gate": False,
        "min_score": 50,
    })
    monkeypatch.setattr(ap, "_in_window", lambda: True)
    monkeypatch.setattr(ap, "_open_autopilot_trades", lambda: [])
    monkeypatch.setattr(ap, "_top_sectors", lambda: ([], "ok"))
    reason = ap._passes_gates("RELIANCE", 70.0, 0.2, "Energy", volume_ratio=1.5)
    assert reason and "no positive sector" in reason


def test_market_scan_feed_helper_calls_on_setups(monkeypatch):
    mss = _load("mss_under_test", "scan/market_scan_service.py")
    called = {"on_setups": 0, "review": 0}

    class FakeAP:
        @staticmethod
        def on_setups(rows):
            called["on_setups"] += 1
            assert any(r.get("verdict") == "BUY" for r in rows)

        @staticmethod
        def review_cycle():
            called["review"] += 1

    monkeypatch.setitem(sys.modules, "execution.autopilot", FakeAP)
    monkeypatch.setitem(sys.modules, "execution", type(sys)("execution"))
    sys.modules["execution"].autopilot = FakeAP

    class FakeSession:
        @staticmethod
        def in_market_open():
            return True

    monkeypatch.setitem(sys.modules, "core.market_session", FakeSession)
    mss._feed_autopilot_from_scan(
        {
            "records": [
                {
                    "symbol": "ABC",
                    "verdict": "BUY",
                    "entry": 100,
                    "stop": 95,
                    "score": 70,
                    "edge_r": 0.2,
                }
            ]
        }
    )
    assert called["on_setups"] == 1
    assert called["review"] == 1


def test_diagnose_silence_reports_disarmed(monkeypatch, tmp_path):
    ap = sys.modules.get("ap_under_test") or _load("ap_under_test", "execution/autopilot.py")
    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    ap._state = dict(ap._DEFAULTS)
    ap._state["armed"] = False
    ap._state["disarmed_reason"] = "user"
    monkeypatch.setattr(ap, "reject_funnel", lambda: {"considered": 0, "rejects": {}})

    class FakeScanStore:
        @staticmethod
        def load_scan():
            return {"scanned_at": "2026-08-11", "records": [{"verdict": "BUY", "chase_risk": False}]}

    monkeypatch.setitem(sys.modules, "product.scan_store", FakeScanStore)
    d = ap.diagnose_silence()
    assert d["armed"] is False
    assert any("DISARMED" in b for b in d["blockers"])
