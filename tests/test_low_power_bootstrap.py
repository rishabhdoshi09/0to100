"""Low-power must not starve autopilot: market-scan bootstrap stays on."""
from __future__ import annotations

from pathlib import Path


def test_low_power_launcher_uses_complete_stack():
    root = Path(__file__).resolve().parents[1]
    low = (root / "scripts" / "run_quantterm_low_power.sh").read_text(encoding="utf-8")
    env = (root / "scripts" / "apply_low_power_env.sh").read_text(encoding="utf-8")
    assert "run_quantterm_complete.sh" in low
    assert "apply_low_power_env.sh" in low
    assert "QT_LOW_POWER=1" in env
    assert "QT_DISABLE_AUTO_MARKET_SCAN" not in env.split("export")[0] or "unset QT_DISABLE_AUTO_MARKET_SCAN" in env
    assert "unset QT_DISABLE_AUTO_MARKET_SCAN" in env
    assert "unset QT_DISABLE_AUTO_LONG_TERM" in env
    assert "autopilot" in env.lower() or "market scan" in env.lower()


def test_low_power_does_not_skip_market_scan_bootstrap(tmp_path, monkeypatch):
    import operations.market_ops as mo

    monkeypatch.setenv("QT_LOW_POWER", "1")
    monkeypatch.delenv("QT_DISABLE_AUTO_MARKET_SCAN", raising=False)
    monkeypatch.setattr(mo, "ROOT", tmp_path)
    product = tmp_path / "logs" / "product"
    product.mkdir(parents=True)
    # Stale / missing scan artifact → bootstrap should enqueue MARKET_SCAN
    class FakeStore:
        def __init__(self):
            self.kinds = []

        def enqueue(self, kind, **kwargs):
            self.kinds.append(kind)
            return {"operation_id": kind}, True

    worker = mo.MarketOperationsWorker.__new__(mo.MarketOperationsWorker)
    worker.store = FakeStore()
    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        lambda load_cache=False: {"ready": True, "sessions": 100, "cache_exists": True, "csv_files": 100},
        raising=False,
    )
    # Patch import path used inside _bootstrap
    import sys
    import types

    fake = types.ModuleType("data.bhavcopy_runtime")
    fake.status = lambda load_cache=False: {
        "ready": True, "sessions": 100, "cache_exists": True, "csv_files": 100,
    }
    monkeypatch.setitem(sys.modules, "data.bhavcopy_runtime", fake)

    queued = mo.MarketOperationsWorker._bootstrap(worker)
    assert mo.MARKET_SCAN in queued
    assert mo.MARKET_SCAN in worker.store.kinds


def test_eco_honors_qt_low_power(monkeypatch):
    import core.eco as eco

    monkeypatch.delenv("QT_ECO", raising=False)
    monkeypatch.setenv("QT_LOW_POWER", "1")
    assert eco.eco_on() is True
    assert eco.workers(8) == 2
