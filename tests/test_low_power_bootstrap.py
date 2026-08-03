"""Low-power mode must not auto-queue heavy scans at worker bootstrap."""
from __future__ import annotations

from pathlib import Path

from operations.store import OperationStore


def test_low_power_skips_heavy_bootstrap_scans(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO

    monkeypatch.setenv("QT_LOW_POWER", "1")
    monkeypatch.setattr(MO, "ROOT", tmp_path)
    monkeypatch.setattr(MO, "LOCK_PATH", tmp_path / "market_ops" / "worker.lock")
    product = tmp_path / "logs" / "product"
    product.mkdir(parents=True)
    # Missing scan artifacts would normally enqueue MARKET_SCAN / LONG_TERM / US.
    worker = MO.MarketOperationsWorker(OperationStore(tmp_path / "jobs.db"))

    def history_status(*, load_cache: bool = False):
        return {
            "ready": False,
            "cache_exists": True,
            "csv_files": 200,
            "sessions": 0,
        }

    monkeypatch.setattr("data.bhavcopy_runtime.status", history_status)
    queued = worker._bootstrap()
    assert MO.MARKET_SCAN not in queued
    assert MO.LONG_TERM_SCAN not in queued
    assert MO.US_DATA_PREPARE not in queued
    assert MO.US_MARKET_SCAN not in queued


def test_scanner_honours_qt_scan_workers(monkeypatch):
    monkeypatch.setenv("QT_SCAN_WORKERS", "2")
    from scan.unified_scanner import UnifiedScanner

    assert UnifiedScanner()._max_workers == 2
