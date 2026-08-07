"""Market-scan progress must stay visible — no silent freezes at 0/N."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from operations.store import OperationStore, RUNNING


def test_unified_scanner_reports_progress_frequently(monkeypatch):
    import pandas as pd

    from scan.unified_scanner import UnifiedScanner

    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    close = pd.Series(range(100, 100 + len(idx)), dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1_000_000.0] * len(idx),
        },
        index=idx,
    )
    symbols = [f"S{i:03d}" for i in range(25)]

    monkeypatch.setattr(
        "scan.bulk_fetcher.cached_symbols",
        lambda: list(symbols),
    )
    monkeypatch.setattr(
        "scan.bulk_fetcher.get_cached",
        lambda _s: frame,
    )
    monkeypatch.setattr(
        "scan.unified_scanner._nifty_return_30d",
        lambda: 0.02,
    )

    events: list[tuple[int, int]] = []

    def progress(current, total):
        events.append((int(current), int(total)))

    scanner = UnifiedScanner(max_workers=4)
    # Force a deterministic lightweight analyze so we only assert progress cadence.
    monkeypatch.setattr(scanner, "_analyze", lambda _sym, _df: None)

    out = scanner.scan(symbols, progress=progress, skip_prefetch=True)
    assert out == []
    assert events, "progress callback must fire"
    assert events[0][0] == 0
    assert events[-1][0] == events[-1][1] == 25
    # More frequent than the old every-40 cadence for a 25-name universe.
    assert len(events) >= 4


def test_market_scan_service_honours_skip_scanner_prefetch(monkeypatch):
    from scan import market_scan_service as MSS
    import product.scan_store as store

    calls = {"prefetch": 0, "scan_kwargs": None}

    def fake_universe():
        return {"AAA": "A", "BBB": "B"}

    def fake_prefetch(symbols, *, progress=None):
        calls["prefetch"] += 1
        if progress:
            progress(0, len(symbols))
        return len(symbols)

    class FakeScanner:
        def scan(self, symbols, progress=None, *, skip_prefetch: bool = False):
            calls["scan_kwargs"] = {"skip_prefetch": skip_prefetch, "n": len(symbols)}
            if progress:
                progress(len(symbols), len(symbols))
            return []

    monkeypatch.setattr(store, "build_scan_payload", lambda *_a, **_k: {
        "summary": {"with_any_setup": 0},
        "records": [],
        "universe_size": 2,
    })
    monkeypatch.setattr(store, "save_scan", lambda *_a, **_k: None)

    report = MSS.run_whole_market_scan(
        universe_provider=fake_universe,
        prefetch_fn=fake_prefetch,
        scanner=FakeScanner(),
        fno_provider=lambda: set(),
        save=False,
        skip_scanner_prefetch=True,
    )
    assert report.ok
    assert calls["prefetch"] == 1
    assert calls["scan_kwargs"]["skip_prefetch"] is True


def test_history_lock_wait_emits_progress(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO

    store = OperationStore(tmp_path / "ops.db")
    worker = MO.MarketOperationsWorker(store)
    queued, _ = store.enqueue("MARKET_SCAN", lane="market_scan")
    leased = store.lease_next("market_scan", worker_pid=1)
    assert leased is not None

    # Hold the history lock in another thread so the scan must wait + announce.
    held = threading.Event()
    release = threading.Event()

    def holder():
        worker._history_lock.acquire()
        held.set()
        release.wait(timeout=5)
        worker._history_lock.release()

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(timeout=2)

    stages: list[str] = []
    original = worker._progress

    def capture(operation_id, stage, message, current=None, total=None):
        stages.append(str(stage))
        original(operation_id, stage, message, current, total)

    worker._progress = capture  # type: ignore[method-assign]
    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        lambda load_cache=True: {"ready": True, "sessions": 200, "symbols": 50},
    )

    def run_ensure():
        worker._ensure_history(leased["operation_id"])

    runner = threading.Thread(target=run_ensure, daemon=True)
    runner.start()
    # Give the waiter at least one 0.5s timeout cycle.
    time.sleep(0.7)
    release.set()
    runner.join(timeout=3)
    thread.join(timeout=3)

    assert "WAITING_HISTORY" in stages
    row = store.get(leased["operation_id"])
    assert row is not None
    assert row["status"] == RUNNING
