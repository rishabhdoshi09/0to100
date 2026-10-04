"""Production cache preparation uses Kite once across broad scans and F&O."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

from scan import bulk_fetcher as bulk
from research.intelligence.data import snapshot_store as snapshots


@pytest.fixture
def kite_store(monkeypatch, tmp_path):
    store = snapshots.SnapshotStore(tmp_path / "snapshots")
    monkeypatch.setattr(snapshots, "SnapshotStore", lambda *a, **k: store)
    monkeypatch.setattr(bulk, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(bulk, "_kite_cache", {})
    monkeypatch.setattr(bulk, "_kite_snapshot_id", "")
    monkeypatch.setattr(bulk, "_kite_loaded_snapshot_id", "", raising=False)
    monkeypatch.setattr(bulk, "_KITE_LOAD_CHUNK_ROWS", 23, raising=False)
    return store


def activate(store, names=("AAA", "BBB"), price=100, days=45):
    start = date(2026, 7, 1)
    rows = [(name, (start + timedelta(days=i)).isoformat(), price, price+2,
             price-1, price+1, 1000, "EQ") for name in names for i in range(days)]
    sid = store.commit_snapshot(rows, extra_manifest={"source": "kite"})
    store.activate_snapshot(sid)
    return sid


def test_ready_store_adopts_kite_not_the_unrelated_bhavcopy_cache(kite_store, monkeypatch):
    sid = activate(kite_store)
    monkeypatch.setattr("data.bhavcopy_runtime.status", lambda **k: {
        "ready": True, "symbols": 3401, "sessions": 1821,
    })
    monkeypatch.setattr(bulk, "_bhav_symbols", lambda: {"PUBLIC"})
    bulk._kite_cache["ADHOC"] = object()
    assert bulk.adopt_ready_store(overlay_live=False) == 2
    assert bulk.cached_symbols() == ["AAA", "BBB"]
    assert bulk.get_cached("AAA").attrs["quantterm_snapshot_id"] == sid


def test_concurrent_reads_and_gaps_load_one_verified_snapshot(kite_store, monkeypatch):
    sid = activate(kite_store)
    calls = []
    original_verify = kite_store.verify_snapshot

    def verify(snapshot_id):
        calls.append(snapshot_id)
        return original_verify(snapshot_id)

    monkeypatch.setattr(kite_store, "verify_snapshot", verify)
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(bulk.get_cached, ["AAA", "BBB", "GAP"] * 6))
    assert calls == [sid]
    for symbol, frame in zip(["AAA", "BBB", "GAP"] * 6, frames):
        if symbol == "GAP":
            assert frame is None
        else:
            assert frame is not None
            assert len(frame) == 45
            assert frame.index.is_unique
            assert frame.attrs["quantterm_source"] == "kite_snapshot"


def test_bulk_load_does_not_construct_the_point_in_time_row_object(kite_store, monkeypatch):
    activate(kite_store)
    from research.intelligence.data import snapshot

    def forbidden(*args, **kwargs):
        raise AssertionError("bulk cache must not duplicate the snapshot as Python row dictionaries")

    monkeypatch.setattr(snapshot, "Snapshot", forbidden)
    assert bulk.get_cached("AAA") is not None
    assert bulk.get_cached("BBB") is not None


def test_pointer_rotation_clears_frames_and_cached_gaps(kite_store):
    first = activate(kite_store, names=("AAA",))
    assert bulk.get_cached("AAA").attrs["quantterm_snapshot_id"] == first
    assert bulk.get_cached("BBB") is None
    second = activate(kite_store, names=("BBB",), price=200)
    assert bulk.get_cached("AAA") is None
    frame = bulk.get_cached("BBB")
    assert frame.attrs["quantterm_snapshot_id"] == second
    assert float(frame["close"].iloc[-1]) == 201
    (kite_store.root / "ACTIVE").unlink()
    assert bulk.get_cached("BBB") is None
    assert bulk.cached_symbols() == []


def test_rotation_during_load_never_publishes_old_frames(kite_store, monkeypatch):
    import json

    first = activate(kite_store, names=("AAA",))
    second = activate(kite_store, names=("BBB",), price=200)
    kite_store.activate_snapshot(first)
    original_verify = kite_store.verify_snapshot

    def rotate(snapshot_id):
        result = original_verify(snapshot_id)
        if snapshot_id == first:
            (kite_store.root / "ACTIVE").write_text(json.dumps({"snapshot_id": second}))
        return result

    monkeypatch.setattr(kite_store, "verify_snapshot", rotate)
    assert bulk.get_cached("AAA") is None
    assert bulk.cached_symbols() == []
    assert bulk.get_cached("BBB").attrs["quantterm_snapshot_id"] == second


def test_market_worker_scans_the_snapshot_universe_despite_warm_public_history(kite_store, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from operations.market_ops import MarketOperationsWorker
    from scan import market_scan_service
    from scan.unified_scanner import UnifiedScanner

    names = [f"STOCK{i:03}" for i in range(240)]
    sid = activate(kite_store, names=names, days=70)
    monkeypatch.setattr(bulk, "_KITE_LOAD_CHUNK_ROWS", 100_000)
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr("data.bhavcopy_runtime.status", lambda **k: {
        "ready": True, "symbols": 3401, "sessions": 1821,
    })
    monkeypatch.setattr(bulk, "_bhav_symbols", lambda: set(names))
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: names)
    monkeypatch.setattr("core.regime_engine.compute_regime", lambda: SimpleNamespace(market_regime=""))
    monkeypatch.setattr("core.regime_engine.peek_cached_regime", lambda: None)
    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *a, **k: [])
    monkeypatch.setattr("scan.unified_scanner._nifty_return_30d", lambda: 0)
    monkeypatch.setattr(market_scan_service, "_saved_priority_inputs", lambda: ({}, {}, {}, []))
    scanner = UnifiedScanner(calibration_snapshot={"multipliers": {}})
    seen = []

    def analyze(symbol, frame):
        assert frame.attrs["quantterm_snapshot_id"] == sid
        seen.append(symbol)
        return None

    monkeypatch.setattr(scanner, "_analyze", analyze)
    original_scan = market_scan_service.run_whole_market_scan

    def run_scan(**kwargs):
        return original_scan(
            universe_provider=lambda: {name: name for name in names},
            prefetch_fn=kwargs["prefetch_fn"], scanner=scanner,
            fno_provider=lambda: set(), progress_callback=kwargs["progress_callback"],
            save=False,
        )

    monkeypatch.setattr(market_scan_service, "run_whole_market_scan", run_scan)
    worker = MarketOperationsWorker.__new__(MarketOperationsWorker)
    monkeypatch.setattr(worker, "_require_current_history", lambda op: {"sessions": 1821})
    monkeypatch.setattr(worker, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_notify_scan_telegram", lambda payload: {})
    result = worker._run_market_scan({"operation_id": "test", "payload": {}})
    assert set(seen) == set(names)
    assert len(seen) == 240
    assert result["scanned"] == 240
