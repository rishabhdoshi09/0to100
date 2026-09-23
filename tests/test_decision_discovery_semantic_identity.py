from __future__ import annotations

import product.decision_discovery_store as store
import product.long_term_store as long_term_store


def test_discovery_survives_timestamp_only_long_term_refresh(tmp_path, monkeypatch):
    target = tmp_path / "startup_trade_discovery.json"
    monkeypatch.setattr(store, "_path", lambda: target)

    current = {
        "scanned_at": "2026-09-23T04:55:00+00:00",
        "records": [{"symbol": "AAA", "combined_score": 72.0}],
        "summary": {"candidates": 1},
    }
    monkeypatch.setattr(long_term_store, "load_long_term_scan", lambda: dict(current))

    board = {"available": True, "best_trades": [{"symbol": "AAA"}], "actionable": 1}
    store.save(
        board,
        scan_scanned_at="scan-1",
        long_term_scanned_at=current["scanned_at"],
        thesis_hash="thesis-1",
    )

    current["scanned_at"] = "2026-09-23T04:56:00+00:00"
    assert store.load(
        scan_scanned_at="scan-1",
        long_term_scanned_at=current["scanned_at"],
        thesis_hash="thesis-1",
    ) == board


def test_discovery_fails_closed_on_material_long_term_change(tmp_path, monkeypatch):
    target = tmp_path / "startup_trade_discovery.json"
    monkeypatch.setattr(store, "_path", lambda: target)

    current = {
        "scanned_at": "2026-09-23T04:55:00+00:00",
        "records": [{"symbol": "AAA", "combined_score": 72.0}],
        "summary": {"candidates": 1},
    }
    monkeypatch.setattr(long_term_store, "load_long_term_scan", lambda: dict(current))

    store.save(
        {"available": True, "best_trades": [{"symbol": "AAA"}]},
        scan_scanned_at="scan-1",
        long_term_scanned_at=current["scanned_at"],
        thesis_hash="thesis-1",
    )

    current["scanned_at"] = "2026-09-23T04:56:00+00:00"
    current["records"] = [{"symbol": "BBB", "combined_score": 80.0}]
    assert store.load(
        scan_scanned_at="scan-1",
        long_term_scanned_at=current["scanned_at"],
        thesis_hash="thesis-1",
    ) is None
