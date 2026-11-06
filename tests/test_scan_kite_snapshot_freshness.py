"""Regression gates for a Nov-06 saved scan pointing at a superseded Kite snapshot.

A fresh NSE bhavcopy must never certify historical prices from an older
active Kite snapshot. No broker access or live orders are involved.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

from product import scan_store as S


class _Snapshot:
    def __init__(self, root, active):
        self.root = root
        self.active = active

    def get_active_snapshot(self):
        return self.active


def _install_snapshot(monkeypatch, tmp_path, *, active="new", day="2026-11-06"):
    from research.intelligence.data import snapshot_store

    root = tmp_path / "snapshots"
    directory = root / active
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "source": "kite",
        "last_trading_date": day,
        "instrument_count": 2428,
    }))
    fake = _Snapshot(root, active)
    monkeypatch.setattr(snapshot_store, "SnapshotStore", lambda: fake)
    return fake


def _scan(*, sid="new", day="2026-11-06", source="kite_snapshot"):
    return {
        "schema_version": 2,
        "source_snapshot_id": sid,
        "scanned_at": "2026-11-06T13:16:18+00:00",
        "provenance": {
            "market_session_date": day,
            "price_data_as_of": day,
            "expected_session_date": day,
            "data_freshness": "HISTORY_CURRENT",
            "data_current": True,
            "price_source": source,
        },
        "records": [{"symbol": "KRISHANA", "price": 212.89, "verdict": "BUY"}],
    }


def test_old_scan_is_rejected_after_new_snapshot_activation(monkeypatch, tmp_path):
    _install_snapshot(monkeypatch, tmp_path, active="0d50c14e22b06718")
    payload = _scan(sid="2ee235366f4818f5", source="official_nse_bhavcopy")
    valid, reason = S.scan_source_status(payload)
    assert valid is False
    assert reason == "SNAPSHOT_ID_MISMATCH"

    saved = tmp_path / "latest_momentum_scan.json"
    S.save_scan(payload, saved)
    assert S.load_scan(saved) is None


def test_matching_kite_snapshot_must_use_actual_date_and_source(monkeypatch, tmp_path):
    _install_snapshot(monkeypatch, tmp_path, active="0d50c14e22b06718")
    assert S.scan_source_status(_scan(sid="0d50c14e22b06718")) == (True, "")
    assert S.scan_source_status(
        _scan(sid="0d50c14e22b06718", day="2026-10-19")
    ) == (False, "SNAPSHOT_DATE_MISMATCH")
    assert S.scan_source_status(
        _scan(sid="0d50c14e22b06718", source="official_nse_bhavcopy")
    ) == (False, "SCAN_SOURCE_MISLABELLED")


def test_kite_manifest_unavailable_is_not_an_approved_scan(monkeypatch, tmp_path):
    from research.intelligence.data import snapshot_store

    fake = _Snapshot(tmp_path / "missing", "new")
    monkeypatch.setattr(snapshot_store, "SnapshotStore", lambda: fake)
    valid, reason = S.scan_source_status(_scan(sid="new"))
    assert not valid
    assert reason == "SNAPSHOT_VERIFICATION_FAILED"


class _Scanner:
    def scan(self, symbols, *, progress=None, prefetch=False):
        return [{
            "symbol": "AAA", "score": 80, "verdict": "BUY",
            "signals": ["MOMENTUM"], "reasons": ["qualified"], "price": 100,
        }]


class _Coverage:
    def finalize(self, _results, *, cached, walked_total):
        return {"summary": {"checked": 1, "state": "COMPLETE", "failed": 0}, "ledger": []}


@contextmanager
def _observe(*_args, **_kwargs):
    yield _Coverage()


def test_snapshot_switch_during_scan_blocks_publication(monkeypatch):
    from scan import market_scan_service as market
    from scan import scan_coverage

    seen = iter(["old", "new"])
    monkeypatch.setattr(market, "_active_snapshot_id", lambda: next(seen))
    monkeypatch.setattr(market, "_saved_priority_inputs", lambda: (None, None, None, []))
    monkeypatch.setattr(scan_coverage, "observe_scanner", _observe)
    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: ["AAA"])

    report = market.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda *_a, **_k: None,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert not report.ok
    assert report.error_code == "SCAN_SNAPSHOT_CHANGED"
    assert report.payload == {}
