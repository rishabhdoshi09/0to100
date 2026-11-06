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


def test_health_does_not_claim_evidence_ready_for_stale_kite_scan(monkeypatch, tmp_path):
    from product.startup_check import _scan_evidence_status

    _install_snapshot(monkeypatch, tmp_path, active="0d50c14e22b06718")
    old = _scan(sid="2ee235366f4818f5", source="official_nse_bhavcopy")
    old["as_of_session"] = "2026-11-06"
    status, detail = _scan_evidence_status(old, expected_session="2026-11-06")
    assert status == "STALE"
    assert "SNAPSHOT_ID_MISMATCH" in detail

    current = _scan(sid="0d50c14e22b06718")
    current["as_of_session"] = "2026-11-06"
    status, detail = _scan_evidence_status(current, expected_session="2026-11-06")
    assert status == "READY"


def test_recommendations_hide_old_cards_when_scan_authority_is_missing(monkeypatch):
    from types import SimpleNamespace
    from product import recommendations_liveness as R

    monkeypatch.setattr(
        "product.recommendations_store.load_recommendations",
        lambda: {"records_status": "READY", "categories": [
            {"id": "buy", "cards": [{"symbol": "KRISHANA", "verdict": "BUY"}]}
        ]},
    )
    monkeypatch.setattr(
        R, "_ensure_rebuild",
        lambda *_a: (_ for _ in ()).throw(AssertionError("must not rebuild from stale scan")),
    )
    core = SimpleNamespace(
        _scan_payload=lambda: {"available": False},
        _long_term_payload=lambda: {"available": True},
    )
    result = R.build_fast_response(core)
    assert result["records_status"] == "FAILED"
    assert "stale" in result["error"].lower()
    assert all(not group.get("cards") for group in result.get("categories", []))


def _prepare_kite_publication(monkeypatch, tmp_path, *, kite_date):
    from types import SimpleNamespace
    from scan import market_scan_service as market
    from scan import scan_coverage

    _install_snapshot(monkeypatch, tmp_path, active="new", day=kite_date)
    monkeypatch.setattr(market, "_saved_priority_inputs", lambda: (None, None, None, []))
    monkeypatch.setattr(scan_coverage, "observe_scanner", _observe)
    monkeypatch.setattr(scan_coverage, "save_audit", lambda *_: None)
    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        S, "_history_freshness",
        lambda: ({
            "available_session": "2026-11-05",
            "expected_latest_completed_session": "2026-11-06",
            "reason_code": "HISTORY_STALE",
            "current": False,
        }, ""),
    )
    monkeypatch.setattr("product.sepa_setup.persist_public_best_setups", lambda *_: None)
    monkeypatch.setattr(
        "scan.long_term_service.overlay_long_term_from_market_scan",
        lambda *_a, **_kw: SimpleNamespace(status="SUCCEEDED", payload={}, error_code=""),
    )
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_desks_from_market_scan",
        lambda *_: {},
    )
    saved = []
    monkeypatch.setattr(S, "save_scan", lambda payload: saved.append(payload))
    return market, saved


def test_fresh_kite_snapshot_beats_lagging_bhavcopy_provenance(monkeypatch, tmp_path):
    market, saved = _prepare_kite_publication(
        monkeypatch, tmp_path, kite_date="2026-11-06",
    )
    result = market.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda *_a, **_kw: None,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert result.ok
    assert len(saved) == 1
    prov = saved[0]["provenance"]
    assert saved[0]["source_snapshot_id"] == "new"
    assert prov["price_data_as_of"] == "2026-11-06"
    assert prov["price_source"] == "kite_snapshot"
    assert prov["data_current"] is True
    assert prov["data_freshness"] == "KITE_SNAPSHOT_CURRENT"


def test_stale_kite_snapshot_is_not_published_even_if_scan_runs(monkeypatch, tmp_path):
    market, saved = _prepare_kite_publication(
        monkeypatch, tmp_path, kite_date="2026-11-05",
    )
    result = market.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda *_a, **_kw: None,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert result.ok is False
    assert result.error_code == "SCAN_INPUT_STALE"
    assert saved == []


def test_production_paper_cycle_blocks_old_recommendations_before_entries(monkeypatch):
    from datetime import datetime, timezone
    from research.autonomy import jobs as J

    sent = []
    deps = J.Deps.__new__(J.Deps)
    deps.now_ist = lambda: datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
    deps.holidays = lambda: set()
    deps.run_paper_cycle = lambda ok, reason, phase, failures: (
        sent.append((ok, reason)) or {"eligibility": "NO_ELIGIBLE_TRADE"}
    )
    monkeypatch.setattr(J, "_entry_reason", lambda *_a: (True, "", "intraday"))
    monkeypatch.setattr(J, "_paper_market_data_source", lambda *_a: (True, "snapshot:new"))
    monkeypatch.setattr(
        "product.scan_store.load_scan", lambda: {
            "scanned_at": "2026-11-06T13:00:00+00:00",
            "source_snapshot_id": "new",
        },
    )
    monkeypatch.setattr(
        "product.recommendations_store.load_recommendations", lambda: {
            "scan_scanned_at": "2026-10-23T13:00:00+00:00",
        },
    )

    result = J.run_paper_cycle(J._Ctx(deps))
    assert sent == [(False, "STALE_SCAN_SNAPSHOT")]
    assert result.new_entries_allowed is False
    assert result.metadata["eligibility"] == "DATA_UNAVAILABLE"
    assert result.metadata["entry_block_reason"] == "STALE_SCAN_SNAPSHOT"


def test_production_paper_cycle_allows_verified_matching_scan_identity(monkeypatch):
    from datetime import datetime, timezone
    from research.autonomy import jobs as J

    sent = []
    deps = J.Deps.__new__(J.Deps)
    deps.now_ist = lambda: datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
    deps.holidays = lambda: set()
    deps.run_paper_cycle = lambda ok, reason, phase, failures: (
        sent.append((ok, reason)) or {"eligibility": "NO_ELIGIBLE_TRADE"}
    )
    monkeypatch.setattr(J, "_entry_reason", lambda *_a: (True, "", "intraday"))
    monkeypatch.setattr(J, "_paper_market_data_source", lambda *_a: (True, "snapshot:new"))
    monkeypatch.setattr(
        "product.scan_store.load_scan", lambda: {
            "scanned_at": "2026-11-06T13:00:00+00:00",
            "source_snapshot_id": "new",
        },
    )
    monkeypatch.setattr(
        "product.recommendations_store.load_recommendations", lambda: {
            "scan_scanned_at": "2026-11-06T13:00:00+00:00",
        },
    )

    result = J.run_paper_cycle(J._Ctx(deps))
    assert sent == [(True, "")]
    assert result.new_entries_allowed is True


def test_discovery_freshness_uses_verified_kite_instead_of_lagging_bhavcopy(
    monkeypatch, tmp_path
):
    from product import desk_pipeline
    from research.intelligence.data import nse_calendar

    _install_snapshot(monkeypatch, tmp_path, active="new", day="2026-11-06")
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    location = tmp_path / "product" / "latest_momentum_scan.json"
    S.save_scan(_scan(sid="new"), location)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda **_kw: {"usable_for_scan": False, "current": False},
    )
    # The scan's broker authority is independently verifiable. Missing public
    # NSE archive does not justify a false negative in discovery readiness.
    monkeypatch.setattr(
        nse_calendar, "snapshot_freshness",
        lambda *_args, **_kwargs: {"fresh": True},
    )
    assert desk_pipeline.scan_is_fresh() is True

    # A once-current Kite snapshot must still expire under the exchange
    # calendar: cached scan timestamps cannot bypass future session rollovers.
    monkeypatch.setattr(
        nse_calendar, "snapshot_freshness",
        lambda *_args, **_kwargs: {"fresh": False},
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_discovery_freshness_rejects_rotated_or_unverified_kite_scan(
    monkeypatch, tmp_path
):
    from product import desk_pipeline
    from research.intelligence.data import nse_calendar

    snapshot = _install_snapshot(monkeypatch, tmp_path, active="new")
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    location = tmp_path / "product" / "latest_momentum_scan.json"
    S.save_scan(_scan(sid="old"), location)
    monkeypatch.setattr(
        nse_calendar, "snapshot_freshness",
        lambda *_args, **_kwargs: {"fresh": True},
    )
    assert desk_pipeline.scan_is_fresh() is False
    S.save_scan(_scan(sid="new", source="official_nse_bhavcopy"), location)
    assert desk_pipeline.scan_is_fresh() is False

    S.save_scan(_scan(sid="new"), location)
    snapshot.active = "old"
    assert desk_pipeline.scan_is_fresh() is False
