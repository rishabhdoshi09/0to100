"""Scan Now writes recos + today's pulse; page-open reads those files."""
from __future__ import annotations

from product.desk_scan_overlays import persist_desks_from_market_scan
from product.recommendations_store import (
    load_recommendations,
    reco_matches_scan,
    save_recommendations,
)


def test_recommendations_store_roundtrip(tmp_path):
    path = tmp_path / "latest_recommendations.json"
    payload = {
        "schema_version": 4,
        "categories": [{"id": "wealth_builders", "count": 0, "cards": []}],
        "scan_scanned_at": "S1",
        "long_term_scanned_at": "L1",
    }
    save_recommendations(payload, path)
    loaded = load_recommendations(path)
    assert loaded["scan_scanned_at"] == "S1"
    assert reco_matches_scan(loaded, scan_scanned_at="S1", long_term_scanned_at="L1")
    assert not reco_matches_scan(loaded, scan_scanned_at="S2", long_term_scanned_at="L1")


def test_old_schema_is_not_loaded(tmp_path):
    path = tmp_path / "latest_recommendations.json"
    save_recommendations({"schema_version": 3, "categories": []}, path)
    assert load_recommendations(path) is None


def test_persist_desks_saves_recos_and_rebuilds_pulse(monkeypatch, tmp_path):
    saved = {}

    def fake_build_reco(**kwargs):
        assert kwargs.get("deep_confirm") is True
        assert kwargs.get("persist_ledger") is True
        return {
            "schema_version": 4,
            "categories": [{"id": "wealth_builders", "count": 1, "cards": [{"symbol": "AAA"}]}],
            "scan_meta": {"assigned_count": 1},
            "scan_scanned_at": "S",
            "long_term_scanned_at": "L",
            "lifecycle": {"active": [], "closed": []},
        }

    def fake_build_reports(**kwargs):
        assert kwargs.get("rebuild") is True
        saved["pulse"] = True
        return {"reports": []}

    monkeypatch.setattr("product.long_term_store.load_long_term_scan", lambda: {"scanned_at": "L", "records": []})
    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        fake_build_reco,
    )
    monkeypatch.setattr(
        "product.recommendations_workspace.build_market_reports_workspace",
        fake_build_reports,
    )
    monkeypatch.setattr("product.recommendations_store.DEFAULT_RECO_PATH", tmp_path / "recos.json")
    monkeypatch.setattr("product.recommendations_store.save_recommendations", lambda payload, path=None: saved.setdefault("reco", dict(payload)))

    out = persist_desks_from_market_scan({"scanned_at": "S", "records": [{"symbol": "AAA"}]})
    assert out["recommendations"] == "saved"
    assert out["market_reports"] == "saved"
    assert saved["reco"]["from_saved_market_scan"] is True
    assert saved["pulse"] is True


def test_persist_desks_failure_is_status_not_raise(monkeypatch):
    monkeypatch.setattr("product.long_term_store.load_long_term_scan", lambda: {})
    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("reco boom")),
    )
    monkeypatch.setattr(
        "product.recommendations_workspace.build_market_reports_workspace",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("pulse boom")),
    )
    out = persist_desks_from_market_scan({"records": []})
    assert out["recommendations"] == "RuntimeError"
    assert out["market_reports"] == "RuntimeError"


def test_recommendations_get_is_cache_only_when_file_matches(monkeypatch):
    saved = {
        "schema_version": 4,
        "categories": [{"id": "wealth_builders", "count": 0, "cards": []}],
        "scan_scanned_at": "S",
        "long_term_scanned_at": "L",
    }
    monkeypatch.setattr("product.observer_api.core._scan_payload", lambda: {"scanned_at": "S", "records": [{"symbol": "AAA"}]})
    monkeypatch.setattr("product.observer_api.core._long_term_payload", lambda: {"scanned_at": "L"})
    monkeypatch.setattr("product.recommendations_store.load_recommendations", lambda: saved)
    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        lambda **_k: (_ for _ in ()).throw(AssertionError("GET must not rebuild")),
    )
    from product.observer_api import recommendations_workspace
    assert recommendations_workspace(refresh=False) is saved


def test_recommendations_get_keeps_saved_file_when_scan_missing(monkeypatch):
    saved = {
        "schema_version": 4,
        "categories": [{"id": "wealth_builders", "count": 1, "cards": [{"symbol": "AAA"}]}],
        "scan_scanned_at": "yesterday",
        "long_term_scanned_at": "yesterday",
    }
    monkeypatch.setattr("product.observer_api.core._scan_payload", lambda: {"scanned_at": "", "records": []})
    monkeypatch.setattr("product.observer_api.core._long_term_payload", lambda: {"scanned_at": ""})
    monkeypatch.setattr("product.recommendations_store.load_recommendations", lambda: saved)
    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        lambda **_k: (_ for _ in ()).throw(AssertionError("do not wipe recos")),
    )
    from product.observer_api import recommendations_workspace
    assert recommendations_workspace(refresh=False) is saved
