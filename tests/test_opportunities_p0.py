"""P0 9 Nov 2026: a completed scan never reached the Opportunities page.

Reproduced against the real FastAPI app (not mocked responses) with a saved scan
and recommendation projection produced by the real production pipeline
(``tests/fixtures/opportunities``; see its README):

* The canonical launcher serves ``api.app:app``. There, two synchronous
  ``GET /api/recommendations-workspace`` routes were registered
  (``product.observer_api`` first, ``api.runtime`` shadowed). The persisted-first
  route was swapped in only by a startup hook that silently did nothing unless
  ``terminal_product_api_parallel`` happened to be the launch module, so the
  one-command stack rebuilt the whole workspace inside the HTTP request after
  every new scan.
* The browser could not judge recommendation currency from the Opportunities
  API itself; it borrowed it from the 15-second dashboard read cache.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).parent / "fixtures" / "opportunities"
PATH = "/api/recommendations-workspace"


def _get_routes(app):
    return [
        route for route in app.router.routes
        if getattr(route, "path", None) == PATH and "GET" in (getattr(route, "methods", None) or set())
    ]


@pytest.fixture
def saved_pipeline(monkeypatch, tmp_path):
    """Serve the real-pipeline scan, funds and recommendation files to the real loaders.

    The sandbox that produced the fixture had no official bhavcopy store, so its
    provenance honestly says HISTORY_NOT_READY. The incident scan was a verified,
    current one; stamp the provenance a current official-history scan records.
    """
    from product import long_term_store, recommendations_store
    from product.scan_store import PRICE_SOURCE

    scan_path = tmp_path / "latest_momentum_scan.json"
    raw = json.loads((FIXTURE / "latest_momentum_scan.json").read_text(encoding="utf-8"))
    session = raw["provenance"]["expected_session_date"]
    raw["provenance"].update({
        "data_current": True, "data_freshness": "CURRENT", "market_session_date": session,
        "price_data_as_of": session, "price_source": PRICE_SOURCE, "sessions_behind": 0,
        "provenance_available": True, "provenance_reason": "",
    })
    scan_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("QT_SCAN_PATH", str(scan_path))
    real_reco, real_lt = recommendations_store.load_recommendations, long_term_store.load_long_term_scan
    monkeypatch.setattr(recommendations_store, "load_recommendations",
                        lambda path=FIXTURE / "latest_recommendations.json": real_reco(path))
    monkeypatch.setattr(long_term_store, "load_long_term_scan",
                        lambda path=FIXTURE / "latest_long_term_scan.json": real_lt(path))
    from product import desk_pipeline
    real = {"scan_is_fresh": desk_pipeline.scan_is_fresh,
            "scan_payload_is_fresh": desk_pipeline.scan_payload_is_fresh}
    monkeypatch.setattr(desk_pipeline, "scan_is_fresh", lambda: True)
    monkeypatch.setattr(desk_pipeline, "scan_payload_is_fresh", lambda *_a, **_k: True)
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    return {"scan": scan, "scan_path": scan_path, "real": real}


def test_every_launch_module_serves_one_canonical_recommendations_route():
    import api.app
    import terminal_product_api_parallel

    for app in (api.app.app, terminal_product_api_parallel.app):
        routes = _get_routes(app)
        assert len(routes) == 1, [f"{r.endpoint.__module__}.{r.endpoint.__name__}" for r in routes]
        assert routes[0].endpoint is api.app.recommendations_workspace
    # No startup hook may swap the route later depending on sys.modules.
    hooks = [getattr(fn, "__name__", "") for fn in api.app.app.router.on_startup]
    assert not any("recommendation" in name for name in hooks), hooks


def test_current_projection_is_served_with_server_display_authority(saved_pipeline):
    from api.app import app

    started = time.perf_counter()
    response = TestClient(app).get(PATH)
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    payload = response.json()
    cards = [c for cat in payload["categories"] for c in cat.get("cards") or []]
    assert cards, "the published projection for the current scan must reach the page"
    authority = payload["display_authority"]
    assert authority["status"] == "CURRENT"
    assert authority["scan_current"] is True
    assert authority["scan_scanned_at"] == saved_pipeline["scan"]["scanned_at"]
    assert payload["scan_scanned_at"] == authority["scan_scanned_at"]
    assert authority["evaluated_at"]
    assert elapsed < 5.0
    # The scan's own publication record travels with the answer.
    assert authority["publication"]["recommendations"] == "saved"
    assert authority["publication"]["scan_scanned_at"] == authority["scan_scanned_at"]


def test_failed_publication_is_explained_not_silent(saved_pipeline, monkeypatch):
    from api.app import app

    scan = dict(saved_pipeline["scan"])
    scan["scanned_at"] = "2026-11-09T16:45:00+00:00"
    scan["desk_overlays"] = {"recommendations": "error", "scan_scanned_at": scan["scanned_at"],
                             "recommendations_error": {"error_code": "DESK_PERSIST_FAILED",
                                                       "error_type": "OSError",
                                                       "error_message": "disk full"}}
    saved_pipeline["scan_path"].write_text(json.dumps(scan), encoding="utf-8")
    monkeypatch.setattr("product.recommendations_liveness._ensure_rebuild", lambda *a, **k: False)
    authority = TestClient(app).get(PATH).json()["display_authority"]
    assert authority["status"] == "REFRESHING"
    assert authority["publication"]["recommendations"] == "error"
    assert authority["publication"]["error"]["error_code"] == "DESK_PERSIST_FAILED"


def test_new_scan_generation_is_never_rebuilt_inside_the_request(saved_pipeline, monkeypatch):
    """After a new scan the saved projection no longer matches. The read must return
    at once with an explicit REFRESHING state and build only in the background."""
    from api.app import app
    from product import recommendations_liveness as RL
    from product import recommendations_workspace as RW

    scan = dict(saved_pipeline["scan"])
    scan["scanned_at"] = "2026-11-09T16:30:00+00:00"          # a newer generation
    saved_pipeline["scan_path"].write_text(json.dumps(scan), encoding="utf-8")
    request_thread = []
    built_in = []
    release = threading.Event()

    def slow_build(**_kwargs):
        built_in.append(threading.current_thread().name)
        release.wait(5)
        raise RuntimeError("background build stopped by the test")

    monkeypatch.setattr(RW, "build_recommendations_workspace", slow_build)
    monkeypatch.setattr(RL, "_state_path", lambda: Path(saved_pipeline["scan_path"]).with_name("rebuild_state.json"))
    monkeypatch.setattr(RL, "_lock_path", lambda: Path(saved_pipeline["scan_path"]).with_name("rebuild.lock"))

    def spy(*args, **kwargs):
        request_thread.append(threading.current_thread().name)
        return real_fast(*args, **kwargs)

    real_fast = RL.build_fast_response
    monkeypatch.setattr(RL, "build_fast_response", spy)
    try:
        started = time.perf_counter()
        response = TestClient(app).get(PATH)
        elapsed = time.perf_counter() - started
    finally:
        release.set()
    assert response.status_code == 200
    payload = response.json()
    assert elapsed < 3.0, f"the read waited for a rebuild ({elapsed:.1f}s)"
    assert payload["display_authority"]["status"] == "REFRESHING"
    assert all(not cat.get("cards") for cat in payload["categories"]), "previous-generation cards leaked"
    deadline = time.time() + 3
    while not built_in and time.time() < deadline:
        time.sleep(0.02)
    assert built_in == ["quantterm-recommendations-rebuild"]
    assert request_thread and request_thread[0] != built_in[0]


def test_stale_official_scan_is_not_presented_as_current(saved_pipeline, monkeypatch):
    """An old-day scan: its saved BUY cards are withheld by the server, nothing is
    rebuilt from it, and the page is told why (not "rebuilding")."""
    from api.app import app
    from product import recommendations_liveness as RL

    monkeypatch.setattr("product.desk_pipeline.scan_payload_is_fresh", lambda *_a, **_k: False)
    monkeypatch.setattr(RL, "_ensure_rebuild",
                        lambda *a, **k: pytest.fail("a stale scan must not start a rebuild"))
    payload = TestClient(app).get(PATH).json()
    authority = payload["display_authority"]
    assert authority["status"] == "SCAN_NOT_CURRENT"
    assert authority["scan_current"] is False
    assert authority["scan_scanned_at"] == saved_pipeline["scan"]["scanned_at"]
    assert payload["records_status"] == "STALE"
    assert payload["rebuilding"] is False
    assert all(not cat.get("cards") for cat in payload["categories"]), "old-day cards leaked"
    assert "rebuilding" not in payload["cmp_note"].lower()


def test_invalidated_scan_provenance_withholds_cards_with_authority(saved_pipeline):
    from api.app import app

    scan = dict(saved_pipeline["scan"])
    scan["provenance"] = {**dict(scan.get("provenance") or {}), "data_current": False}
    saved_pipeline["scan_path"].write_text(json.dumps(scan), encoding="utf-8")
    payload = TestClient(app).get(PATH).json()
    assert payload["records_status"] == "STALE"
    assert payload["display_authority"]["status"] == "SCAN_NOT_CURRENT"
    assert all(not cat.get("cards") for cat in payload["categories"])


def test_missing_scan_withholds_saved_cards(saved_pipeline, monkeypatch):
    """Superseded the deleted observer handler, which served yesterday's saved
    cards when the scan was missing."""
    from api.app import app

    monkeypatch.setenv("QT_SCAN_PATH", str(saved_pipeline["scan_path"].with_name("absent.json")))
    payload = TestClient(app).get(PATH).json()
    assert payload["display_authority"]["status"] in {"FAILED", "SCAN_UNAVAILABLE"}
    assert all(not cat.get("cards") for cat in payload["categories"])


def test_scan_freshness_reads_the_same_scan_artifact_as_every_other_reader(monkeypatch, tmp_path):
    """desk_pipeline.scan_is_fresh hard-coded logs/product/latest_momentum_scan.json and
    ignored QT_SCAN_PATH, the override every other scan reader honours."""
    from product import desk_pipeline

    target = tmp_path / "scan.json"
    target.write_bytes((FIXTURE / "latest_momentum_scan.json").read_bytes())
    monkeypatch.setenv("QT_SCAN_PATH", str(target))
    seen = []
    import product.scan_store as store
    real = store.load_scan
    monkeypatch.setattr(store, "load_scan", lambda path=None: seen.append(Path(path)) or real(path))
    desk_pipeline.scan_is_fresh()
    assert seen and seen[0] == target


def test_freshness_is_judged_on_the_same_scan_artifact_that_is_served(saved_pipeline, monkeypatch):
    """Read-time TOCTOU. The freshness verdict and the served scan must come from
    one parse of one artifact. Here the scan file is replaced right after the
    freshness check by an older-session scan with the same scanned_at (so the
    saved projection still "matches"); it must never be served as CURRENT."""
    from api.app import app
    from product import desk_pipeline

    session = saved_pipeline["scan"]["provenance"]["expected_session_date"]
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda **_k: {"usable_for_scan": True, "current": True, "available_session": session},
    )
    old = json.loads(saved_pipeline["scan_path"].read_text(encoding="utf-8"))
    old["provenance"] = {**old["provenance"], "scan_id": "older-session",
                         "market_session_date": "2026-11-06", "price_data_as_of": "2026-11-06"}

    def swap_after(real):
        def judged(*args, **kwargs):
            verdict = real(*args, **kwargs)
            saved_pipeline["scan_path"].write_text(json.dumps(old), encoding="utf-8")
            return verdict
        return judged

    # This test needs the real freshness authority, not the fixture's stub.
    for name, real in saved_pipeline["real"].items():
        monkeypatch.setattr(desk_pipeline, name, swap_after(real))

    authority = TestClient(app).get(PATH).json()["display_authority"]
    # The artifact that was judged fresh is the one served; the swapped-in
    # older scan is picked up (and judged) by the next read, not this one.
    assert authority["status"] == "CURRENT", authority
    assert authority["scan_market_session"] == session, authority
    assert authority["scan_id"] != "older-session", authority
    nxt = TestClient(app).get(PATH).json()["display_authority"]
    assert nxt["status"] == "SCAN_NOT_CURRENT" and nxt["scan_market_session"] == "2026-11-06", nxt


def test_absent_scan_is_reported_as_no_scan_not_as_stale(saved_pipeline, monkeypatch):
    from api.app import app

    monkeypatch.setenv("QT_SCAN_PATH", str(saved_pipeline["scan_path"].with_name("absent.json")))
    monkeypatch.setattr("product.desk_pipeline.scan_payload_is_fresh", saved_pipeline["real"]["scan_payload_is_fresh"])
    payload = TestClient(app).get(PATH).json()
    assert payload["display_authority"]["status"] == "SCAN_UNAVAILABLE"
    assert all(not cat.get("cards") for cat in payload["categories"])
