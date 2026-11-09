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
    """Serve the real-pipeline scan, funds and recommendation files to the real loaders."""
    from product import long_term_store, recommendations_store

    scan_path = tmp_path / "latest_momentum_scan.json"
    scan_path.write_bytes((FIXTURE / "latest_momentum_scan.json").read_bytes())
    monkeypatch.setenv("QT_SCAN_PATH", str(scan_path))
    real_reco, real_lt = recommendations_store.load_recommendations, long_term_store.load_long_term_scan
    monkeypatch.setattr(recommendations_store, "load_recommendations",
                        lambda path=FIXTURE / "latest_recommendations.json": real_reco(path))
    monkeypatch.setattr(long_term_store, "load_long_term_scan",
                        lambda path=FIXTURE / "latest_long_term_scan.json": real_lt(path))
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    return {"scan": scan, "scan_path": scan_path}


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
    from api.app import app

    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: False)
    payload = TestClient(app).get(PATH).json()
    assert payload["display_authority"]["status"] == "SCAN_NOT_CURRENT"
    assert payload["display_authority"]["scan_current"] is False


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
