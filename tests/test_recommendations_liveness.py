from __future__ import annotations


from product import recommendations_liveness as RL


class _Core:
    @staticmethod
    def _scan_payload():
        return {"scanned_at": "2026-09-04T15:40:00+05:30", "records": [{"symbol": "AAA"}]}

    @staticmethod
    def _long_term_payload():
        return {"scanned_at": "2026-09-04T16:00:00+05:30", "records": []}


def _saved():
    return {
        "schema_version": 4,
        "generated_at": "2026-09-04T16:01:00+05:30",
        "scan_scanned_at": "2026-09-04T15:40:00+05:30",
        "long_term_scanned_at": "2026-09-04T16:00:00+05:30",
        "records_status": "CURRENT",
        "same_ist_day": True,
        "cmp_note": "persisted",
        "methods_note": "persisted",
        "ensemble": {"high_conviction_count": 0, "good_setup_count": 0, "empty_high_conviction": True},
        "categories": [],
        "lifecycle": {"active": [], "closed": [], "active_count": 0, "closed_count": 0},
        "disclaimer": "paper only",
    }


def test_matching_recommendations_are_served_without_rebuild(monkeypatch):
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    monkeypatch.setattr(RS, "load_recommendations", lambda: _saved())
    monkeypatch.setattr(RS, "reco_matches_scan", lambda *args, **kwargs: True)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))

    def _must_not_rebuild(*_args, **_kwargs):
        raise AssertionError("matching persisted recommendations must not rebuild in request path")

    monkeypatch.setattr(RL, "_ensure_rebuild", _must_not_rebuild)
    result = RL.build_fast_response(_Core())
    assert result["records_status"] == "CURRENT"
    assert result["cmp_note"] == "persisted"


def test_stale_recommendations_schedule_background_rebuild_and_return_immediately(monkeypatch):
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    calls = []
    monkeypatch.setattr(RS, "load_recommendations", lambda: _saved())
    monkeypatch.setattr(RS, "reco_matches_scan", lambda *args, **kwargs: False)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(RL, "_ensure_rebuild", lambda scan, long_term: calls.append((scan, long_term)) or True)

    result = RL.build_fast_response(_Core())
    assert len(calls) == 1
    assert result["records_status"] == "REFRESHING"
    assert result["rebuilding"] is True
    assert "have been withheld" in result["cmp_note"]
    assert result["categories"] and all(not c["cards"] for c in result["categories"])
    assert result["lifecycle"]["active"] == []


def test_old_buy_cards_are_withheld_while_rebuilding(monkeypatch):
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    stale = _saved()
    stale["categories"] = [{
        "id": "momentum_breakouts",
        "count": 1,
        "cards": [{"symbol": "AAA", "action_badge": "Buy", "entry": 100}],
    }]
    stale["lifecycle"] = {
        "active": [{"symbol": "AAA", "action_badge": "Buy"}],
        "closed": [], "active_count": 1, "closed_count": 0,
    }
    stale["ensemble"]["high_conviction_count"] = 1
    monkeypatch.setattr(RS, "load_recommendations", lambda: stale)
    monkeypatch.setattr(RS, "reco_matches_scan", lambda *args, **kwargs: False)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(RL, "_ensure_rebuild", lambda *_args: True)

    result = RL.build_fast_response(_Core())

    assert result["records_status"] == "REFRESHING"
    assert result["ensemble"]["high_conviction_count"] == 0
    assert result["categories"] and all(c["count"] == 0 and not c["cards"] for c in result["categories"])
    assert result["lifecycle"]["active"] == []
    assert result["scan_scanned_at"] == _Core._scan_payload()["scanned_at"]


def test_failed_rebuild_does_not_restore_old_buy_cards(monkeypatch):
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    stale = _saved()
    stale["categories"] = [{"id": "momentum_breakouts", "count": 1, "cards": [{"symbol": "AAA", "action_badge": "Buy"}]}]
    monkeypatch.setattr(RS, "load_recommendations", lambda: stale)
    monkeypatch.setattr(RS, "reco_matches_scan", lambda *_a, **_k: False)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(RL, "_ensure_rebuild", lambda *_args: False)
    monkeypatch.setattr(RL, "_read_state", lambda: {
        **RL._target(_Core._scan_payload(), _Core._long_term_payload()),
        "status": "FAILED",
        "error": "test rebuild failure",
    })

    result = RL.build_fast_response(_Core())

    assert result["records_status"] == "FAILED"
    assert result["rebuilding"] is False
    assert result["categories"] and all(not c["cards"] for c in result["categories"])
    assert "test rebuild failure" in result["rebuild_error"]


def test_missing_recommendations_return_truthful_empty_refreshing_shape(monkeypatch):
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    calls = []
    monkeypatch.setattr(RS, "load_recommendations", lambda: None)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(RL, "_ensure_rebuild", lambda scan, long_term: calls.append((scan, long_term)) or True)

    result = RL.build_fast_response(_Core())
    assert len(calls) == 1
    assert result["records_status"] == "REFRESHING"
    assert result["rebuilding"] is True
    assert result["categories"] is not None
    assert "no recommendation was fabricated" in result["disclaimer"].lower()


# The startup-time route swap was removed: api.app registers the one canonical
# route at import. See tests/test_opportunities_p0.py::
# test_every_launch_module_serves_one_canonical_recommendations_route.


def test_canonical_api_get_uses_persisted_first_recommendations(monkeypatch):
    """The deployed api.app, not only a dormant compatibility hook, must use the fast route."""
    import importlib

    canonical = importlib.import_module("api.app")
    routes = [
        item for item in canonical.app.router.routes
        if getattr(item, "path", None) == "/api/recommendations-workspace"
        and "GET" in set(getattr(item, "methods", set()) or set())
    ]
    assert len(routes) == 1
    assert routes[0].name == "recommendations_workspace_persisted_first"

    from product import recommendations_workspace as RW
    from product import desk_pipeline

    monkeypatch.setattr(desk_pipeline, "scan_payload_is_fresh", lambda *_a, **_k: True)

    def forbidden_sync_builder(**_kwargs):
        raise AssertionError("GET must never rebuild all recommendations synchronously")

    monkeypatch.setattr(RW, "build_recommendations_workspace", forbidden_sync_builder)
    monkeypatch.setattr(
        RL, "build_fast_response",
        lambda _core, attach_authority=None, **_kw: {"records_status": "REFRESHING", "categories": []},
    )
    assert routes[0].endpoint()["records_status"] == "REFRESHING"


def test_stale_scan_provenance_withholds_saved_buy_without_rebuild(monkeypatch):
    """Same saved scan identity is insufficient when its data provenance was invalidated."""
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    stale = _saved()
    stale["categories"] = [
        {"id": "momentum_breakouts", "count": 1,
         "cards": [{"symbol": "AAA", "action_badge": "Buy", "entry": 100}]},
    ]
    monkeypatch.setattr(RS, "load_recommendations", lambda: stale)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(
        RL, "_ensure_rebuild",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("stale source cannot start rebuild")),
    )

    class StaleCore(_Core):
        @staticmethod
        def _scan_payload():
            return {
                **_Core._scan_payload(),
                "available": True,
                "provenance": {"data_current": False, "source_snapshot_id": "stale"},
            }

    result = RL.build_fast_response(StaleCore())
    assert result["records_status"] == "STALE"
    assert not any(category.get("cards") for category in result["categories"])
    assert result["lifecycle"]["active"] == []


def test_canonical_get_withholds_old_day_buy_even_when_saved_provenance_was_once_current(monkeypatch):
    """A 6-Nov scan can stay persisted on 9-Nov; calendar proof must be rechecked."""
    import importlib

    from product import desk_pipeline

    canonical = importlib.import_module("api.app")
    route = next(
        row for row in canonical.app.router.routes
        if getattr(row, "path", None) == "/api/recommendations-workspace"
        and "GET" in set(getattr(row, "methods", set()) or set())
    )

    import json
    import os
    from pathlib import Path

    # A scan that was current on 6 Nov, still on disk on 9 Nov.
    Path(os.environ["QT_SCAN_PATH"]).write_text(json.dumps({
        "schema_version": 2,
        "scanned_at": "2026-11-06T16:00:00+05:30",
        "records": [{"symbol": "AAA"}],
        "provenance": {"scan_id": "nov6", "data_current": True,
                       "market_session_date": "2026-11-06", "price_data_as_of": "2026-11-06"},
    }), encoding="utf-8")
    monkeypatch.setattr(desk_pipeline, "scan_payload_is_fresh", lambda *_a, **_k: False)
    monkeypatch.setattr(
        RL, "build_fast_response",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("old-day saved BUY cards must not reach presentation")
        ),
    )
    response = route.endpoint()
    assert response["records_status"] == "STALE"
    assert response["categories"] and all(c["count"] == 0 and not c["cards"] for c in response["categories"])
    assert response["lifecycle"]["active"] == []
    assert "No current verified NSE scan" in response["error"]
    assert response["display_authority"]["status"] == "SCAN_NOT_CURRENT"
    assert response["display_authority"]["scan_market_session"] == "2026-11-06"
    assert response["rebuilding"] is False


def test_background_rebuild_never_overwrites_a_newer_pipeline_publication(monkeypatch, tmp_path):
    """The scan pipeline saves generation B's recommendations, then B's scan.
    A background rebuild for the older generation A that finishes in between
    saw scan A still on disk and overwrote B's published recommendations with
    A's. Publication (pipeline) and the rebuild's generation check + save must
    be one critical section."""
    import contextlib
    import json
    import threading
    import time
    from pathlib import Path

    from product import long_term_store, recommendations_store as RS, scan_store
    from product import recommendations_workspace as RW

    reco_path = tmp_path / "latest_recommendations.json"
    scan_path = tmp_path / "latest_momentum_scan.json"
    monkeypatch.setenv("QT_RECO_PUBLISH_LOCK", str(tmp_path / "publish.lock"))
    monkeypatch.setattr(RL, "_state_path", lambda: tmp_path / "rebuild_state.json")
    real_save_reco, real_load_reco = RS.save_recommendations, RS.load_recommendations
    monkeypatch.setattr(RS, "save_recommendations", lambda payload, path=reco_path: real_save_reco(payload, path))
    monkeypatch.setattr(RS, "load_recommendations", lambda path=reco_path: real_load_reco(path))
    monkeypatch.setattr(scan_store, "load_scan", lambda path=scan_path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(long_term_store, "load_long_term_scan", lambda *a, **k: {"scanned_at": "LT"})

    def scan(gen):
        return {"schema_version": 2, "scanned_at": gen, "records": [], "provenance": {}}

    def reco(gen):
        return {"schema_version": 4, "categories": [], "scan_scanned_at": gen, "long_term_scanned_at": "LT"}

    scan_path.write_text(json.dumps(scan("A")))
    monkeypatch.setattr(RW, "build_recommendations_workspace", lambda **_k: reco("A"))

    reco_saved, go = threading.Event(), threading.Event()

    def pipeline_b():
        lock = getattr(RS, "publication_lock", None)
        with (lock() if lock else contextlib.nullcontext()):
            RS.save_recommendations(reco("B"))
            reco_saved.set()
            go.wait(5)
            scan_path.write_text(json.dumps(scan("B")))

    publisher = threading.Thread(target=pipeline_b)
    publisher.start()
    assert reco_saved.wait(5)
    rebuild = threading.Thread(target=RL._build_and_persist, args=(scan("A"), {"scanned_at": "LT"}, None))
    rebuild.start()
    time.sleep(0.3)          # the rebuild is ready to commit while B is mid-publication
    go.set()
    publisher.join(5)
    rebuild.join(5)
    assert json.loads(reco_path.read_text())["scan_scanned_at"] == "B"
    assert json.loads((tmp_path / "rebuild_state.json").read_text())["status"] == "SUPERSEDED"


def test_a_deterministically_failing_rebuild_backs_off_instead_of_rebuilding_every_30s(monkeypatch, tmp_path):
    """A projection that always fails was rebuilt every 30 s for as long as the
    page was open -- a full projection build each time on an 8 GB host. Retries
    back off per generation (30 s, 60 s, 120 s ... capped) and reset when the
    generation changes."""
    from product import recommendations_workspace as RW

    monkeypatch.setattr(RL, "_state_path", lambda: tmp_path / "rebuild_state.json")
    monkeypatch.setattr(RL, "_lock_path", lambda: tmp_path / "rebuild.lock")
    monkeypatch.setattr(RL, "_current_generation_matches", lambda *_a: True)
    monkeypatch.setattr(RW, "build_recommendations_workspace",
                        lambda **_k: (_ for _ in ()).throw(RuntimeError("deterministic projection bug")))
    clock = [1_000_000.0]
    monkeypatch.setattr(RL.time, "time", lambda: clock[0])
    scan, lt = {"scanned_at": "S1"}, {"scanned_at": "L1"}

    waits = []
    for _ in range(5):
        claim = RL._claim_rebuild(scan, lt)          # the real claim path
        assert claim is not None
        RL._build_and_persist(dict(scan), dict(lt), claim)
        assert RL._read_state()["status"] == "FAILED"
        wait = 0
        while RL._recent_failed_target(scan, lt):
            clock[0] += 1
            wait += 1
        waits.append(wait)
    assert waits[0] == 30
    assert waits == sorted(waits) and waits[-1] > waits[0] * 4, waits
    assert max(waits) <= 900
    # A new generation is retried at once.
    assert RL._recent_failed_target({"scanned_at": "S2"}, lt) is False
