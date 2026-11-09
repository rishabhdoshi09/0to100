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

    monkeypatch.setattr(desk_pipeline, "scan_is_fresh", lambda: True)

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

    monkeypatch.setattr(desk_pipeline, "scan_is_fresh", lambda: False)
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
    assert response["rebuilding"] is False
