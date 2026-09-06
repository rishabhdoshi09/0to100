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
    assert "last persisted projection" in result["cmp_note"]


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
    assert "not fabricated" in result["disclaimer"].lower()
