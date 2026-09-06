from __future__ import annotations

import time

from product import recommendations_liveness as RL


class _Core:
    @staticmethod
    def _scan_payload():
        return {"scanned_at": "2026-09-04T15:40:00+05:30", "records": [{"symbol": "AAA"}]}

    @staticmethod
    def _long_term_payload():
        return {"scanned_at": "2026-09-04T16:00:00+05:30", "records": []}


def _paths(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RECOMMENDATIONS_REBUILD_LOCK", str(tmp_path / "rebuild.lock"))
    monkeypatch.setenv("QT_RECOMMENDATIONS_REBUILD_STATE", str(tmp_path / "rebuild.json"))


def test_rebuild_claim_is_process_exclusive(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    scan = _Core._scan_payload()
    long_term = _Core._long_term_payload()

    first = RL._claim_rebuild(scan, long_term)
    assert first is not None
    try:
        assert RL._claim_rebuild(scan, long_term) is None
        state = RL._read_state()
        assert state["status"] == "RUNNING"
        assert state["scan_scanned_at"] == scan["scanned_at"]
    finally:
        RL._release_claim(first)

    second = RL._claim_rebuild(scan, long_term)
    assert second is not None
    RL._release_claim(second)


def test_failed_current_generation_is_explicit_not_refreshing_forever(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    saved = {
        "schema_version": 4,
        "generated_at": "2026-09-03T16:01:00+05:30",
        "scan_scanned_at": "2026-09-03T15:40:00+05:30",
        "long_term_scanned_at": "2026-09-03T16:00:00+05:30",
        "records_status": "CURRENT",
        "same_ist_day": False,
        "cmp_note": "old",
        "methods_note": "old",
        "ensemble": {"high_conviction_count": 0, "good_setup_count": 0, "empty_high_conviction": True},
        "categories": [],
        "lifecycle": {"active": [], "closed": [], "active_count": 0, "closed_count": 0},
        "disclaimer": "paper only",
    }
    monkeypatch.setattr(RS, "load_recommendations", lambda: saved)
    monkeypatch.setattr(RS, "reco_matches_scan", lambda *args, **kwargs: False)
    monkeypatch.setattr(RW, "slim_workspace_for_desk", lambda payload: dict(payload))
    monkeypatch.setattr(RL, "_ensure_rebuild", lambda *_args, **_kwargs: False)
    RL._write_state({
        "schema_version": 1,
        **RL._target(_Core._scan_payload(), _Core._long_term_payload()),
        "status": "FAILED",
        "finished_at": time.time(),
        "error": "projection exploded",
    })

    result = RL.build_fast_response(_Core())
    assert result["records_status"] == "FAILED"
    assert result["rebuilding"] is False
    assert result["rebuild_error"] == "projection exploded"
    assert "NOT current" in result["cmp_note"]


def test_superseded_rebuild_cannot_overwrite_newer_generation(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    from product import recommendations_store as RS
    from product import recommendations_workspace as RW

    calls = []
    monkeypatch.setattr(RW, "build_recommendations_workspace", lambda **_kwargs: {"schema_version": 4, "categories": []})
    monkeypatch.setattr(RS, "save_recommendations", lambda payload: calls.append(payload))
    monkeypatch.setattr(RL, "_current_generation_matches", lambda *_args, **_kwargs: False)

    scan = _Core._scan_payload()
    long_term = _Core._long_term_payload()
    claim = RL._claim_rebuild(scan, long_term)
    assert claim is not None
    RL._build_and_persist(scan, long_term, claim)

    assert calls == []
    state = RL._read_state()
    assert state["status"] == "SUPERSEDED"
