from __future__ import annotations

from product import desk_pipeline as pipeline
from operations.store import SUCCEEDED


class _RecentSuccessStore:
    def __init__(self, successful_kind: str):
        self.successful_kind = successful_kind

    def latest(self, kind: str):
        if kind != self.successful_kind:
            return None
        return {
            "kind": kind,
            "status": SUCCEEDED,
            "updated_at": 999_999.0,
        }


def _pin_clock(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.time, "time", lambda: 1_000_000.0)


def test_missing_required_session_bypasses_recent_data_prepare_success(monkeypatch):
    """A prior refresh cannot satisfy a newly-required exchange session."""
    _pin_clock(monkeypatch)
    monkeypatch.setattr(pipeline, "prices_kind_due", lambda: pipeline.DATA_PREPARE)
    store = _RecentSuccessStore(pipeline.DATA_PREPARE)

    assert pipeline._kind_for_step("prices", store) == pipeline.DATA_PREPARE


def test_missing_required_session_bypasses_recent_fno_success(monkeypatch):
    """F&O freshness must never mask stale authoritative price history."""
    _pin_clock(monkeypatch)
    monkeypatch.setattr(pipeline, "prices_kind_due", lambda: pipeline.DATA_PREPARE)
    store = _RecentSuccessStore(pipeline.FNO_REFRESH)

    assert pipeline._kind_for_step("prices", store) == pipeline.DATA_PREPARE


def test_current_history_fno_refresh_still_respects_success_ttl(monkeypatch):
    """Only the mandatory history boundary bypasses TTL suppression."""
    _pin_clock(monkeypatch)
    monkeypatch.setattr(pipeline, "prices_kind_due", lambda: pipeline.FNO_REFRESH)
    store = _RecentSuccessStore(pipeline.FNO_REFRESH)

    assert pipeline._kind_for_step("prices", store) is None
