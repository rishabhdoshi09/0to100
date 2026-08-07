from __future__ import annotations

from datetime import datetime

from research.autonomy import job_store as JS
from research.autonomy import operational_guards as OG


class _Deps:
    def now_ist(self):
        return datetime(2026, 7, 31, 23, 0)

    def holidays(self):
        return set()


class _Ctx:
    deps = _Deps()
    active_failures = set()
    owner_paused = False


def test_empty_off_session_paper_cycle_is_fast_noop(monkeypatch):
    monkeypatch.setattr(OG, "_open_position_count", lambda: 0)

    def must_not_run(_ctx):
        raise AssertionError("research-scale paper cycle should not run")

    result = OG.guarded_paper_cycle(_Ctx(), original_handler=must_not_run)

    assert result.status == JS.SUCCEEDED
    assert result.metadata["fast_path"] is True
    assert result.metadata["open_positions"] == 0
    assert result.metadata["session_phase"] == "off_session"
    assert result.new_entries_allowed is False


def test_guard_preserves_position_management_when_positions_exist(monkeypatch):
    monkeypatch.setattr(OG, "_open_position_count", lambda: 1)
    marker = object()

    def original(ctx):
        assert isinstance(ctx, _Ctx)
        return marker

    assert OG.guarded_paper_cycle(_Ctx(), original_handler=original) is marker


def test_guard_fails_closed_when_position_state_unknown(monkeypatch):
    monkeypatch.setattr(OG, "_open_position_count", lambda: None)
    marker = object()

    assert OG.guarded_paper_cycle(_Ctx(), original_handler=lambda _ctx: marker) is marker
