"""Critical paper-cycle authority failures must not open recommendation paper trades."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from research.autonomy import jobs as JOBS
from research.autonomy import job_store as JS


def _wired_deps(monkeypatch, *, core_status="NO_ACTION", auth_error=False):
    from research.auto_research import scheduler

    class Brain:
        intel_book = object()
        def run_intelligence_cycle_day(self, **kwargs):
            return {
                "status": core_status,
                "as_of_date": "2026-11-06",
                "positions_opened": [],
                "eligibility": "NO_ELIGIBLE_TRADE",
            }
        def is_paper_auto_enabled(self):
            if auth_error:
                raise OSError("preference authority unavailable")
            return True
        def _save_intel_book(self):
            return True

    brain = Brain()
    monkeypatch.setattr(scheduler, "get_brain", lambda: brain)
    deps = object.__new__(JOBS.Deps)
    deps.live_feed = None
    notifications = []
    deps.telegram = SimpleNamespace(
        notify_paper_cycle=lambda result, **kw: notifications.append(result),
    )
    return deps, notifications


@pytest.mark.parametrize("blocked", [
    "BLOCKED_RECOVERY", "BLOCKED_DURABILITY", "FAILED_SAFE", "SKIPPED_LOCKED",
])
def test_upstream_blocked_intelligence_never_invokes_reco_paper(monkeypatch, blocked):
    from product import paper_autopilot as PA

    calls = []
    monkeypatch.setattr(
        PA, "run_reco_paper_cycle",
        lambda **kw: calls.append(kw) or {},
    )
    deps, notifications = _wired_deps(monkeypatch, core_status=blocked)
    result = deps.run_paper_cycle(True)

    assert calls == []
    assert result["eligibility"] == "BLOCKED_SAFETY"
    assert result["new_entries_allowed"] is False
    assert result["reco_autopilot"]["status"] == "BLOCKED_UPSTREAM"
    assert notifications


def test_paper_permission_lookup_failure_never_defaults_to_on(monkeypatch):
    from product import paper_autopilot as PA

    calls = []
    monkeypatch.setattr(
        PA, "run_reco_paper_cycle",
        lambda **kw: calls.append(kw) or {},
    )
    deps, _ = _wired_deps(monkeypatch, auth_error=True)
    result = deps.run_paper_cycle(True)

    assert calls == []
    assert result["status"] == "FAILED_SAFE"
    assert result["eligibility"] == "BLOCKED_SAFETY"
    assert "PAPER_AUTO_AUTHORITY_UNAVAILABLE" in result["reco_autopilot"]["error"]


def test_valid_paper_permission_still_reaches_recommendation_runner(monkeypatch):
    from product import paper_autopilot as PA

    seen = []
    def run(**kwargs):
        seen.append(kwargs)
        return {
            "eligibility": "NO_ELIGIBLE_TRADE",
            "positions_opened": [],
            "evolution": {"paper_book_durable_saved": True},
        }

    monkeypatch.setattr(PA, "run_reco_paper_cycle", run)
    deps, _ = _wired_deps(monkeypatch)
    result = deps.run_paper_cycle(True)

    assert len(seen) == 1
    assert seen[0]["paper_enabled"] is True
    assert seen[0]["entries_allowed"] is True
    assert result["status"] == "NO_ACTION"


def test_supervisor_does_not_mark_real_upstream_failure_succeeded(monkeypatch):
    deps = object.__new__(JOBS.Deps)
    deps.now_ist = lambda: datetime(2026, 11, 6, 12, 30)
    deps.holidays = lambda: set()
    deps.run_paper_cycle = lambda *args: {
        "status": "BLOCKED_DURABILITY",
        "error": "simulated disk loss",
        "eligibility": "BLOCKED_SAFETY",
    }

    monkeypatch.setattr(JOBS, "_entry_reason", lambda *args: (False, "test", "intraday"))
    monkeypatch.setattr(JOBS, "_paper_market_data_source", lambda *args: (True, "fixture"))
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))

    assert result.status == JS.PERMANENT_FAILED
    assert result.new_entries_allowed is False
    assert result.error_code == "PAPER_CYCLE_UPSTREAM_UNSAFE"
    assert result.metadata["intelligence_status"] == "BLOCKED_DURABILITY"
