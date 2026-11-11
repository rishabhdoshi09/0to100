"""Research Director served view (owner's Mac 2026-11-11: 11 s cold build,
Research workspace timed out as "state is unavailable")."""
from __future__ import annotations

import pytest


@pytest.fixture()
def RD(monkeypatch):
    import product.research_director_status as rd

    rd._served.update({"payload": None, "built_at": 0.0, "error": ""})
    rd._refreshing.clear()
    builds = []

    def build():
        builds.append(1)
        return {"n": len(builds), "live_locked": "STALE-COPY"}

    monkeypatch.setattr(rd, "build_research_director_status", build)
    monkeypatch.setattr(rd, "live_safety_projection", lambda: {"live_locked": True, "live_lock_status": "LOCKED"})
    # Run background refreshes inline so the test is deterministic.
    monkeypatch.setattr(rd, "_refresh_in_background",
                        lambda: (rd._refreshing.set(), rd._rebuild(), True)[-1])
    rd.builds = builds
    yield rd
    rd._served.update({"payload": None, "built_at": 0.0, "error": ""})
    rd._refreshing.clear()


def test_first_call_builds_and_later_calls_are_served(RD):
    first = RD.served_research_director_status(now=1000.0)
    assert first["n"] == 1 and RD.builds == [1]
    second = RD.served_research_director_status(now=RD._served["built_at"] + 5)
    assert second["n"] == 1 and RD.builds == [1]
    assert second["served_from_cache"] is True and second["cache_refreshing"] is False


def test_stale_view_is_served_then_refreshed(RD):
    RD.served_research_director_status()
    old_built = RD._served["built_at"]
    stale = RD.served_research_director_status(now=old_built + RD.REFRESH_AFTER_S + 1)
    assert stale["cache_age_s"] >= RD.REFRESH_AFTER_S
    assert RD.builds == [1, 1]                      # refresh triggered
    assert RD.served_research_director_status()["n"] == 2


def test_live_safety_is_never_served_from_cache(RD):
    out = RD.served_research_director_status()
    assert out["live_locked"] is True and out["live_lock_status"] == "LOCKED"


def test_warm_starts_a_build_only_when_empty(RD):
    RD.warm_research_director_status()
    assert RD.builds == [1]
    RD.warm_research_director_status()
    assert RD.builds == [1]


def test_failed_rebuild_keeps_last_good_view(RD, monkeypatch):
    RD.served_research_director_status()
    monkeypatch.setattr(RD, "build_research_director_status",
                        lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    out = RD.served_research_director_status(now=RD._served["built_at"] + RD.REFRESH_AFTER_S + 1)
    assert out["n"] == 1 and "locked" in RD._served["error"]
