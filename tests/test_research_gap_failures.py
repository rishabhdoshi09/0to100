"""Sticky CA / universe incompletes must not permanently red the heartbeat."""
from __future__ import annotations

from research.autonomy import health as H
from research.autonomy import jobs as JOBS
from research.autonomy import job_store as JS
from research.autonomy.supervisor import Supervisor


def test_blocked_ca_clears_sticky_failure_not_sets_it():
    class Deps:
        def corporate_actions_status(self):
            return {"available": False, "events": 0, "symbols": 0}

        def ensure_corporate_actions(self):
            return self.corporate_actions_status()

    result = JOBS.run_corporate_actions(JOBS._Ctx(Deps()))
    assert result.status == JS.BLOCKED
    assert H.CA_INCOMPLETE in result.clears
    assert not result.failures


def test_blocked_universe_clears_sticky_failure():
    class Deps:
        def universe_history_status(self):
            return {
                "survivorship_complete": False,
                "symbols": ["AAA"],
                "research_grade": False,
                "note": "survivors only",
            }

        def ensure_universe_history(self):
            return self.universe_history_status()

    result = JOBS.run_universe_history(JOBS._Ctx(Deps()))
    assert result.status == JS.BLOCKED
    assert H.UNIVERSE_INCOMPLETE in result.clears
    assert not result.failures


def test_supervisor_heartbeat_drops_sticky_research_gaps(tmp_path):
    root = tmp_path / "auto"
    root.mkdir()
    (root / "failures.json").write_text(
        '["corporate_actions_incomplete","universe_history_incomplete","news_unavailable"]',
        encoding="utf-8",
    )
    (root / "state.json").write_text("{}", encoding="utf-8")
    (root / "owner_state.json").write_text(
        '{"paper_auto_enabled": true, "new_entries_paused": false, "halted": false}',
        encoding="utf-8",
    )

    class Deps(JOBS.Deps):
        def corporate_actions_status(self):
            return {"available": False, "events": 0}

        def universe_history_status(self):
            return {"survivorship_complete": False, "symbols": []}

    sup = Supervisor(root=root, owner="test", deps=Deps(root))
    assert H.CA_INCOMPLETE not in sup.failures
    assert H.UNIVERSE_INCOMPLETE not in sup.failures
    assert H.NEWS_UNAVAILABLE in sup.failures

    sup.failures.add(H.CA_INCOMPLETE)
    sup.failures.add(H.UNIVERSE_INCOMPLETE)
    sup.heartbeat()
    assert H.CA_INCOMPLETE not in sup.failures
    assert H.UNIVERSE_INCOMPLETE not in sup.failures
    assert H.NEWS_UNAVAILABLE in sup.failures


def test_ca_incomplete_does_not_limit_paper_entries():
    caps = H.capabilities([H.CA_INCOMPLETE])
    assert caps["new_paper_entries"] == H.ALLOWED
    assert caps["research"] == H.LIMITED
