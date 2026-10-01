"""Durable PAPER owner -> deferred Evolution handoff tests."""
from __future__ import annotations

from types import SimpleNamespace

from product.evolution import deferred_work as DW
from research.auto_research.paper_book import PaperBook
from research.autonomy import jobs as JOBS


class _Telegram:
    def notify_paper_cycle(self, *_args, **_kwargs):
        return None


def _prepared(tmp_path):
    return DW.prepare_work(
        bundle={
            "domain": "EQUITY",
            "as_of": "2026-09-30",
            "champion_policy_id": "CHAMP",
            "snapshots": {},
            "champion_rows": {},
        },
        champion_policy_fingerprint="fp-owner",
        challenger_policies=[],
        individual_decisions_by_symbol={},
        pre_mutation_book_snapshot=PaperBook(capital=100_000).snapshot(),
        held_sector_by_symbol={},
        correlations={},
        max_new=1,
        regime="RISK_ON",
    )


def _deps():
    deps = object.__new__(JOBS.Deps)
    deps.live_feed = None
    deps.telegram = _Telegram()
    return deps


def test_deferred_work_released_only_after_durable_paper_save(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    prepared = _prepared(tmp_path)
    work_id = prepared["work_id"]
    events: list[str] = []

    class Brain:
        intel_book = PaperBook(capital=100_000)

        def run_intelligence_cycle_day(self, **_kwargs):
            return {"as_of_date": "2026-09-30", "positions_opened": []}

        def is_paper_auto_enabled(self):
            return True

        def _save_intel_book(self):
            # The critical proof: research is not READY before the durable
            # PAPER owner reaches its atomic save boundary.
            assert DW.get_work(work_id)["status"] == "PREPARED"
            events.append("durable_save")
            return True

    import research.auto_research.scheduler as scheduler
    import product.paper_autopilot as PA

    monkeypatch.setattr(scheduler, "get_brain", lambda: Brain())

    def fake_reco(**_kwargs):
        events.append("paper_cycle_returned")
        return {
            "positions_opened": [("ensemble", "TCS")],
            "taken": [{"symbol": "TCS"}],
            "rejections": [],
            "waits": [],
            "final_decision": "ENTER_NOW",
            "eligibility": "TRADED",
            "cycle_reasons": [],
            "reason_counts": {},
            "summary": "taken=1",
            "evolution": {
                "deferred_work_id": work_id,
                "deferred_work_status": "PREPARED",
            },
        }

    monkeypatch.setattr(PA, "run_reco_paper_cycle", fake_reco)

    result = _deps().run_paper_cycle(True)
    assert events == ["paper_cycle_returned", "durable_save"]
    assert DW.get_work(work_id)["status"] == "READY"
    assert (
        result["reco_autopilot"]["evolution"]["deferred_work_status"]
        == "READY"
    )


def test_failed_durable_save_abandons_research_and_surfaces_paper_error(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    prepared = _prepared(tmp_path)
    work_id = prepared["work_id"]

    class Brain:
        intel_book = PaperBook(capital=100_000)

        def run_intelligence_cycle_day(self, **_kwargs):
            return {"as_of_date": "2026-09-30", "positions_opened": []}

        def is_paper_auto_enabled(self):
            return True

        def _save_intel_book(self):
            return False

    import research.auto_research.scheduler as scheduler
    import product.paper_autopilot as PA

    monkeypatch.setattr(scheduler, "get_brain", lambda: Brain())
    monkeypatch.setattr(
        PA,
        "run_reco_paper_cycle",
        lambda **_kwargs: {
            "positions_opened": [("ensemble", "TCS")],
            "taken": [{"symbol": "TCS"}],
            "rejections": [],
            "waits": [],
            "final_decision": "ENTER_NOW",
            "eligibility": "TRADED",
            "cycle_reasons": [],
            "reason_counts": {},
            "summary": "taken=1",
            "evolution": {"deferred_work_id": work_id},
        },
    )

    result = _deps().run_paper_cycle(True)
    assert DW.get_work(work_id)["status"] == "ABANDONED"
    assert "durable intel_book save failed" in result["reco_autopilot"]["error"]
