"""product/evolution/autonomy_hook.py -- the production wiring seam called
from research/autonomy/jobs.py's Deps.run_paper_cycle() right after the REAL
paper cycle has already run. Entirely best-effort: must never raise into its
caller, and must bootstrap a Champion the first time it runs.
"""
from __future__ import annotations

from unittest import mock

from product.evolution import autonomy_hook, policy_registry


def _card(symbol, **over):
    card = {
        "symbol": symbol, "entry": 100.0, "stop": 95.0, "target": 115.0,
        "setup_label": "VCP", "sector": "IT", "rs_percentile": 70, "volume_ratio": 1.2,
        "extension_pct": 1.0, "entry_state": "ready", "as_of": "2026-09-30",
        "reco_tier": "high_conviction",
        "methods": [{"id": "rs", "status": "pass", "points": 80}],
    }
    card.update(over)
    return card


def _fake_recommendations_payload(cards):
    return {
        "schema_version": 4, "generated_at": "2026-09-30T10:00:00",
        "scan_scanned_at": "2026-09-30T10:00:00",
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }


def test_bootstraps_a_champion_when_none_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    assert policy_registry.current_champion(policy_registry.EQUITY) is None

    cards = [_card("RELIANCE")]
    reco = {"taken": [{"symbol": "RELIANCE", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}]}

    with mock.patch("product.recommendations_store.load_recommendations", return_value=_fake_recommendations_payload(cards)):
        result = autonomy_hook.run_tournament_for_reco_cycle(reco, as_of="2026-09-30")

    assert result is not None
    assert result["champion_policy_id"] == autonomy_hook.CHAMPION_BOOTSTRAP_ID
    champ = policy_registry.current_champion(policy_registry.EQUITY)
    assert champ is not None and champ["policy_id"] == autonomy_hook.CHAMPION_BOOTSTRAP_ID
    seeded = policy_registry.active_challengers(policy_registry.EQUITY)
    assert len(seeded) >= 5

    # A second call reuses the SAME champion and seed population -- never duplicates.
    with mock.patch("product.recommendations_store.load_recommendations", return_value=_fake_recommendations_payload(cards)):
        autonomy_hook.run_tournament_for_reco_cycle(reco, as_of="2026-09-30")
    assert len(policy_registry.list_policies(domain=policy_registry.EQUITY, status=policy_registry.CHAMPION)) == 1
    assert len(policy_registry.active_challengers(policy_registry.EQUITY)) == len(seeded)


def test_real_champion_decision_is_frozen_from_the_reco_cycle_output(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    cards = [_card("RELIANCE"), _card("TCS")]
    reco = {
        "taken": [{"symbol": "RELIANCE", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}],
        "rejections": [{"symbol": "TCS", "decision": "BLOCK", "reason_code": "ENTRY_TOO_EXTENDED", "selection_score": 10.0}],
        "waits": [],
    }
    with mock.patch("product.recommendations_store.load_recommendations", return_value=_fake_recommendations_payload(cards)):
        result = autonomy_hook.run_tournament_for_reco_cycle(reco, as_of="2026-09-30")

    assert result is not None
    by_symbol = {r["symbol"]: r for r in result["results"]}
    assert by_symbol["RELIANCE"]["champion"]["decision"] == "ENTER_NOW"
    assert by_symbol["TCS"]["champion"]["decision"] == "REJECT"
    assert by_symbol["TCS"]["champion"]["reason_code"] == "ENTRY_TOO_EXTENDED"


def test_never_raises_when_recommendations_cannot_be_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    with mock.patch("product.recommendations_store.load_recommendations", side_effect=RuntimeError("disk gone")):
        result = autonomy_hook.run_tournament_for_reco_cycle({"taken": []}, as_of="2026-09-30")
    assert result is None


def test_returns_none_quietly_when_no_candidates_this_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    with mock.patch("product.recommendations_store.load_recommendations", return_value=_fake_recommendations_payload([])):
        result = autonomy_hook.run_tournament_for_reco_cycle({"taken": []}, as_of="2026-09-30")
    assert result is None


def test_real_paper_cycle_handler_survives_a_broken_tournament_hook(tmp_path, monkeypatch):
    """The actual research/autonomy/jobs.py wiring point: even if the
    Evolution Engine hook raises, Deps.run_paper_cycle's real result must be
    returned unaffected (section 32 applied at the real call site, not just
    inside the hook's own try/except)."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import research.autonomy.jobs as JOBS

    class _FakeBrain:
        intel_book = object()

        def is_paper_auto_enabled(self):
            return True

        def run_intelligence_cycle_day(self, **kwargs):
            return {"as_of_date": "2026-09-30", "positions_opened": []}

        def _save_intel_book(self):
            pass

    deps = JOBS.Deps.__new__(JOBS.Deps)
    deps.live_feed = None

    class _FakeTelegram:
        def notify_paper_cycle(self, *a, **k):
            pass

    deps.telegram = _FakeTelegram()

    with mock.patch("research.auto_research.scheduler.get_brain", return_value=_FakeBrain()), \
         mock.patch("product.paper_autopilot.run_reco_paper_cycle", return_value={
             "taken": [], "rejections": [], "waits": [], "final_decision": "NO_TRADE",
             "eligibility": "NO_ELIGIBLE_TRADE", "cycle_reasons": [], "reason_counts": {}, "summary": "",
         }), \
         mock.patch(
             "product.evolution.autonomy_hook.run_tournament_for_reco_cycle",
             side_effect=RuntimeError("evolution engine exploded"),
         ):
        result = deps.run_paper_cycle(True)

    assert isinstance(result, dict)
    assert result.get("as_of_date") == "2026-09-30"
