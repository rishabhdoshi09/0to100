"""Tournament orchestrator: consensus, ranking-cap, and the "grade decisions
not taken" requirement (sections 8, 16, 37, 38).
"""
from __future__ import annotations

from unittest import mock

from product.evolution import grading, policy_eval, policy_registry, scorecard, shadow_decisions, tournament


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


def _register(name, domain=policy_registry.EQUITY, status=policy_registry.CHALLENGER, weights=None, parent=None):
    return policy_registry.register_policy(
        policy_id=name, domain=domain, hypothesis=f"test hypothesis for {name}",
        weights=weights or {}, status=status, parent_policy_id=parent,
    )


# ── consensus (section 16, 37) ──────────────────────────────────────────────

def test_consensus_all_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("C1", parent="CHAMP")
    _register("C2", parent="CHAMP")

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    consensus = result["results"][0]["consensus"]
    assert consensus["qualified_count"] == 3
    assert consensus["consensus_pct"] == 100.0


def test_consensus_half_disagree(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    # A policy that hard-rejects via a hostile min_empirical_sample gate.
    _register("REJECTOR", parent="CHAMP", weights={"min_empirical_sample": 9999})
    _register("ACCEPTOR", parent="CHAMP", weights={})

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    consensus = result["results"][0]["consensus"]
    assert consensus["qualified_count"] == 3
    # Champion + ACCEPTOR select, REJECTOR does not -> 2/3
    assert consensus["selecting_count"] == 2
    assert abs(consensus["consensus_pct"] - 66.7) < 0.2
    assert consensus["main_dissent_reason"] == "POLICY_MIN_SAMPLE_NOT_MET"


def test_consensus_one_outlier(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    for i in range(5):
        _register(f"C{i}", parent="CHAMP")
    _register("OUTLIER", parent="CHAMP", weights={"min_empirical_sample": 9999})

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    consensus = result["results"][0]["consensus"]
    assert consensus["qualified_count"] == 7
    assert consensus["selecting_count"] == 6


def test_consensus_denominator_excludes_retired_and_rejected_policies(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("ACTIVE", parent="CHAMP")
    _register("DEAD", parent="CHAMP", status=policy_registry.RETIRED)
    _register("BAD", parent="CHAMP", status=policy_registry.REJECTED)

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    assert result["challengers_evaluated"] == ["ACTIVE"]
    consensus = result["results"][0]["consensus"]
    assert consensus["qualified_count"] == 2  # champion + ACTIVE only


# ── grading decisions not taken (section 8, 38) ─────────────────────────────

def test_champion_rejects_challenger_takes_and_it_wins_counts_for_challenger(tmp_path, monkeypatch):
    """Champion rejects A. Challenger takes A. A becomes a winner. The
    difference must count toward the Challenger's incremental evidence."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("BOLD_CHALLENGER", parent="CHAMP", weights={"relative_strength_mult": 2.0})

    cards = [_card("TCS")]
    # Champion's REAL production decision rejected it (too extended, say).
    champion_decisions = {"TCS": {"decision": "BLOCK", "reason_code": "ENTRY_TOO_EXTENDED", "selection_score": 20.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    row = result["results"][0]
    assert row["champion"]["decision"] == "REJECT"
    assert row["challengers"]["BOLD_CHALLENGER"]["decision"] == "ENTER_NOW"

    # The market later proves it was a big winner -- grade BOTH with the
    # SAME realized price path (the Common Outcome Resolver principle).
    with mock.patch.object(grading, "first_touch_path", return_value=(115.0, 15.0, 1)):
        champ_graded = grading.grade_shadow_decision(row["champion"])
        chal_graded = grading.grade_shadow_decision(row["challengers"]["BOLD_CHALLENGER"])

    assert champ_graded["classification"] == "MISSED_WINNER"
    assert chal_graded["classification"] == "WINNER_TAKEN"
    assert chal_graded["counterfactual_R"] > 0

    paired = scorecard.paired_comparison("CHAMP", "BOLD_CHALLENGER")
    assert paired["paired_snapshots"] == 1
    assert paired["incremental_expectancy_R"] > 0
    assert paired["agreement"]["challenger_only"] == 1


def test_champion_takes_challenger_rejects_and_it_loses_credits_challenger(tmp_path, monkeypatch):
    """Champion takes B. Challenger rejects B. B loses. That must count as
    the Challenger's correct avoidance."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("CAUTIOUS_CHALLENGER", parent="CHAMP", weights={"min_empirical_sample": 9999})

    cards = [_card("INFY")]
    champion_decisions = {"INFY": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 80.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    row = result["results"][0]
    assert row["champion"]["decision"] == "ENTER_NOW"
    assert row["challengers"]["CAUTIOUS_CHALLENGER"]["decision"] == "REJECT"

    with mock.patch.object(grading, "first_touch_path", return_value=(95.0, -5.0, 0)):
        champ_graded = grading.grade_shadow_decision(row["champion"])
        chal_graded = grading.grade_shadow_decision(row["challengers"]["CAUTIOUS_CHALLENGER"])

    assert champ_graded["classification"] == "LOSER_TAKEN"
    # product.counterfactual_learning's pre-existing taxonomy labels a clear
    # stop-level decline as CORRECT_REJECTION here (AVOIDED_LOSER is its
    # label for a milder decline below -5% that doesn't reach the stop) --
    # either way this is unambiguously "good for the rejecting policy".
    assert chal_graded["classification"] == "CORRECT_REJECTION"

    paired = scorecard.paired_comparison("CHAMP", "CAUTIOUS_CHALLENGER")
    assert paired["incremental_expectancy_R"] > 0  # challenger's R (0 risk taken) beats champion's realized loss
    assert paired["agreement"]["champion_only"] == 1


# ── ranking cap applies per-policy, not globally (section 5/21 portfolio seam) ──

def test_ranking_cap_lets_a_policy_select_a_different_top_pick_than_champion(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("RS_HEAVY", parent="CHAMP", weights={"relative_strength_mult": 5.0})

    cards = [
        _card("HIGH_RS", rs_percentile=95),
        _card("LOW_RS", rs_percentile=20),
    ]
    # Realistic: this is what run_reco_paper_cycle's OWN ranking cap already
    # produced upstream (only the top-ranked name keeps ENTER_NOW; the
    # Champion's real selection_score ranked LOW_RS higher here).
    champion_decisions = {
        "HIGH_RS": {"decision": "BLOCK", "reason_code": "NOT_TOP_RANKED_THIS_CYCLE", "selection_score": 50.0},
        "LOW_RS": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 95.0},
    }
    result = tournament.run_tournament_cycle(
        cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30", max_new=1,
    )
    by_symbol = {r["symbol"]: r for r in result["results"]}
    # Champion (using its own real decision scores) picked LOW_RS.
    assert by_symbol["LOW_RS"]["champion"]["decision"] == "ENTER_NOW"
    assert by_symbol["HIGH_RS"]["champion"]["decision"] == "REJECT"
    # RS_HEAVY's own ranking (5x RS weight) should flip the pick to HIGH_RS.
    assert by_symbol["HIGH_RS"]["challengers"]["RS_HEAVY"]["decision"] == "ENTER_NOW"
    assert by_symbol["LOW_RS"]["challengers"]["RS_HEAVY"]["decision"] == "REJECT"


# ── wall-clock budget: a slow/hung Challenger can never stall the real
# Champion path indefinitely (section 30/31 performance constraint) ─────────

def test_slow_challenger_is_skipped_once_budget_exhausted_never_blocks_champion(tmp_path, monkeypatch):
    import time

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("SLOW", parent="CHAMP")
    _register("FAST", parent="CHAMP")

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}

    def _evaluator(snapshots, challenger_policy):
        if challenger_policy["policy_id"] == "SLOW":
            time.sleep(0.2)
        return [
            {"symbol": s["symbol"], "decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "adjusted_score": 10.0}
            for s in snapshots
        ]

    result = tournament.run_tournament_cycle(
        cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30",
        max_seconds=0.05, challenger_batch_evaluator=_evaluator,
    )
    # The Champion's own row is frozen regardless -- the real decision this
    # cycle made is never affected by how long Challenger research takes.
    assert result["results"][0]["champion"]["decision"] == "ENTER_NOW"
    # SLOW ran (registered first) and ate the whole budget; FAST never got
    # its turn and is explicitly reported as skipped, not silently dropped.
    assert "SLOW" in result["challengers_evaluated"]
    assert "FAST" in result["challengers_skipped"]
    assert result["tournament_budget_exhausted"] is True


def test_default_budget_does_not_affect_normal_fast_cycles(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    _register("CHAMP", status=policy_registry.CHAMPION)
    _register("C1", parent="CHAMP")
    _register("C2", parent="CHAMP")

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}
    result = tournament.run_tournament_cycle(cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30")
    assert result["tournament_budget_exhausted"] is False
    assert set(result["challengers_evaluated"]) == {"C1", "C2"}
    assert result["challengers_skipped"] == []
