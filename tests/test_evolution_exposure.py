"""Section 5: distinguishing "Champion looks better because it received more
observable decisions" from "Champion genuinely performs better on the same
opportunities." Promotion itself is already immune (scorecard.paired_
comparison only ever compares the shared intersection) -- this diagnostic
makes the coverage gap visible to an operator, which nothing did before.
"""
from __future__ import annotations

from product.evolution import policy_registry as PR
from product.evolution import shadow_decisions as SD
from product.evolution.exposure import exposure_report, LOW_COVERAGE_THRESHOLD


def _seed(policy_id, sid, r, *, decision="ENTER_NOW"):
    row = {
        "schema_version": 1, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
        "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
        "decision": decision, "reason_code": "ELIGIBLE" if decision == "ENTER_NOW" else "REJECT",
        "adjusted_score": 50.0, "breakdown": {}, "entry": 100.0, "stop": 95.0, "target": 110.0,
        "sector": "IT", "setup_label": "VCP", "regime": "TRENDING_BULL", "frozen_at": "x", "fingerprint": "x",
        "outcome": {"forward_return_pct": r * 5, "not_pnl": True},
        "classification": "WINNER_TAKEN" if r > 0 else "LOSER_TAKEN",
        "graded_at": "x", "not_pnl": True, "is_champion_decision": policy_id.endswith("CHAMP"),
        "counterfactual_R": r,
    }
    SD.save_graded_decision(row)


def test_empty_domain_reports_no_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    report = exposure_report(PR.EQUITY)
    assert report["champion_policy_id"] is None
    assert report["rows"] == []


def test_full_coverage_challenger_is_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="FULL_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="FULL_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="FULL_CHAMP")
    for i in range(20):
        _seed("FULL_CHAMP", f"snap_{i}", 0.1)
        _seed("FULL_CHAL", f"snap_{i}", 0.2)

    report = exposure_report(PR.EQUITY)
    row = next(r for r in report["rows"] if r["policy_id"] == "FULL_CHAL")
    assert row["champion_total_snapshots"] == 20
    assert row["challenger_total_snapshots"] == 20
    assert row["paired_snapshots"] == 20
    assert row["coverage_of_champion_opportunities"] == 1.0
    assert row["coverage_of_own_opportunities"] == 1.0
    assert row["narrow_sample_risk"] is False


def test_narrow_coverage_challenger_is_flagged(tmp_path, monkeypatch):
    """A Challenger that was only ever evaluated on a small fraction of what
    the Champion saw (e.g. repeatedly skipped by the tournament's per-cycle
    budget/cap) must be flagged, even though its own paired comparison math
    is still fair on the snapshots it does share."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="NARROW_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="NARROW_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="NARROW_CHAMP")
    # Champion evaluated on 50 opportunities total...
    for i in range(50):
        _seed("NARROW_CHAMP", f"snap_{i}", 0.1)
    # ...but the challenger only ever got evaluated on 5 of them.
    for i in range(5):
        _seed("NARROW_CHAL", f"snap_{i}", 0.2)

    report = exposure_report(PR.EQUITY)
    row = next(r for r in report["rows"] if r["policy_id"] == "NARROW_CHAL")
    assert row["champion_total_snapshots"] == 50
    assert row["paired_snapshots"] == 5
    assert row["coverage_of_champion_opportunities"] == 0.1
    assert row["coverage_of_champion_opportunities"] < LOW_COVERAGE_THRESHOLD
    assert row["narrow_sample_risk"] is True


def test_evolution_lab_board_surfaces_exposure_per_challenger(tmp_path, monkeypatch):
    from product.evolution.api import evolution_lab_board

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="BOARD_CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="BOARD_CHAL", domain=PR.EQUITY, hypothesis="test", weights={}, status=PR.CHALLENGER, parent_policy_id="BOARD_CHAMP")
    for i in range(10):
        _seed("BOARD_CHAMP", f"snap_{i}", 0.1)
    for i in range(2):
        _seed("BOARD_CHAL", f"snap_{i}", 0.2)

    board = evolution_lab_board(PR.EQUITY)
    row = next(r for r in board["challenger_leaderboard"] if r["policy_id"] == "BOARD_CHAL")
    assert row["exposure"] is not None
    assert row["exposure"]["paired_snapshots"] == 2
    assert row["exposure"]["narrow_sample_risk"] is True
