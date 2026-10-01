"""product/evolution/api.py -- the Evolution Lab board aggregator, and its
terminal_product_api.py wiring."""
from __future__ import annotations

from product.evolution import policy_registry as PR
from product.evolution.api import evolution_lab_board


def test_board_with_no_champion_yet_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    board = evolution_lab_board(PR.EQUITY)
    assert board["champion"] is None
    assert board["challenger_leaderboard"] == []
    assert board["live_locked"] is True
    assert board["live_execution_authorized"] is False


def test_board_reports_champion_and_leaderboard_sorted_by_incremental_value(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="WEAK", domain=PR.EQUITY, hypothesis="weak variant", weights={}, status=PR.CHALLENGER, parent_policy_id="CHAMP")
    PR.register_policy(policy_id="STRONG", domain=PR.EQUITY, hypothesis="strong variant", weights={"relative_strength_mult": 2.0}, status=PR.CHALLENGER, parent_policy_id="CHAMP")

    from product.evolution import shadow_decisions as SD

    def seed(policy_id, sid, r):
        SD.save_graded_decision({
            "schema_version": 1, "shadow_id": f"{policy_id}:{sid}", "policy_id": policy_id,
            "market_snapshot_id": sid, "domain": PR.EQUITY, "symbol": "SYM", "as_of": "2026-09-30",
            "decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "adjusted_score": 50.0, "breakdown": {},
            "entry": 100.0, "stop": 95.0, "target": 110.0, "sector": "IT", "setup_label": "VCP",
            "regime": "TRENDING_BULL", "frozen_at": "x", "fingerprint": "x",
            "outcome": {"forward_return_pct": r * 5}, "classification": "WINNER_TAKEN" if r > 0 else "LOSER_TAKEN",
            "graded_at": "x", "not_pnl": True, "is_champion_decision": policy_id == "CHAMP",
            "counterfactual_R": r,
        })

    for i in range(10):
        seed("CHAMP", f"snap_{i}", 0.0)
        seed("WEAK", f"snap_{i}", -0.5)
        seed("STRONG", f"snap_{i}", 1.0)

    board = evolution_lab_board(PR.EQUITY)
    assert board["champion"]["policy_id"] == "CHAMP"
    ids = [row["policy_id"] for row in board["challenger_leaderboard"]]
    assert ids == ["STRONG", "WEAK"]  # best incremental value first
    assert board["challenger_leaderboard"][0]["paired_vs_champion"]["incremental_expectancy_R"] > 0
    assert board["challenger_leaderboard"][1]["paired_vs_champion"]["incremental_expectancy_R"] < 0


def test_retired_and_rejected_policies_are_excluded_from_the_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(policy_id="CHAMP", domain=PR.EQUITY, hypothesis="baseline", weights={}, status=PR.CHAMPION)
    PR.register_policy(policy_id="DEAD", domain=PR.EQUITY, hypothesis="dead", weights={}, status=PR.RETIRED, parent_policy_id="CHAMP")
    board = evolution_lab_board(PR.EQUITY)
    assert board["challenger_leaderboard"] == []
    assert board["retired_count"] == 1


def test_evolution_lab_endpoint_is_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import terminal_product_api as api

    routes = {getattr(r, "path", "") for r in api.app.routes}
    assert "/api/evolution-lab" in routes
