"""Phase 11 — execution-aware learning, PIT correlation, promotion governance."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from product.champion_challenger import ChampionChallengerEngine
from product.evidence_integrity import settled_learning_result
from product.learning_policy_store import load_policies
from product.pit_correlation import build_pit_correlations
from product.portfolio_selection_authority import allocate
from product.promotion_governance import promotion_board


def test_settled_learning_preserves_gross_and_uses_more_conservative_adjusted_r():
    trade = {
        "symbol": "TCS",
        "entry_price": 100.0,
        "exit_price": 115.0,
        "stop_price": 94.0,
        "qty": 10,
        "realized_R": 2.5,
    }
    out = settled_learning_result(trade, {"entry_fill": 100.0, "stop": 94.0, "qty": 10})
    assert out["gross_realized_R"] == 2.5
    assert out["execution_adjusted_available"] is True
    assert out["execution_adjusted_R"] < 2.5
    assert out["policy_realized_R"] <= out["gross_realized_R"]
    assert out["paper_fill_unchanged"] is True
    assert out["affects_orders"] is False
    assert "slippage_unknown_not_treated_as_zero" in out["warnings"]


def test_settled_learning_missing_execution_geometry_stays_gross_only():
    out = settled_learning_result(
        {"symbol": "TCS", "realized_R": 1.2, "entry_price": 100, "exit_price": 108, "stop_price": 94},
        {},
    )
    assert out["gross_realized_R"] == 1.2
    assert out["execution_adjusted_R"] is None
    assert out["policy_realized_R"] == 1.2
    assert out["execution_adjusted_available"] is False
    assert "qty" in out["missing"]


def test_paper_learning_policy_records_gross_and_execution_adjusted_r(tmp_path, monkeypatch):
    from product.paper_learning_loop import ingest_closed_trade, record_taken_evidence

    evidence_path = tmp_path / "taken.jsonl"
    monkeypatch.setenv("QT_TAKEN_EVIDENCE", str(evidence_path))
    policy_path = tmp_path / "policies.json"
    record_taken_evidence([
        {
            "symbol": "TCS",
            "setup_label": "VCP",
            "sector": "Technology",
            "entry_state": "ready",
            "entry": 100.0,
            "entry_fill": 100.0,
            "stop": 94.0,
            "target": 115.0,
            "qty": 10,
            "regime": "RISK_ON",
        }
    ], as_of="2026-09-01")
    ingest_closed_trade(
        {
            "symbol": "TCS",
            "entry_price": 100.0,
            "exit_price": 115.0,
            "stop_price": 94.0,
            "qty": 10,
            "entry_date": "2026-09-01",
            "exit_date": "2026-09-02",
            "exit_reason": "TARGET",
            "realized_R": 2.5,
            "pnl": 150.0,
        },
        path=policy_path,
        floors={"experimental": 1, "eligible": 2, "active": 3, "conditional_mult": 1},
    )
    policies = load_policies(policy_path)["policies"]
    setup = next(p for p in policies if p["policy_id"] == "SETUP::VCP")
    assert setup["gross_realized_R"] == 2.5
    assert setup["execution_adjusted_R"] is not None
    assert setup["policy_realized_R"] <= 2.5
    assert setup["last_observation_R"] == round(setup["policy_realized_R"], 4)
    assert setup["evidence_source"] == "paper_forward_taken_execution_adjusted"
    assert setup["paper_fill_unchanged"] is True


def test_pit_correlation_uses_only_local_official_data_and_respects_cutoff(monkeypatch):
    from data import bhavcopy_store as store

    dates = pd.bdate_range("2026-01-01", periods=100)
    a = pd.DataFrame({"close": [100.0 + i for i in range(100)]}, index=dates)
    b = pd.DataFrame({"close": [200.0 + 2 * i for i in range(100)]}, index=dates)
    monkeypatch.setattr(store, "is_ready", lambda: True)
    monkeypatch.setattr(store, "get_ohlcv", lambda symbol: a.copy() if symbol == "AAA" else b.copy() if symbol == "BBB" else None)
    out = build_pit_correlations(["AAA", "BBB"], as_of="2026-04-30", window=60, min_periods=30)
    assert out["source"] == "official_nse_bhavcopy_local"
    assert out["network_used"] is False
    assert out["point_in_time"] is True
    assert out["as_of"] == "2026-04-30"
    assert out["pair_samples"]["AAA|BBB"] >= 30
    assert out["correlations"]["AAA|BBB"] > 0.99


def test_portfolio_authority_automatically_consumes_local_pit_correlation(monkeypatch):
    from product import pit_correlation

    monkeypatch.setattr(
        pit_correlation,
        "correlations_for_candidates",
        lambda rows, held_symbols=None, **kwargs: {
            "source": "official_nse_bhavcopy_local",
            "point_in_time": True,
            "as_of": "2026-09-01",
            "coverage": 1.0,
            "network_used": False,
            "correlations": {"TCS|WIPRO": 0.95},
        },
    )
    held = type("P", (), {"symbol": "TCS", "sector": "Technology"})()
    book = type("B", (), {"open": {("QT", "TCS"): held}, "capital": 100_000, "max_positions": 5})()
    rows = [{
        "symbol": "WIPRO",
        "selection_score": 90,
        "sector": "Technology",
        "reco_tier": "high_conviction",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "volume_ratio": 1.2,
        "dd_verdict": "PASS",
    }]
    out = allocate(rows, book=book, correlations=None)
    assert out[0].reason_code == "CORRELATION_CAP"
    assert out[0].fields["correlation_source"] == "official_nse_bhavcopy_local"
    assert out[0].fields["network_used"] is False


def _register_positive(eng: ChampionChallengerEngine, cid: str) -> None:
    eng.register(
        challenger_id=cid,
        hypothesis="execution-aware challenger",
        changed_behavior="ranking",
        rules={"ranking_weights": {"VCP": 1.1}},
        training_data="train",
        validation_data="validation",
        oos_data="oos",
    )


def test_gross_positive_challenger_cannot_promote_without_execution_evidence(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    _register_positive(eng, "gross_only")
    before = eng.champion()["rules_hash"]
    for _ in range(35):
        eng.record_forward("gross_only", pnl=0.6, split="oos")
    comparison = eng.compare("gross_only")
    assert comparison["expectancy"] > 0
    assert comparison["execution_adjusted_n"] == 0
    blocked = eng.promote("gross_only", adversarial_status="SURVIVED")
    assert blocked["promoted"] is False
    assert "EXECUTION_EVIDENCE_INCOMPLETE" in blocked["reasons"]
    assert eng.champion()["rules_hash"] == before


def test_gross_edge_that_dies_after_execution_cannot_promote(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    _register_positive(eng, "cost_killed")
    before = eng.champion()["rules_hash"]
    for _ in range(35):
        eng.record_forward("cost_killed", pnl=0.5, execution_adjusted_pnl=-0.05, split="oos")
    comparison = eng.compare("cost_killed")
    assert comparison["expectancy"] > 0
    assert comparison["execution_adjusted_expectancy"] < 0
    blocked = eng.promote("cost_killed", adversarial_status="SURVIVED")
    assert blocked["promoted"] is False
    assert "GROSS_EDGE_DID_NOT_SURVIVE_EXECUTION" in blocked["reasons"]
    assert eng.champion()["rules_hash"] == before


def test_positive_execution_adjusted_challenger_remains_explicitly_promotable(tmp_path):
    eng = ChampionChallengerEngine(tmp_path / "ch.json")
    _register_positive(eng, "survivor")
    before = eng.champion()["rules_hash"]
    for _ in range(35):
        eng.record_forward("survivor", pnl=0.6, execution_adjusted_pnl=0.35, split="oos")
    comparison = eng.compare("survivor")
    assert comparison["execution_adjusted_coverage"] == 1.0
    result = eng.promote("survivor", adversarial_status="SURVIVED")
    assert result["promoted"] is True
    assert result["explicit"] is True
    assert result["previous_rules_hash"] == before
    assert result["execution_adjusted_expectancy"] > 0


def test_promotion_board_is_fail_closed_and_never_enables_live_money():
    board = promotion_board([
        {
            "component": "Execution Reality",
            "status": "SHADOW",
            "forward_n": 42,
            "gross_expectancy": 0.4,
            "execution_adjusted_expectancy": -0.1,
            "execution_adjusted_coverage": 1.0,
            "adversarial_status": "SURVIVED",
        },
        {
            "component": "Regime 2.0",
            "status": "SHADOW",
            "forward_n": 55,
            "gross_expectancy": 0.35,
            "execution_adjusted_expectancy": 0.2,
            "execution_adjusted_coverage": 0.95,
            "adversarial_status": "SURVIVED",
        },
    ])
    assert board["live_locked"] is True
    first = board["components"][0]
    assert first["decision"] == "KEEP_SHADOW"
    assert "GROSS_EDGE_DID_NOT_SURVIVE_EXECUTION" in first["blockers"]
    second = board["components"][1]
    assert second["decision"] == "ELIGIBLE"
    assert second["explicit_promotion_required"] is True
