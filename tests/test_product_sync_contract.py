from __future__ import annotations

import json

from product.fundamental_intelligence import build_fundamental_intelligence
from product import research_status as RS
from product.strategy_contract import (
    production_strategies,
    strategy_for_category,
    parity_for_strategy,
)


def test_production_strategy_registry_has_unique_versioned_rules():
    rows = production_strategies()
    assert {row["category_id"] for row in rows} == {
        "wealth_builders", "super_trends", "momentum_breakouts", "recovery_setups"
    }
    assert len({row["strategy_id"] for row in rows}) == len(rows)
    assert len({row["rules_hash"] for row in rows}) == len(rows)
    assert all(row["version"] >= 1 and row["universe"] == "NSE_EQ" for row in rows)


def test_backtest_parity_fails_closed_then_accepts_exact_hash(monkeypatch):
    import research.registry as REG

    strategy = strategy_for_category("momentum_breakouts")
    assert strategy is not None
    monkeypatch.setattr(REG, "list_experiments", lambda: [])
    missing = parity_for_strategy(strategy)
    assert missing["status"] == "UNVERIFIED"
    assert missing["evidence"] is None

    monkeypatch.setattr(REG, "list_experiments", lambda: [{
        "hypothesis_id": "EXP-EXACT",
        "status": "PROMOTED",
        "code_hash": strategy.rules_hash,
        "evaluated_at": "2026-09-01T10:00:00",
        "data_window": '{"from":"2022-01-01","to":"2026-08-31"}',
        "result": (
            '{"strategy_id":"%s","strategy_version":%d,'
            '"n_trades":128,"net_expectancy_R":0.74,"max_drawdown":0.18}'
            % (strategy.strategy_id, strategy.version)
        ),
    }])
    exact = parity_for_strategy(strategy)
    assert exact["status"] == "VERIFIED"
    assert exact["experiment_id"] == "EXP-EXACT"
    assert exact["evidence"]["n_trades"] == 128
    assert exact["evidence"]["net_expectancy_R"] == 0.74


def test_backtest_parity_rejects_old_rules_hash(monkeypatch):
    import research.registry as REG

    strategy = strategy_for_category("super_trends")
    assert strategy is not None
    monkeypatch.setattr(REG, "list_experiments", lambda: [{
        "hypothesis_id": "OLD",
        "status": "PROMOTED",
        "code_hash": "old-rules",
        "result": '{"n_trades":999,"net_expectancy_R":99}',
    }])
    result = parity_for_strategy(strategy)
    assert result["status"] == "UNVERIFIED"
    assert result["evidence"] is None


def test_recommendation_ledger_freezes_strategy_version_and_rules_hash(tmp_path):
    from product.reco_ledger import append_recommendations

    strategy = strategy_for_category("momentum_breakouts")
    assert strategy is not None
    path = tmp_path / "reco.jsonl"
    card = {
        "symbol": "AAA",
        "category_id": "momentum_breakouts",
        "reco_tier": "good_setup",
        "primary_thesis": "Breakout with independent confirmation",
        "methods": [],
        "entry": 100,
        "stop": 95,
        "target": 112,
        "cmp": 99,
    }
    assert append_recommendations([card], scan_scanned_at="2026-09-01T10:00:00Z", path=path) == path
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["schema_version"] == 3
    frozen = record["cards"][0]["strategy"]
    assert frozen["strategy_id"] == strategy.strategy_id
    assert frozen["strategy_version"] == strategy.version
    assert frozen["rules_hash"] == strategy.rules_hash


def test_research_status_exposes_real_blockers_without_claiming_learning(monkeypatch):
    monkeypatch.setattr(RS, "_production_strategy_status", lambda: {
        "production_strategy_count": 4,
        "verified_backtest_parity_count": 1,
        "unverified_backtest_parity_count": 3,
        "strategies": [],
    })
    monkeypatch.setattr(RS, "_research_overview", lambda: {
        "research_health": {
            "experiments_awaiting_validation": 2,
            "experiments_promoted": 5,
            "experiments_rejected": 9,
            "beliefs_active": 3,
            "beliefs_watch": 2,
            "beliefs_retired": 4,
            "promoted_this_week": 0,
            "retired_this_week": 1,
            "calibration": {"n": 0},
        },
        "knowledge_growth": {"net_knowledge_gain": -1, "avg_evidence_per_belief": 10},
        "edge_health": {"tracked_signals": 8, "durable": 2, "decaying": 1, "dead": 1, "recovering": 1},
        "research_debt": {"drift_alerts_unresolved": 1},
        "data_health": {"total_observations": 50, "on_current_schema": True},
    })
    monkeypatch.setattr(RS, "_decision_status", lambda: {
        "surfaced_history": 4,
        "latest_scan_decisions": 100,
        "settled_sample_size": 0,
        "performance_claim_allowed": False,
        "performance_label": "UNAVAILABLE",
    })
    status = RS.build_research_status()
    assert status["state"] == "ATTENTION"
    assert status["production"]["unverified_backtest_parity"] == 3
    assert status["experiments"]["awaiting_validation"] == 2
    assert status["decisions"]["performance_claim_allowed"] is False
    assert any("BACKTEST PARITY" in blocker for blocker in status["blockers"])
    assert any("No settled decision sample" in blocker for blocker in status["blockers"])


def test_fundamental_intelligence_projects_existing_research_without_inventing():
    stock = {"symbol": "AAA", "company": "Alpha", "fundamentals": {"company_about": "Alpha makes widgets."}}
    dd = {
        "symbol": "AAA",
        "company": "Alpha",
        "profile": {"business_model": "Manufacturing"},
        "framework": {"id": "industrial", "label": "Industrials", "sub_sector": "Components", "blurb": "Industrial framework"},
        "fundamental_quality": {"score": 82, "label": "Strong", "coverage_pct": 75, "score_coverage_pct": 70, "explain": "Measured evidence."},
        "research_coverage": {"coverage_pct": 75},
        "decision_coverage": {"coverage_pct": 80},
        "decision_coverage_pct": 80,
        "kpis": [
            {
                "id": "order_book", "label": "Order book", "pillar": "growth", "available": True,
                "trend": "improving", "snapshot": {"current": 1000, "current_period": "FY26"},
                "fact": "Order book: 1000 Cr", "interpretation": "Order book grew.",
                "provenance": {"source": "Company filing", "source_url": "https://example.test/filing", "confidence": "high"},
            },
            {
                "id": "cash_conversion", "label": "Cash conversion", "pillar": "cash", "available": False,
                "trend": "unknown", "fact": "Data unavailable",
            },
        ],
        "cash_flow_quality": "Mixed",
        "balance_sheet_quality": "Adequate",
        "growth_quality": "Improving",
        "financial_strength": "Strong",
        "earnings_quality": "Improving",
        "governance_risk": "Low",
        "red_flags": [{"id": "receivables", "severity": "watch", "kind": "accounting", "title": "Receivables rising", "source": "Annual report"}],
        "concerns": ["Cash conversion weakened YoY."],
        "critical_metrics_missing": [{"id": "cash_conversion", "label": "Cash conversion"}],
        "missing_evidence": [{"key": "management_commentary"}],
        "extracted_guidance": [{"metric": "Revenue growth", "excerpt": "Expect 20% growth", "source": "Concall"}],
        "valuation": [{"id": "pe", "label": "P/E", "value": 32, "source": "Cached fundamentals"}],
        "peers": [{"symbol": "BBB"}],
        "sources": [{"source": "Company filing"}],
        "filings": [{"title": "FY26 Results"}],
        "thesis": {"summary": "Growth is improving."},
    }
    dossier = build_fundamental_intelligence(stock, dd)
    assert dossier["available"] is True
    assert dossier["fundamental_score"]["score"] == 82
    assert dossier["fundamental_score"]["missing_is_zero"] is False
    assert dossier["business_specific_kpis"]["measured_count"] == 1
    assert dossier["business_specific_kpis"]["missing"][0]["label"] == "Cash conversion"
    assert dossier["management_execution"]["guidance"][0]["execution_status"] == "UNMEASURED"
    assert any(item["type"] == "MISSING_CRITICAL_EVIDENCE" for item in dossier["thesis_breakers"])
    assert dossier["places_orders"] is False


def test_fundamental_intelligence_empty_state_is_honest():
    dossier = build_fundamental_intelligence({"symbol": "AAA"}, {})
    assert dossier["available"] is False
    assert dossier["score"] is None
    assert "will not infer" in dossier["message"]
