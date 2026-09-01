"""Phase 7 — Opportunity / regret engine."""

from __future__ import annotations

from product.opportunity_regret import (
    CORRECT_ABSTENTION,
    DD_FAILURE,
    DISCOVERY_FAILURE,
    ENTRY_TIMING_FAILURE,
    INCONCLUSIVE,
    PORTFOLIO_ALLOCATION_FAILURE,
    RANKING_FAILURE,
    evaluate_day,
)


def test_regret_classifies_failure_sources_without_rewriting_history():
    out = evaluate_day(
        as_of="2026-09-01",
        taken=[{"symbol": "TCS", "forward_return_pct": 2.0}],
        rejected=[
            {"symbol": "AAA", "reason_code": "DD_GATE_FAILED", "forward_return_pct": 8.0},
            {"symbol": "BBB", "reason_code": "DD_GATE_FAILED", "forward_return_pct": -6.0},
            {"symbol": "CCC", "reason_code": "LOW_QUALITY_SETUP", "forward_return_pct": 7.0},
        ],
        waits=[{"symbol": "DDD", "reason_code": "ENTRY_TOO_EXTENDED", "forward_return_pct": 6.0}],
        not_surfaced=[{"symbol": "EEE", "reason_code": "NOT_SURFACED", "forward_return_pct": 9.0}],
        competing=[{"symbol": "FFF", "reason_code": "NOT_TOP_OF_PORTFOLIO", "forward_return_pct": 8.0}],
    )
    by = {r["symbol"]: r["failure_source"] for r in out["rows"]}
    assert by["AAA"] == DD_FAILURE
    assert by["BBB"] == CORRECT_ABSTENTION
    assert by["CCC"] == RANKING_FAILURE
    assert by["DDD"] == ENTRY_TIMING_FAILURE
    assert by["EEE"] == DISCOVERY_FAILURE
    assert by["FFF"] == PORTFOLIO_ALLOCATION_FAILURE
    assert out["affects_hard_controls"] is False
    assert all(r["rewrote_historical_decision"] is False for r in out["rows"])
    assert out["research_hypotheses"]


def test_missing_forward_return_is_inconclusive():
    out = evaluate_day(rejected=[{"symbol": "ZZZ", "reason_code": "DD_GATE_FAILED"}])
    assert out["rows"][0]["failure_source"] == INCONCLUSIVE
    assert out["rows"][0]["forward_return_pct"] is None
