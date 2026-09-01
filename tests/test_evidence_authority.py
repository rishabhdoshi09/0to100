from __future__ import annotations

import json

from product.evidence_authority import evidence_scorecard


def test_evidence_score_keeps_unknown_separate_from_failure():
    card = {
        "methods": [
            {"id": "tape", "label": "Tape", "status": "pass", "points": 90, "detail": "clean"},
            {"id": "sepa", "label": "SEPA", "status": "unknown", "points": None, "detail": "missing"},
            {"id": "funds", "label": "Funds", "status": "fail", "points": 0, "detail": "weak"},
            {"id": "trend", "label": "Trend", "status": "pass", "points": 80, "detail": "above averages"},
            {"id": "rs", "label": "RS", "status": "unknown", "points": None, "detail": "missing"},
            {"id": "ev", "label": "Live EV", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "conviction", "label": "Conviction", "status": "pass", "points": 75, "detail": "good setup"},
            {"id": "case", "label": "Case memory", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "sector", "label": "Sector", "status": "pass", "points": 80, "detail": "leader"},
        ]
    }
    score = evidence_scorecard(card)
    assert score["passed"] == 4
    assert score["failed"] == 1
    assert score["unknown"] == 4
    assert 0 < score["coverage_pct"] < 100
    assert score["score"] is not None
    assert "not a win probability" in score["disclaimer"]
    empirical = next(c for c in score["components"] if c["id"] == "empirical")
    assert empirical["score"] is None
    assert empirical["unknown"] == 2


def test_decision_journal_combines_surfaced_and_latest_scan(monkeypatch, tmp_path):
    from product import evidence_authority as ea

    ledger = tmp_path / "reco.jsonl"
    ledger.write_text(json.dumps({
        "recorded_at": "2026-08-29T03:00:00+00:00",
        "scan_scanned_at": "2026-08-29T02:59:00+00:00",
        "cards": [{"symbol": "AAA", "tier": "good_setup", "thesis": "Two families agree"}],
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(ea, "_read_reco_ledger", lambda path=None, max_lines=500: [json.loads(ledger.read_text())])
    monkeypatch.setattr(
        "scan.scan_coverage.load_audit",
        lambda: {
            "generated_at": "2026-08-29T03:01:00+00:00",
            "summary": {"requested": 2, "checked": 2},
            "ledger": [
                {"symbol": "AAA", "status": "QUALIFIED", "reason": "setup"},
                {"symbol": "BBB", "status": "NO_SETUP", "reason": "fully checked"},
            ],
        },
    )
    monkeypatch.setattr(ea, "performance_summary", lambda: {"sample_size": 0})
    journal = ea.build_decision_journal(limit=20)
    assert journal["scan_summary"]["requested"] == 2
    assert any(row["kind"] == "SURFACED" and row["symbol"] == "AAA" for row in journal["entries"])
    assert any(row["decision"] == "NO_SETUP" and row["symbol"] == "BBB" for row in journal["entries"])


def test_performance_summary_refuses_to_invent_empty_metrics(monkeypatch):
    from product import evidence_authority as ea
    monkeypatch.setattr(
        "core.signal_outcome_tracker.get_accuracy_report",
        lambda: {"wins": 0, "losses": 0, "open_signals": 3, "system_edge": 0, "avg_win_pct": 0, "avg_loss_pct": 0},
    )
    monkeypatch.setattr("core.signal_outcome_tracker.get_recent_signals", lambda limit=5000: [])
    out = ea.performance_summary()
    assert out["sample_size"] == 0
    assert out["hit_rate_pct"] is None
    assert out["expectancy_pct"] is None
    assert out["avg_gain_pct"] is None
    assert out["avg_loss_pct"] is None
    assert out["max_drawdown_pct"] is None
    assert out["benchmark_comparison"] is None
