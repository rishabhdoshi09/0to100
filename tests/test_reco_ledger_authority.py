from __future__ import annotations

import json


def _methods():
    return [
        {"id": "tape", "label": "Tape", "status": "pass", "points": 90, "detail": "clean"},
        {"id": "sepa", "label": "SEPA", "status": "unknown", "points": None, "detail": "missing"},
        {"id": "funds", "label": "Funds", "status": "pass", "points": 80, "detail": "quality"},
        {"id": "trend", "label": "Trend", "status": "pass", "points": 85, "detail": "uptrend"},
        {"id": "rs", "label": "RS", "status": "pass", "points": 75, "detail": "leader"},
        {"id": "ev", "label": "Live EV", "status": "unknown", "points": None, "detail": "n<30"},
        {"id": "conviction", "label": "Conviction", "status": "pass", "points": 70, "detail": "ready"},
        {"id": "case", "label": "Case", "status": "unknown", "points": None, "detail": "n<30"},
        {"id": "sector", "label": "Sector", "status": "pass", "points": 80, "detail": "leader"},
    ]


def test_recommendation_ledger_freezes_evidence_and_trade_levels(tmp_path):
    from product.reco_ledger import LEDGER_VERSION, append_recommendations

    path = tmp_path / "reco.jsonl"
    card = {
        "symbol": "AAA",
        "reco_tier": "good_setup",
        "primary_thesis": "Price leadership + quality",
        "entry_state": "ready",
        "entry": 100,
        "stop": 94,
        "target": 115,
        "cmp": 101,
        "methods": _methods(),
        "families": [{"id": "price_leadership", "status": "pass"}],
        "family_confirms": 2,
        "allows_recommend": True,
    }
    append_recommendations([card], scan_scanned_at="2026-08-29T03:00:00+00:00", path=path)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["schema_version"] == LEDGER_VERSION
    assert LEDGER_VERSION >= 3
    saved = row["cards"][0]
    assert saved["entry"] == 100
    assert saved["stop"] == 94
    assert saved["target"] == 115
    assert saved["allows_recommend"] is True
    assert saved["dd_status"] == "pass"
    score = saved["evidence_scorecard"]
    assert score["score"] is not None
    assert 0 < score["coverage_pct"] < 100
    assert score["unknown"] == 3
    assert {c["id"] for c in score["components"]} == {
        "price_structure", "fundamentals", "market_sector", "empirical", "setup_risk"
    }


def test_old_ledger_rows_remain_readable_without_frozen_score(monkeypatch):
    from product import evidence_authority as ea

    monkeypatch.setattr(ea, "_read_reco_ledger", lambda: [{
        "schema_version": 1,
        "recorded_at": "2026-08-20T00:00:00+00:00",
        "cards": [{"symbol": "OLD", "tier": "good_setup", "thesis": "legacy"}],
    }])
    rows = ea._recommendation_history("OLD")
    assert len(rows) == 1
    assert rows[0]["score_frozen_at_decision"] is False
    assert rows[0]["evidence_score"] is None


def test_schema_v2_ledger_rows_remain_readable(monkeypatch):
    from product import evidence_authority as ea

    monkeypatch.setattr(ea, "_read_reco_ledger", lambda: [{
        "schema_version": 2,
        "recorded_at": "2026-08-28T00:00:00+00:00",
        "cards": [{
            "symbol": "TCS",
            "tier": "high_conviction",
            "thesis": "v2 row",
            "entry": 100,
            "stop": 95,
            "target": 120,
            "evidence_scorecard": {"score": 70, "coverage_pct": 60, "quality": "PARTIAL"},
        }],
    }])
    rows = ea._recommendation_history("TCS")
    assert len(rows) == 1
    assert rows[0]["score_frozen_at_decision"] is True
    assert rows[0]["evidence_score"] == 70
    assert rows[0]["entry"] == 100
    assert rows[0]["stop"] == 95
