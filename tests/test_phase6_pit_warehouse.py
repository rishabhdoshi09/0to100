"""Phase 6 — PIT evidence warehouse, temporal query, harvest, no lookahead."""
from __future__ import annotations

import json
from pathlib import Path

from product.pit_availability import (
    PIT_MARKET_ONLY,
    PIT_PARTIAL,
    PIT_STRONG,
    PIT_UNVERIFIED,
)
from product.pit_coverage import category_coverage, overall_replay_grade
from product.pit_ingest import harvest_symbol, ingest_announcement_row, ingest_result_row
from product.pit_query import (
    get_financial_snapshot,
    get_research_snapshot,
    get_sector_context,
    pit_research_inputs,
)
from product.pit_warehouse import (
    DOC_QUARTERLY_RESULT,
    get_evidence,
    get_evidence_raw,
    persist,
    record_conflict,
    resolve_by_authority,
)
from product.pit_versions import current_versions


def test_warehouse_rejects_future_filing(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-08-01",
        "available_from": "2026-08-01",
        "source": "NSE financial results",
        "source_identity": "nse_result:future-q",
        "extracted": {"numbers_parsed": False},
    }, path=db)
    later = get_evidence("INFY", as_of="2026-07-01", path=db)
    now = get_evidence("INFY", as_of="2026-08-01", path=db)
    assert later == []
    assert len(now) == 1
    raw = get_evidence_raw("INFY", path=db)
    assert len(raw) == 1


def test_period_end_alone_is_not_indexed(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "TCS",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "period_end": "2026-03-31",
        "publication_date": "",
        "available_from": "",
        "source": "guess",
        "source_identity": "period-only",
    }, path=db)
    rows = get_evidence("TCS", as_of="2026-06-12", path=db)
    assert rows == []
    raw = get_evidence_raw("TCS", path=db)
    assert raw[0]["pit_status"] == PIT_UNVERIFIED


def test_acquired_later_does_not_hide_published_earlier(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": "CORPORATE_ANNOUNCEMENT",
        "publication_date": "2025-05-14",
        "available_from": "2025-05-14",
        "acquired_at": "2026-09-03T00:00:00+00:00",
        "source": "NSE corporate announcements",
        "source_identity": "nse_ann:backfill-later",
    }, path=db)
    rows = get_evidence("INFY", as_of="2025-06-01", path=db)
    assert len(rows) == 1
    assert rows[0]["acquired_at"].startswith("2026-09-03")


def test_revisions_are_separate_rows(tmp_path):
    db = tmp_path / "pit.db"
    first = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-15",
        "available_from": "2026-04-15",
        "period_end": "2026-03-31",
        "source": "NSE financial results",
        "source_identity": "nse_result:q1",
        "revision": 1,
        "extracted": {"pat": 100, "numbers_parsed": True},
    }, path=db)
    second = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-07-01",
        "available_from": "2026-07-01",
        "period_end": "2026-03-31",
        "source": "NSE financial results",
        "source_identity": "nse_result:q1",
        "revision": 2,
        "supersedes": first["evidence_id"],
        "extracted": {"pat": 90, "numbers_parsed": True},
    }, path=db)
    assert first["evidence_id"] != second["evidence_id"]
    mid = get_evidence("INFY", as_of="2026-05-01", path=db)
    late = get_evidence("INFY", as_of="2026-07-02", path=db)
    assert len(mid) == 1
    assert mid[0]["extracted"]["pat"] == 100
    assert len(late) == 2


def test_conflicts_keep_both_records(tmp_path):
    db = tmp_path / "pit.db"
    left = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-15",
        "available_from": "2026-04-15",
        "source": "NSE financial results",
        "source_identity": "nse_result:a",
        "source_trust": 100,
        "extracted": {"pat": 10},
    }, path=db)
    right = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-16",
        "available_from": "2026-04-16",
        "source": "screener-like",
        "source_identity": "agg:a",
        "source_trust": 70,
        "extracted": {"pat": 12},
    }, path=db)
    left["value"] = 10
    right["value"] = 12
    left["fact_key"] = "pat"
    right["fact_key"] = "pat"
    resolved = resolve_by_authority(left, right, path=db)
    assert resolved["winner"]["evidence_id"] == left["evidence_id"]
    assert len(get_evidence("INFY", as_of="2026-04-20", path=db)) == 2
    record_conflict({
        "symbol": "INFY", "fact_key": "pat",
        "left_evidence_id": left["evidence_id"],
        "right_evidence_id": right["evidence_id"],
        "left_value": 10, "right_value": 12,
        "winner_evidence_id": left["evidence_id"],
        "resolution": "higher_source_trust",
    }, path=db)


def test_harvest_infy_announcement_dates(tmp_path):
    folder = tmp_path / "INFY" / "autonomy"
    folder.mkdir(parents=True)
    folder.joinpath("nse_0.json").write_text(json.dumps([
        {
            "sort_date": "2026-08-01 15:34:10",
            "attchmntText": "Infosys Limited financial results",
            "seq_id": "1",
            "symbol": "INFY",
        },
        {
            "sort_date": "2026-06-20 10:00:00",
            "attchmntText": "Investor conference",
            "seq_id": "2",
            "symbol": "INFY",
        },
    ]), encoding="utf-8")
    folder.joinpath("nse_1.json").write_text(json.dumps([
        {
            "filingDate": "-",
            "fromDate": "01-Apr-2026",
            "toDate": "30-Jun-2026",
            "seqNumber": "x",
            "symbol": "INFY",
        },
        {
            "filingDate": "12-Oct-2012 09:25",
            "fromDate": "01-Apr-2012",
            "toDate": "30-Sep-2012",
            "seqNumber": "99728",
            "symbol": "INFY",
        },
    ]), encoding="utf-8")
    db = tmp_path / "pit.db"
    harvest_symbol("INFY", folder=folder, warehouse_path=db)
    before = get_evidence("INFY", as_of="2026-06-10", path=db)
    after = get_evidence("INFY", as_of="2026-08-01", path=db)
    assert all(r["available_from"] <= "2026-06-10" for r in before)
    assert any(r["available_from"] == "2026-08-01" for r in after)
    assert not any(r["available_from"] == "2026-08-01" for r in before)
    unverified = [r for r in get_evidence_raw("INFY", path=db) if r["pit_status"] == PIT_UNVERIFIED]
    assert unverified
    snap = get_financial_snapshot("INFY", as_of="2026-08-01", path=db)
    assert snap["available"] is True
    assert snap["numbers_parsed"] is False


def test_screener_is_not_in_pit_inputs(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": "CORPORATE_ANNOUNCEMENT",
        "publication_date": "2026-06-01",
        "available_from": "2026-06-01",
        "source": "NSE corporate announcements",
        "source_identity": "nse_ann:x",
        "extracted": {"headline": "Update"},
    }, path=db)
    inputs = pit_research_inputs("INFY", as_of="2026-06-12", path=db)
    assert inputs["raw_fundamentals"]["data"] == {}
    assert inputs["raw_fundamentals"]["point_in_time"] is True
    assert inputs["scan_payload"]["records"] == []


def test_pit_strong_is_not_two_pdfs(tmp_path):
    db = tmp_path / "pit.db"
    for i in range(3):
        persist({
            "symbol": "INFY",
            "evidence_type": "CORPORATE_ANNOUNCEMENT",
            "publication_date": f"2026-06-0{i+1}",
            "available_from": f"2026-06-0{i+1}",
            "source": "NSE corporate announcements",
            "source_identity": f"nse_ann:{i}",
        }, path=db)
    grade = overall_replay_grade("INFY", as_of="2026-06-12", market_bars_ok=True, path=db)
    assert grade["grade"] == PIT_PARTIAL
    assert grade["comparable_to_forward"] is False
    assert grade["production_comparable"] is False


def test_sector_context_is_not_a_family_confirm():
    ctx = get_sector_context("INFY", as_of="2026-06-12")
    assert ctx["usable_as_family_confirm"] is False
    assert ctx["classification_versioned"] is False
    assert ctx["status"] in {"UNVERIFIED", "SECTOR_MEMBERSHIP_APPROXIMATE", "UNAVAILABLE"}


def test_research_snapshot_keeps_unknown(tmp_path):
    db = tmp_path / "pit.db"
    snap = get_research_snapshot("TCS", as_of="2026-06-12", path=db)
    assert snap["quality_label"] == "Unmeasured"
    assert "business_quality_score" in snap["unknown"]


def test_versions_are_persisted():
    v = current_versions().as_dict()
    assert v["committee_version"]
    assert v["pit_contract_version"]
    assert v["warehouse_schema_version"]


def test_category_coverage_does_not_collapse(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": "CORPORATE_ANNOUNCEMENT",
        "publication_date": "2026-06-01",
        "available_from": "2026-06-01",
        "source": "NSE corporate announcements",
        "source_identity": "nse_ann:cov",
    }, path=db)
    cov = category_coverage("INFY", as_of="2026-06-12", market_bars_ok=True, path=db)
    assert cov["categories"]["MARKET_DATA"] == "STRONG"
    assert cov["categories"]["FINANCIALS"] == "UNAVAILABLE"
    assert cov["categories"]["ANNOUNCEMENTS"] in {"PARTIAL", "STRONG"}
    assert "FINANCIALS" in cov["missing"]


def test_ingest_does_not_guess_annual_report_date(tmp_path):
    from product.pit_ingest import ingest_annual_report_index

    db = tmp_path / "pit.db"
    ingest_annual_report_index("INFY", [{"fileName": "INFY_AR_2025.pdf"}], warehouse_path=db)
    assert get_evidence("INFY", as_of="2026-09-01", path=db) == []
    raw = get_evidence_raw("INFY", path=db)
    assert raw[0]["reason_code"] == "PUBLICATION_DATE_UNKNOWN"


def test_committee_as_of_does_not_call_defaults(monkeypatch, tmp_path):
    from product.decision_committee import evaluate_committee
    from product.paper_autopilot import ENTER_NOW

    called = {"defaults": 0}

    def boom(*_a, **_k):
        called["defaults"] += 1
        raise AssertionError("live defaults leaked into PIT committee")

    monkeypatch.setattr("product.due_diligence.engine._defaults", boom)

    class Paper:
        decision = ENTER_NOW
        reason_code = "ELIGIBLE"
        detail = ""

    monkeypatch.setattr("product.decision_committee.evaluate_candidate", lambda *a, **k: Paper())
    rec = evaluate_committee(
        {
            "symbol": "INFY",
            "reco_tier": "watch",
            "entry_state": "ready",
            "entry": 100, "stop": 90, "target": 120,
            "methods": [{"id": "sepa", "status": "pass"}],
        },
        load_research=True,
        as_of="2026-06-12",
    )
    assert called["defaults"] == 0
    assert rec.references.get("pit_as_of") == "2026-06-12"
    assert rec.decision != "BUY" or rec.effective_confirmation_count >= 2
