"""Phase A — official XBRL facts, PIT cutoff, no Screener leak."""
from __future__ import annotations

from product.pit_availability import PIT_STRONG, PIT_PARTIAL, PIT_UNVERIFIED
from product.pit_coverage import overall_replay_grade
from product.pit_financials import get_fact, get_financial_snapshot_v2
from product.pit_query import pit_research_inputs
from product.pit_warehouse import DOC_QUARTERLY_RESULT, get_evidence, persist
from product.pit_xbrl import parse_xbrl
from product.pit_parser_qa import validate_xbrl_text
from product.pit_backfill import _persist_parsed_xbrl

INFY_XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns:in-bse-fin="http://www.bseindia.com/xbrl/fin">
  <context id="OneD"><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></context>
  <context id="OneI"><instant>2026-06-30</instant></context>
  <LevelOfRounding>Crores</LevelOfRounding>
  <NatureOfReportStandaloneConsolidated>Consolidated</NatureOfReportStandaloneConsolidated>
  <DateOfBoardMeetingWhenFinancialResultsWereApproved>2026-07-23</DateOfBoardMeetingWhenFinancialResultsWereApproved>
  <DateOfStartOfReportingPeriod>2026-04-01</DateOfStartOfReportingPeriod>
  <DateOfEndOfReportingPeriod>2026-06-30</DateOfEndOfReportingPeriod>
  <RevenueFromOperations contextRef="OneD" unitRef="INR" decimals="-7">399570000000</RevenueFromOperations>
  <ProfitBeforeTax contextRef="OneD" unitRef="INR" decimals="-7">97840000000</ProfitBeforeTax>
  <ProfitLossForPeriod contextRef="OneD" unitRef="INR" decimals="-7">72490000000</ProfitLossForPeriod>
  <FinanceCosts contextRef="OneD" unitRef="INR" decimals="-7">1200000000</FinanceCosts>
  <BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" unitRef="INRPerShare" decimals="2">17.87</BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
</xbrl>
"""


BANK_XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrl>
  <context id="OneD"><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></context>
  <LevelOfRounding>Crores</LevelOfRounding>
  <InterestEarned contextRef="OneD" unitRef="INR" decimals="-5">793627800000</InterestEarned>
  <ProfitLossForThePeriod contextRef="OneD" unitRef="INR" decimals="-5">190597200000</ProfitLossForThePeriod>
  <ProfitLossFromOrdinaryActivitiesBeforeTax contextRef="OneD" unitRef="INR" decimals="-6">251083000000</ProfitLossFromOrdinaryActivitiesBeforeTax>
  <PercentageOfGrossNpa contextRef="OneD" unitRef="pure" decimals="INF">0.0117</PercentageOfGrossNpa>
</xbrl>
"""


def test_bank_xbrl_maps_interest_earned_and_does_not_scale_npa():
    parsed = parse_xbrl(BANK_XBRL)
    assert parsed["ok"] is True
    assert parsed["facts"]["revenue"] == 79362.78
    assert parsed["facts"]["pat"] == 19059.72
    assert parsed["facts"]["gnpa_pct"] == 0.0117


def test_parse_infy_q1_fy27_matches_official_xbrl():
    parsed = parse_xbrl(INFY_XBRL)
    assert parsed["ok"] is True
    assert parsed["facts"]["revenue"] == 39957.0
    assert parsed["facts"]["pat"] == 7249.0
    assert parsed["facts"]["basic_eps"] == 17.87
    qa = validate_xbrl_text(INFY_XBRL, {"revenue": 39957, "pat": 7249, "basic_eps": 17.87}, symbol="INFY")
    assert qa["n_matched"] == 3
    assert qa["parsing_success_is_not_correctness"] is True


def test_q1_fy27_invisible_before_broadcast(tmp_path):
    db = tmp_path / "pit.db"
    _persist_parsed_xbrl(
        "INFY",
        xml_text=INFY_XBRL,
        publication="2026-07-23",
        period_end="2026-06-30",
        source_url="https://nsearchives.nseindia.com/example.xml",
        source_identity="nse_xbrl:test:infy-q1",
        warehouse_path=db,
    )
    early = get_evidence("INFY", as_of="2026-06-10", path=db)
    late = get_evidence("INFY", as_of="2026-08-01", path=db)
    assert early == []
    assert len(late) == 1
    fact_early = get_fact("INFY", "revenue", as_of="2026-06-10", path=db)
    fact_late = get_fact("INFY", "revenue", as_of="2026-08-01", path=db)
    assert fact_early["available"] is False
    assert fact_late["value"] == 39957.0
    snap = get_financial_snapshot_v2("INFY", as_of="2026-08-01", path=db)
    assert snap["numbers_parsed"] is True
    assert snap["derived"]["pat_margin_pct"] == 18.14


def test_period_end_alone_still_unverified(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "period_end": "2026-06-30",
        "publication_date": "",
        "available_from": "",
        "source": "filename-guess",
        "source_identity": "guess",
        "extracted": {"facts": {"revenue": 1}, "numbers_parsed": True},
    }, path=db)
    assert get_evidence("INFY", as_of="2026-09-01", path=db) == []
    from product.pit_warehouse import get_evidence_raw
    assert get_evidence_raw("INFY", path=db)[0]["pit_status"] == PIT_UNVERIFIED


def test_revisions_do_not_overwrite(tmp_path):
    db = tmp_path / "pit.db"
    first = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-23",
        "available_from": "2026-04-23",
        "period_end": "2026-03-31",
        "source": "NSE XBRL",
        "source_identity": "nse_xbrl:q4",
        "revision": 1,
        "extracted": {"numbers_parsed": True, "facts": {"revenue": 100.0}},
    }, path=db)
    second = persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-24",
        "available_from": "2026-04-24",
        "period_end": "2026-03-31",
        "source": "NSE XBRL",
        "source_identity": "nse_xbrl:q4",
        "revision": 2,
        "extracted": {"numbers_parsed": True, "facts": {"revenue": 99.0}},
    }, path=db)
    assert first["evidence_id"] != second["evidence_id"]
    mid = get_fact("INFY", "revenue", as_of="2026-04-23", path=db)
    late = get_fact("INFY", "revenue", as_of="2026-04-25", path=db)
    assert mid["value"] == 100.0
    assert late["value"] == 99.0


def test_screener_unused_when_xbrl_present(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-23",
        "available_from": "2026-04-23",
        "period_end": "2026-03-31",
        "source": "NSE XBRL",
        "source_identity": "nse_xbrl:q4b",
        "extracted": {
            "numbers_parsed": True,
            "facts": {"revenue": 40986.0, "pat": 8090.0, "pbt": 11000.0},
        },
    }, path=db)
    inputs = pit_research_inputs("INFY", as_of="2026-06-12", path=db)
    assert inputs["raw_fundamentals"]["point_in_time"] is True
    assert inputs["raw_fundamentals"]["source"] == "pit_warehouse_xbrl"
    assert inputs["raw_fundamentals"]["data"]["quarterly_results"]
    grade = overall_replay_grade("INFY", as_of="2026-06-12", market_bars_ok=True, path=db)
    assert grade["production_comparable"] is False
    assert grade["grade"] == PIT_PARTIAL


def test_two_parsed_periods_can_be_production_comparable(tmp_path):
    db = tmp_path / "pit.db"
    persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-01-14",
        "available_from": "2026-01-14",
        "period_end": "2025-12-31",
        "source": "NSE XBRL",
        "source_identity": "nse_xbrl:q3",
        "extracted": {
            "numbers_parsed": True,
            "facts": {"revenue": 37996.0, "pat": 6811.0, "pbt": 9000.0, "finance_costs": 50.0},
        },
    }, path=db)
    persist({
        "symbol": "INFY",
        "evidence_type": DOC_QUARTERLY_RESULT,
        "publication_date": "2026-04-23",
        "available_from": "2026-04-23",
        "period_end": "2026-03-31",
        "source": "NSE XBRL",
        "source_identity": "nse_xbrl:q4b",
        "extracted": {
            "numbers_parsed": True,
            "facts": {"revenue": 40986.0, "pat": 8090.0, "pbt": 11000.0, "finance_costs": 60.0},
        },
    }, path=db)
    inputs = pit_research_inputs("INFY", as_of="2026-06-12", path=db)
    assert inputs["raw_fundamentals"]["point_in_time"] is True
    grade = overall_replay_grade("INFY", as_of="2026-06-12", market_bars_ok=True, path=db)
    assert grade["production_comparable"] is True
    assert grade["grade"] == PIT_STRONG
    assert grade["n_parsed_results"] >= 2
