"""Tests for PIT fundamentals/events foundation (network-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.pit_events import get_events, validate_rows, write_events
from data.pit_fundamentals import get_fundamentals, validate_rows as validate_fund
from data.nse_results_ingest import parse_xbrl_metrics, results_to_event_rows
from research.intelligence import data_state as DS
from research.intelligence.data.pit_contract import (
    DOMAIN_EVENTS,
    DOMAIN_FUNDAMENTALS,
    PitContract,
)
from research.intelligence.data.snapshot_store import SnapshotStore


def test_event_rows_require_available_at():
    rows = validate_rows([
        {"symbol": "AAA", "available_at": "2024-01-15", "event_type": "EARNINGS_RESULT"},
        {"symbol": "BBB", "fetched_at": "2024-01-15", "event_type": "EARNINGS_RESULT"},  # no available_at
    ])
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["available_at"] == "2024-01-15"


def test_fundamentals_refuse_fetched_at_only():
    rows = validate_fund([
        {"symbol": "AAA", "fetched_at": "2024-01-15", "basic_eps": 1.2},
        {"symbol": "AAA", "available_at": "2024-01-10", "basic_eps": 1.1, "profit_after_tax": 100},
    ])
    assert len(rows) == 1
    assert rows[0]["available_at"] == "2024-01-10"
    assert rows[0]["basic_eps"] == 1.1


def test_results_to_event_rows_uses_broadcast():
    raw = [{
        "symbol": "RELIANCE",
        "isin": "INE002A01018",
        "broadCastDate": "16-Jan-2025 20:20:21",
        "exchdisstime": "16-Jan-2025 20:20:54",
        "period": "Quarterly",
        "fromDate": "01-Oct-2024",
        "toDate": "31-Dec-2024",
        "consolidated": "Consolidated",
        "seqNumber": "1",
        "xbrl": "https://example.invalid/x.xml",
    }]
    ev = results_to_event_rows(raw)
    assert len(ev) == 1
    assert ev[0]["event_type"] == "EARNINGS_RESULT"
    assert ev[0]["available_at"].startswith("2025-01-16")


def test_parse_xbrl_metrics_minimal():
    xml = b"""<?xml version='1.0'?>
    <xbrl xmlns='http://www.xbrl.org/2003/instance'
          xmlns:in='http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin'>
      <in:RevenueFromOperations contextRef='c1' unitRef='u'>100.5</in:RevenueFromOperations>
      <in:BasicEarningsLossPerShareFromContinuingOperations contextRef='c1' unitRef='u'>2.5</in:BasicEarningsLossPerShareFromContinuingOperations>
    </xbrl>
    """
    m = parse_xbrl_metrics(xml)
    assert m["revenue_from_operations"] == 100.5
    assert m["basic_eps"] == 2.5


def test_xbrl_prefers_oned_over_fourd_ytd():
    """FourD often reuses the quarter dates for a YTD total. Do not take it."""
    xml = b"""<?xml version='1.0'?>
    <xbrl xmlns='http://www.xbrl.org/2003/instance'
          xmlns:in='http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin'>
      <in:RevenueFromOperations contextRef='FourD' unitRef='u'>900</in:RevenueFromOperations>
      <in:RevenueFromOperations contextRef='OneD' unitRef='u'>300</in:RevenueFromOperations>
      <in:ProfitBeforeTax contextRef='FourD' unitRef='u'>90</in:ProfitBeforeTax>
      <in:ProfitBeforeTax contextRef='OneD' unitRef='u'>30</in:ProfitBeforeTax>
    </xbrl>
    """
    m = parse_xbrl_metrics(xml)
    assert m["revenue_from_operations"] == 300.0
    assert m["profit_before_tax"] == 30.0


def test_pit_contract_fundamentals_and_events(tmp_path):
    # Minimal OHLCV snapshot
    store = SnapshotStore(tmp_path / "snaps")
    sid = store.commit_snapshot([
        ("AAA", "2024-01-01", 10, 11, 9, 10, 1000, "EQ"),
        ("AAA", "2024-01-02", 10, 11, 9, 10.5, 1000, "EQ"),
        ("AAA", "2024-01-03", 10, 11, 9, 11, 1000, "EQ"),
    ])
    ev_path = tmp_path / "events.json"
    write_events([
        {"symbol": "AAA", "available_at": "2024-01-01", "event_type": "EARNINGS_RESULT", "headline": "Q"},
        {"symbol": "AAA", "available_at": "2024-01-03", "event_type": "EARNINGS_RESULT", "headline": "Q2"},
    ], path=ev_path, source="operator")
    fund_path = tmp_path / "fund.json"
    fund_path.write_text(json.dumps({
        "schema_version": 1,
        "source": "operator",
        "rows": [
            {"symbol": "AAA", "available_at": "2024-01-01", "basic_eps": 1.0, "profit_after_tax": 10},
            {"symbol": "AAA", "available_at": "2024-01-03", "basic_eps": 2.0, "profit_after_tax": 20},
        ],
    }), encoding="utf-8")

    pit = PitContract.from_store(
        store, sid, events_path=ev_path, fundamentals_path=fund_path,
    )
    early_e = pit.as_of(DOMAIN_EVENTS, when="2024-01-02", symbol="AAA")
    assert early_e.status == DS.READY
    assert len(early_e.data) == 1
    assert early_e.data[0]["available_at"] == "2024-01-01"

    early_f = pit.as_of(DOMAIN_FUNDAMENTALS, when="2024-01-02", symbol="AAA")
    assert early_f.status == DS.READY
    assert early_f.data["basic_eps"] == 1.0

    late_f = pit.as_of(DOMAIN_FUNDAMENTALS, when="2024-01-03", symbol="AAA")
    assert late_f.data["basic_eps"] == 2.0

    # No ledger → still NOT_PIT_SAFE (operational cache wall)
    pit2 = PitContract.from_store(store, sid, fundamentals_path=tmp_path / "missing.json")
    assert pit2.as_of(DOMAIN_FUNDAMENTALS, when="2024-01-02", symbol="AAA").status == DS.NOT_PIT_SAFE


def test_get_events_and_fundamentals_helpers(tmp_path):
    p = tmp_path / "e.json"
    write_events([
        {"symbol": "ZZZ", "available_at": "2023-05-01", "event_type": "EARNINGS_RESULT"},
    ], path=p, source="operator")
    assert len(get_events("ZZZ", "2023-04-30", path=p)) == 0
    assert len(get_events("ZZZ", "2023-05-01", path=p)) == 1

    fp = tmp_path / "f.json"
    from data.pit_fundamentals import write_fundamentals
    write_fundamentals([
        {"symbol": "ZZZ", "available_at": "2023-05-01", "basic_eps": 3.3},
    ], path=fp, source="operator")
    assert get_fundamentals("ZZZ", "2023-04-30", path=fp) is None
    assert get_fundamentals("ZZZ", "2023-05-01", path=fp)["basic_eps"] == 3.3
