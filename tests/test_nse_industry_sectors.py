"""Sector coverage from NSE's own industry label (audit QT-REC-012).

The static comment-parsed map covered 337 symbols; TCS, HDFCBANK, PNB and
most PSU banks had no sector, so sector packs and the portfolio sector cap
were blind to them.
"""
from __future__ import annotations

import json

import pytest


ROWS = [
    {"symbol": "NIFTY 500", "meta": {}},                      # index row, no industry
    {"symbol": "PNB", "meta": {"industry": "BANKS"}},
    {"symbol": "TCS", "meta": {"industry": "COMPUTERS - SOFTWARE"}},
    {"symbol": "BIKAJI", "meta": {"industry": "  FOOD PRODUCTS "}},
    {"symbol": "ODD", "meta": {"industry": "-"}},
    "not-a-row",
]


def test_industries_extracted_from_index_rows():
    from data.nse_industry import industries_from_index_rows

    assert industries_from_index_rows(ROWS) == {
        "PNB": "BANKS", "TCS": "COMPUTERS - SOFTWARE", "BIKAJI": "FOOD PRODUCTS"}


@pytest.mark.parametrize("label,group", [
    ("BANKS", "Banking & Finance"),
    ("HOUSING FINANCE", "Housing Finance"),
    ("FINANCE", "NBFC"),
    ("COMPUTERS - SOFTWARE", "IT / Software"),
    ("PHARMACEUTICALS", "Pharma & Healthcare"),
    ("TYRES", "Tyres"),
    ("AUTOMOBILES - 4 WHEELERS", "Auto & Auto Ancillary"),
    ("STEEL AND STEEL PRODUCTS", "Metals & Mining"),
    ("REFINERIES", "Energy & Power"),
    ("EDIBLE OIL", "FMCG / Consumer Staples"),
    ("SHIPPING", "Logistics & Transport"),
    ("SOMETHING NEW", "Something New"),
    ("", ""),
])
def test_nse_labels_map_onto_static_groups(label, group):
    from data.nse_industry import normalise_industry

    assert normalise_industry(label) == group


def test_record_is_merge_only_and_reports_changes(tmp_path):
    from data.nse_industry import industry_of, record_industries

    path = tmp_path / "nse_industry_map.json"
    assert record_industries({"PNB": "BANKS"}, path=path) == 1
    assert record_industries({"PNB": "BANKS"}, path=path) == 0          # unchanged
    assert record_industries({"TCS": "COMPUTERS - SOFTWARE"}, path=path) == 1
    stored = json.loads(path.read_text())["industries"]
    assert stored == {"PNB": "BANKS", "TCS": "COMPUTERS - SOFTWARE"}
    assert industry_of("pnb", path=path) == "BANKS"
    assert industry_of("MISSING", path=path) == ""


def test_record_never_raises_on_unwritable_target(tmp_path):
    from data.nse_industry import record_industries

    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert record_industries({"PNB": "BANKS"}, path=blocker / "map.json") == 0


def test_sector_of_falls_back_to_official_industry(tmp_path, monkeypatch):
    import data.nse_industry as ni
    from scan.sector_heat import sector_of

    path = tmp_path / "nse_industry_map.json"
    ni.record_industries({"PNB": "BANKS", "ZZNEWCO": "PAINTS"}, path=path)
    monkeypatch.setattr(ni, "map_path", lambda: path)
    assert sector_of("PNB") == "Banking & Finance"
    assert sector_of("ZZNEWCO") == "Paints"
    # The static comment map still wins where it has an answer.
    assert sector_of("EMAMILTD") == "FMCG / Consumer Staples"
    assert sector_of("NOPE") == ""


def test_weekly_refresh_records_and_then_stays_fresh(tmp_path):
    from data.nse_industry import industry_of, refresh_industry_map

    path = tmp_path / "nse_industry_map.json"
    fetched = []

    def fetch():
        fetched.append(1)
        return [ROWS]

    first = refresh_industry_map(path=path, fetch=fetch)
    assert first == {"symbols": 3, "changed": 3}
    assert industry_of("TCS", path=path) == "COMPUTERS - SOFTWARE"
    assert refresh_industry_map(path=path, fetch=fetch)["skipped"] == "fresh"
    assert fetched == [1]
    assert refresh_industry_map(path=path, fetch=fetch, force=True)["changed"] == 0
    assert fetched == [1, 1]


def test_refresh_without_network_or_rows_is_a_no_op(tmp_path, monkeypatch):
    from data.nse_industry import refresh_industry_map

    path = tmp_path / "nse_industry_map.json"
    assert refresh_industry_map(path=path)["skipped"] == "network_disabled"   # under pytest
    assert refresh_industry_map(path=path, fetch=lambda: [[]])["skipped"] == "no_rows"
    assert not path.exists()


CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Punjab National Bank,Financial Services,PNB,EQ,INE160A01022
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
UltraTech Cement Ltd.,Construction Materials,ULTRACEMCO,EQ,INE481G01011
,,,,
"""


def test_constituent_csv_rows_feed_the_same_pipeline():
    from data.nse_industry import industries_from_index_rows, normalise_industry, rows_from_constituent_csv

    found = industries_from_index_rows(rows_from_constituent_csv(CSV))
    assert found == {"PNB": "Financial Services", "TCS": "Information Technology",
                     "RELIANCE": "Oil Gas & Consumable Fuels", "ULTRACEMCO": "Construction Materials"}
    assert normalise_industry("Financial Services") == "Banking & Finance"
    assert normalise_industry("Oil Gas & Consumable Fuels") == "Energy & Power"
    assert normalise_industry("Information Technology") == "IT / Software"
    assert normalise_industry("Construction Materials") == "Cement"


def test_csv_fallback_used_when_api_returns_nothing(monkeypatch):
    import data.nse_industry as ni

    monkeypatch.setattr(ni, "_fetch_api_rows", lambda: [])
    monkeypatch.setattr(ni, "_fetch_constituent_rows", lambda: [ni.rows_from_constituent_csv(CSV)])
    assert len(ni._fetch_index_rows()[0]) == 4
