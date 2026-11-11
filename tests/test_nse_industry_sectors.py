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
