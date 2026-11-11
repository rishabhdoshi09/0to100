"""Official-fundamentals truth: units, ROE/ROCE math, YoY, re-parse, merge.

Regression for the 2026-11-11 audit of the production warehouse:
- "Lakhs"-rounded XBRL filings were stored 100x too large (v1 parser);
- ROE fell back to paid-up capital (TCS 3707%, SBIN 2747%);
- "YoY" compared against the sequential quarter;
- the official profit_loss table fed quarterly values into a "3y CAGR";
- a stale cached ROE survived when the new derivation had none.
"""
from __future__ import annotations

from pathlib import Path

import pytest

LAKH_XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:in-capmkt="http://example/in-capmkt">
  <xbrli:context id="OneD"><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="OneI"><xbrli:period><xbrli:instant>2026-06-30</xbrli:instant></xbrli:period></xbrli:context>
  <in-capmkt:LevelOfRounding contextRef="OneD">Lakhs</in-capmkt:LevelOfRounding>
  <in-capmkt:ReportingQuarter contextRef="OneD">Quarterly</in-capmkt:ReportingQuarter>
  <in-capmkt:RevenueFromOperations contextRef="OneD" decimals="-3" unitRef="INR">610265000</in-capmkt:RevenueFromOperations>
  <in-capmkt:ProfitBeforeTax contextRef="OneD" decimals="-3" unitRef="INR">120000000</in-capmkt:ProfitBeforeTax>
  <in-capmkt:ProfitLossForPeriod contextRef="OneD" decimals="-3" unitRef="INR">90000000</in-capmkt:ProfitLossForPeriod>
  <in-capmkt:FinanceCosts contextRef="OneD" decimals="-3" unitRef="INR">10000000</in-capmkt:FinanceCosts>
  <in-capmkt:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" decimals="2" unitRef="INRPerShare">9.0</in-capmkt:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
  <in-capmkt:SomeUnmappedRatio contextRef="OneD" decimals="INF" unitRef="pure">0.015</in-capmkt:SomeUnmappedRatio>
  <in-capmkt:Equity contextRef="OneI" decimals="-3" unitRef="INR">1752464000</in-capmkt:Equity>
  <in-capmkt:BorrowingsNoncurrent contextRef="OneI" decimals="-3" unitRef="INR">2159000</in-capmkt:BorrowingsNoncurrent>
  <in-capmkt:BorrowingsCurrent contextRef="OneI" decimals="-3" unitRef="INR">1712000</in-capmkt:BorrowingsCurrent>
</xbrli:xbrl>
"""


# ---------------------------------------------------------------- parser
def test_lakh_rounded_filing_is_stored_in_crore():
    from product.pit_xbrl import PARSER_VERSION, parse_xbrl

    parsed = parse_xbrl(LAKH_XBRL)
    assert PARSER_VERSION == "pit_xbrl.v2"
    # 610,265,000 INR is Rs 61.03 cr whatever the presentation rounding says.
    assert parsed["facts"]["revenue"] == pytest.approx(61.0265)
    assert parsed["facts"]["basic_eps"] == 9.0          # per-share untouched


def test_unmapped_tags_never_become_a_null_fact():
    from product.pit_xbrl import parse_xbrl

    facts = parse_xbrl(LAKH_XBRL)["facts"]
    assert "null" not in facts and None not in facts and "None" not in facts


def test_balance_sheet_borrowings_become_total_debt():
    from product.pit_xbrl import parse_xbrl

    facts = parse_xbrl(LAKH_XBRL)["facts"]
    assert facts["bs_equity"] == pytest.approx(175.2464)
    assert facts["bs_total_debt"] == pytest.approx(0.3871)


# ------------------------------------------------------- legacy normalise
def test_v1_lakh_rows_are_divided_by_100_on_read():
    from product.pit_financials import normalise_legacy_units

    out = normalise_legacy_units(
        {"revenue": 6102.65, "basic_eps": 9.0, "gnpa_pct": 0.012, "null": 0.015},
        parser_version="pit_xbrl.v1", rounding="Lakhs")
    assert out["revenue"] == pytest.approx(61.0265)
    assert out["basic_eps"] == 9.0
    assert out["gnpa_pct"] == 0.012
    assert "null" not in out


def test_crore_and_v2_rows_are_untouched():
    from product.pit_financials import normalise_legacy_units

    assert normalise_legacy_units({"revenue": 61.0}, parser_version="pit_xbrl.v1",
                                  rounding="Crores")["revenue"] == 61.0
    assert normalise_legacy_units({"revenue": 61.0}, parser_version="pit_xbrl.v2",
                                  rounding="Lakhs")["revenue"] == 61.0


def test_bank_placeholder_zero_npa_is_unknown_not_perfect():
    from product.pit_financials import normalise_legacy_units

    out = normalise_legacy_units({"gnpa_pct": 0.0, "roa": 0.0, "interest_earned": 10.0})
    assert "gnpa_pct" not in out and "roa" not in out


# ------------------------------------------------------------ derivations
def _row(end: str, facts: dict, *, months: int = 3, eid: str = "") -> dict:
    return {"period_end": end, "facts": facts, "period_months": months,
            "evidence_id": eid or end, "available_from": end}


def _quarter(end, revenue, pat, pbt=None, fin=1.0, eps=1.0, **bs):
    facts = {"revenue": revenue, "pat": pat, "pbt": pbt if pbt is not None else pat * 1.3,
             "finance_costs": fin, "basic_eps": eps, "paid_up_equity": 10.0, "face_value": 1.0}
    facts.update(bs)
    return _row(end, facts)


def test_roe_never_uses_paid_up_capital():
    from product.pit_financials import _derive

    latest = _quarter("2026-06-30", 100.0, 20.0)       # paid-up 10, no net worth
    derived = _derive(latest, None, rows=[latest])
    assert "roe_approx_pct" not in derived              # unknown, not 800%


def test_roe_uses_ttm_profit_over_net_worth():
    from product.pit_financials import _derive

    rows = [
        _quarter("2026-06-30", 100.0, 25.0, bs_equity=400.0),
        _quarter("2026-03-31", 100.0, 25.0),
        _quarter("2025-12-31", 100.0, 25.0),
        _quarter("2025-09-30", 100.0, 25.0),
    ]
    derived = _derive(rows[0], rows[1], rows=rows)
    assert derived["roe_basis"] == "TTM"
    assert derived["roe_approx_pct"] == pytest.approx(25.0)   # 100 / 400
    assert derived["ttm_eps"] == pytest.approx(4.0)


def test_single_quarter_roe_is_annualised():
    from product.pit_financials import _derive

    latest = _quarter("2026-06-30", 100.0, 10.0, bs_equity=400.0)
    derived = _derive(latest, None, rows=[latest])
    assert derived["roe_basis"] == "ANNUALISED_3M"
    assert derived["roe_approx_pct"] == pytest.approx(10.0)   # 10*4 / 400


def test_yoy_is_same_quarter_last_year_not_sequential():
    from product.pit_financials import _derive

    rows = [
        _quarter("2026-06-30", 120.0, 12.0),
        _quarter("2026-03-31", 200.0, 30.0),   # sequential quarter
        _quarter("2025-06-30", 100.0, 10.0),   # same quarter last year
    ]
    derived = _derive(rows[0], rows[1], rows=rows)
    assert derived["revenue_yoy_pct"] == pytest.approx(20.0)
    assert derived["pat_yoy_pct"] == pytest.approx(20.0)
    assert derived["revenue_qoq_pct"] == pytest.approx(-40.0)


def test_no_yoy_claim_without_last_years_quarter():
    from product.pit_financials import _derive

    rows = [_quarter("2026-06-30", 120.0, 12.0), _quarter("2026-03-31", 200.0, 30.0)]
    derived = _derive(rows[0], rows[1], rows=rows)
    assert "revenue_yoy_pct" not in derived


def test_growth_from_a_loss_base_is_undefined():
    from product.pit_financials import _derive

    rows = [_quarter("2026-06-30", 120.0, 12.0), _quarter("2025-06-30", 100.0, -5.0)]
    assert "pat_yoy_pct" not in _derive(rows[0], None, rows=rows)


def test_bank_gets_no_ebit_or_roce():
    from product.pit_financials import _derive

    latest = _row("2026-06-30", {"interest_earned": 100.0, "revenue": 100.0, "pat": 20.0,
                                 "pbt": 26.0, "finance_costs": 60.0, "bs_equity": 400.0,
                                 "bs_total_debt": 50.0})
    derived = _derive(latest, None, rows=[latest])
    assert derived["is_financial"] is True
    assert "roce_approx_pct" not in derived and "ebit_cr" not in derived
    assert "interest_coverage" not in derived


def test_roce_needs_capital_employed():
    from product.pit_financials import _derive

    no_debt = _quarter("2026-06-30", 100.0, 10.0, pbt=14.0, fin=1.0, bs_equity=100.0)
    assert "roce_approx_pct" not in _derive(no_debt, None, rows=[no_debt])
    with_debt = _quarter("2026-06-30", 100.0, 10.0, pbt=14.0, fin=1.0,
                         bs_equity=100.0, bs_total_debt=50.0)
    derived = _derive(with_debt, None, rows=[with_debt])
    # EBIT 15/qtr -> 60/yr over capital employed 150
    assert derived["roce_approx_pct"] == pytest.approx(40.0)
    assert derived["debt_to_equity"] == pytest.approx(0.5)
    assert derived["interest_coverage"] == pytest.approx(15.0)


def test_annual_profit_loss_only_sums_complete_fiscal_years():
    from product.pit_financials import screener_tables

    rows = [
        _quarter("2026-06-30", 130.0, 13.0),
        _quarter("2026-03-31", 120.0, 12.0),
        _quarter("2025-12-31", 110.0, 11.0),
        _quarter("2025-09-30", 100.0, 10.0),
        _quarter("2025-06-30", 90.0, 9.0),
    ]
    sales = screener_tables(rows)["profit_loss"][0]
    assert sales == {"row_label": "Sales+", "Mar 2026": pytest.approx(420.0)}


# ------------------------------------------------- warehouse + re-parse
def _persist_v1(db: Path, *, eid_identity="nse_xbrl:integrated:1001", revenue=6102.65):
    from product.pit_warehouse import DOC_QUARTERLY_RESULT, persist

    return persist({
        "symbol": "MACPOWER", "evidence_type": DOC_QUARTERLY_RESULT,
        "period_start": "2026-04-01", "period_end": "2026-06-30",
        "publication_date": "2026-07-29", "filing_date": "2026-07-29",
        "available_from": "2026-07-29", "source": "NSE XBRL",
        "source_url": "https://nsearchives.nseindia.com/x.xml",
        "source_identity": eid_identity, "parser_version": "pit_xbrl.v1",
        "extracted": {"numbers_parsed": True, "parser_version": "pit_xbrl.v1",
                      "rounding": "Lakhs", "reporting_period": "Quarterly",
                      "nature": "Standalone",
                      "facts": {"revenue": revenue, "pat": 900.0, "paid_up_equity": 1000.42,
                                "face_value": 10.0, "null": 0.0}},
    }, path=db)


def test_parsed_results_normalise_v1_and_skip_superseded(tmp_path):
    from product.pit_financials import parsed_results
    from product.pit_warehouse import DOC_QUARTERLY_RESULT, persist

    db = tmp_path / "pit.db"
    old = _persist_v1(db)
    rows = parsed_results("MACPOWER", as_of="2026-11-10", path=db)
    assert rows[0]["facts"]["revenue"] == pytest.approx(61.0265)
    persist({**{k: old[k] for k in ("symbol", "evidence_type", "period_start", "period_end",
                                    "publication_date", "filing_date", "available_from",
                                    "source", "source_url", "source_identity")},
             "parser_version": "pit_xbrl.v2", "revision": 2, "supersedes": old["evidence_id"],
             "extracted": {"numbers_parsed": True, "parser_version": "pit_xbrl.v2",
                           "rounding": "Lakhs", "facts": {"revenue": 61.5, "pat": 9.0}}},
            path=db)
    rows = parsed_results("MACPOWER", as_of="2026-11-10", path=db)
    assert [r["facts"]["revenue"] for r in rows] == [61.5]


def test_reparse_writes_superseding_revision_once(tmp_path):
    from product.pit_financials import parsed_results
    from product.pit_reparse import reparse_legacy_xbrl

    db = tmp_path / "pit.db"
    _persist_v1(db)
    root = tmp_path / "research_evidence"
    (root / "MACPOWER" / "autonomy").mkdir(parents=True)
    (root / "MACPOWER" / "autonomy" / "nse_xbrl_integrated_1001.xml").write_text(LAKH_XBRL)

    first = reparse_legacy_xbrl(warehouse_path=db, evidence_root=root)
    assert first["reparsed"] == 1 and first["borrowings_recovered"] == 1
    rows = parsed_results("MACPOWER", as_of="2026-11-10", path=db)
    assert len(rows) == 1
    assert rows[0]["facts"]["revenue"] == pytest.approx(61.0265)
    assert rows[0]["facts"]["bs_total_debt"] == pytest.approx(0.3871)
    assert rows[0]["available_from"] == "2026-07-29"         # availability unchanged

    second = reparse_legacy_xbrl(warehouse_path=db, evidence_root=root)
    assert second["candidates"] == 0 and second["reparsed"] == 0


def test_reparse_dry_run_writes_nothing(tmp_path):
    from product.pit_reparse import reparse_legacy_xbrl

    db = tmp_path / "pit.db"
    _persist_v1(db)
    root = tmp_path / "research_evidence"
    (root / "MACPOWER").mkdir(parents=True)
    (root / "MACPOWER" / "nse_xbrl_integrated_1001.xml").write_text(LAKH_XBRL)
    report = reparse_legacy_xbrl(warehouse_path=db, evidence_root=root, dry_run=True)
    assert report["reparsed"] == 1
    assert reparse_legacy_xbrl(warehouse_path=db, evidence_root=root, dry_run=True)["candidates"] == 1


# ------------------------------------------------------- fetcher / scorer
def test_stale_cached_official_ratio_is_dropped_on_merge():
    from fundamentals.fetcher import _merge_official

    cached = {"roe": 2747.45, "key_ratios": [{"name": "ROE", "value": "17"}]}
    merged = _merge_official(cached, {"source_tier": "official", "quarterly_results": [{"x": 1}],
                                      "is_financial": True})
    assert "roe" not in merged
    assert merged["key_ratios"]                       # secondary context kept


def test_official_ratios_win_over_cached_key_ratios():
    from screener.engine import _extract_fundamentals

    data = {"official": True, "source_tier": "official", "roe": 25.0,
            "key_ratios": [{"name": "ROE", "value": "3707"}],
            "sales_growth_yoy": 12.5, "ttm_eps": 15.35, "is_financial": True}
    out = _extract_fundamentals(data)
    assert out["roe"] == 25.0
    assert out["sales_growth_yoy"] == 12.5 and out["ttm_eps"] == 15.35
    assert out["is_financial"] is True


def test_pe_is_recomputed_from_price_and_official_ttm_eps():
    from scan.long_term_service import _with_current_pe

    out = _with_current_pe({"pe": 80.0, "ttm_eps": 15.35}, 307.0)
    assert out["pe"] == pytest.approx(20.0)
    assert "pe" not in _with_current_pe({"pe": 80.0, "ttm_eps": -2.0}, 307.0)
    assert _with_current_pe({"pe": 80.0}, 307.0)["pe"] == 80.0   # no EPS: unchanged


def test_filing_flags_bank_even_without_sector_name():
    from scan.long_term_service import score_current_fundamentals

    fund = {"is_financial": True, "roe": 16.0, "roce": 1.0, "debt_to_equity": 9.0}
    scored = score_current_fundamentals(fund, sector="—")
    assert not any("debt/equity" in r for r in scored["risks"])
    assert not any("ROCE" in r for r in scored["risks"])


def test_yoy_growth_only_fills_a_missing_3y_cagr_at_lower_weight():
    from scan.long_term_service import score_current_fundamentals

    base = {"roe": 22.0}
    yoy = score_current_fundamentals({**base, "sales_growth_yoy": 20.0})
    three = score_current_fundamentals({**base, "sales_growth_3y": 20.0})
    both = score_current_fundamentals({**base, "sales_growth_3y": 20.0, "sales_growth_yoy": -50.0})
    assert yoy["coverage"] < three["coverage"]        # weaker evidence, less weight
    assert both["coverage"] == three["coverage"]      # 3y wins when present


BANK_XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:in-bse-fin="http://example/in-bse-fin">
  <xbrli:context id="OneD"><xbrli:period><xbrli:startDate>2026-07-01</xbrli:startDate><xbrli:endDate>2026-09-30</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="OneI"><xbrli:period><xbrli:instant>2026-09-30</xbrli:instant></xbrli:period></xbrli:context>
  <in-bse-fin:LevelOfRounding contextRef="OneD">Lakhs</in-bse-fin:LevelOfRounding>
  <in-bse-fin:InterestEarned contextRef="OneD" decimals="-5" unitRef="INR">335891700000</in-bse-fin:InterestEarned>
  <in-bse-fin:ProfitLossForThePeriod contextRef="OneD" decimals="-5" unitRef="INR">58147200000</in-bse-fin:ProfitLossForThePeriod>
  <in-bse-fin:Capital contextRef="OneI" decimals="-5" unitRef="INR">22985900000</in-bse-fin:Capital>
  <in-bse-fin:ReservesAndSurplus contextRef="OneI" decimals="-5" unitRef="INR">1399482300000</in-bse-fin:ReservesAndSurplus>
  <in-bse-fin:Assets contextRef="OneI" decimals="-5" unitRef="INR">19209351600000</in-bse-fin:Assets>
</xbrli:xbrl>
"""


def test_bank_net_worth_is_capital_plus_reserves():
    from product.pit_financials import _derive
    from product.pit_xbrl import parse_xbrl

    parsed = parse_xbrl(BANK_XBRL)
    row = {"period_end": "2026-09-30", "facts": parsed["facts"], "period_months": 3}
    derived = _derive(row, None, rows=[row])
    assert derived["is_financial"] is True
    assert derived["net_worth_cr"] == pytest.approx(142246.82)
    # 5,814.72 cr a quarter, annualised, over 1.42 lakh cr of net worth.
    assert derived["roe_approx_pct"] == pytest.approx(16.35, abs=0.01)
