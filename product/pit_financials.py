"""PIT financial facts and derived ratios.

Ratios are recomputed from warehouse facts whose available_from <= T.
Current Screener TTM is never read.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from product.pit_warehouse import DOC_QUARTERLY_RESULT, get_evidence

PARSER_VERSION = "pit_financials.v2"
# v2: lakh-rounded legacy rows normalised to crore, true YoY (same period one
# year earlier, never the sequential quarter), ROE on net worth (never paid-up
# capital) with TTM / annualised profit, ROCE only on capital employed.
DERIVATION_VERSION = PARSER_VERSION

# Bank fields that consolidated filings routinely leave as 0.0. An exact zero
# NPA / ROA / CET1 for a bank is a blank in the filing, not a measurement.
_BANK_PLACEHOLDER_ZERO = frozenset({"gnpa", "nnpa", "gnpa_pct", "nnpa_pct", "roa", "cet1_ratio"})
_JUNK_FACT_KEYS = frozenset({"", "null", "None", "bs_None", "bs_null"})


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return round(num / den, 4)


def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev in (None, 0):
        return None
    return round((cur - prev) / abs(prev) * 100.0, 2)


def normalise_legacy_units(facts: Mapping[str, Any], *, parser_version: str = "",
                           rounding: str = "") -> dict[str, Any]:
    """Return facts in crore with placeholder/junk keys removed.

    ``pit_xbrl.v1`` divided INR by 1e5 for filings whose LevelOfRounding was
    "Lakhs", so their monetary facts are 100x too large. Per-share values and
    ratios were never scaled and stay untouched.
    """
    from product.pit_xbrl import LEGACY_LAKH_SCALED_VERSIONS, PER_SHARE, RATIOS

    lakh_scaled = (
        str(parser_version or "") in LEGACY_LAKH_SCALED_VERSIONS
        and "lakh" in str(rounding or "").lower()
    )
    out: dict[str, Any] = {}
    for key, value in dict(facts or {}).items():
        key = str(key)
        if key in _JUNK_FACT_KEYS:
            continue
        base = key[3:] if key.startswith("bs_") else key
        if base in _BANK_PLACEHOLDER_ZERO and _f(value) == 0.0:
            continue
        if lakh_scaled and base not in PER_SHARE and base not in RATIOS:
            num = _f(value)
            if num is not None:
                value = round(num / 100.0, 4)
        out[key] = value
    return out


def _period_months(reporting_period: Any) -> int:
    """Length of the *facts* window. Integrated filings carry the current
    quarter (context OneD); SME filers report half-yearly."""
    label = str(reporting_period or "").lower()
    if "half" in label:
        return 6
    if "year" in label or "annual" in label:
        return 12
    return 3


def parsed_results(symbol: str, *, as_of: str, path=None) -> list[dict[str, Any]]:
    rows = []
    evidence = get_evidence(symbol, as_of=as_of, evidence_types=(DOC_QUARTERLY_RESULT,), path=path)
    # A re-parse of the same filing is stored as a new revision that names the
    # row it replaces; the replaced row stays for audit but is never read.
    superseded = {str(item.get("supersedes")) for item in evidence if item.get("supersedes")}
    for item in evidence:
        if str(item.get("evidence_id") or "") in superseded:
            continue
        extracted = dict(item.get("extracted") or {})
        if not extracted.get("numbers_parsed"):
            continue
        facts = normalise_legacy_units(
            extracted.get("facts") or {},
            parser_version=str(extracted.get("parser_version") or item.get("parser_version") or ""),
            rounding=str(extracted.get("rounding") or ""),
        )
        if not facts:
            continue
        nature = str(extracted.get("nature") or "").lower()
        rows.append({
            **item,
            "facts": facts,
            "rounding": extracted.get("rounding") or "",
            "reporting_period": extracted.get("reporting_period") or "",
            "period_months": _period_months(extracted.get("reporting_period")),
            "source_tags": extracted.get("source_tags") or {},
            "nature": extracted.get("nature") or "",
            "consolidated": nature.startswith("consol"),
            "confidence": extracted.get("confidence"),
            "parser_version": extracted.get("parser_version") or item.get("parser_version"),
        })
    preferred: list[dict[str, Any]] = []
    seen_period = set()
    consol = [r for r in rows if r.get("consolidated")]
    pool = consol or rows
    pool.sort(key=lambda r: str(r.get("available_from") or ""), reverse=True)
    for row in pool:
        period = str(row.get("period_end") or "")
        if period in seen_period:
            continue
        seen_period.add(period)
        preferred.append(row)
    return preferred


def get_fact(symbol: str, field: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Latest eligible value for one field. Future artifacts are invisible."""
    for row in parsed_results(symbol, as_of=as_of, path=path):
        facts = row.get("facts") or {}
        if field in facts:
            return {
                "symbol": str(symbol).upper(),
                "field": field,
                "as_of": str(as_of)[:10],
                "value": facts[field],
                "available": True,
                "available_from": row.get("available_from"),
                "period_end": row.get("period_end"),
                "evidence_id": row.get("evidence_id"),
                "source_tag": (row.get("source_tags") or {}).get(field),
                "source": row.get("source"),
                "pit_status": "AVAILABLE_AT_T",
            }
    return {
        "symbol": str(symbol).upper(),
        "field": field,
        "as_of": str(as_of)[:10],
        "value": None,
        "available": False,
        "pit_status": "UNAVAILABLE_AT_T",
    }


def _months(row: Mapping[str, Any] | None) -> int:
    try:
        return int((row or {}).get("period_months") or 3)
    except (TypeError, ValueError):
        return 3


def _find_period(rows: Sequence[Mapping[str, Any]], target: date, months: int,
                 tolerance_days: int = 25) -> Mapping[str, Any] | None:
    for row in rows:
        end = _day(row.get("period_end"))
        if end and _months(row) == months and abs((end - target).days) <= tolerance_days:
            return row
    return None


def _shift(end: date, months: int) -> date:
    return date.fromordinal(end.toordinal() - int(round(months * 30.44)))


def _ttm_block(rows: Sequence[Mapping[str, Any]], latest: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """Consecutive filings covering the trailing twelve months, latest first."""
    end = _day(latest.get("period_end"))
    months = _months(latest)
    if end is None or months not in (3, 6):
        return None
    block = [latest]
    for step in range(1, 12 // months):
        row = _find_period(rows, _shift(end, months * step), months)
        if row is None:
            return None
        block.append(row)
    return block


def _ttm_sum(block: Sequence[Mapping[str, Any]] | None, field: str) -> float | None:
    if not block:
        return None
    vals = [_f((row.get("facts") or {}).get(field)) for row in block]
    if any(v is None for v in vals):
        return None
    return round(sum(vals), 4)


def _balance_sheet(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None, str]:
    """Latest net worth and borrowings at or before the latest filing.

    Net worth is shareholders' equity (share capital + other equity). Paid-up
    capital alone is NOT net worth: using it inflated ROE 10-100x (TCS 3707%).
    """
    for row in rows:
        facts = row.get("facts") or {}
        equity = _f(facts.get("bs_equity"))
        if equity is None:
            cap, other = _f(facts.get("bs_equity_share_capital")), _f(facts.get("bs_other_equity"))
            equity = None if cap is None or other is None else cap + other
        if equity is None:
            continue
        debt = _f(facts.get("bs_total_debt"))
        if debt is None:
            debt = _f(facts.get("total_debt"))
        return equity, debt, str(row.get("period_end") or "")
    return None, None, ""


def _growth_from_positive(cur: float | None, prev: float | None) -> float | None:
    """Growth is undefined from a loss/zero base; never report a sign flip as growth."""
    if cur is None or prev is None or prev <= 0:
        return None
    return round((cur - prev) / prev * 100.0, 2)


def _derive(latest: Mapping[str, Any], previous: Mapping[str, Any] | None,
            *, rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    history = list(rows) if rows is not None else [r for r in (latest, previous) if r]
    facts = dict(latest.get("facts") or {})
    prev = dict((previous or {}).get("facts") or {})
    months = _months(latest)
    annualise = 12.0 / months if months in (3, 6, 12) else None
    financial = any("interest_earned" in (row.get("facts") or {}) for row in history)

    revenue = _f(facts.get("revenue") or facts.get("total_income"))
    pat = _f(facts.get("pat"))
    pbt = _f(facts.get("pbt"))
    finance = _f(facts.get("finance_costs"))
    dep = _f(facts.get("depreciation"))
    cfo = _f(facts.get("cfo"))
    paid = _f(facts.get("paid_up_equity"))
    face = _f(facts.get("face_value"))
    cash = _f(facts.get("bs_cash"))
    ebit = None if financial or pbt is None or finance is None else pbt + finance
    ebitda = None if ebit is None or dep is None else ebit + dep
    shares = _ratio(paid, face)

    end = _day(latest.get("period_end"))
    yoy_row = _find_period(history, _shift(end, 12), months) if end else None
    yoy = dict((yoy_row or {}).get("facts") or {})
    block = _ttm_block(history, latest)

    ttm_pat = _ttm_sum(block, "pat")
    ttm_revenue = _ttm_sum(block, "revenue")
    ttm_pbt = _ttm_sum(block, "pbt")
    ttm_finance = _ttm_sum(block, "finance_costs")
    ttm_eps = _ttm_sum(block, "basic_eps")
    annual_pat = ttm_pat if ttm_pat is not None else (pat * annualise if pat is not None and annualise else None)
    roe_basis = "TTM" if ttm_pat is not None else (f"ANNUALISED_{months}M" if annual_pat is not None else "")
    if financial:
        annual_ebit = annual_finance = None
    elif ttm_pbt is not None and ttm_finance is not None:
        annual_ebit, annual_finance = ttm_pbt + ttm_finance, ttm_finance
    else:
        annual_ebit = ebit * annualise if ebit is not None and annualise else None
        annual_finance = finance * annualise if finance is not None and annualise else None

    net_worth, debt, bs_as_of = _balance_sheet(history)
    has_equity = net_worth is not None and net_worth > 0
    roe = round(annual_pat / net_worth * 100.0, 2) if has_equity and annual_pat is not None else None
    capital_employed = net_worth + debt if has_equity and debt is not None and debt >= 0 else None
    roce = (round(annual_ebit / capital_employed * 100.0, 2)
            if capital_employed and annual_ebit is not None and not financial else None)
    interest_cover = (round(annual_ebit / annual_finance, 2)
                      if annual_ebit is not None and annual_finance and annual_finance > 0 else None)

    derived = {
        "is_financial": financial,
        "period_months": months,
        "pat_margin_pct": None if revenue is None or pat is None else round(pat / revenue * 100.0, 2),
        "pbt_margin_pct": None if revenue is None or pbt is None else round(pbt / revenue * 100.0, 2),
        "ebit_cr": ebit,
        "ebitda_cr": ebitda,
        "ebit_margin_pct": None if revenue is None or ebit is None else round(ebit / revenue * 100.0, 2),
        # YoY compares the same period one year earlier; sequential change is
        # reported separately and never under a "yoy" name.
        "revenue_yoy_pct": _growth_from_positive(revenue, _f(yoy.get("revenue") or yoy.get("total_income"))),
        "pat_yoy_pct": _growth_from_positive(pat, _f(yoy.get("pat"))),
        "yoy_period_end": (yoy_row or {}).get("period_end"),
        "revenue_qoq_pct": _growth(revenue, _f(prev.get("revenue") or prev.get("total_income"))),
        "pat_qoq_pct": _growth_from_positive(pat, _f(prev.get("pat"))),
        "ttm_revenue_cr": ttm_revenue,
        "ttm_pat_cr": ttm_pat,
        "ttm_eps": ttm_eps,
        "ttm_periods": [r.get("period_end") for r in block] if block else None,
        "net_worth_cr": round(net_worth, 4) if net_worth is not None else None,
        "debt_cr": debt,
        "balance_sheet_as_of": bs_as_of or None,
        "debt_to_equity": _ratio(debt, net_worth) if has_equity else None,
        "roe_approx_pct": roe,
        "roe_basis": roe_basis if roe is not None else None,
        "roce_approx_pct": roce,
        "interest_coverage": interest_cover,
        "cash_conversion": _ratio(cfo, pat),
        "shares_outstanding_cr": shares,
        "share_count_change_pct": _growth(shares, _ratio(_f(prev.get("paid_up_equity")), _f(prev.get("face_value")))),
        "cash_cr": cash,
        "derivation_version": DERIVATION_VERSION,
        "source_fact_ids": {
            "latest": latest.get("evidence_id"),
            "previous": (previous or {}).get("evidence_id"),
            "yoy": (yoy_row or {}).get("evidence_id"),
        },
    }
    return {k: v for k, v in derived.items() if v is not None or k == "source_fact_ids"}


def _period_label(period_end: str) -> str:
    d = _day(period_end)
    if not d:
        return period_end or "period"
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{months[d.month - 1]} {d.year}"


def _annual_profit_loss(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fiscal years (April-March) whose every quarter/half is on file."""
    years: dict[str, dict[str, float]] = {}
    for row in rows:
        end = _day(row.get("period_end"))
        if end is None or end.month != 3:
            continue
        block = _ttm_block(rows, row)
        if not block:
            continue
        label = _period_label(str(row.get("period_end") or ""))
        sums = {f: _ttm_sum(block, f) for f in ("revenue", "pat", "pbt")}
        years.setdefault(label, {k: v for k, v in sums.items() if v is not None})
    ordered = sorted(years, key=lambda lab: int(lab.split()[-1]))

    def annual(field: str, label: str) -> dict[str, Any]:
        out: dict[str, Any] = {"row_label": label}
        for lab in ordered:
            if field in years[lab]:
                out[lab] = years[lab][field]
        return out

    return [annual("revenue", "Sales+"), annual("pat", "Net Profit+"), annual("pbt", "Profit before tax")]


def screener_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Shape PIT facts like Investigate tables so the same quality rules run."""
    chronological = list(reversed(list(rows)[:8]))
    labels = [_period_label(str(r.get("period_end") or "")) for r in chronological]

    def series(field: str, label: str) -> dict[str, Any]:
        out: dict[str, Any] = {"row_label": label}
        for lab, row in zip(labels, chronological):
            val = (row.get("facts") or {}).get(field)
            if val is not None:
                out[lab] = val
        return out

    quarterly = [
        series("revenue", "Sales+"),
        series("pat", "Net Profit+"),
        series("pbt", "Profit before tax"),
        series("finance_costs", "Interest"),
        series("depreciation", "Depreciation"),
        series("employee_expense", "Employee cost"),
    ]
    opm: dict[str, Any] = {"row_label": "OPM %"}
    for lab, row in zip(labels, chronological):
        facts = row.get("facts") or {}
        rev = _f(facts.get("revenue"))
        pbt = _f(facts.get("pbt"))
        fin = _f(facts.get("finance_costs"))
        if rev and pbt is not None:
            ebit = pbt + (fin or 0)
            opm[lab] = round(ebit / rev * 100.0, 2)
    quarterly.append(opm)
    return {
        "quarterly_results": quarterly,
        # Annual (complete fiscal-year) sums only. Feeding quarterly values
        # here made every downstream "3y CAGR" a ratio of two quarters.
        "profit_loss": _annual_profit_loss(rows),
        "balance_sheet": [
            series("bs_total_assets", "Total Assets"),
            series("bs_equity", "Equity"),
            series("total_debt", "Borrowings"),
            series("bs_cash", "Cash Equivalents"),
            series("paid_up_equity", "Equity Share Capital"),
        ],
        "cash_flow": [
            series("cfo", "Cash from Operating Activity+"),
            series("cfi", "Cash from Investing Activity"),
            series("cff", "Cash from Financing Activity"),
        ],
    }


def get_financial_snapshot_v2(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    rows = parsed_results(symbol, as_of=as_of, path=path)
    latest = rows[0] if rows else None
    previous = None
    if latest:
        for row in rows[1:]:
            if str(row.get("period_end") or "") != str(latest.get("period_end") or ""):
                previous = row
                break
    derived = _derive(latest, previous, rows=rows) if latest else {}
    tables = screener_tables(rows) if rows else {}
    stale = False
    if latest:
        pub = _day(latest.get("available_from"))
        t = _day(as_of)
        if pub and t and (t - pub).days > 400:
            stale = True
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "available": bool(latest),
        "numbers_parsed": bool(latest),
        "n_parsed_results": len(rows),
        "latest_publication": (latest or {}).get("available_from"),
        "latest_period_end": (latest or {}).get("period_end"),
        "latest_source_url": (latest or {}).get("source_url"),
        "latest_evidence_id": (latest or {}).get("evidence_id"),
        "nature": (latest or {}).get("nature"),
        "facts": (latest or {}).get("facts") or {},
        "derived": derived,
        "tables": tables,
        "stale_for_production": stale,
        "quality_status": "UNKNOWN" if not latest or stale else "MEASURED",
        "parser_version": PARSER_VERSION,
        "note": (
            "No parsed official XBRL/HTML result on or before T."
            if not latest else
            "Derived only from filings published on or before T."
        ),
    }


def get_business_snapshot(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Answer only questions the available facts can support. Unknown stays unknown."""
    fin = get_financial_snapshot_v2(symbol, as_of=as_of, path=path)
    facts = dict(fin.get("facts") or {})
    derived = dict(fin.get("derived") or {})
    answered: dict[str, Any] = {}
    unknown = []
    if facts.get("revenue") is not None:
        answered["revenue"] = facts["revenue"]
    else:
        unknown.append("revenue")
    if derived.get("pat_margin_pct") is not None:
        answered["pat_margin_pct"] = derived["pat_margin_pct"]
    else:
        unknown.append("margins")
    if derived.get("revenue_yoy_pct") is not None:
        answered["growth"] = derived["revenue_yoy_pct"]
    else:
        unknown.append("growth")
    if derived.get("debt_to_equity") is not None:
        answered["leverage"] = derived["debt_to_equity"]
    else:
        unknown.append("leverage")
    if facts.get("employee_expense") is not None and facts.get("revenue"):
        answered["employee_intensity_pct"] = round(facts["employee_expense"] / facts["revenue"] * 100.0, 2)
    else:
        unknown.append("cost_structure")
    for name in ("segment_mix", "order_book", "geographic_mix", "customer_dependence", "capacity"):
        unknown.append(name)
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "answered": answered,
        "unknown": unknown,
        "pit_unavailable": [] if fin.get("available") else ["company_financials"],
        "quality_label": "Unmeasured" if not answered else "Partial",
        "note": "Framework questions without a dated fact remain UNKNOWN.",
    }
