"""Parse official NSE XBRL / Integrated Filing instances.

Does not guess. Tag local-names are mapped; unmapped numeric tags stay in
extra. Dimensional breakdown contexts are ignored. Publication dates come
from exchange metadata, never from the financial period label.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Mapping

PARSER_VERSION = "pit_xbrl.v1"

# Local-name → canonical field. Works for in-capmkt and in-bse-fin.
TAG_MAP = {
    "RevenueFromOperations": "revenue",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "ProfitBeforeExceptionalItemsAndTax": "pbt_before_exceptional",
    "ExceptionalItemsBeforeTax": "exceptional_items",
    "ProfitBeforeTax": "pbt",
    "ProfitLossForPeriod": "pat",
    "ProfitLossForPeriodFromContinuingOperations": "pat_continuing",
    "FinanceCosts": "finance_costs",
    "EmployeeBenefitExpense": "employee_expense",
    "DepreciationDepletionAndAmortisationExpense": "depreciation",
    "Expenses": "total_expenses",
    "CostOfMaterialsConsumed": "materials",
    "PurchasesOfStockInTrade": "purchases",
    "CurrentTax": "current_tax",
    "DeferredTax": "deferred_tax",
    "TaxExpense": "tax",
    "PaidUpValueOfEquityShareCapital": "paid_up_equity",
    "FaceValueOfEquityShareCapital": "face_value",
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "basic_eps",
    "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "diluted_eps",
    "BasicEarningsLossPerShareFromContinuingOperations": "basic_eps_continuing",
    "DilutedEarningsLossPerShareFromContinuingOperations": "diluted_eps_continuing",
    "AmountOfTotalFinancialIndebtednessOfTheListedEntityIncludingShortTermAndLongTermDebtAtTheEndOfPeriod": "total_debt",
    "PropertyPlantAndEquipment": "ppe",
    "TotalAssets": "total_assets",
    "Equity": "equity",
    "EquityShareCapital": "equity_share_capital",
    "OtherEquity": "other_equity",
    "Inventories": "inventories",
    "TradeReceivables": "trade_receivables",
    "TradePayables": "trade_payables",
    "CashAndCashEquivalents": "cash",
    "NetCashFlowsFromUsedInOperatingActivities": "cfo",
    "NetCashFlowsFromUsedInInvestingActivities": "cfi",
    "NetCashFlowsFromUsedInFinancingActivities": "cff",
}

PER_SHARE = {
    "face_value", "basic_eps", "diluted_eps",
    "basic_eps_continuing", "diluted_eps_continuing",
}

MAIN_CONTEXTS = ("OneD", "FourD", "OneI", "PY_I")


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    if ":" in tag:
        return tag.split(":", 1)[-1]
    return tag


def _f(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip().replace(",", "")
    if raw in {"", "NaN", "INF", "-INF"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _scale_to_crore(value: float, *, decimals: str, rounding: str) -> float:
    """XBRL values are INR. QuantTerm stores crore for P&L/BS levels."""
    rounding = (rounding or "").lower()
    if "crore" in rounding or decimals in {"-7", "-6"}:
        return round(value / 1e7, 4)
    if "lakh" in rounding:
        return round(value / 1e5, 4)
    return round(value / 1e7, 4)


def parse_xbrl(xml_text: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {
            "ok": False,
            "numbers_parsed": False,
            "reason_code": "PARSER_FAILED",
            "errors": [f"xml:{exc}"],
            "parser_version": PARSER_VERSION,
            "confidence": 0.0,
        }

    contexts: dict[str, dict[str, str]] = {}
    rounding = ""
    reporting_period = ""
    nature = ""
    audited = ""
    board_date = ""
    period_start = ""
    period_end = ""
    facts_by_ctx: dict[str, dict[str, Any]] = {}
    extra: dict[str, Any] = {}

    for node in root.iter():
        name = _local(node.tag)
        if name == "context":
            cid = node.attrib.get("id") or ""
            start = end = instant = ""
            for child in node.iter():
                loc = _local(child.tag)
                if loc == "startDate" and child.text:
                    start = child.text.strip()[:10]
                elif loc == "endDate" and child.text:
                    end = child.text.strip()[:10]
                elif loc == "instant" and child.text:
                    instant = child.text.strip()[:10]
            contexts[cid] = {"start": start, "end": end, "instant": instant}
            continue
        if name == "LevelOfRounding" and node.text:
            rounding = node.text.strip()
        if name == "TypeOfReportingPeriod" and node.text:
            reporting_period = node.text.strip()
        if name == "NatureOfReportStandaloneConsolidated" and node.text:
            nature = node.text.strip()
        if name == "WhetherResultsAreAuditedOrUnaudited" and node.text:
            audited = node.text.strip()
        if name == "DateOfBoardMeetingWhenFinancialResultsWereApproved" and node.text:
            board_date = node.text.strip()[:10]
        if name == "DateOfStartOfReportingPeriod" and node.text:
            period_start = node.text.strip()[:10]
        if name == "DateOfEndOfReportingPeriod" and node.text:
            period_end = node.text.strip()[:10]

        ctx = node.attrib.get("contextRef")
        if not ctx:
            continue
        if ctx not in MAIN_CONTEXTS:
            continue
        value = _f(node.text)
        if value is None:
            continue
        field = TAG_MAP.get(name)
        unit = node.attrib.get("unitRef") or ""
        decimals = node.attrib.get("decimals") or ""
        if field in PER_SHARE or unit == "INRPerShare":
            scaled = value
            unit_out = "INR_per_share"
        elif unit in {"INR", "pure", ""} and field:
            scaled = _scale_to_crore(value, decimals=decimals, rounding=rounding)
            unit_out = "INR_crore"
        elif field:
            scaled = value
            unit_out = unit or "raw"
        else:
            extra.setdefault(ctx, {})[name] = value
            continue
        bucket = facts_by_ctx.setdefault(ctx, {})
        bucket[field] = {
            "value": scaled,
            "raw": value,
            "tag": name,
            "unit": unit_out,
            "decimals": decimals,
            "context": ctx,
        }

    current = dict(facts_by_ctx.get("OneD") or facts_by_ctx.get("FourD") or {})
    bs = dict(facts_by_ctx.get("OneI") or {})
    flat: dict[str, float] = {}
    identities: dict[str, str] = {}
    for key, item in {**current, **{f"bs_{k}": v for k, v in bs.items()}}.items():
        if not isinstance(item, Mapping):
            continue
        val = item.get("value")
        if isinstance(val, (int, float)):
            flat[key] = float(val)
            identities[key] = str(item.get("tag") or "")

    n_core = sum(1 for k in ("revenue", "pat", "pbt") if k in flat)
    ok = n_core >= 1
    if not period_end:
        period_end = (contexts.get("OneD") or {}).get("end") or (contexts.get("OneI") or {}).get("instant") or ""
    if not period_start:
        period_start = (contexts.get("OneD") or {}).get("start") or ""

    ebit = None
    if "pbt" in flat and "finance_costs" in flat:
        ebit = round(flat["pbt"] + flat["finance_costs"], 4)
    ebitda = None
    if ebit is not None and "depreciation" in flat:
        ebitda = round(ebit + flat["depreciation"], 4)

    confidence = 0.0
    if ok:
        confidence = min(0.95, 0.45 + 0.08 * n_core + 0.03 * min(len(flat), 12))
        if nature.lower().startswith("consol"):
            confidence = min(0.97, confidence + 0.05)

    return {
        "ok": ok,
        "numbers_parsed": ok,
        "parser_version": PARSER_VERSION,
        "confidence": round(confidence, 3),
        "errors": errors,
        "rounding": rounding,
        "reporting_period": reporting_period,
        "nature": nature,
        "audited": audited,
        "board_date": board_date,
        "period_start": period_start,
        "period_end": period_end,
        "facts": flat,
        "source_tags": identities,
        "ebit_cr": ebit,
        "ebitda_cr": ebitda,
        "n_fields": len(flat),
        "reason_code": "" if ok else "PARSER_FAILED",
    }
