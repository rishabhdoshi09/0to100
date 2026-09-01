"""Company Intelligence moat layer — extends existing DD, does not compete with it.

Applies metrics only when the business framework says they are appropriate.
Source hierarchy: official filing → reputable public secondary → last-good.
Missing stays missing (never zero). DD may downgrade/block; it cannot invent BUY.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.provenance import conflict_record, material_disagreement, provenance, resolve_fact
from product.due_diligence.quality_rules import _ratio, _snap, _tables

LENDING = {"bank", "nbfc", "nbfc_gold", "nbfc_housing", "life_insurance", "general_insurance"}
COMMODITY = {"metals", "mining", "oil_gas", "refining", "cement"}
ASSET_LIGHT = {"it", "software_product", "amc", "exchange", "broker", "media"}

# Industrial cash/ROIC thresholds are the wrong picture for these frameworks.
SKIP_GENERIC_INDUSTRIAL = LENDING | COMMODITY | ASSET_LIGHT | {"airlines"}

INVENTS_BUY = False


def metric_applicable(metric_id: str, framework_id: str) -> bool:
    fw = str(framework_id or "generic")
    industrial = {
        "cfo_conversion", "fcf", "working_capital", "receivable_stress",
        "inventory_stress", "roic", "incremental_roic", "capex_productivity",
    }
    if metric_id in industrial and fw in (LENDING | {"life_insurance", "general_insurance"}):
        return False
    if metric_id in {"inventory_stress"} and fw in ASSET_LIGHT:
        return False
    if metric_id in {"roic", "incremental_roic"} and fw in COMMODITY:
        return False  # commodity ROIC is cycle-dominated; do not apply industrial cutoffs
    if metric_id in {"nim", "gnpa"} and fw not in LENDING:
        return False
    return True


def _n(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _metric(
    metric_id: str,
    *,
    framework_id: str,
    value: float | None,
    source: Mapping[str, Any] | None,
    unit: str,
    formula: str,
    assumptions: str,
) -> dict[str, Any]:
    applicable = metric_applicable(metric_id, framework_id)
    if not applicable:
        return {
            "id": metric_id,
            "applicable": False,
            "available": False,
            "value": None,
            "unit": unit,
            "status": "NOT_APPLICABLE",
            "formula": formula,
            "assumptions": assumptions,
            "provenance": provenance(value=None, source="framework", confidence="none",
                                     raw_reference=f"not applicable to {framework_id}"),
        }
    if value is None:
        return {
            "id": metric_id,
            "applicable": True,
            "available": False,
            "value": None,  # missing stays missing — never 0
            "unit": unit,
            "status": "UNKNOWN",
            "formula": formula,
            "assumptions": assumptions,
            "provenance": provenance(value=None, source=(source or {}).get("source") or "unavailable",
                                     confidence="none", raw_reference="missing evidence"),
        }
    return {
        "id": metric_id,
        "applicable": True,
        "available": True,
        "value": value,
        "unit": unit,
        "status": "MEASURED",
        "formula": formula,
        "assumptions": assumptions,
        "provenance": source or provenance(value=value, source="tables_on_file", confidence="medium"),
    }


def promise_vs_actual(
    guidance: Sequence[Mapping[str, Any]] | None,
    actuals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare extracted management promises with later prints. Missing ≠ miss."""
    guidance = list(guidance or [])
    actuals = dict(actuals or {})
    if not guidance:
        return {
            "status": "UNKNOWN",
            "detail": "no management promise on file",
            "missed": False,
            "delivered": False,
        }
    hits = 0
    misses = 0
    unknown = 0
    details = []
    for item in guidance:
        metric = str(item.get("metric") or item.get("guidance_metric") or "unspecified")
        promised = _n(item.get("value") or item.get("guidance_value") or item.get("promised"))
        actual = _n(actuals.get(metric) or actuals.get(metric.lower()) or item.get("actual"))
        if promised is None or actual is None:
            unknown += 1
            details.append({"metric": metric, "status": "UNKNOWN"})
            continue
        # Promised growth/level vs actual: miss if actual below 80% of promise when promise>0
        if promised > 0 and actual < 0.8 * promised:
            misses += 1
            details.append({"metric": metric, "promised": promised, "actual": actual, "status": "MISSED"})
        else:
            hits += 1
            details.append({"metric": metric, "promised": promised, "actual": actual, "status": "DELIVERED"})
    if misses and not hits:
        status = "MISSED"
    elif misses:
        status = "MIXED"
    elif hits:
        status = "DELIVERED"
    else:
        status = "UNKNOWN"
    return {
        "status": status,
        "detail": f"hits={hits} misses={misses} unknown={unknown}",
        "missed": bool(misses),
        "delivered": bool(hits) and not misses,
        "items": details,
    }


def company_intelligence_moat(
    raw: Mapping[str, Any] | None,
    *,
    framework_id: str = "generic",
    findings: Sequence[Mapping[str, Any]] | None = None,
    guidance: Sequence[Mapping[str, Any]] | None = None,
    source_conflicts: Sequence[Mapping[str, Any]] | None = None,
    official: Mapping[str, Any] | None = None,
    secondary: Mapping[str, Any] | None = None,
    last_good: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = dict(raw or {})
    tables = _tables(raw)
    fw = str(framework_id or "generic")
    flags: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    def add_from_tables(metric_id: str, needles: tuple[str, ...], table: str, unit: str, formula: str) -> dict[str, Any]:
        snap = _snap(tables, table, needles)
        fact = resolve_fact(
            official=(official or {}).get(metric_id) if official else None,
            secondary=(secondary or {}).get(metric_id) if secondary else None,
            last_good=(last_good or {}).get(metric_id) if last_good else None,
            on_file={"value": snap.get("current"), "source": "tables_on_file",
                     "period": snap.get("current_period") or ""},
        )
        value = _n(fact.get("value"))
        if value is None:
            value = _n(snap.get("current"))
            if value is not None:
                fact = provenance(value=value, period=str(snap.get("current_period") or ""),
                                  source="tables_on_file", confidence="medium")
        m = _metric(
            metric_id, framework_id=fw, value=value, source=fact if isinstance(fact, dict) else None,
            unit=unit, formula=formula, assumptions=f"framework={fw}",
        )
        m["yoy_change"] = snap.get("yoy_change") if m["applicable"] else None
        m["qoq_change"] = snap.get("qoq_change") if m["applicable"] else None
        metrics.append(m)
        return m

    cfo = add_from_tables("cfo", ("cash from operating",), "cash_flow", "₹ cr", "cash_flow.CFO")
    pat = add_from_tables("pat", ("net profit", "profit after tax"), "profit_loss", "₹ cr", "P&L PAT")
    sales = add_from_tables("revenue", ("sales", "revenue"), "profit_loss", "₹ cr", "P&L sales")
    recv = add_from_tables("receivables", ("trade receivables", "receivables"), "balance_sheet", "₹ cr", "BS receivables")
    inv = add_from_tables("inventory", ("inventories", "inventory"), "balance_sheet", "₹ cr", "BS inventory")
    debt = add_from_tables("debt", ("borrowings", "total borrowings"), "balance_sheet", "₹ cr", "BS borrowings")
    equity = add_from_tables("equity", ("equity capital", "share capital", "reserves"), "balance_sheet", "₹ cr", "BS equity")
    interest = add_from_tables("interest", ("interest",), "profit_loss", "₹ cr", "P&L interest")
    ebit = add_from_tables("ebit", ("operating profit", "ebit"), "profit_loss", "₹ cr", "P&L EBIT")
    capex = add_from_tables("capex", ("cash from investing",), "cash_flow", "₹ cr", "CF investing")
    shares = add_from_tables("share_capital", ("equity share capital", "share capital"), "balance_sheet", "₹ cr", "dilution proxy")
    promoter = add_from_tables("promoter_holding", ("promoters", "promoter"), "shareholding", "%", "shareholding.promoter")
    pledge = add_from_tables("pledging", ("pledge", "pledged"), "shareholding", "%", "shareholding.pledge")
    roce = add_from_tables("roce", ("roce", "return on capital"), "profit_loss", "%", "ROCE print")
    roe = add_from_tables("roe", ("roe", "return on equity"), "profit_loss", "%", "ROE print")

    cfo_pat = _ratio(cfo.get("value"), pat.get("value")) if cfo.get("applicable") and pat.get("applicable") else None
    metrics.append(_metric(
        "cfo_conversion", framework_id=fw, value=cfo_pat,
        source=provenance(value=cfo_pat, source="derived", confidence="medium"),
        unit="x", formula="CFO / PAT", assumptions="not scored for banks/NBFC/insurance",
    ))
    fcf = None
    if cfo.get("value") is not None and capex.get("value") is not None and metric_applicable("fcf", fw):
        fcf = round(float(cfo["value"]) + float(capex["value"]), 4)
    metrics.append(_metric(
        "fcf", framework_id=fw, value=fcf,
        source=provenance(value=fcf, source="derived", confidence="medium"),
        unit="₹ cr", formula="CFO + investing cash", assumptions="investing cash typically negative",
    ))
    coverage = _ratio(ebit.get("value"), interest.get("value"))
    metrics.append(_metric(
        "interest_coverage", framework_id=fw, value=coverage,
        source=provenance(value=coverage, source="derived", confidence="medium"),
        unit="x", formula="EBIT / interest", assumptions="missing stays missing",
    ))
    roic = roce.get("value") if metric_applicable("roic", fw) else None
    metrics.append(_metric(
        "roic", framework_id=fw, value=roic if metric_applicable("roic", fw) else None,
        source=roce.get("provenance"),
        unit="%", formula="ROCE used as ROIC proxy when ROIC print absent",
        assumptions="commodity/lending frameworks skip industrial ROIC cutoffs",
    ))

    conv = next((m for m in metrics if m["id"] == "cfo_conversion"), None)
    if conv and conv.get("applicable") and conv.get("value") is not None and conv["value"] < 0.5 and (pat.get("value") or 0) > 0:
        flags.append({
            "id": "cash_conversion_deterioration",
            "severity": "warning",
            "dd_effect": "PENALIZE",
            "evidence": f"CFO/PAT={conv['value']}",
        })
    if conv and conv.get("applicable") and conv.get("value") is not None and conv["value"] < 0.2 and (pat.get("value") or 0) > 0:
        flags[-1]["severity"] = "critical"
        flags[-1]["dd_effect"] = "BLOCK"

    if shares.get("yoy_change") is not None and shares["yoy_change"] > 10:
        flags.append({
            "id": "dilution",
            "severity": "warning",
            "dd_effect": "PENALIZE",
            "evidence": f"share capital YoY {shares['yoy_change']}",
        })
    pledge_val = pledge.get("value")
    if pledge_val is not None and pledge_val >= 20:
        flags.append({
            "id": "pledging",
            "severity": "critical",
            "dd_effect": "BLOCK",
            "evidence": f"pledge={pledge_val}%",
        })
    promoter_change = promoter.get("yoy_change")
    if promoter_change is None:
        promoter_change = promoter.get("qoq_change")
    if promoter_change is not None and promoter_change < -3:
        flags.append({
            "id": "promoter_holding_trend",
            "severity": "warning",
            "dd_effect": "PENALIZE",
            "evidence": f"promoter YoY {promoter['yoy_change']}",
        })

    promises = promise_vs_actual(guidance, {
        "sales": sales.get("value"),
        "revenue": sales.get("value"),
        "pat": pat.get("value"),
    })
    if promises.get("missed"):
        flags.append({
            "id": "management_promise_vs_actual",
            "severity": "warning",
            "dd_effect": "PENALIZE",
            "evidence": promises.get("detail"),
        })

    # Capital allocation: capex rising while FCF deeply negative and sales not growing
    sales_change = sales.get("yoy_change")
    if sales_change is None:
        sales_change = sales.get("qoq_change")
    if (
        metric_applicable("capex_productivity", fw)
        and capex.get("value") is not None
        and fcf is not None
        and fcf < 0
        and sales_change is not None
        and sales_change < 0
    ):
        flags.append({
            "id": "capital_allocation_deterioration",
            "severity": "warning",
            "dd_effect": "PENALIZE",
            "evidence": "negative FCF with declining sales while investing cash is being spent",
        })

    dd_effect = "NEUTRAL"
    if any(f.get("dd_effect") == "BLOCK" for f in flags):
        dd_effect = "BLOCK"
    elif any(f.get("dd_effect") == "PENALIZE" for f in flags):
        dd_effect = "PENALIZE"

    conflicts = list(source_conflicts or [])
    if official and secondary:
        for key in set(official) | set(secondary):
            a = (official or {}).get(key) if isinstance(official, Mapping) else None
            b = (secondary or {}).get(key) if isinstance(secondary, Mapping) else None
            av = a.get("value") if isinstance(a, Mapping) else a
            bv = b.get("value") if isinstance(b, Mapping) else b
            if av is not None and bv is not None and material_disagreement(av, bv):
                pref = resolve_fact(official=a if isinstance(a, Mapping) else {"value": av, "source": "official"},
                                    secondary=b if isinstance(b, Mapping) else {"value": bv, "source": "secondary"})
                conflicts.append(conflict_record(str(key), pref, b if isinstance(b, Mapping) else provenance(value=bv, source="secondary")))

    by_id = {m["id"]: m for m in metrics}
    stale = [
        m["id"] for m in metrics
        if isinstance(m.get("provenance"), dict) and m["provenance"].get("stale")
    ]

    return {
        "schema_version": 1,
        "invents_buy": INVENTS_BUY,
        "cannot_create_buy": True,
        "framework_id": fw,
        "dd_effect": dd_effect,
        "metrics": metrics,
        "flags": flags,
        "promise_vs_actual": promises,
        "source_conflicts": conflicts,
        "stale_last_good": stale,
        "missing_stays_unknown": True,
        "zero_never_imputed": True,
        "by_id": by_id,
        "findings_echo": len(list(findings or [])),
    }
