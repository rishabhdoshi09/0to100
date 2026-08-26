"""Deterministic cash-flow and balance-sheet quality rules. No estimates."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.series import dated_series, find_row, snapshot


def _tables(raw: Mapping[str, Any]) -> dict[str, list]:
    return {
        "quarterly_results": list(raw.get("quarterly_results") or []),
        "profit_loss": list(raw.get("profit_loss") or []),
        "cash_flow": list(raw.get("cash_flow") or []),
        "balance_sheet": list(raw.get("balance_sheet") or []),
        "shareholding": list(raw.get("shareholding") or []),
    }


def _snap(tables: Mapping[str, Sequence], table: str, needles: tuple[str, ...], *, kind: str = "level") -> dict[str, Any]:
    row = find_row(tables.get(table), needles)
    return snapshot(dated_series(row), kind=kind)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 3)


def _skip_cfo_conversion(framework_id: str) -> bool:
    try:
        from product.due_diligence.frameworks import get_framework
        return bool(get_framework(framework_id).get("skip_cfo_conversion"))
    except Exception:
        return framework_id in {"bank", "nbfc"}


def cash_flow_quality(raw: Mapping[str, Any], *, framework_id: str) -> dict[str, Any]:
    """Banks / NBFCs / insurers skip CFO/PAT cash conversion — it is the wrong picture."""
    if _skip_cfo_conversion(framework_id):
        return {
            "applicable": False,
            "label": "Not applicable",
            "detail": "Cash-conversion ratios are not scored for banks / NBFCs / insurers.",
            "flags": [],
            "metrics": [],
        }
    tables = _tables(raw)
    cfo = _snap(tables, "cash_flow", ("cash from operating",))
    pat = _snap(tables, "profit_loss", ("net profit", "profit after tax"))
    if pat.get("current") is None:
        pat = _snap(tables, "quarterly_results", ("net profit",))
    capex = _snap(tables, "cash_flow", ("cash from investing", "net cash used in investing"))
    recv = _snap(tables, "balance_sheet", ("trade receivables", "receivables"))
    sales = _snap(tables, "profit_loss", ("sales", "revenue"))
    if sales.get("current") is None:
        sales = _snap(tables, "quarterly_results", ("sales", "revenue"))

    metrics: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    def add(metric_id: str, label: str, snap: Mapping[str, Any], unit: str) -> None:
        metrics.append({
            "id": metric_id,
            "label": label,
            "available": snap.get("current") is not None,
            "current": snap.get("current"),
            "period": snap.get("current_period") or "",
            "previous": snap.get("previous"),
            "yoy_change": snap.get("yoy_change"),
            "unit": unit,
            "fact": (
                f"{label}: {snap.get('current')} {unit} ({snap.get('current_period')})"
                if snap.get("current") is not None else
                "Data unavailable"
            ),
        })

    add("cfo", "Cash from operations", cfo, "₹ cr")
    add("pat_cf", "PAT (cash-quality pair)", pat, "₹ cr")
    add("capex", "Investing cash / capex proxy", capex, "₹ cr")
    add("receivables", "Trade receivables", recv, "₹ cr")
    add("sales_cf", "Revenue (cash-quality pair)", sales, "₹ cr")

    cfo_pat = _ratio(cfo.get("current"), pat.get("current"))
    metrics.append({
        "id": "cfo_to_pat",
        "label": "CFO / PAT",
        "available": cfo_pat is not None,
        "current": cfo_pat,
        "period": cfo.get("current_period") or pat.get("current_period") or "",
        "unit": "x",
        "fact": f"CFO / PAT = {cfo_pat}x" if cfo_pat is not None else "Calculation not possible",
        "formula": (
            f"{cfo.get('current')} / {pat.get('current')} = {cfo_pat}"
            if cfo_pat is not None else
            "Calculation not possible — CFO or PAT missing."
        ),
    })
    fcf = None
    if cfo.get("current") is not None and capex.get("current") is not None:
        # Investing cash is typically negative when capex is spent.
        fcf = round(float(cfo["current"]) + float(capex["current"]), 2)
    metrics.append({
        "id": "fcf",
        "label": "Free cash flow proxy (CFO + investing cash)",
        "available": fcf is not None,
        "current": fcf,
        "unit": "₹ cr",
        "fact": f"FCF proxy: {fcf} ₹ cr" if fcf is not None else "Calculation not possible",
    })

    if cfo_pat is not None and cfo_pat < 0.5 and (pat.get("current") or 0) > 0:
        flags.append({
            "id": "cf-cfo-below-pat",
            "severity": "warning",
            "rule": "CFO significantly below PAT",
            "triggered_value": cfo_pat,
            "threshold": 0.5,
            "evidence": f"CFO/PAT = {cfo_pat}x in {cfo.get('current_period') or 'latest period'}.",
            "source": "Cash-flow and P&L tables on file",
        })
    if (
        cfo.get("yoy_change") is not None
        and pat.get("yoy_change") is not None
        and pat["yoy_change"] > 5
        and cfo["yoy_change"] < -5
    ):
        flags.append({
            "id": "cf-pat-up-cfo-down",
            "severity": "warning",
            "rule": "PAT increasing but CFO persistently weak",
            "triggered_value": {"pat_yoy_pct": pat["yoy_change"], "cfo_yoy_pct": cfo["yoy_change"]},
            "threshold": {"pat_yoy_pct": 5, "cfo_yoy_pct": -5},
            "evidence": f"PAT YoY {pat['yoy_change']:+.1f}% while CFO YoY {cfo['yoy_change']:+.1f}%.",
            "source": "Cash-flow and P&L tables on file",
        })
    if (
        recv.get("yoy_change") is not None
        and sales.get("yoy_change") is not None
        and recv["yoy_change"] > sales["yoy_change"] + 15
        and recv["yoy_change"] > 20
    ):
        flags.append({
            "id": "cf-receivables-vs-sales",
            "severity": "monitor",
            "rule": "Receivables growing much faster than revenue",
            "triggered_value": {"receivables_yoy_pct": recv["yoy_change"], "sales_yoy_pct": sales["yoy_change"]},
            "threshold": {"gap_pp": 15, "recv_yoy_pct": 20},
            "evidence": (
                f"Receivables YoY {recv['yoy_change']:+.1f}% vs revenue YoY {sales['yoy_change']:+.1f}%."
            ),
            "source": "Balance-sheet and results tables on file",
        })

    label = "Unmeasured"
    if any(m.get("available") for m in metrics):
        if any(f.get("severity") == "warning" for f in flags):
            label = "Weak"
        elif flags:
            label = "Watch"
        else:
            label = "Adequate"
            if cfo_pat is not None and cfo_pat >= 0.9:
                label = "Strong"
    return {
        "applicable": True,
        "label": label,
        "detail": (
            "Rule-based cash conversion from tables on file. Missing prints stay missing."
        ),
        "flags": flags,
        "metrics": metrics,
    }


def balance_sheet_quality(raw: Mapping[str, Any], *, framework_id: str) -> dict[str, Any]:
    tables = _tables(raw)
    debt = _snap(tables, "balance_sheet", ("borrowings", "total borrowings"))
    equity = _snap(tables, "balance_sheet", ("equity capital", "share capital", "reserves"))
    interest = _snap(tables, "profit_loss", ("interest",))
    ebit = _snap(tables, "profit_loss", ("operating profit", "ebit"))
    inventory = _snap(tables, "balance_sheet", ("inventories", "inventory"))
    flags: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    def add(metric_id: str, label: str, snap: Mapping[str, Any], unit: str) -> None:
        metrics.append({
            "id": metric_id,
            "label": label,
            "available": snap.get("current") is not None,
            "current": snap.get("current"),
            "period": snap.get("current_period") or "",
            "yoy_change": snap.get("yoy_change"),
            "unit": unit,
            "fact": (
                f"{label}: {snap.get('current')} {unit}"
                if snap.get("current") is not None else
                "Data unavailable"
            ),
        })

    add("debt", "Borrowings", debt, "₹ cr")
    add("inventory", "Inventory", inventory, "₹ cr")
    de = _ratio(debt.get("current"), equity.get("current")) if not _skip_cfo_conversion(framework_id) else None
    metrics.append({
        "id": "debt_equity",
        "label": "Debt / equity",
        "available": de is not None,
        "current": de,
        "unit": "x",
        "fact": f"Debt/equity = {de}x" if de is not None else (
            "Not used for banks / NBFCs / insurers" if _skip_cfo_conversion(framework_id) else "Calculation not possible"
        ),
    })
    coverage = _ratio(ebit.get("current"), interest.get("current"))
    metrics.append({
        "id": "interest_coverage",
        "label": "Interest coverage (EBIT / interest)",
        "available": coverage is not None,
        "current": coverage,
        "unit": "x",
        "fact": f"Interest coverage = {coverage}x" if coverage is not None else "Calculation not possible",
        "formula": (
            f"{ebit.get('current')} / {interest.get('current')} = {coverage}"
            if coverage is not None else
            "Calculation not possible — EBIT or interest missing."
        ),
    })

    if not _skip_cfo_conversion(framework_id):
        if debt.get("yoy_change") is not None and debt["yoy_change"] > 40:
            flags.append({
                "id": "bs-debt-spike",
                "severity": "warning",
                "rule": "Sharp debt increase",
                "triggered_value": debt["yoy_change"],
                "threshold": 40,
                "evidence": f"Borrowings YoY {debt['yoy_change']:+.1f}%.",
                "source": "Balance-sheet table on file",
            })
        if coverage is not None and coverage < 2:
            flags.append({
                "id": "bs-interest-cover",
                "severity": "warning",
                "rule": "Declining / low interest coverage",
                "triggered_value": coverage,
                "threshold": 2,
                "evidence": f"EBIT / interest = {coverage}x.",
                "source": "P&L table on file",
            })
        if inventory.get("yoy_change") is not None and inventory["yoy_change"] > 35:
            flags.append({
                "id": "bs-inventory-build",
                "severity": "monitor",
                "rule": "Unusual inventory build",
                "triggered_value": inventory["yoy_change"],
                "threshold": 35,
                "evidence": f"Inventory YoY {inventory['yoy_change']:+.1f}%.",
                "source": "Balance-sheet table on file",
            })

    label = "Unmeasured"
    if any(m.get("available") for m in metrics):
        if any(f.get("severity") == "warning" for f in flags):
            label = "Weak"
        elif flags:
            label = "Watch"
        else:
            label = "Strong" if (coverage is None or coverage >= 4) else "Adequate"
    if _skip_cfo_conversion(framework_id):
        label = "See sector KPIs" if label == "Unmeasured" else label
    return {
        "applicable": True,
        "label": label,
        "detail": "Rule-based leverage, coverage and working-capital checks from tables on file.",
        "flags": flags,
        "metrics": metrics,
    }


def growth_quality(findings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    growth = [f for f in findings if f.get("pillar") == "growth" and f.get("available")]
    profit = [f for f in findings if f.get("pillar") == "profitability" and f.get("available")]
    flags: list[str] = []
    for finding in growth + profit:
        snap = dict(finding.get("snapshot") or {})
        points = list(snap.get("points") or [])
        if len(points) >= 3:
            values = [p.get("value") for p in points if p.get("value") is not None]
            if values and any(v is not None and v < 0 for v in values[-4:]):
                flags.append(f"{finding['label']}: negative print in the recent series.")
        if finding.get("trend") == "improving" and (snap.get("yoy_change") or 0) > 8:
            flags.append(f"{finding['label']}: accelerating versus year-ago.")
        if finding.get("trend") == "deteriorating":
            flags.append(f"{finding['label']}: decelerating versus year-ago.")
    label = "Unmeasured"
    if growth or profit:
        deterior = sum(1 for f in growth + profit if f.get("trend") == "deteriorating")
        improve = sum(1 for f in growth + profit if f.get("trend") == "improving")
        if improve and not deterior:
            label = "Improving"
        elif deterior and not improve:
            label = "Deteriorating"
        elif improve > deterior:
            label = "Improving"
        else:
            label = "Mixed"
    return {"label": label, "notes": flags[:8], "n_growth": len(growth), "n_profit": len(profit)}
