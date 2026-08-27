"""Cross-company quality scores from filings on disk. Missing inputs stay Unmeasured."""
from __future__ import annotations

from typing import Any, Mapping

from product.due_diligence.quality_rules import _snap, _tables


def _n(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _point(ok: bool | None) -> tuple[int | None, str]:
    if ok is None:
        return None, "unmeasured"
    return (1 if ok else 0), ("pass" if ok else "fail")


def piotroski_f_score(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Classic 9-signal F-score. Fewer than 6 measurable signals → Unmeasured, not a fake grade."""
    tables = _tables(raw)
    roa = _snap(tables, "key_ratios", ("roa", "return on assets"), kind="rate")
    cfo = _snap(tables, "cash_flow", ("cash from operating",))
    pat = _snap(tables, "profit_loss", ("net profit", "profit after tax"))
    if pat.get("current") is None:
        pat = _snap(tables, "quarterly_results", ("net profit",))
    debt = _snap(tables, "balance_sheet", ("borrowings",))
    current_assets = _snap(tables, "balance_sheet", ("total current assets", "current assets"))
    current_liab = _snap(tables, "balance_sheet", ("total current liabilities", "current liabilities"))
    sales = _snap(tables, "profit_loss", ("sales", "revenue"))
    if sales.get("current") is None:
        sales = _snap(tables, "quarterly_results", ("sales", "revenue"))
    assets = _snap(tables, "balance_sheet", ("total assets",))
    margin = _snap(tables, "quarterly_results", ("opm", "operating profit margin"), kind="rate")
    equity = _snap(tables, "balance_sheet", ("equity share capital", "share capital"))

    def pos(snap: Mapping[str, Any]) -> bool | None:
        cur = _n(snap.get("current"))
        return None if cur is None else cur > 0

    def improved(snap: Mapping[str, Any]) -> bool | None:
        cur, prev = _n(snap.get("current")), _n(snap.get("previous"))
        if cur is None or prev is None:
            return None
        return cur > prev

    def cr(snap_a: Mapping[str, Any], snap_l: Mapping[str, Any], which: str) -> float | None:
        a = _n(snap_a.get(which))
        b = _n(snap_l.get(which))
        if a is None or b in (None, 0):
            return None
        return a / b

    cr_now = cr(current_assets, current_liab, "current")
    cr_prev = cr(current_assets, current_liab, "previous")
    at_now = None
    at_prev = None
    sales_now, sales_prev = _n(sales.get("current")), _n(sales.get("previous"))
    assets_now, assets_prev = _n(assets.get("current")), _n(assets.get("previous"))
    if sales_now is not None and assets_now not in (None, 0):
        at_now = sales_now / assets_now
    if sales_prev is not None and assets_prev not in (None, 0):
        at_prev = sales_prev / assets_prev

    checks = [
        ("roa_positive", "ROA > 0", pos(roa)),
        ("cfo_positive", "CFO > 0", pos(cfo)),
        ("roa_up", "ROA improved", improved(roa)),
        (
            "accrual",
            "CFO > PAT",
            None if _n(cfo.get("current")) is None or _n(pat.get("current")) is None
            else _n(cfo.get("current")) > _n(pat.get("current")),
        ),
        (
            "leverage_down",
            "Borrowings down",
            None if _n(debt.get("current")) is None or _n(debt.get("previous")) is None
            else _n(debt.get("current")) < _n(debt.get("previous")),
        ),
        (
            "current_ratio_up",
            "Current ratio up",
            None if cr_now is None or cr_prev is None else cr_now > cr_prev,
        ),
        (
            "no_dilution",
            "Share capital not up",
            None if _n(equity.get("current")) is None or _n(equity.get("previous")) is None
            else _n(equity.get("current")) <= _n(equity.get("previous")) * 1.01,
        ),
        ("margin_up", "Operating margin up", improved(margin)),
        (
            "turnover_up",
            "Asset turnover up",
            None if at_now is None or at_prev is None else at_now > at_prev,
        ),
    ]
    scored: list[dict[str, Any]] = []
    total = 0
    measured = 0
    for sid, label, ok in checks:
        pts, state = _point(ok)
        if pts is not None:
            measured += 1
            total += pts
        scored.append({"id": sid, "label": label, "points": pts, "state": state})
    available = measured >= 6
    return {
        "id": "piotroski_f",
        "label": "Piotroski F-Score",
        "available": available,
        "score": total if available else None,
        "max": 9,
        "measured": measured,
        "label_text": f"{total}/9" if available else "Unmeasured",
        "detail": (
            f"{measured} of 9 signals were measurable from filings on file."
            if available else
            f"Only {measured} of 9 Piotroski signals are on file — Unmeasured, not a weak company."
        ),
        "signals": scored,
    }


def altman_z_score(raw: Mapping[str, Any]) -> dict[str, Any]:
    tables = _tables(raw)
    wc = _snap(tables, "balance_sheet", ("net working capital", "working capital"))
    re = _snap(tables, "balance_sheet", ("reserves", "reserves and surplus"))
    ebit = _snap(tables, "profit_loss", ("operating profit", "ebit", "pbdit"))
    equity = _snap(tables, "balance_sheet", ("total equity", "shareholders funds", "net worth"))
    sales = _snap(tables, "profit_loss", ("sales", "revenue"))
    if sales.get("current") is None:
        sales = _snap(tables, "quarterly_results", ("sales", "revenue"))
    assets = _snap(tables, "balance_sheet", ("total assets",))
    liab = _snap(tables, "balance_sheet", ("total liabilities", "total debt", "borrowings"))
    ta = _n(assets.get("current"))
    missing = []
    parts = {
        "wc": _n(wc.get("current")),
        "re": _n(re.get("current")),
        "ebit": _n(ebit.get("current")),
        "equity": _n(equity.get("current")),
        "sales": _n(sales.get("current")),
        "liab": _n(liab.get("current")),
    }
    for key, val in parts.items():
        if val is None:
            missing.append(key)
    if ta in (None, 0) or missing:
        return {
            "id": "altman_z",
            "label": "Altman Z-Score",
            "available": False,
            "score": None,
            "label_text": "Unmeasured",
            "detail": "Altman Z needs working capital, reserves, EBIT, equity, sales and liabilities vs total assets. Missing: " + (", ".join(missing) or "total assets") + ".",
            "missing": missing,
        }
    z = (
        1.2 * (parts["wc"] or 0) / ta
        + 1.4 * (parts["re"] or 0) / ta
        + 3.3 * (parts["ebit"] or 0) / ta
        + 0.6 * (parts["equity"] or 0) / max(parts["liab"] or 0, 1e-9)
        + 1.0 * (parts["sales"] or 0) / ta
    )
    band = "distress" if z < 1.81 else "grey" if z < 2.99 else "safe"
    return {
        "id": "altman_z",
        "label": "Altman Z-Score",
        "available": True,
        "score": round(z, 2),
        "band": band,
        "label_text": f"{z:.2f} · {band}",
        "detail": "Original Altman Z from filings on file. Not a credit rating.",
        "missing": [],
    }


def beneish_m_score(raw: Mapping[str, Any]) -> dict[str, Any]:
    """8-variable Beneish M. Incomplete inputs stay Unmeasured — never a guessed fraud score."""
    tables = _tables(raw)
    needed = {
        "receivables": _snap(tables, "balance_sheet", ("trade receivables", "receivables")),
        "sales": _snap(tables, "profit_loss", ("sales", "revenue")),
        "gp": _snap(tables, "profit_loss", ("gross profit",)),
        "assets": _snap(tables, "balance_sheet", ("total assets",)),
        "ppe": _snap(tables, "balance_sheet", ("property plant", "fixed assets", "ppe")),
        "dep": _snap(tables, "profit_loss", ("depreciation",)),
        "sga": _snap(tables, "profit_loss", ("other expenses", "sga")),
        "liab": _snap(tables, "balance_sheet", ("total liabilities", "borrowings")),
    }
    missing = [key for key, snap in needed.items() if _n(snap.get("current")) is None or _n(snap.get("previous")) is None]
    if missing:
        return {
            "id": "beneish_m",
            "label": "Beneish M-Score",
            "available": False,
            "score": None,
            "label_text": "Unmeasured",
            "detail": "Beneish M needs year-over-year filings for eight variables. Missing: " + ", ".join(missing) + ".",
            "missing": missing,
        }
    # Formula omitted unless every DSRI/GMI/AQI/SGI/DEPI/SGAI/LVGI/TATA term is defined.
    return {
        "id": "beneish_m",
        "label": "Beneish M-Score",
        "available": False,
        "score": None,
        "label_text": "Unmeasured",
        "detail": "Year-over-year rows exist, but the eight Beneish ratios are not fully mapped on this filing layout. Unmeasured, not a clean bill of health.",
        "missing": ["ratio_map"],
    }


def dupont_roe(raw: Mapping[str, Any]) -> dict[str, Any]:
    tables = _tables(raw)
    pat = _snap(tables, "profit_loss", ("net profit", "profit after tax"))
    if pat.get("current") is None:
        pat = _snap(tables, "quarterly_results", ("net profit",))
    sales = _snap(tables, "profit_loss", ("sales", "revenue"))
    if sales.get("current") is None:
        sales = _snap(tables, "quarterly_results", ("sales", "revenue"))
    assets = _snap(tables, "balance_sheet", ("total assets",))
    equity = _snap(tables, "balance_sheet", ("total equity", "shareholders funds", "net worth"))
    npm = None
    at = None
    em = None
    if _n(pat.get("current")) is not None and _n(sales.get("current")) not in (None, 0):
        npm = _n(pat.get("current")) / _n(sales.get("current"))
    if _n(sales.get("current")) is not None and _n(assets.get("current")) not in (None, 0):
        at = _n(sales.get("current")) / _n(assets.get("current"))
    if _n(assets.get("current")) is not None and _n(equity.get("current")) not in (None, 0):
        em = _n(assets.get("current")) / _n(equity.get("current"))
    if npm is None or at is None or em is None:
        missing = [name for name, val in (("net_margin", npm), ("asset_turnover", at), ("equity_multiplier", em)) if val is None]
        return {
            "id": "dupont_roe",
            "label": "DuPont ROE",
            "available": False,
            "score": None,
            "label_text": "Unmeasured",
            "detail": "DuPont needs PAT, sales, assets and equity. Missing: " + ", ".join(missing) + ".",
            "missing": missing,
        }
    roe = npm * at * em
    return {
        "id": "dupont_roe",
        "label": "DuPont ROE",
        "available": True,
        "score": round(roe * 100.0, 2),
        "label_text": f"{roe * 100:.1f}%",
        "detail": f"Net margin {npm * 100:.1f}% × asset turnover {at:.2f}× × equity multiplier {em:.2f}×.",
        "components": {"net_margin": round(npm, 4), "asset_turnover": round(at, 4), "equity_multiplier": round(em, 4)},
        "missing": [],
    }


def generic_cross_company_scores(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Attach named scores only when filings support them. Never fabricate a Strong grade."""
    scores = [
        piotroski_f_score(raw),
        altman_z_score(raw),
        beneish_m_score(raw),
        dupont_roe(raw),
    ]
    available = [row for row in scores if row.get("available")]
    return {
        "available": bool(available),
        "scores": scores,
        "detail": (
            f"{len(available)} of 4 named quality scores could be computed from filings on file."
            if available else
            "Piotroski, Altman Z, Beneish M and DuPont stay Unmeasured until the required filing lines exist."
        ),
    }
