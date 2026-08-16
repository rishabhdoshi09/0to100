"""Buy thesis for one clicked name — why it is on the desk, with live layers.

Never invents prices, sales, or a book. Missing layers stay missing and are
fetched from the highest-grade source still allowed: Kite depth → NSE
quote-equity → Screener/Yahoo fundamentals resolver.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.research_levels import attach_research_levels
from product.stock_workspace import build_stock_workspace, clean_symbol


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        n = float(value)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _why_chosen(scan: Mapping[str, Any], long_row: Mapping[str, Any], tech: Mapping[str, Any]) -> list[str]:
    bullets: list[str] = []
    reasons = scan.get("reasons") or scan.get("why") or []
    if isinstance(reasons, str) and reasons.strip():
        bullets.append(reasons.strip())
    elif isinstance(reasons, list):
        bullets.extend(str(item).strip() for item in reasons[:4] if str(item).strip())
    signals = [str(s).replace("_", " ") for s in (scan.get("signals") or []) if s]
    if signals:
        bullets.append("Scanner tags: " + ", ".join(signals[:6]))
    grade = str(scan.get("breakout_grade") or "").upper()
    if grade:
        bullets.append(f"Breakout grade {grade}" + (
            f" · conviction {scan.get('breakout_conviction')}"
            if scan.get("breakout_conviction") not in (None, "") else ""
        ))
    vol = _f(scan.get("volume_ratio") or tech.get("volume_ratio"))
    if vol is not None:
        bullets.append(f"Volume {vol:.1f}× the 20-day average")
    rsi = _f(scan.get("rsi") or tech.get("rsi14"))
    if rsi is not None:
        bullets.append(f"RSI {rsi:.0f}")
    cls = str(long_row.get("classification") or "")
    if cls:
        cov = long_row.get("fundamental_coverage")
        cov_s = f" · coverage {round(float(cov) * 100)}%" if cov not in (None, "") else ""
        bullets.append(f"Long-term class {cls.replace('_', ' ')}{cov_s}")
    factors = list(long_row.get("quality_factors") or [])[:3]
    bullets.extend(str(f) for f in factors if f)
    timing = str(long_row.get("timing") or "")
    if timing:
        bullets.append(f"Timing: {timing.replace('_', ' ').title()}")
    trend = str(tech.get("trend_explanation") or tech.get("trend") or "")
    if trend:
        bullets.append(trend)
    if not bullets:
        bullets.append("On the desk from the latest scan — open layers below for the evidence.")
    seen: set[str] = set()
    out: list[str] = []
    for item in bullets:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    out = out[:7]
    out.append("Research candidate, not an order. Invalidation is the stop.")
    return out


def _sales_from_raw(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    data = dict((raw_record or {}).get("data") or {})
    rows = list(data.get("profit_loss") or [])
    series: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("") or row.get("row_label") or "").lower()
        if "sales" not in label and "revenue" not in label:
            continue
        for key, value in row.items():
            if key in ("", "row_label"):
                continue
            num = _f(str(value).replace(",", "").replace("₹", "").replace("%", ""))
            if num is None:
                continue
            series.append({"period": str(key), "sales_cr": num})
        break
    cagr = None
    if len(series) >= 2:
        start, end = series[0]["sales_cr"], series[-1]["sales_cr"]
        years = max(1, len(series) - 1)
        if start and start > 0 and end is not None and end >= 0:
            try:
                cagr = round(((end / start) ** (1.0 / years) - 1.0) * 100.0, 1)
            except Exception:
                cagr = None
    fetched = str((raw_record or {}).get("fetched_at") or "")
    return {
        "available": bool(series),
        "cagr_3y": cagr,
        "series": series[-6:],
        "source": "screener" if series else "",
        "as_of": fetched,
        "note": (
            "Annual sales from the company filings pack (Screener)."
            if series else
            "Sales history not in cache yet — fetch fills this from Screener / Yahoo."
        ),
    }


def _order_book(symbol: str) -> dict[str, Any]:
    try:
        from product.breakout_quality import enrich_optional_context
        kite_book = dict((enrich_optional_context(symbol) or {}).get("order_book") or {})
        if kite_book.get("status") not in (None, "", "unavailable"):
            kite_book["source"] = kite_book.get("source") or "kite"
            kite_book["available"] = True
            return kite_book
    except Exception:
        pass
    try:
        from data.nse_live import fetch_market_depth
        return fetch_market_depth(symbol)
    except Exception as exc:
        return {
            "available": False,
            "status": "unavailable",
            "note": f"Order book unavailable ({type(exc).__name__})",
            "source": "",
            "bids": [],
            "asks": [],
        }


def _plan(scan: Mapping[str, Any], long_row: Mapping[str, Any], tech: Mapping[str, Any]) -> dict[str, Any]:
    row = attach_research_levels({
        **dict(long_row or {}),
        **dict(scan or {}),
        "price": scan.get("price") or long_row.get("price") or tech.get("close"),
        "atr": tech.get("atr14") or scan.get("atr") or long_row.get("atr"),
        "atr_pct": tech.get("atr_pct") or scan.get("atr_pct"),
        "vol_pct": long_row.get("vol_pct") or scan.get("vol_pct"),
    })
    buy = _f(row.get("entry"))
    stop = _f(row.get("stop"))
    target = _f(row.get("target"))
    upside = _f(row.get("upside_from_buy_pct"))
    return {
        "buy": buy,
        "stop": stop,
        "target": target,
        "upside_from_buy_pct": upside,
        "levels_source": str(row.get("levels_source") or ""),
    }


def build_buy_thesis(symbol: str, *, fetch_missing: bool = False) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    fetched = {"fundamentals": False, "source": "", "message": ""}
    if fetch_missing:
        try:
            from fundamentals.lazy import ensure_deep_fundamentals
            data = ensure_deep_fundamentals(symbol, force_refresh=False)
            fetched = {
                "fundamentals": bool(data),
                "source": str((data or {}).get("_qt_source") or (data or {}).get("source") or "resolver"),
                "message": "Filled from Screener / Yahoo / cache" if data else "Resolver returned nothing",
            }
        except Exception as exc:
            fetched = {
                "fundamentals": False,
                "source": "",
                "message": f"Could not fetch filings: {type(exc).__name__}: {exc}",
            }

    workspace = build_stock_workspace(symbol)
    scan = dict(workspace.get("scanner") or {})
    long_row = dict(workspace.get("long_term") or {})
    tech = dict(workspace.get("technical") or {})
    fund = dict(workspace.get("fundamentals") or {})
    raw = {}
    try:
        from reporting.evidence_intake import load_raw_fundamentals
        raw = load_raw_fundamentals(symbol) or {}
    except Exception:
        raw = {}
    sales = _sales_from_raw(raw)
    if sales.get("cagr_3y") is None:
        for metric in fund.get("metrics") or []:
            if isinstance(metric, Mapping) and metric.get("key") == "sales_growth_3y" and metric.get("value") is not None:
                sales["cagr_3y"] = metric.get("value")
                sales["available"] = True
                break
    plan = _plan(scan, long_row, tech)
    fund_metrics = [
        m for m in (fund.get("metrics") or [])
        if isinstance(m, Mapping) and m.get("key") in {
            "market_cap", "pe", "roe", "roce", "sales_growth_3y",
            "profit_growth_3y", "debt_to_equity", "promoter_holding",
        }
    ]
    headline = (
        str(workspace.get("summary") or "")
        or "Clicked name — evidence layers below. Not a buy instruction."
    )
    return {
        "schema_version": 1,
        "symbol": symbol,
        "company": workspace.get("company") or symbol,
        "sector": workspace.get("sector") or "",
        "state": workspace.get("state"),
        "headline": headline,
        "why": _why_chosen(scan, long_row, tech),
        "plan": plan,
        "technical": {
            "available": bool(tech.get("available")),
            "close": tech.get("close"),
            "latest_date": tech.get("latest_date"),
            "trend": tech.get("trend"),
            "trend_explanation": tech.get("trend_explanation"),
            "rsi14": tech.get("rsi14"),
            "volume_ratio": tech.get("volume_ratio"),
            "from_high_pct": tech.get("from_high_pct"),
        },
        "fundamentals": {
            "available": bool(fund.get("available")),
            "coverage_pct": fund.get("coverage_pct") or 0,
            "classification": fund.get("classification") or long_row.get("classification") or "",
            "quality_factors": fund.get("quality_factors") or [],
            "risk_flags": fund.get("risk_flags") or [],
            "metrics": fund_metrics,
            "fetched_at": fund.get("fetched_at") or "",
            "about": (fund.get("company_about") or "")[:400],
        },
        "sales": sales,
        "order_book": _order_book(symbol),
        "gaps": workspace.get("gaps") or [],
        "fetched": fetched,
        "confidence_pct": workspace.get("confidence_pct"),
    }
