"""Health for Active Buys — technicals + fundamentals, never a sell ticket.

Technicals (official bhav):
  • EMA 20 / 50 / 200 stack
  • Major swing support (20d / 60d lows)
  • Volume dump / distribution day
  • Optional user stop breach / entry drawdown

Fundamentals (Screener cache only — no invent, no live scrape on every refresh):
  • Key ratios (P/E, ROE, debt when present)
  • Sales / profit trend flags from cached profit-loss table
  • Missing cache stays MISSING (open Stock Intelligence to fetch)

Live LTP overlays when available; charts stay EOD. Never places orders.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np


SEVERITY_RANK = {"critical": 0, "warn": 1, "info": 2, "good": 3, "unknown": 4}

# Short TTL so Active Buys page + App heartbeat don't thrash bhav on every nav.
_EVAL_CACHE: dict[str, Any] = {"ts": 0.0, "key": "", "payload": None}
_EVAL_CACHE_TTL_S = 12.0


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _ema(series: np.ndarray, span: int) -> float | None:
    if series is None or len(series) < span:
        return None
    alpha = 2.0 / (span + 1.0)
    value = float(series[0])
    for price in series[1:]:
        value = alpha * float(price) + (1.0 - alpha) * value
    return round(value, 2)


def _swing_support(lows: np.ndarray, lookback: int) -> float | None:
    if lows is None or len(lows) < max(5, lookback // 2):
        return None
    window = lows[-min(len(lows), lookback) :]
    return round(float(np.nanmin(window)), 2)


def _ratio_map(key_ratios: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in key_ratios or []:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("label") or "").strip()
        value = row.get("value")
        if name and value not in (None, ""):
            out[name.lower()] = str(value).strip()
    return out


def _parse_number(text: Any) -> float | None:
    try:
        if text in (None, ""):
            return None
        cleaned = re.sub(r"[^\d.\-]", "", str(text).replace(",", ""))
        if cleaned in {"", "-", "."}:
            return None
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _find_ratio(ratios: Mapping[str, str], *needles: str) -> float | None:
    for key, value in ratios.items():
        if any(n in key for n in needles):
            return _parse_number(value)
    return None


def _pl_series(table: Any, *needles: str) -> list[float]:
    for row in table or []:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("row_label") or row.get("label") or "").strip().lower()
        if not any(n in label for n in needles):
            continue
        values: list[float] = []
        for key, raw in row.items():
            if str(key).lower() in {"", "row_label", "label", "particulars", "particular"}:
                continue
            num = _parse_number(raw)
            if num is not None:
                values.append(num)
        return values
    return []


def fundamentals_snapshot(symbol: str) -> dict[str, Any]:
    """Cache-only fundamental lens for Active Buys. Missing stays missing."""
    sym = str(symbol or "").strip().upper()
    out: dict[str, Any] = {
        "available": False,
        "status": "MISSING",
        "severity": "unknown",
        "ratios": {},
        "flags": [],
        "about": "",
        "fetched_at": "",
        "freshness": "MISSING",
        "note": "Fundamentals not in Screener cache — open Stock Intelligence → Retry fundamentals.",
    }
    try:
        from reporting.evidence_intake import load_raw_fundamentals

        raw = load_raw_fundamentals(sym, auto_fetch=False)
    except Exception as exc:
        out["note"] = f"Fundamentals unavailable ({exc})"
        return out
    if not raw.get("available"):
        return out
    data = dict(raw.get("data") or {})
    ratios = _ratio_map(data.get("key_ratios"))
    pe = _find_ratio(ratios, "stock p/e", "p/e", "pe")
    roe = _find_ratio(ratios, "roe", "return on equity")
    roce = _find_ratio(ratios, "roce")
    debt = _find_ratio(ratios, "debt to equity", "debt/equity", "d/e")
    sales_growth = _find_ratio(ratios, "sales growth", "revenue growth")
    profit_growth = _find_ratio(ratios, "profit growth", "np growth")

    sales = _pl_series(data.get("profit_loss"), "sales", "revenue")
    profits = _pl_series(data.get("profit_loss"), "net profit", "profit after tax", "pat")
    sales_yoy = None
    profit_yoy = None
    if len(sales) >= 2 and sales[-2] not in (0, None):
        sales_yoy = round((sales[-1] / sales[-2] - 1.0) * 100.0, 1)
    if len(profits) >= 2 and profits[-2] not in (0, None):
        profit_yoy = round((profits[-1] / profits[-2] - 1.0) * 100.0, 1)

    flags: list[dict[str, str]] = []
    score = 0

    def flag(sev: str, code: str, text: str) -> None:
        nonlocal score
        flags.append({"severity": sev, "code": code, "text": text})
        score += {"critical": 30, "warn": 15, "info": 6, "good": 0}.get(sev, 0)

    if pe is not None and pe < 0:
        flag("warn", "NEG_PE", f"P/E is negative ({pe:.1f}) — earnings currently not supporting the price.")
    elif pe is not None and pe >= 80:
        flag("warn", "RICH_PE", f"P/E looks rich at {pe:.1f}x — valuation risk if growth disappoints.")
    if debt is not None and debt >= 1.5:
        flag("warn", "HIGH_DEBT", f"Debt/equity {debt:.2f} — balance-sheet leverage is elevated.")
    if roe is not None and roe < 8:
        flag("info", "LOW_ROE", f"ROE {roe:.1f}% — capital efficiency looks modest on cached figures.")
    elif roe is not None and roe >= 18:
        flag("good", "SOLID_ROE", f"ROE {roe:.1f}% — quality signal from cached fundamentals.")
    yoy_sales = sales_growth if sales_growth is not None else sales_yoy
    yoy_profit = profit_growth if profit_growth is not None else profit_yoy
    if yoy_sales is not None and yoy_sales <= -10:
        flag("warn", "SALES_SHRINK", f"Sales growth about {yoy_sales:+.1f}% — top-line is contracting.")
    elif yoy_sales is not None and yoy_sales >= 15:
        flag("good", "SALES_GROWTH", f"Sales growth about {yoy_sales:+.1f}% on cached periods.")
    if yoy_profit is not None and yoy_profit <= -15:
        flag("warn", "PROFIT_SHRINK", f"Profit growth about {yoy_profit:+.1f}% — earnings momentum is weak.")
    elif yoy_profit is not None and yoy_profit >= 15:
        flag("good", "PROFIT_GROWTH", f"Profit growth about {yoy_profit:+.1f}% on cached periods.")

    if any(f["severity"] == "critical" for f in flags):
        severity = "critical"
    elif any(f["severity"] == "warn" for f in flags):
        severity = "warn"
    elif any(f["severity"] == "good" for f in flags) and not any(f["severity"] in {"warn", "critical"} for f in flags):
        severity = "good"
    elif flags:
        severity = "info"
    else:
        severity = "info"
        flags.append(
            {
                "severity": "info",
                "code": "FUND_PRESENT",
                "text": "Fundamentals cache present — no automatic red flags from available ratios.",
            }
        )

    about = str(data.get("about") or "").strip()
    return {
        "available": True,
        "status": str(raw.get("freshness") or "UNKNOWN"),
        "severity": severity,
        "risk_score": min(100, score),
        "ratios": {
            "pe": pe,
            "roe": roe,
            "roce": roce,
            "debt_to_equity": debt,
            "sales_growth_pct": yoy_sales,
            "profit_growth_pct": yoy_profit,
        },
        "flags": flags,
        "about": about[:280],
        "fetched_at": str(raw.get("fetched_at") or ""),
        "freshness": str(raw.get("freshness") or ""),
        "note": "From Screener cache only — missing ratios stay blank; not a valuation call.",
    }


def evaluate_symbol(
    symbol: str,
    *,
    entry_price: float | None = None,
    stop_price: float | None = None,
    live_price: float | None = None,
) -> dict[str, Any]:
    """Return technical + fundamental health for one Active Buy symbol."""
    sym = str(symbol or "").strip().upper()
    out: dict[str, Any] = {
        "symbol": sym,
        "available": False,
        "severity": "unknown",
        "status_label": "INCOMPLETE",
        "price": None,
        "eod_close": None,
        "live_price": live_price,
        "price_source": "unavailable",
        "warnings": [],
        "supports": {},
        "averages": {},
        "structure": {},
        "technicals": {"available": False},
        "fundamentals": fundamentals_snapshot(sym),
        "vs_entry_pct": None,
        "honesty": "Technicals + fundamentals research warning only — not a sell order.",
    }
    try:
        from data.bhavcopy_runtime import get_ohlcv
    except Exception:
        try:
            from data.bhavcopy_store import get_ohlcv
        except Exception as exc:
            out["warnings"].append(f"Price history unavailable ({exc})")
            return out

    try:
        frame = get_ohlcv(sym)
    except Exception as exc:
        out["warnings"].append(f"Could not load history ({exc})")
        return out
    if frame is None or len(frame) < 30:
        out["warnings"].append("Not enough official daily history for a reliable check (need ~30+ sessions).")
        return out

    data = frame.sort_index()
    close = data["close"].astype(float).values
    high = data["high"].astype(float).values if "high" in data.columns else close
    low = data["low"].astype(float).values if "low" in data.columns else close
    vol = data["volume"].astype(float).values if "volume" in data.columns else np.zeros(len(close))

    eod = float(close[-1])
    px = float(live_price) if live_price and live_price > 0 else eod
    price_source = "live" if live_price and live_price > 0 else "eod"

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    support_20 = _swing_support(low, 20)
    support_60 = _swing_support(low, 60)
    resistance_20 = float(np.nanmax(high[-20:])) if len(high) >= 20 else None
    resistance_60 = float(np.nanmax(high[-60:])) if len(high) >= 60 else None

    avg_vol20 = float(np.nanmean(vol[-21:-1])) if len(vol) >= 22 and np.nanmean(vol[-21:-1]) > 0 else None
    vol_ratio = round(float(vol[-1]) / avg_vol20, 2) if avg_vol20 else None
    chg_1d = round((close[-1] / close[-2] - 1.0) * 100.0, 2) if len(close) >= 2 else None
    chg_5d = round((close[-1] / close[-6] - 1.0) * 100.0, 2) if len(close) >= 6 else None

    # Structure: lower highs over last ~3 swings (simple 10-bar peaks)
    structure_notes: list[str] = []
    peaks: list[float] = []
    for i in range(10, len(high) - 1):
        if high[i] >= high[i - 1] and high[i] >= high[i + 1] and high[i] >= float(np.max(high[i - 5 : i + 1])):
            peaks.append(float(high[i]))
    if len(peaks) >= 3 and peaks[-1] < peaks[-2] < peaks[-3]:
        structure_notes.append("Lower highs forming — supply is stacking")

    warnings: list[dict[str, str]] = []
    score = 0  # higher = worse

    def warn(sev: str, code: str, text: str) -> None:
        nonlocal score
        warnings.append({"severity": sev, "code": code, "text": text})
        score += {"critical": 40, "warn": 20, "info": 8}.get(sev, 0)

    # Moving averages
    if ema20 is not None and px < ema20:
        gap = (ema20 - px) / ema20 * 100.0
        sev = "warn" if gap >= 1.0 else "info"
        warn(sev, "BELOW_EMA20", f"Price is below the 20-day average (₹{ema20:,.1f}) by {gap:.1f}%. Short-term trend is weakening.")
    if ema50 is not None and px < ema50:
        gap = (ema50 - px) / ema50 * 100.0
        sev = "critical" if gap >= 3.0 else "warn"
        warn(sev, "BELOW_EMA50", f"Price is below the 50-day average (₹{ema50:,.1f}) by {gap:.1f}%. Medium-term trend is damaged.")
    if ema200 is not None and px < ema200:
        gap = (ema200 - px) / ema200 * 100.0
        sev = "critical" if gap >= 2.0 else "warn"
        warn(sev, "BELOW_EMA200", f"Price is below the 200-day average (₹{ema200:,.1f}) by {gap:.1f}%. Long-term support is broken.")
    if ema20 and ema50 and ema200 and px < ema20 < ema50 < ema200:
        warn("critical", "DEATH_STACK", "Price and averages are in a full downtrend stack (price < 20 < 50 < 200).")

    # Major support breaks
    if support_20 is not None and px < support_20 * 0.998:
        gap = (support_20 - px) / support_20 * 100.0
        sev = "critical" if (vol_ratio or 0) >= 1.5 or gap >= 1.5 else "warn"
        warn(sev, "SUPPORT_20D", f"Price broke the 20-session low support near ₹{support_20:,.1f} ({gap:.1f}% below).")
    if support_60 is not None and px < support_60 * 0.998:
        gap = (support_60 - px) / support_60 * 100.0
        warn("critical", "SUPPORT_60D", f"Price broke the 60-session low support near ₹{support_60:,.1f} ({gap:.1f}% below). Major swing support lost.")

    # Volume dump on a down day
    if chg_1d is not None and chg_1d <= -1.5 and vol_ratio is not None and vol_ratio >= 1.8:
        warn("warn", "VOLUME_DUMP", f"Down day {chg_1d:+.1f}% on {vol_ratio:.1f}× average volume — distribution / selling pressure.")
    elif chg_1d is not None and chg_1d <= -3.0:
        warn("warn", "HARD_DOWN_DAY", f"Hard down day {chg_1d:+.1f}% — check whether support still holds.")

    if chg_5d is not None and chg_5d <= -8.0:
        warn("warn", "WEAK_5D", f"Down {chg_5d:+.1f}% over 5 sessions — momentum is clearly negative.")

    for note in structure_notes:
        warn("info", "LOWER_HIGHS", note)

    # User levels
    if stop_price and stop_price > 0 and px < stop_price:
        warn("critical", "STOP_BREACH", f"Live/EOD price ₹{px:,.1f} is below your stop ₹{stop_price:,.1f}. Plan is broken — review the position.")
    if entry_price and entry_price > 0:
        vs = round((px / entry_price - 1.0) * 100.0, 2)
        out["vs_entry_pct"] = vs
        if vs <= -8.0:
            warn("critical", "ENTRY_DRAWDOWN", f"About {vs:.1f}% below your entry ₹{entry_price:,.1f}.")
        elif vs <= -4.0:
            warn("warn", "ENTRY_DRAWDOWN", f"About {vs:.1f}% below your entry ₹{entry_price:,.1f}.")

    # Overall severity
    if any(w["severity"] == "critical" for w in warnings):
        severity, label = "critical", "AT RISK"
    elif any(w["severity"] == "warn" for w in warnings):
        severity, label = "warn", "WEAKENING"
    elif warnings:
        severity, label = "info", "WATCH"
    else:
        severity, label = "good", "HEALTHY"
        warnings.append(
            {
                "severity": "good",
                "code": "STACK_OK",
                "text": "Price is holding above key averages and recent swing supports on available data.",
            }
        )

    latest_index = data.index[-1]
    latest_date = str(getattr(latest_index, "date", lambda: latest_index)())

    technicals = {
        "available": True,
        "severity": severity,
        "status_label": label,
        "risk_score": min(100, score),
        "averages": {
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
        },
        "supports": {
            "swing_20d": support_20,
            "swing_60d": support_60,
        },
        "resistances": {
            "swing_20d": round(resistance_20, 2) if resistance_20 else None,
            "swing_60d": round(resistance_60, 2) if resistance_60 else None,
        },
        "structure": {
            "chg_1d_pct": chg_1d,
            "chg_5d_pct": chg_5d,
            "volume_ratio": vol_ratio,
            "notes": structure_notes,
        },
        "warnings": warnings,
        "as_of": latest_date,
        "note": "Official daily history — charts stay EOD even when LTP overlays.",
    }
    fundamentals = out.get("fundamentals") or fundamentals_snapshot(sym)
    # Blend: technicals drive primary label; fund warn/critical escalate one notch.
    fund_sev = str(fundamentals.get("severity") or "unknown")
    blend = severity
    if severity == "good" and fund_sev == "warn":
        blend, label = "warn", "WEAKENING"
    elif severity in {"good", "info"} and fund_sev == "critical":
        blend, label = "critical", "AT RISK"
    elif severity == "info" and fund_sev == "warn":
        blend, label = "warn", "WEAKENING"

    out.update(
        {
            "available": True,
            "severity": blend,
            "status_label": label,
            "price": round(px, 2),
            "eod_close": round(eod, 2),
            "live_price": round(float(live_price), 2) if live_price else None,
            "price_source": price_source,
            "as_of": latest_date,
            "warnings": warnings,
            "risk_score": min(100, score + int(fundamentals.get("risk_score") or 0) // 2),
            "supports": technicals["supports"],
            "resistances": technicals.get("resistances") or {},
            "averages": technicals["averages"],
            "structure": technicals["structure"],
            "technicals": technicals,
            "fundamentals": fundamentals,
        }
    )
    return out


def evaluate_book(
    items: list[Mapping[str, Any]] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate all active buy-book rows. Returns sorted unhealthy-first list.

    Results are cached briefly so nav/heartbeat does not re-scan bhav every click.
    Pass force=True after add/remove or when the UI taps Refresh.
    """
    from product.buy_book import list_active, empty_book
    import time

    rows = list(items) if items is not None else list_active()
    cache_key = "|".join(
        f"{r.get('id')}:{r.get('symbol')}:{r.get('entry_price')}:{r.get('stop_price')}:{r.get('quantity')}:{r.get('updated_at')}"
        for r in rows
    )
    now = time.monotonic()
    cached = _EVAL_CACHE.get("payload")
    if (
        not force
        and cached is not None
        and _EVAL_CACHE.get("key") == cache_key
        and (now - float(_EVAL_CACHE.get("ts") or 0.0)) < _EVAL_CACHE_TTL_S
    ):
        return cached

    symbols = sorted({str(r.get("symbol") or "").upper() for r in rows if r.get("symbol")})
    live: dict[str, dict] = {}
    if symbols:
        try:
            from data.live_quotes import get_live_quotes

            live = get_live_quotes(symbols, ttl=8.0) or {}
        except Exception:
            live = {}

    evaluated: list[dict[str, Any]] = []
    for row in rows:
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        ltp = _f((live.get(sym) or {}).get("price"))
        health = evaluate_symbol(
            sym,
            entry_price=_f(row.get("entry_price")),
            stop_price=_f(row.get("stop_price")),
            live_price=ltp,
        )
        structure = health.get("structure") or {}
        entry = _f(row.get("entry_price"))
        qty = _f(row.get("quantity"))
        price = _f(health.get("price"))
        vs_entry = _f(health.get("vs_entry_pct"))
        est_pnl = None
        if entry and entry > 0 and qty and qty > 0 and price and price > 0:
            est_pnl = round((price - entry) * qty, 2)
        fundamentals = health.get("fundamentals") or {}
        technicals = health.get("technicals") or {}
        evaluated.append(
            {
                **dict(row),
                "health": health,
                "severity": health.get("severity"),
                "status_label": health.get("status_label"),
                "price": health.get("price"),
                "vs_entry_pct": health.get("vs_entry_pct"),
                "chg_1d_pct": structure.get("chg_1d_pct"),
                "chg_5d_pct": structure.get("chg_5d_pct"),
                "est_pnl": est_pnl,
                "result_label": _result_label(vs_entry, has_entry=entry is not None and entry > 0),
                "technicals": technicals,
                "fundamentals": fundamentals,
                "tech_label": technicals.get("status_label") or health.get("status_label"),
                "fund_label": (
                    "MISSING"
                    if not fundamentals.get("available")
                    else str(fundamentals.get("severity") or "unknown").upper()
                ),
            }
        )

    evaluated.sort(
        key=lambda r: (
            SEVERITY_RANK.get(str(r.get("severity")), 9),
            -float(r.get("health", {}).get("risk_score") or 0),
            str(r.get("symbol") or ""),
        )
    )
    summary = {
        "total": len(evaluated),
        "critical": sum(1 for r in evaluated if r.get("severity") == "critical"),
        "warn": sum(1 for r in evaluated if r.get("severity") == "warn"),
        "info": sum(1 for r in evaluated if r.get("severity") == "info"),
        "good": sum(1 for r in evaluated if r.get("severity") == "good"),
        "unknown": sum(1 for r in evaluated if r.get("severity") == "unknown"),
    }
    results = _results_summary(evaluated)
    base = empty_book()
    payload = {
        "available": True,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
        "items": evaluated,
        "places_orders": False,
        "live_locked": True,
        "honesty": base["honesty"],
        "cached": False,
    }
    _EVAL_CACHE["ts"] = now
    _EVAL_CACHE["key"] = cache_key
    _EVAL_CACHE["payload"] = {**payload, "cached": True}
    return payload


def invalidate_eval_cache() -> None:
    _EVAL_CACHE["ts"] = 0.0
    _EVAL_CACHE["key"] = ""
    _EVAL_CACHE["payload"] = None


def _result_label(vs_entry: float | None, *, has_entry: bool) -> str:
    if not has_entry or vs_entry is None:
        return "NO ENTRY"
    if vs_entry >= 0.5:
        return "UP"
    if vs_entry <= -0.5:
        return "DOWN"
    return "FLAT"


def _results_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Book-level stock results from user entry vs current print — never invents fills."""
    with_entry = [r for r in rows if _f(r.get("vs_entry_pct")) is not None]
    missing_entry = len(rows) - len(with_entry)
    ups = sum(1 for r in with_entry if float(r["vs_entry_pct"]) >= 0.5)
    downs = sum(1 for r in with_entry if float(r["vs_entry_pct"]) <= -0.5)
    flats = len(with_entry) - ups - downs
    avg_vs = None
    if with_entry:
        avg_vs = round(sum(float(r["vs_entry_pct"]) for r in with_entry) / len(with_entry), 2)
    pnl_rows = [r for r in rows if _f(r.get("est_pnl")) is not None]
    est_pnl_total = round(sum(float(r["est_pnl"]) for r in pnl_rows), 2) if pnl_rows else None
    return {
        "with_entry": len(with_entry),
        "missing_entry": missing_entry,
        "up": ups,
        "down": downs,
        "flat": flats,
        "avg_vs_entry_pct": avg_vs,
        "est_pnl_total": est_pnl_total,
        "honesty": (
            "Results = your entry vs live LTP or EOD close. "
            "Qty rupee P&L is an estimate you typed — not broker demat truth. "
            "Missing entry stays missing."
        ),
    }


def refresh_book_research(
    symbols: list[str] | None = None,
    *,
    force_fundamentals: bool = False,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Fetch Screener fundamentals + warm official technical history for Active Buys.

    Opt-in from UI (sync checkbox or Fetch fund+tech). Never invents ratios/prices.
    Low-power caps the batch so old Macs are not hammered by Screener.
    """
    import os

    from product.buy_book import list_active

    if symbols is None:
        wanted = sorted({str(r.get("symbol") or "").upper() for r in list_active() if r.get("symbol")})
    else:
        wanted = sorted({str(s or "").strip().upper() for s in symbols if str(s or "").strip()})

    low_power = str(os.getenv("QT_LOW_POWER", "") or "").strip().lower() in {"1", "true", "yes"}
    if max_symbols is None:
        max_symbols = 12 if low_power else 40
    capped = wanted[: max(1, int(max_symbols))]
    skipped = wanted[len(capped) :]

    # Warm official bhav into this process so technicals can score.
    bhav_status: dict[str, Any] = {}
    try:
        from data.bhavcopy_runtime import ensure_loaded, get_ohlcv

        bhav_status = ensure_loaded(rebuild_from_local=True)
    except Exception as exc:
        get_ohlcv = None  # type: ignore[assignment]
        bhav_status = {"ready": False, "error": str(exc)}

    fund_ok = 0
    fund_fail = 0
    fund_cached = 0
    tech_ok = 0
    tech_thin = 0
    rows_out: list[dict[str, Any]] = []

    for sym in capped:
        row: dict[str, Any] = {
            "symbol": sym,
            "fundamentals": {"ok": False, "source": "", "message": ""},
            "technicals": {"ok": False, "bars": 0, "message": ""},
        }
        # ── Fundamentals (Screener lazy fetch) ─────────────────────────────
        try:
            from fundamentals.lazy import ensure_deep_fundamentals
            from fundamentals.cache import FundamentalsCache

            cache = FundamentalsCache()
            had_cache = bool(cache.has(sym)) and not force_fundamentals
            data = ensure_deep_fundamentals(sym, force_refresh=bool(force_fundamentals))
            ok = bool(data)
            row["fundamentals"] = {
                "ok": ok,
                "source": "cache" if had_cache else "screener",
                "message": "Fundamentals ready" if ok else "Empty fundamentals payload",
            }
            if ok and had_cache:
                fund_cached += 1
                fund_ok += 1
            elif ok:
                fund_ok += 1
            else:
                fund_fail += 1
        except Exception as exc:
            fund_fail += 1
            row["fundamentals"] = {
                "ok": False,
                "source": "error",
                "message": str(exc)[:160],
            }

        # ── Technicals (official daily history warm + length check) ────────
        bars = 0
        try:
            if get_ohlcv is not None:
                frame = get_ohlcv(sym)
                bars = int(len(frame)) if frame is not None else 0
            if bars >= 30:
                tech_ok += 1
                row["technicals"] = {
                    "ok": True,
                    "bars": bars,
                    "message": f"{bars} official daily bars ready",
                }
            else:
                tech_thin += 1
                row["technicals"] = {
                    "ok": False,
                    "bars": bars,
                    "message": (
                        f"Only {bars} bar(s) — need ~30+ sessions in bhav store"
                        if bars
                        else "No official daily history in bhav store yet"
                    ),
                }
        except Exception as exc:
            tech_thin += 1
            row["technicals"] = {"ok": False, "bars": 0, "message": str(exc)[:160]}

        rows_out.append(row)

    invalidate_eval_cache()
    return {
        "accepted": True,
        "requested": len(wanted),
        "processed": len(capped),
        "skipped": skipped,
        "force_fundamentals": bool(force_fundamentals),
        "low_power": low_power,
        "bhav": bhav_status,
        "fundamentals": {
            "ok": fund_ok,
            "cached": fund_cached,
            "failed": fund_fail,
        },
        "technicals": {
            "ok": tech_ok,
            "thin_or_missing": tech_thin,
        },
        "rows": rows_out,
        "places_orders": False,
        "honesty": (
            "Fetched Screener fundamentals (cache-first) and warmed official bhav technicals "
            "for Active Buys. Missing stays missing — never invents PE or prices. Paper-first."
        ),
        "message": (
            f"Research refresh: fund ok {fund_ok}/{len(capped)} "
            f"(cached {fund_cached}, failed {fund_fail}) · "
            f"tech ready {tech_ok}/{len(capped)}"
            + (f" · skipped {len(skipped)} (cap {max_symbols})" if skipped else "")
        ),
    }
