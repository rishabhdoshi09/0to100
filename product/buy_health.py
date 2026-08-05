"""Technical health for Active Buys — warnings only, never a sell ticket.

Evaluates each user-added buy against:
  • EMA 20 / 50 / 200 stack (from official bhav history)
  • Major swing support (20d / 60d lows)
  • Volume dump / distribution day
  • Optional user stop breach
  • Optional entry drawdown

Missing history stays missing. Live LTP overlays when available; charts stay EOD.
"""
from __future__ import annotations

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


def evaluate_symbol(
    symbol: str,
    *,
    entry_price: float | None = None,
    stop_price: float | None = None,
    live_price: float | None = None,
) -> dict[str, Any]:
    """Return health snapshot for one symbol from bhav + optional live print."""
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
        "vs_entry_pct": None,
        "honesty": "Research warning only — not a sell order.",
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

    out.update(
        {
            "available": True,
            "severity": severity,
            "status_label": label,
            "price": round(px, 2),
            "eod_close": round(eod, 2),
            "live_price": round(float(live_price), 2) if live_price else None,
            "price_source": price_source,
            "as_of": latest_date,
            "warnings": warnings,
            "risk_score": min(100, score),
            "supports": {
                "swing_20d": support_20,
                "swing_60d": support_60,
            },
            "averages": {
                "ema20": ema20,
                "ema50": ema50,
                "ema200": ema200,
            },
            "structure": {
                "chg_1d_pct": chg_1d,
                "chg_5d_pct": chg_5d,
                "volume_ratio": vol_ratio,
                "notes": structure_notes,
            },
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
