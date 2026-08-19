"""Market-monitor context from official NSE history — no Yahoo, no scrape.

Stage (Weinstein/Minervini 1–4) from the same SMA stack SEPA already computed.
Relative strength vs Nifty 50 from the official index store vs the stock's
bhavcopy close. Index strip (Nifty / Bank Nifty / VIX) is the same store.

Missing history stays missing. Not a measured QuantTerm edge.
"""
from __future__ import annotations

from typing import Any, Mapping

LOOKBACK = 63  # ~3 months of sessions
INDEX_STRIP_NOTE = (
    "Nifty 50, Bank Nifty, and India VIX from the official NSE index store — "
    "the same history used for regime and the gauntlet, not a live scrape."
)
RS_NOTE = (
    "Relative strength is the stock's ~63-session return minus Nifty 50, "
    "in percentage points, from official OHLCV. Research context, not a buy."
)


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _close_series(frame: Any):
    if frame is None or len(frame) == 0:
        return None
    data = frame
    try:
        data = frame.sort_index()
    except Exception:
        pass
    for col in ("close", "Close"):
        if col in data.columns:
            try:
                s = data[col].astype(float).dropna()
            except Exception:
                return None
            return s if len(s) else None
    return None


def period_return(close, lookback: int = LOOKBACK) -> float | None:
    if close is None or len(close) < lookback + 1:
        return None
    start = float(close.iloc[-lookback - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        return None
    return (end / start) - 1.0


def rs_vs_benchmark(stock_frame: Any, bench_frame: Any, *, lookback: int = LOOKBACK) -> dict[str, Any]:
    """Stock 63-session return minus Nifty, in percentage points."""
    stock_ret = period_return(_close_series(stock_frame), lookback)
    bench_ret = period_return(_close_series(bench_frame), lookback)
    if stock_ret is None or bench_ret is None:
        return {
            "available": False,
            "lookback": lookback,
            "stock_pct": None,
            "benchmark_pct": None,
            "excess_pp": None,
            "label": "UNKNOWN",
            "note": "Need ~3 months of official stock and Nifty history.",
        }
    excess = (stock_ret - bench_ret) * 100.0
    if excess >= 5:
        label, note = "LEADER", "Outperforming Nifty 50 over ~3 months."
    elif excess <= -5:
        label, note = "LAGGARD", "Underperforming Nifty 50 over ~3 months."
    else:
        label, note = "IN LINE", "Moving with Nifty 50 over ~3 months."
    return {
        "available": True,
        "lookback": lookback,
        "stock_pct": round(stock_ret * 100.0, 1),
        "benchmark_pct": round(bench_ret * 100.0, 1),
        "excess_pp": round(excess, 1),
        "label": label,
        "note": note,
        "benchmark": "Nifty 50",
        "source": "nse_index_store",
    }


def classify_stage(
    levels: Mapping[str, Any] | None,
    *,
    sma200_rising: bool | None,
) -> dict[str, Any]:
    """Weinstein/Minervini stage from price vs 50/200 DMA. Unknown if SMAs missing."""
    levels = dict(levels or {})
    price = _f(levels.get("price"))
    sma50 = _f(levels.get("sma50"))
    sma200 = _f(levels.get("sma200"))
    if price is None or sma50 is None or sma200 is None:
        return {
            "id": "unknown",
            "label": "STAGE ?",
            "note": "Need 200 sessions before a stage call is honest.",
        }
    stacked = price > sma50 > sma200
    below = price < sma50 < sma200
    if stacked and sma200_rising is True:
        return {
            "id": "stage_2",
            "label": "STAGE 2",
            "note": "Price above 50-DMA above 200-DMA, and the 200-DMA is rising.",
        }
    if stacked:
        return {
            "id": "stage_2_early",
            "label": "STAGE 2?",
            "note": "Averages are stacked, but the 200-DMA slope is unknown or not rising yet.",
        }
    if below and sma200_rising is False:
        return {
            "id": "stage_4",
            "label": "STAGE 4",
            "note": "Price below 50-DMA below a falling 200-DMA — decline, not a swing long.",
        }
    if price > sma200 and price < sma50:
        return {
            "id": "stage_3",
            "label": "STAGE 3",
            "note": "Still above the 200-DMA but lost the 50-DMA — topping / extended.",
        }
    if price < sma200 and price > sma50:
        return {
            "id": "stage_1",
            "label": "STAGE 1",
            "note": "Below the 200-DMA while the 50-DMA is trying — base, not Stage 2.",
        }
    if price > sma200:
        return {
            "id": "stage_2_messy",
            "label": "STAGE 2?",
            "note": "Above the 200-DMA but the 50/200 stack is mixed.",
        }
    return {
        "id": "stage_4_messy",
        "label": "STAGE 4?",
        "note": "Below the 200-DMA without a clean declining stack.",
    }


def sma200_is_rising(sepa: Mapping[str, Any] | None) -> bool | None:
    for item in (sepa or {}).get("criteria") or []:
        if item.get("id") == "sma200_rising":
            passed = item.get("passed")
            if passed is True:
                return True
            if passed is False:
                return False
            return None
    return None


def nifty_frame() -> Any:
    try:
        from data.index_store import get_index_ohlcv
        return get_index_ohlcv("^NSEI")
    except Exception:
        return None


def index_strip() -> list[dict[str, Any]]:
    """Last official print for Nifty 50, Bank Nifty, India VIX."""
    specs = (
        ("^NSEI", "NIFTY 50"),
        ("^NSEBANK", "BANK NIFTY"),
        ("^INDIAVIX", "INDIA VIX"),
    )
    rows: list[dict[str, Any]] = []
    try:
        from data.index_store import get_index_ohlcv
    except Exception:
        return rows
    for ticker, label in specs:
        try:
            frame = get_index_ohlcv(ticker)
        except Exception:
            frame = None
        close = _close_series(frame)
        if close is None or len(close) < 1:
            rows.append({"id": ticker, "label": label, "close": None, "change_pct": None, "available": False})
            continue
        last = float(close.iloc[-1])
        chg = None
        if len(close) >= 2 and float(close.iloc[-2]) > 0:
            chg = round((last / float(close.iloc[-2]) - 1.0) * 100.0, 2)
        rows.append({
            "id": ticker,
            "label": label,
            "close": round(last, 2),
            "change_pct": chg,
            "available": True,
            "source": "nse_index_store",
        })
    return rows


def rs_rank(sepa: Mapping[str, Any] | None) -> int:
    """Tie-break only. SEPA score still ranks first."""
    label = str(((sepa or {}).get("rs") or {}).get("label") or "").upper()
    if label == "LEADER":
        return 2
    if label == "IN LINE":
        return 1
    if label == "LAGGARD":
        return -1
    return 0


def volume_pattern(frame: Any) -> dict[str, Any]:
    """Accumulation / distribution / dry-up from official volume — no scrape."""
    close = _close_series(frame)
    if frame is None or close is None or len(close) < 20:
        return {
            "available": False,
            "label": "UNKNOWN",
            "note": "Need ~20 sessions of official volume.",
            "rvol": None,
            "up_down": None,
        }
    data = frame
    try:
        data = frame.sort_index()
    except Exception:
        pass
    vol_col = None
    for name in ("volume", "Volume"):
        if name in getattr(data, "columns", []):
            vol_col = name
            break
    if vol_col is None:
        return {
            "available": False,
            "label": "UNKNOWN",
            "note": "Volume is not on the official history file.",
            "rvol": None,
            "up_down": None,
        }
    try:
        vol = data[vol_col].astype(float).dropna()
    except Exception:
        return {
            "available": False,
            "label": "UNKNOWN",
            "note": "Volume column is unreadable.",
            "rvol": None,
            "up_down": None,
        }
    aligned = close.align(vol, join="inner")
    close_a, vol_a = aligned[0], aligned[1]
    if len(vol_a) < 20:
        return {
            "available": False,
            "label": "UNKNOWN",
            "note": "Need ~20 sessions of official volume.",
            "rvol": None,
            "up_down": None,
        }
    last_vol = float(vol_a.iloc[-1])
    avg20 = float(vol_a.iloc[-20:].mean())
    rvol = (last_vol / avg20) if avg20 > 0 else None
    look = min(16, len(close_a) - 1)
    rets = close_a.diff().iloc[-look:]
    vwin = vol_a.iloc[-look:]
    up_vol = float(vwin[rets > 0].sum())
    dn_vol = float(vwin[rets < 0].sum())
    ratio = (up_vol / dn_vol) if dn_vol > 0 else None
    dryup = False
    if len(vol_a) >= 50:
        recent = float(vol_a.iloc[-10:].mean())
        base = float(vol_a.iloc[-50:-10].mean())
        dryup = base > 0 and recent < base * 0.70
    if dryup and ratio is not None and ratio >= 1.4:
        label, note = "ACCUMULATION", "Volume dried up while up-day volume still leads."
    elif ratio is not None and ratio <= 0.7:
        label, note = "DISTRIBUTION", "Down-day volume is leading — supply, not a quiet base."
    elif dryup:
        label, note = "DRY-UP", "Recent volume is quiet vs the prior month."
    elif rvol is not None and rvol >= 1.5 and float(close_a.iloc[-1]) >= float(close_a.iloc[-2]):
        label, note = "EXPANSION", "Today's volume is running hot on an up close."
    else:
        label, note = "MIXED", "No clean accumulation or distribution print."
    return {
        "available": True,
        "label": label,
        "note": note,
        "rvol": round(rvol, 2) if rvol is not None else None,
        "up_down": round(ratio, 2) if ratio is not None else None,
        "dryup": dryup,
        "source": "nse_bhavcopy",
    }


def breakout_readiness(frame: Any, sepa: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """0–100 research template: proximity + tightness + volume + 50-DMA.

    Not a measured QuantTerm edge. Missing pieces award 0, never invented.
    """
    close = _close_series(frame)
    if close is None or len(close) < 20:
        return {
            "available": False,
            "score": 0,
            "max_score": 100,
            "label": "UNKNOWN",
            "note": "Need official history before a breakout-readiness read.",
        }
    data = frame
    try:
        data = frame.sort_index()
    except Exception:
        pass
    high = data["high"].astype(float) if "high" in getattr(data, "columns", []) else close
    low = data["low"].astype(float) if "low" in getattr(data, "columns", []) else close
    price = float(close.iloc[-1])
    win = min(252, len(high))
    high_52w = float(high.tail(win).max())
    below = ((1.0 - price / high_52w) * 100.0) if high_52w > 0 else None
    awarded = 0
    parts: list[str] = []
    if below is not None and below <= 8:
        awarded += 30
        parts.append(f"{below:.1f}% below 52-week high")
    elif below is not None and below <= 15:
        awarded += 15
        parts.append(f"{below:.1f}% below 52-week high")
    elif below is not None:
        parts.append(f"{below:.1f}% below 52-week high — not a leading break")

    ranges = (high - low).astype(float).dropna()
    tight = False
    if len(ranges) >= 7:
        last_range = float(ranges.iloc[-1])
        avg20 = float(ranges.iloc[-20:].mean()) if len(ranges) >= 20 else float(ranges.mean())
        nr7 = last_range > 0 and last_range <= float(ranges.iloc[-7:].min()) + 1e-9
        coil = avg20 > 0 and last_range <= avg20 * 0.60
        tight = nr7 or coil
        if tight:
            awarded += 25
            parts.append("Range is coiled (NR7 or <60% of recent average range)")
        else:
            parts.append("Range is not coiled yet")

    vol = volume_pattern(frame)
    if vol.get("dryup"):
        awarded += 20
        parts.append("Volume dry-up")
    elif vol.get("rvol") is not None and float(vol["rvol"]) >= 1.2:
        awarded += 15
        parts.append(f"Volume {vol['rvol']}× the 20-day average")

    sma50 = _f((sepa or {}).get("levels", {}).get("sma50")) if sepa else None
    if sma50 is None and len(close) >= 50:
        sma50 = float(close.iloc[-50:].mean())
    if sma50 is not None and price > sma50:
        awarded += 25
        parts.append("Price holds above the 50-DMA")
    elif sma50 is not None:
        parts.append("Price is below the 50-DMA")

    if awarded >= 70:
        label = "READY"
    elif awarded >= 45:
        label = "COILING"
    else:
        label = "EARLY"
    return {
        "available": True,
        "score": int(awarded),
        "max_score": 100,
        "label": label,
        "note": "; ".join(parts) if parts else "Breakout-readiness is a published template, not a buy.",
        "below_52w_pct": round(below, 1) if below is not None else None,
        "tight": tight,
        "source": "nse_bhavcopy",
    }


def sepa_modules(sepa: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Six Reco-style analyser cards from fields already on the SEPA payload."""
    sepa = dict(sepa or {})
    by_id = {str(c.get("id")): c for c in (sepa.get("criteria") or [])}
    near = by_id.get("near_52w_high") or {}
    rs = dict(sepa.get("rs") or {})
    stage = dict(sepa.get("stage") or {})
    vol = dict(sepa.get("volume") or {})
    brk = dict(sepa.get("breakout") or {})
    score = sepa.get("score")
    passed = sepa.get("passed")
    total = sepa.get("total")

    def _state(ok: bool | None) -> str:
        if ok is True:
            return "pass"
        if ok is False:
            return "fail"
        return "unknown"

    trend_ok = bool(sepa.get("available")) and int(score or 0) >= 40
    return [
        {
            "id": "trend_template",
            "n": 1,
            "title": "Trend Template",
            "detail": (
                f"{score}/100 · {passed}/{total} Stage-2 rules"
                if score is not None
                else "SEPA unavailable"
            ),
            "note": "Price vs 50/150/200 DMA — Minervini published template.",
            "state": _state(trend_ok if sepa.get("available") else None),
        },
        {
            "id": "near_52w",
            "n": 2,
            "title": "52-Week High Proximity",
            "detail": str(near.get("detail") or "52-week high not on file."),
            "note": str(near.get("note") or "Leaders live near highs."),
            "state": _state(near.get("passed")),
        },
        {
            "id": "rs_nifty",
            "n": 3,
            "title": "Relative Strength vs Nifty",
            "detail": (
                f"{rs.get('label')} · {rs.get('excess_pp'):+g} pp / 63 sessions"
                if rs.get("available") and rs.get("excess_pp") is not None
                else str(rs.get("note") or "Need stock and Nifty history.")
            ),
            "note": str(rs.get("note") or ""),
            "state": _state(
                True if rs.get("label") == "LEADER"
                else False if rs.get("label") == "LAGGARD"
                else None if not rs.get("available")
                else True
            ),
        },
        {
            "id": "volume",
            "n": 4,
            "title": "Volume Pattern",
            "detail": (
                f"{vol.get('label')}"
                + (f" · rvol {vol.get('rvol')}×" if vol.get("rvol") is not None else "")
            ) if vol.get("available") else str(vol.get("note") or "Volume not on file."),
            "note": str(vol.get("note") or ""),
            "state": _state(
                True if vol.get("label") in {"ACCUMULATION", "DRY-UP", "EXPANSION"}
                else False if vol.get("label") == "DISTRIBUTION"
                else None if not vol.get("available")
                else True
            ),
        },
        {
            "id": "stage",
            "n": 5,
            "title": "Stage Analysis",
            "detail": str(stage.get("label") or "STAGE ?"),
            "note": str(stage.get("note") or "Weinstein/Minervini 1–4 from the 50/200 stack."),
            "state": _state(
                True if str(stage.get("id") or "") == "stage_2"
                else False if str(stage.get("id") or "").startswith("stage_4")
                else None if str(stage.get("id") or "") in {"", "unknown"}
                else True
            ),
        },
        {
            "id": "breakout",
            "n": 6,
            "title": "Breakout Readiness",
            "detail": (
                f"{brk.get('score')}/100 · {brk.get('label')}"
                if brk.get("available")
                else str(brk.get("note") or "Not enough history.")
            ),
            "note": str(brk.get("note") or "Proximity + coil + volume + 50-DMA. Research, not a buy."),
            "state": _state(
                True if brk.get("label") == "READY"
                else False if brk.get("label") == "EARLY"
                else None if not brk.get("available")
                else True
            ),
        },
    ]


def attach_context(sepa: dict[str, Any], stock_frame: Any, bench_frame: Any = None) -> dict[str, Any]:
    """Add stage + RS onto a SEPA payload. Mutates and returns it."""
    rising = sma200_is_rising(sepa)
    sepa["stage"] = classify_stage(sepa.get("levels") or {}, sma200_rising=rising)
    bench = bench_frame
    if bench is None:
        bench = nifty_frame()
    sepa["rs"] = rs_vs_benchmark(stock_frame, bench)
    sepa["volume"] = volume_pattern(stock_frame)
    sepa["breakout"] = breakout_readiness(stock_frame, sepa)
    sepa["modules"] = sepa_modules(sepa)
    return sepa
