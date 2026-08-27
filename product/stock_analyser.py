"""Per-symbol Minervini-style analyser shown when any stock is opened.

Seven live checks (52-week location, relative strength, candle, VCP coil,
liquidity, Weinstein stage) scored 0–100 from official NSE history.
This is a research card, not a buy ticket and not a QuantTerm backtest edge.

The DMA trend-template scorer in ``product.sepa_setup`` still ranks Best Setups.
This module is the click-through card: quote strip + pass/fail checklist.
"""
from __future__ import annotations

from typing import Any, Mapping

NEAR_52W_PCT = 25.0
OFF_52W_LOW_PCT = 25.0
TIGHT_RANGE_PCT = 2.5
COIL_VS_AVG = 0.60
MIN_TURNOVER_CR = 10.0
MIN_VOLUME = 2_000_000  # 20 lakh shares
MIN_VOLUME_FOR_TURNOVER = 1_000_000  # 10 lakh — rupee notional alone is not a liquid book
ANALYSER_TOTAL = 7
ANALYSER_MAX = 100

CRITERIA: tuple[dict[str, Any], ...] = (
    {"id": "near_52w_high", "title": "52-Week High Proximity", "points": 20,
     "rule": "Close within 25% of the 52-week high."},
    {"id": "off_52w_low", "title": "52-Week Low Distance", "points": 20,
     "rule": "Close at least 25% above the 52-week low."},
    {"id": "relative_strength", "title": "Relative Strength vs Nifty 500", "points": 15,
     "rule": "Last session beats the official Nifty 500 (else Nifty 50) benchmark."},
    {"id": "intraday", "title": "Intraday Price Action", "points": 10,
     "rule": "Last session closes in the upper half of the day's range."},
    {"id": "vcp", "title": "Volatility Contraction (VCP Signal)", "points": 10,
     "rule": "Day range is tight (≤2.5%) or coiled vs the 20-day average range."},
    {"id": "liquidity", "title": "Liquidity & Volume", "points": 10,
     "rule": "Turnover at least ₹10 Cr or volume at least 20 lakh shares."},
    {"id": "stage", "title": "Weinstein Stage Analysis", "points": 15,
     "rule": "Stage 2 — price above a rising 50/200 DMA stack."},
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


def _sma(close, window: int) -> float | None:
    if close is None or len(close) < window:
        return None
    try:
        val = float(close.iloc[-window:].mean())
    except Exception:
        return None
    return None if val != val else val


def _sorted(frame: Any):
    try:
        return frame.sort_index()
    except Exception:
        return frame


def _close(frame: Any):
    data = _sorted(frame)
    for col in ("close", "Close"):
        if col in getattr(data, "columns", []):
            try:
                series = data[col].astype(float).dropna()
            except Exception:
                return None, data
            return series if len(series) else None, data
    return None, data


def _criterion(spec_id: str, *, passed: bool | None, detail: str, note: str,
               values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = next(row for row in CRITERIA if row["id"] == spec_id)
    points = int(spec["points"])
    return {
        "id": spec_id,
        "title": spec["title"],
        "rule": spec["rule"],
        "points": points,
        "awarded": points if passed is True else 0,
        "passed": passed,
        "detail": detail,
        "note": note,
        "values": dict(values or {}),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    criteria = [
        _criterion(spec["id"], passed=None, detail="Not enough official history.", note=reason)
        for spec in CRITERIA
    ]
    return {
        "available": False,
        "score": 0,
        "max_score": ANALYSER_MAX,
        "passed": 0,
        "total": ANALYSER_TOTAL,
        "unknown": ANALYSER_TOTAL,
        "verdict": "INCOMPLETE",
        "headline": "INCOMPLETE — NOT ENOUGH HISTORY",
        "advice": reason,
        "method": "minervini_live_analyser_7",
        "disclaimer": (
            "Research overlay on official NSE bhavcopy. Not a QuantTerm backtest edge "
            "and not a buy or sell order."
        ),
        "criteria": criteria,
        "quote": None,
        "benchmark": None,
    }


def _verdict(score: int, passed: int, unknown: int, criteria: list[dict[str, Any]]) -> tuple[str, str, str]:
    if unknown >= 4:
        return (
            "INCOMPLETE",
            "INCOMPLETE — NOT ENOUGH HISTORY",
            "Need official OHLCV before the live Minervini checklist is readable.",
        )
    passed_titles = [str(c["title"]) for c in criteria if c.get("passed") is True]
    failed_ids = {str(c["id"]) for c in criteria if c.get("passed") is False}
    strengths = ", ".join(t.lower() for t in passed_titles[:3])
    if score >= 80 and passed >= 6:
        return (
            "STRONG",
            "READY — SETUP QUALIFIED",
            "The live checklist is intact. This is a research qualify, not a buy ticket "
            "— still confirm stop, volume and chase risk.",
        )
    if score >= 55 and passed >= 3:
        wait = "tight consolidation near the pivot" if "vcp" in failed_ids else "the failed checks to flip"
        lead = (
            f"The stock shows characteristics Minervini looks for ({strengths}), "
            if strengths else "Some Minervini characteristics are present, "
        )
        return (
            "WATCHLIST",
            "WATCHLIST — WAIT FOR SETUP",
            f"{lead}but look for {wait} before entry.",
        )
    if score >= 40:
        return (
            "MIXED",
            "MIXED — WAIT FOR STRUCTURE",
            "Some Stage-2 pieces are present, but this is not a clean setup yet. "
            "Better names sit near 52-week highs with a tight range and leadership vs Nifty.",
        )
    return (
        "WEAK",
        "WEAK — NOT IDEAL FOR SWING",
        "This stock currently does not meet the live Minervini checklist. "
        "Wait for a proper Stage-2 setup or look for better candidates near 52-week highs.",
    )


def _session_return(close) -> float | None:
    if close is None or len(close) < 2:
        return None
    prev = float(close.iloc[-2])
    last = float(close.iloc[-1])
    if prev <= 0:
        return None
    return (last / prev - 1.0) * 100.0


def analyser_benchmark_frame() -> tuple[Any, str]:
    """Prefer Nifty 500 when the official index store has it; else Nifty 50."""
    try:
        from data.index_store import get_index_ohlcv
    except Exception:
        return None, "Nifty 50"
    for ticker, label in (("NIFTY500", "Nifty 500"), ("^NSEI", "Nifty 50")):
        try:
            frame = get_index_ohlcv(ticker)
        except Exception:
            frame = None
        if frame is not None and len(frame):
            return frame, label
    return None, "Nifty 50"


def _quote(data: Any, *, high_52w: float | None, low_52w: float | None,
           volume: float | None, turnover_cr: float | None) -> dict[str, Any] | None:
    try:
        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) >= 2 else None
        close = float(last["close"]) if "close" in data.columns else float(last["Close"])
        prev_close = None
        if prev is not None:
            prev_close = float(prev["close"]) if "close" in data.columns else float(prev["Close"])
        def col(name: str, title: str):
            if name in data.columns:
                return _f(last[name])
            if title in data.columns:
                return _f(last[title])
            return None
        open_px = col("open", "Open")
        high_px = col("high", "High")
        low_px = col("low", "Low")
        change_pct = None
        if prev_close and prev_close > 0:
            change_pct = round((close / prev_close - 1.0) * 100.0, 2)
        as_of = str(getattr(data.index[-1], "date", lambda: data.index[-1])())
        vol_lakh = round(volume / 100_000.0, 1) if volume is not None else None
        return {
            "open": round(open_px, 2) if open_px is not None else None,
            "high": round(high_px, 2) if high_px is not None else None,
            "low": round(low_px, 2) if low_px is not None else None,
            "close": round(close, 2),
            "prev_close": round(prev_close, 2) if prev_close else None,
            "change_pct": change_pct,
            "high_52w": round(high_52w, 2) if high_52w is not None else None,
            "low_52w": round(low_52w, 2) if low_52w is not None else None,
            "volume": int(volume) if volume is not None else None,
            "volume_lakh": vol_lakh,
            "turnover_cr": round(turnover_cr, 2) if turnover_cr is not None else None,
            "as_of": as_of,
        }
    except Exception:
        return None


def analyse_stock(frame: Any, *, bench_frame: Any = None, bench_label: str | None = None) -> dict[str, Any]:
    """Score one official OHLCV frame into the click-through analyser card."""
    if frame is None:
        return _unavailable("Official price history is unavailable.")
    try:
        if len(frame) == 0:
            return _unavailable("Official price history is unavailable.")
    except Exception:
        return _unavailable("Official price history is unreadable.")
    close, data = _close(frame)
    if close is None or len(close) < 20:
        n = 0 if close is None else len(close)
        return _unavailable(
            f"Only {n} sessions on file — the analyser needs at least 20 days, 200 for a full stage call."
        )

    price = float(close.iloc[-1])
    high_col = data["high"].astype(float) if "high" in data.columns else (
        data["High"].astype(float) if "High" in data.columns else close
    )
    low_col = data["low"].astype(float) if "low" in data.columns else (
        data["Low"].astype(float) if "Low" in data.columns else close
    )
    open_col = data["open"].astype(float) if "open" in data.columns else (
        data["Open"].astype(float) if "Open" in data.columns else None
    )
    vol_col = None
    for name in ("volume", "Volume"):
        if name in data.columns:
            vol_col = name
            break
    win = min(252, len(data))
    high_52w = float(high_col.tail(win).max())
    low_52w = float(low_col.tail(win).min())
    last_high = float(high_col.iloc[-1])
    last_low = float(low_col.iloc[-1])
    last_open = float(open_col.iloc[-1]) if open_col is not None else None
    volume = None
    if vol_col is not None:
        try:
            volume = float(data[vol_col].astype(float).iloc[-1])
        except Exception:
            volume = None
    turnover_cr = None
    for name in ("value", "turnover", "Value", "Turnover", "TOTTRDVAL"):
        if name in data.columns:
            try:
                raw = float(data[name].astype(float).iloc[-1])
            except Exception:
                raw = None
            if raw is not None and raw == raw and raw >= 0:
                turnover_cr = raw / 10_000_000.0 if raw > 1_000 else raw
            break
    if turnover_cr is None and volume is not None and volume >= 0:
        turnover_cr = volume * price / 10_000_000.0
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
    range_base = prev_close if prev_close and prev_close > 0 else price
    day_range_pct = ((last_high - last_low) / range_base * 100.0) if range_base else None

    below_high_pct = round((1.0 - price / high_52w) * 100.0, 1) if high_52w > 0 else None
    above_low_pct = round((price / low_52w - 1.0) * 100.0, 1) if low_52w > 0 else None

    # 1. 52-week high
    if below_high_pct is None:
        c1 = _criterion("near_52w_high", passed=None, detail="52-week high is unavailable.",
                        note="Cannot score proximity without a 52-week high.")
    else:
        ok = below_high_pct <= NEAR_52W_PCT
        if below_high_pct <= 8:
            note = "Excellent — within striking distance of highs."
        elif ok:
            note = "Acceptable — still within 25% of the 52-week high."
        else:
            note = "Laggard zone — more than 25% below the 52-week high."
        c1 = _criterion(
            "near_52w_high", passed=ok,
            detail=f"{below_high_pct:.1f}% below 52WH (₹{high_52w:,.2f})",
            note=note,
            values={"below_high_pct": below_high_pct, "high_52w": round(high_52w, 2)},
        )

    # 2. 52-week low
    if above_low_pct is None:
        c2 = _criterion("off_52w_low", passed=None, detail="52-week low is unavailable.",
                        note="Cannot score distance from lows.")
    else:
        ok = above_low_pct >= OFF_52W_LOW_PCT
        if above_low_pct >= 100:
            note = "Strong — stock has doubled from lows, powerful uptrend."
        elif ok:
            note = "Enough distance from the low — not a falling knife."
        else:
            note = "Too close to the 52-week low for a Stage-2 swing."
        c2 = _criterion(
            "off_52w_low", passed=ok,
            detail=f"{above_low_pct:.1f}% above 52WL (₹{low_52w:,.2f})",
            note=note,
            values={"above_low_pct": above_low_pct, "low_52w": round(low_52w, 2)},
        )

    # 3. Relative strength vs Nifty (last session)
    bench = bench_frame
    label = bench_label or "Nifty 50"
    if bench is None:
        bench, label = analyser_benchmark_frame()
    stock_pct = _session_return(close)
    bench_close, _ = _close(bench) if bench is not None else (None, None)
    bench_pct = _session_return(bench_close)
    if stock_pct is None or bench_pct is None:
        c3 = _criterion(
            "relative_strength", passed=None,
            detail=f"Need a last session for the stock and {label}.",
            note="Relative strength stays unknown until both prints exist.",
            values={"benchmark": label},
        )
    else:
        excess = round(stock_pct - bench_pct, 2)
        ok = excess >= 0
        c3 = _criterion(
            "relative_strength", passed=ok,
            detail=(
                f"Stock {stock_pct:+.2f}% vs {label} {bench_pct:+.2f}% "
                f"(RS: {excess:+.2f}%)"
            ),
            note=(
                f"Leading — outperforming {label} on the last session."
                if ok else
                f"Weak — underperforming {label} by {abs(excess):.2f}%."
            ),
            values={
                "stock_pct": round(stock_pct, 2),
                "benchmark_pct": round(bench_pct, 2),
                "excess_pct": excess,
                "benchmark": label,
            },
        )
        c3["title"] = f"Relative Strength vs {label}"

    # 4. Intraday candle
    mid = (last_high + last_low) / 2.0
    if last_high <= last_low:
        c4 = _criterion("intraday", passed=None, detail="Day range is unreadable.",
                        note="High and low are missing or identical.")
    else:
        upper = price >= mid
        bullish = last_open is None or price >= last_open
        ok = upper and bullish
        o_txt = f"₹{last_open:,.2f}" if last_open is not None else "—"
        kind = "Bullish" if ok else "Bearish"
        c4 = _criterion(
            "intraday", passed=ok,
            detail=f"{kind} candle (O: {o_txt} → C: ₹{price:,.2f})",
            note=(
                "Closing in the upper half — buyers in control today."
                if ok else
                "Closing in lower half — sellers dominant today."
            ),
            values={"open": last_open, "close": round(price, 2), "mid": round(mid, 2)},
        )

    # 5. VCP / range contraction
    ranges = (high_col - low_col).astype(float).dropna()
    avg20 = float(ranges.iloc[-20:].mean()) if len(ranges) >= 20 else (
        float(ranges.mean()) if len(ranges) else None
    )
    last_range = float(ranges.iloc[-1]) if len(ranges) else None
    coiled = bool(avg20 and last_range is not None and avg20 > 0 and last_range <= avg20 * COIL_VS_AVG)
    tight = bool(day_range_pct is not None and day_range_pct <= TIGHT_RANGE_PCT)
    if day_range_pct is None:
        c5 = _criterion("vcp", passed=None, detail="Day range is unavailable.",
                        note="Need a high and low for the last session.")
    else:
        ok = tight or coiled
        if tight:
            note = "Tight range — the kind of coil Minervini wants near a pivot."
        elif coiled:
            note = "Range is coiled versus the last 20 sessions — monitor for a break."
        elif day_range_pct <= 5:
            note = "Moderate range — monitor for tightening."
        else:
            note = "Wide day — not a volatility contraction."
        c5 = _criterion(
            "vcp", passed=ok,
            detail=f"Day range: {day_range_pct:.2f}% (₹{last_low:,.2f} – ₹{last_high:,.2f})",
            note=note,
            values={"day_range_pct": round(day_range_pct, 2), "coiled": coiled, "tight": tight},
        )

    # 6. Liquidity
    if volume is None and turnover_cr is None:
        c6 = _criterion("liquidity", passed=None, detail="Volume is not on the official history file.",
                        note="Cannot judge slippage without volume.")
    else:
        ok = (volume is not None and volume >= MIN_VOLUME) or (
            turnover_cr is not None
            and turnover_cr >= MIN_TURNOVER_CR
            and volume is not None
            and volume >= MIN_VOLUME_FOR_TURNOVER
        )
        vol_lakh = (volume / 100_000.0) if volume is not None else 0.0
        turn_txt = f"₹{turnover_cr:.1f} Cr" if turnover_cr is not None else "₹— Cr"
        c6 = _criterion(
            "liquidity", passed=ok,
            detail=f"Turnover: {turn_txt} | Vol: {vol_lakh:.1f}L",
            note=(
                "Enough participation for a swing without obvious thin-book risk."
                if ok else
                "Low liquidity — risky for larger positions, slippage likely."
            ),
            values={"turnover_cr": turnover_cr, "volume": volume},
        )

    # 7. Weinstein stage
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    sma200_prev = _sma(close.iloc[:-21], 200) if len(close) >= 221 else None
    rising = None if sma200 is None or sma200_prev is None else sma200 > sma200_prev
    try:
        from product.monitor_context import classify_stage
        stage = classify_stage(
            {"price": price, "sma50": sma50, "sma200": sma200},
            sma200_rising=rising,
        )
    except Exception:
        stage = {"id": "unknown", "label": "STAGE ?", "note": "Stage unavailable."}
    stage_id = str(stage.get("id") or "")
    ok_stage = stage_id in {"stage_2", "stage_2_early"}
    passed_stage = None if stage_id in {"", "unknown"} else ok_stage
    if stage_id == "stage_2":
        stage_detail = "Stage 2 — Advancing"
    elif stage_id == "stage_2_early":
        stage_detail = "Stage 2? — Early advance"
    else:
        stage_detail = str(stage.get("label") or "STAGE ?")
    c7 = _criterion(
        "stage", passed=passed_stage,
        detail=stage_detail,
        note=(
            "Stock is in the ideal stage per Stan Weinstein."
            if stage_id == "stage_2" else
            str(stage.get("note") or "Weinstein/Minervini 1–4 from the 50/200 stack.")
        ),
        values={"stage_id": stage_id, "sma50": sma50, "sma200": sma200},
    )

    criteria = [c1, c2, c3, c4, c5, c6, c7]
    score = int(sum(int(c["awarded"]) for c in criteria))
    passed_n = sum(1 for c in criteria if c["passed"] is True)
    unknown_n = sum(1 for c in criteria if c["passed"] is None)
    verdict, headline, advice = _verdict(score, passed_n, unknown_n, criteria)
    quote = _quote(data, high_52w=high_52w, low_52w=low_52w, volume=volume, turnover_cr=turnover_cr)
    trend_template = None
    try:
        from product.sepa_setup import score_sepa
        trend_template = score_sepa(frame, bench_frame=bench)
    except Exception:
        trend_template = None
    return {
        "available": True,
        "score": score,
        "max_score": ANALYSER_MAX,
        "passed": passed_n,
        "total": ANALYSER_TOTAL,
        "unknown": unknown_n,
        "verdict": verdict,
        "headline": headline,
        "advice": advice,
        "method": "minervini_live_analyser_7",
        "disclaimer": (
            "Research overlay on official NSE bhavcopy. Not a QuantTerm backtest edge "
            "and not a buy or sell order."
        ),
        "criteria": criteria,
        "quote": quote,
        "stage": stage,
        "benchmark": label,
        "trend_template": trend_template,
    }
