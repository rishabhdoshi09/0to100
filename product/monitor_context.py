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


def attach_context(sepa: dict[str, Any], stock_frame: Any, bench_frame: Any = None) -> dict[str, Any]:
    """Add stage + RS onto a SEPA payload. Mutates and returns it."""
    rising = sma200_is_rising(sepa)
    sepa["stage"] = classify_stage(sepa.get("levels") or {}, sma200_rising=rising)
    bench = bench_frame
    if bench is None:
        bench = nifty_frame()
    sepa["rs"] = rs_vs_benchmark(stock_frame, bench)
    return sepa
