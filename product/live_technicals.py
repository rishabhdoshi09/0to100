"""Current technicals for display — never serve frozen scan RSI/price as truth.

Scan payloads are signal snapshots. Price, RSI and volume ratio change every
session, so radar/scanner reads recompute them from bhavcopy + today's live
bar when the store already has it.

Bulk path: one store overlay, then local frame math only. Never scrape
Google/quote fallbacks per symbol — that freezes the desk for minutes and
makes sniper lanes look empty.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _rsi(close, periods: int = 14) -> float | None:
    try:
        import pandas as pd
        series = close if hasattr(close, "diff") else pd.Series(close, dtype=float)
        series = series.astype(float).dropna()
        if len(series) < periods + 1:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
        last_loss = float(loss.iloc[-1])
        if last_loss == 0:
            return 100.0
        rs = float(gain.iloc[-1]) / last_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)
    except Exception:
        return None


def ensure_live_store_overlay() -> int:
    try:
        from data.nse_live import apply_live_to_store
        return int(apply_live_to_store() or 0)
    except Exception:
        return 0


def _today():
    try:
        from core.market_clock import today_ist
        return today_ist()
    except Exception:
        from datetime import date
        return date.today()


def refresh_row_technicals(
    row: Mapping[str, Any],
    *,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Overwrite price / RSI / volume_ratio from the store (and optional live bar).

    ``allow_network=True`` only for single-symbol views. Radar/scanner bulk
    paths must stay store-local after ``ensure_live_store_overlay()``.
    """
    out = dict(row)
    sym = str(out.get("symbol") or "").strip().upper()
    if not sym:
        return out
    try:
        from data.bhavcopy_runtime import get_ohlcv
        frame = get_ohlcv(sym)
    except Exception:
        return out
    if frame is None or getattr(frame, "empty", True):
        return out

    try:
        last_day = getattr(frame.index[-1], "date", lambda: frame.index[-1])()
        if str(last_day) != str(_today()) and allow_network:
            from data.nse_live import overlay_live_on_frame
            frame, meta = overlay_live_on_frame(frame, sym)
        else:
            from data.nse_live import _is_trading_now
            on_today = str(last_day) == str(_today())
            live_now = bool(on_today and _is_trading_now())
            meta = {
                "live": live_now,
                "price_tag": "LIVE" if live_now else "EOD",
                "source": "store",
            }
    except Exception:
        meta = {"live": False, "price_tag": "EOD", "source": ""}

    try:
        close = frame["close"].astype(float)
        out["price"] = round(float(close.iloc[-1]), 2)
        rsi = _rsi(close)
        if rsi is not None:
            out["rsi"] = rsi
        vol = float(frame["volume"].iloc[-1]) if "volume" in frame.columns else 0.0
        avg20 = _f(out.get("avg_vol20"))
        if avg20 <= 0 and "volume" in frame.columns and len(frame) >= 21:
            avg20 = float(frame["volume"].astype(float).iloc[-21:-1].mean() or 0)
        # Incomplete live prints sometimes land with volume=0 — keep prior ratio.
        if avg20 > 0 and vol > 0:
            out["volume_ratio"] = round(vol / avg20, 2)
        out["price_tag"] = meta.get("price_tag") or ("LIVE" if meta.get("live") else "EOD")
        out["tech_source"] = "live" if meta.get("live") else "eod"
        apply_current_trade_levels(out, frame)
    except Exception:
        pass
    if _f(out.get("price") or out.get("cmp")) > 0:
        apply_current_trade_levels(out, None)
    return out


def _atr14(frame: Any, periods: int = 14) -> float | None:
    try:
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        previous = close.shift(1)
        tr = (high - low).to_frame("a")
        tr["b"] = (high - previous).abs()
        tr["c"] = (low - previous).abs()
        value = tr.max(axis=1).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean().iloc[-1]
        return float(value)
    except Exception:
        return None


def _positive_level(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = _f(row.get(key))
        if value > 0:
            return value
    return 0.0


def apply_current_trade_levels(row: dict[str, Any], frame: Any = None) -> dict[str, Any]:
    """Fill missing entry / stop / target from the latest close.

    Same plan as unified_scanner: stop = entry − 2×ATR, target = entry + 4×ATR
    (5% / 10% fallback). Existing scanner buy-zone levels stay intact.
    """
    price = _f(row.get("price") or row.get("cmp"))
    if price <= 0:
        return row
    entry = _positive_level(row, "entry", "entry_price")
    stop = _positive_level(row, "stop", "stop_price")
    target = _positive_level(row, "target", "target_price")
    if entry > 0 and stop > 0 and target > 0:
        return row
    atr = _atr14(frame) if frame is not None else None
    if atr is None or atr <= 0:
        try:
            atr_pct = _f(row.get("atr_pct"))
            atr = price * atr_pct / 100.0 if atr_pct > 0 else 0.0
        except Exception:
            atr = 0.0
    filled = False
    if entry <= 0:
        entry = round(price, 2)
        row["entry"] = entry
        filled = True
    if atr and atr > 0:
        row["atr"] = round(atr, 2)
        row["atr_pct"] = round(atr / price * 100.0, 2)
        if stop <= 0:
            row["stop"] = round(max(0.01, entry - 2.0 * atr), 2)
            filled = True
        if target <= 0:
            row["target"] = round(entry + 4.0 * atr, 2)
            filled = True
        if filled:
            row["levels_source"] = "current_ohlcv"
    else:
        if stop <= 0:
            row["stop"] = round(entry * 0.95, 2)
            filled = True
        if target <= 0:
            row["target"] = round(entry * 1.10, 2)
            filled = True
        if filled:
            row["levels_source"] = "current_pct"
    return row


def refresh_rows_technicals(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    bulk_overlay: bool = True,
    allow_network: bool = False,
) -> list[dict[str, Any]]:
    items = list(rows)
    if limit is not None:
        items = items[: max(0, int(limit))]
    if bulk_overlay and items:
        ensure_live_store_overlay()
    # After a bulk store overlay, never scrape per symbol.
    network = bool(allow_network) and not bulk_overlay
    return [refresh_row_technicals(r, allow_network=network) for r in items]
