"""Current technicals for display — never serve frozen scan RSI/price as truth.

Scan payloads are signal snapshots. Price, RSI and volume ratio change every
session, so every radar/scanner read recomputes them from bhavcopy + today's
live bar. Simple overwrite. No parallel "scan RSI" UI path.
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


def refresh_row_technicals(row: Mapping[str, Any]) -> dict[str, Any]:
    """Overwrite price / RSI / volume_ratio with the current tape."""
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
        if str(last_day) != str(_today()):
            from data.nse_live import overlay_live_on_frame
            frame, meta = overlay_live_on_frame(frame, sym)
        else:
            from data.nse_live import _is_trading_now
            meta = {
                "live": bool(_is_trading_now()),
                "price_tag": "LIVE" if _is_trading_now() else "EOD",
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
        if avg20 > 0 and vol > 0:
            out["volume_ratio"] = round(vol / avg20, 2)
        out["price_tag"] = meta.get("price_tag") or ("LIVE" if meta.get("live") else "EOD")
        out["tech_source"] = "live" if meta.get("live") else "eod"
    except Exception:
        pass
    return out


def refresh_rows_technicals(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    bulk_overlay: bool = True,
) -> list[dict[str, Any]]:
    items = list(rows)
    if limit is not None:
        items = items[: max(0, int(limit))]
    if bulk_overlay and items:
        ensure_live_store_overlay()
    return [refresh_row_technicals(r) for r in items]
