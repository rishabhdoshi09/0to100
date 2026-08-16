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
        out.update(_structure_from_frame(frame, float(close.iloc[-1])))
    except Exception:
        pass
    return out


def _structure_from_frame(frame, close: float) -> dict[str, float]:
    """Distance from recent highs — used to drop faded scan-time breakouts."""
    try:
        high = frame["high"].astype(float) if "high" in frame.columns else frame["close"].astype(float)
        high = high.dropna()
        if high.empty or close <= 0:
            return {}
        lookback_20 = high.iloc[-20:] if len(high) >= 20 else high
        lookback_52w = high.iloc[-252:] if len(high) >= 252 else high
        h20 = float(lookback_20.max())
        h52 = float(lookback_52w.max())
        out: dict[str, float] = {}
        if h20 > 0:
            out["high_20d"] = round(h20, 2)
            out["pct_below_20d_high"] = round((h20 - close) / h20 * 100.0, 2)
        if h52 > 0:
            out["high_52w"] = round(h52, 2)
            out["pct_below_52w_high"] = round((h52 - close) / h52 * 100.0, 2)
        return out
    except Exception:
        return {}


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
