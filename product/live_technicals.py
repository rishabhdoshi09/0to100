"""Refresh scan-row technicals from official history + today's live bar.

Saved scan RSI/price freeze at scan time. During market hours the radar must
recompute RSI / last price / volume ratio from bhavcopy + live overlay so
cards (e.g. YATHARTH RSI) match the tape — never show yesterday's oscillator
as if it were live.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _rsi_series(close, periods: int = 14) -> float | None:
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
    """One bulk live overlay onto the shared bhav store (Kite/NSE). Idempotent."""
    try:
        from data.nse_live import apply_live_to_store
        return int(apply_live_to_store() or 0)
    except Exception:
        return 0


def refresh_row_technicals(
    row: Mapping[str, Any],
    *,
    overlay_store: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``row`` with live-aware price / RSI / volume_ratio.

    Prefer calling :func:`ensure_live_store_overlay` once per request, then
    pass ``overlay_store=False`` so each row only reads the in-memory frame
    (no per-symbol network). Fail-open to scan fields when history missing.
    """
    out = dict(row)
    sym = str(out.get("symbol") or "").strip().upper()
    if not sym:
        out["tech_source"] = "scan"
        return out

    if overlay_store:
        ensure_live_store_overlay()

    scan_rsi = out.get("rsi")
    scan_price = out.get("price")
    scan_vol = out.get("volume_ratio")
    try:
        from data.bhavcopy_runtime import get_ohlcv
        frame = get_ohlcv(sym)
    except Exception:
        frame = None
    if frame is None or getattr(frame, "empty", True):
        out["tech_source"] = "scan"
        return out

    # If bulk store overlay did not land today's bar, try a single-symbol patch.
    live_meta = {"live": False, "price_tag": "EOD", "source": "", "eod_as_of": ""}
    try:
        from core.market_clock import today_ist
        today = today_ist()
    except Exception:
        from datetime import date as _date
        today = _date.today()
    try:
        last_day = getattr(frame.index[-1], "date", lambda: frame.index[-1])()
        if str(last_day) != str(today):
            from data.nse_live import overlay_live_on_frame
            frame, live_meta = overlay_live_on_frame(frame, sym)
        else:
            live_meta = {
                "live": True,
                "price_tag": "LIVE",
                "source": "store",
                "eod_as_of": str(last_day),
            }
            # Store may already hold today's official EOD after close — still
            # tag as EOD when the bar is the published session file.
            try:
                from data import bhavcopy_store as bs
                with bs._lock:
                    store_last = bs._store_last_day
                if store_last == today:
                    # Could be official EOD or live overlay; prefer LIVE during session.
                    from data.nse_live import _is_trading_now
                    if _is_trading_now():
                        live_meta["price_tag"] = "LIVE"
                        live_meta["live"] = True
                    else:
                        live_meta["price_tag"] = "EOD"
                        live_meta["live"] = False
            except Exception:
                pass
    except Exception:
        pass

    try:
        close = frame["close"].astype(float)
        last_close = float(close.iloc[-1])
        rsi = _rsi_series(close)
        vol = float(frame["volume"].iloc[-1]) if "volume" in frame.columns else 0.0
        avg20 = _f(out.get("avg_vol20"))
        if avg20 <= 0 and "volume" in frame.columns and len(frame) >= 21:
            avg20 = float(frame["volume"].astype(float).iloc[-21:-1].mean() or 0)
        vratio = round(vol / avg20, 2) if avg20 > 0 and vol > 0 else _f(scan_vol)

        out["price"] = round(last_close, 2)
        if rsi is not None:
            out["rsi"] = rsi
            out["rsi_scan"] = scan_rsi
        if vratio > 0:
            out["volume_ratio_scan"] = scan_vol
            out["volume_ratio"] = vratio
        out["tech_source"] = "live" if live_meta.get("live") else "eod"
        out["price_tag"] = live_meta.get("price_tag") or ("LIVE" if live_meta.get("live") else "EOD")
        out["quote_source"] = str(live_meta.get("source") or "")
        out["eod_as_of"] = str(live_meta.get("eod_as_of") or "")
        if scan_price is not None:
            out["price_scan"] = scan_price
    except Exception:
        out["tech_source"] = "scan"
    return out


def refresh_rows_technicals(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    bulk_overlay: bool = True,
) -> list[dict[str, Any]]:
    """Refresh technicals for up to ``limit`` rows after one bulk live overlay."""
    items = list(rows)
    if limit is not None:
        items = items[: max(0, int(limit))]
    if bulk_overlay and items:
        ensure_live_store_overlay()
    return [refresh_row_technicals(r, overlay_store=False) for r in items]
