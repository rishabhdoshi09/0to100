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
        # A partial session print that IS >0 can still look "thin" vs a full
        # average day (0.3× by 10:15) and empty the sniper / Ideas breakout
        # lane. During RTH, never let live volume demote a passing scan ratio.
        prior_ratio = _f(out.get("volume_ratio"))
        if avg20 > 0 and vol > 0:
            live_ratio = vol / avg20
            try:
                from data.nse_live import _is_trading_now
                trading = bool(_is_trading_now())
            except Exception:
                trading = False
            if trading:
                frac = max(_session_frac(), 0.15)
                paced = live_ratio / frac
                out["volume_ratio"] = round(max(prior_ratio, live_ratio, min(paced, 20.0)), 2)
            else:
                out["volume_ratio"] = round(live_ratio, 2)
        out["price_tag"] = meta.get("price_tag") or ("LIVE" if meta.get("live") else "EOD")
        out["tech_source"] = "live" if meta.get("live") else "eod"
        out.update(_structure_from_frame(frame, float(close.iloc[-1])))
    except Exception:
        pass
    return out


def _session_frac() -> float:
    """Fraction of the NSE cash session elapsed (0..1). Fail-open at 1.0."""
    try:
        from core.market_clock import now_ist
        now = now_ist()
        mins = now.hour * 60 + now.minute
        open_m, close_m = 9 * 60 + 15, 15 * 60 + 30
        return max(0.0, min(1.0, (mins - open_m) / (close_m - open_m)))
    except Exception:
        return 1.0


def _structure_from_frame(frame, close: float) -> dict[str, float]:
    """Distance from recent highs — used to drop faded scan-time breakouts.

    Today's incomplete bar must not become the 20-day high. A morning spike
    that gives back would otherwise mark every live breakout as faded and
    empty the sniper lane by midday.
    """
    try:
        high = frame["high"].astype(float) if "high" in frame.columns else frame["close"].astype(float)
        high = high.dropna()
        if high.empty or close <= 0:
            return {}
        try:
            last_day = getattr(frame.index[-1], "date", lambda: frame.index[-1])()
            from core.market_clock import today_ist
            exclude_today = str(last_day) == str(today_ist())
        except Exception:
            exclude_today = False
        prior = high.iloc[:-1] if exclude_today and len(high) > 1 else high
        lookback_20 = prior.iloc[-20:] if len(prior) >= 20 else prior
        lookback_52w = prior.iloc[-252:] if len(prior) >= 252 else prior
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
    head = items
    tail: list[dict[str, Any]] = []
    if limit is not None:
        cap = max(0, int(limit))
        head = items[:cap]
        tail = [dict(r) for r in items[cap:]]
    if bulk_overlay and head:
        ensure_live_store_overlay()
    # After a bulk store overlay, never scrape per symbol.
    network = bool(allow_network) and not bulk_overlay
    refreshed = [refresh_row_technicals(r, allow_network=network) for r in head]
    return _apply_kite_last(refreshed) + tail


def _apply_kite_last(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overwrite displayed last price from Kite when the session is up.

    History / RSI / structure stay on the official frame. The last print
    the trader sees must be Kite, not a frozen bhavcopy close.
    """
    if not rows:
        return rows
    try:
        from data.nse_live import _is_trading_now
        if not _is_trading_now():
            return rows
        from data.kite_client import _fresh_env
        if not (_fresh_env("KITE_ACCESS_TOKEN") or "").strip():
            return rows
        from data.live_quotes import _kite_quotes
        symbols = [str(r.get("symbol") or "").strip().upper() for r in rows]
        symbols = [s for s in symbols if s]
        if not symbols:
            quotes = {}
        else:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                quotes = pool.submit(_kite_quotes, symbols).result(timeout=2.5) or {}
            except FuturesTimeout:
                quotes = {}
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        return rows
    if not quotes:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        q = quotes.get(str(row.get("symbol") or "").strip().upper())
        if not q or not q.get("price") or q.get("source") != "kite":
            out.append(row)
            continue
        updated = dict(row)
        price = round(float(q["price"]), 2)
        updated["price"] = price
        updated["quote_source"] = "kite"
        updated["tech_source"] = "kite"
        try:
            from data.nse_live import _is_trading_now
            updated["price_tag"] = "LIVE" if _is_trading_now() else "KITE"
        except Exception:
            updated["price_tag"] = "KITE"
        high20 = _f(updated.get("high_20d"))
        high52 = _f(updated.get("high_52w"))
        if high20 > 0:
            updated["pct_below_20d_high"] = round((high20 - price) / high20 * 100.0, 2)
        if high52 > 0:
            updated["pct_below_52w_high"] = round((high52 - price) / high52 * 100.0, 2)
        out.append(updated)
    return out
