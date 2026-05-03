"""Dynamic universe provider.

* On startup (or daily 08:00 IST refresh) calls KiteConnect ``instruments("NSE")``,
  filters for cash-equity (segment NSE, instrument_type EQ),
  caches into SQLite ``instruments_cache``.
* ``get_active_universe(max_symbols)`` returns the cached symbols
  (yfinance-friendly suffix ``.NS``).
* Falls back to ``config.yaml`` universe when Kite is unavailable so the
  rest of the system keeps working without broker credentials.
"""
from __future__ import annotations

import os

from sq_ai.portfolio.tracker import PortfolioTracker


def _kite_client():
    api_key = os.environ.get("KITE_API_KEY", "")
    access = os.environ.get("KITE_ACCESS_TOKEN", "")
    if not api_key or not access:
        return None
    try:
        from kiteconnect import KiteConnect  # noqa: WPS433
        k = KiteConnect(api_key=api_key)
        k.set_access_token(access)
        return k
    except Exception as exc:                            # pragma: no cover
        print(f"[universe] kite init failed: {exc}")
        return None


def _filter_eq(instruments: list[dict]) -> list[dict]:
    out = []
    for ins in instruments:
        if ins.get("segment") != "NSE":
            continue
        if ins.get("instrument_type") not in {"EQ", None}:
            continue
        ts = ins.get("tradingsymbol") or ins.get("trading_symbol")
        if not ts:
            continue
        out.append({
            "trading_symbol": ts,
            "instrument_token": ins.get("instrument_token"),
            "name": ins.get("name", ""),
        })
    return out


def refresh_universe(tracker: PortfolioTracker | None = None) -> int:
    """Pull instruments from Kite into SQLite. Returns row count."""
    tracker = tracker or PortfolioTracker()
    k = _kite_client()
    if k is None:
        return 0
    try:
        ins = k.instruments("NSE")
    except Exception as exc:                            # pragma: no cover
        print(f"[universe] kite.instruments failed: {exc}")
        return 0
    rows = _filter_eq(ins)
    return tracker.cache_instruments(rows)


def _yf_suffix(symbol: str) -> str:
    """yfinance NSE symbols need a ``.NS`` suffix; idempotent."""
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def get_active_universe(max_symbols: int = 500,
                        tracker: PortfolioTracker | None = None,
                        fallback_yaml: list[str] | None = None) -> list[str]:
    """Return up to ``max_symbols`` cash-equity tickers (.NS-suffixed)."""
    tracker = tracker or PortfolioTracker()
    cached = tracker.get_cached_instruments()
    if cached:
        symbols = [_yf_suffix(c["trading_symbol"]) for c in cached]
        return symbols[:max_symbols]
    if fallback_yaml:
        return [_yf_suffix(s) for s in fallback_yaml[:max_symbols]]
    return []


__all__ = ["refresh_universe", "get_active_universe"]
