"""
MarketDataProvider — Kite-authoritative for every Kite-covered market field.

All UI/modules that need prices, OHLCV, volume, market depth, OI, or historical
candles should route through this module instead of calling alternate market
providers directly.

Policy:
  1. When a valid Kite credential set is present, Kite is authoritative for
     every dataset Kite exposes. A Kite failure is surfaced honestly; the same
     field is NOT silently substituted from Yahoo/NSE.
  2. Non-Kite providers are fallback only when Kite is unavailable, or for
     datasets Kite does not expose.

Usage:
    from data.market_data import get_provider, get_historical_data
    mdp = get_provider()

    # Single quote
    q = mdp.quote("RELIANCE")
    # {"price": 2450.0, "prev_close": 2420.0, "chg_pct": 1.24, "volume": 1234567}

    # Batch quotes (one API call for Kite)
    quotes = mdp.quotes(["RELIANCE", "TCS", "INFY"])

    # Historical OHLCV (Kite authoritative while Kite is available)
    df = get_historical_data("RELIANCE", interval="day", from_date="2025-01-01", to_date="2025-03-01")
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# yfinance apna '404 / possibly delisted / 1 Failed download' spam khud
# print karta hai — uska logger yahin muzzle karo, hamare debug logs kaafi hain
try:
    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
except Exception:
    pass


def _kite_available() -> bool:
    try:
        from data.kite_client import kite_credentials_available
        return bool(kite_credentials_available())
    except Exception:
        return bool(os.getenv("KITE_API_KEY", "") and os.getenv("KITE_ACCESS_TOKEN", ""))


# ── Kite historical data ──────────────────────────────────────────────────────

_kite_instruments_cache: Optional[pd.DataFrame] = None


def _get_kite_raw():
    from data.kite_client import KiteClient
    return KiteClient()._kite


def _get_instrument_token(symbol: str) -> int:
    global _kite_instruments_cache
    kite = _get_kite_raw()
    if _kite_instruments_cache is None:
        _kite_instruments_cache = pd.DataFrame(kite.instruments("NSE"))
    row = _kite_instruments_cache[_kite_instruments_cache["tradingsymbol"] == symbol]
    if row.empty:
        raise ValueError(f"Instrument {symbol} not found in NSE instrument list")
    return int(row.iloc[0]["instrument_token"])


def get_historical_data_kite(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV from Kite Connect. Raises on failure."""
    if not _kite_available():
        raise ValueError("Kite credentials not set. Add KITE_API_KEY + KITE_ACCESS_TOKEN to .env")
    kite = _get_kite_raw()
    token = _get_instrument_token(symbol)
    from_dt = from_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_dt = to_date or datetime.today().strftime("%Y-%m-%d")
    data = kite.historical_data(token, from_dt, to_dt, interval)
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No data returned from Kite for {symbol}")
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    return df


def _init_yf_cache() -> None:
    """Point yfinance tz-cache at /tmp to avoid SQLite lock errors under concurrent threads."""
    try:
        import os, tempfile, yfinance as yf
        cache_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except Exception:
        pass


_init_yf_cache()


def get_historical_data_yfinance(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV from yfinance (NSE suffix auto-appended)."""
    import yfinance as yf
    _interval_map = {
        "day": "1d", "minute": "1m", "5minute": "5m",
        "15minute": "15m", "60minute": "1h", "week": "1wk", "month": "1mo",
    }
    yf_interval = _interval_map.get(interval, "1d")
    # Index symbols (^NSEI etc.) don't get .NS suffix
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        ticker = symbol
    else:
        ticker = symbol + ".NS"
    from_dt = from_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_dt = to_date or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=from_dt, end=to_dt, interval=yf_interval,
                     progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def get_historical_data(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Unified OHLCV entry point.
    source = "auto"     → Kite when available; no same-field fallback on Kite failure.
                          yfinance is used only when Kite credentials are unavailable.
    source = "kite"     → Kite only (raises on failure)
    source = "yfinance" → explicit yfinance override
    """
    if source == "yfinance":
        return get_historical_data_yfinance(symbol, interval, from_date, to_date)
    if source == "kite":
        return get_historical_data_kite(symbol, interval, from_date, to_date)
    # auto — dead-symbol registry first: a delisted symbol must not cost
    # a Kite miss + a Yahoo 404 on EVERY cycle
    try:
        from data.dead_symbols import is_dead, mark_dead
    except Exception:
        is_dead = lambda s: False           # noqa: E731
        mark_dead = lambda s, r="": None    # noqa: E731
    if is_dead(symbol):
        raise ValueError(f"{symbol}: dead-symbol registry (delisted?) — skip")
    if _kite_available():
        try:
            return get_historical_data_kite(symbol, interval, from_date, to_date)
        except Exception as exc:
            if "not found in NSE instrument" in str(exc):
                mark_dead(symbol, "kite instrument missing")
            log.warning(
                "kite_hist_authoritative_failure",
                symbol=symbol,
                interval=interval,
                error=str(exc)[:160],
            )
            raise
    return get_historical_data_yfinance(symbol, interval, from_date, to_date)


# ── Live quote providers ──────────────────────────────────────────────────────

def _empty_quote() -> dict:
    return {"price": 0.0, "prev_close": 0.0, "chg_pct": 0.0, "volume": 0, "avg_volume": 1}


def _yf_history_closes(symbol: str, days: int) -> list[float]:
    try:
        import yfinance as yf
        h = yf.Ticker(symbol + ".NS").history(period=f"{max(days + 5, 30)}d")
        if h is None or h.empty:
            return []
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]
        return [float(x) for x in h["Close"].dropna().tolist()[-days:]]
    except Exception:
        return []


_KITE_INDEX_SYMBOLS = {
    "NIFTY50": "NSE:NIFTY 50",
    "NIFTY 50": "NSE:NIFTY 50",
    "^NSEI": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "NIFTY BANK": "NSE:NIFTY BANK",
    "^NSEBANK": "NSE:NIFTY BANK",
}


class _KiteProvider:
    def __init__(self):
        from data.kite_client import KiteClient
        self._kite_client = KiteClient()
        self._exchange = os.getenv("KITE_EXCHANGE", "NSE")

    def _key(self, symbol: str) -> str:
        return _KITE_INDEX_SYMBOLS.get(symbol.upper(), (
            symbol if ":" in symbol else f"{self._exchange}:{symbol}"
        ))

    @staticmethod
    def _normalize_quote(payload: dict) -> dict:
        price = float(payload.get("last_price") or 0.0)
        ohlc = payload.get("ohlc") or {}
        prev = float(ohlc.get("close") or price or 0.0)
        volume = float(payload.get("volume") or 0.0)
        chg = float(
            payload.get("change")
            or payload.get("net_change")
            or ((price - prev) / prev * 100 if prev else 0.0)
        )
        return {
            "price": price,
            "prev_close": prev,
            "chg_pct": chg,
            "volume": volume,
            "avg_volume": volume or 1.0,
            "open": float(ohlc.get("open") or price or 0.0),
            "high": float(ohlc.get("high") or price or 0.0),
            "low": float(ohlc.get("low") or price or 0.0),
            "average_price": float(payload.get("average_price") or 0.0),
            "last_quantity": int(payload.get("last_quantity") or 0),
            "buy_quantity": int(payload.get("buy_quantity") or 0),
            "sell_quantity": int(payload.get("sell_quantity") or 0),
            "oi": int(payload.get("oi") or 0),
            "oi_day_high": int(payload.get("oi_day_high") or 0),
            "oi_day_low": int(payload.get("oi_day_low") or 0),
            "depth": payload.get("depth") or {},
            "timestamp": payload.get("timestamp"),
            "last_trade_time": payload.get("last_trade_time"),
            "instrument_token": payload.get("instrument_token"),
            "source": "kite",
        }

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        try:
            key_by_symbol = {symbol: self._key(symbol) for symbol in symbols}
            out: dict[str, dict] = {}
            keys = list(key_by_symbol.values())
            raw: dict = {}
            for start in range(0, len(keys), 500):
                raw.update(self._kite_client.get_quote(keys[start:start + 500]))
            for symbol, key in key_by_symbol.items():
                payload = raw.get(key)
                if not isinstance(payload, dict):
                    out[symbol] = {**_empty_quote(), "source": "kite", "unavailable": True}
                    continue
                out[symbol] = self._normalize_quote(payload)
            return out
        except Exception as exc:
            log.warning("kite_provider_quotes_failed", error=str(exc)[:160])
            return {
                symbol: {**_empty_quote(), "source": "kite", "unavailable": True}
                for symbol in symbols
            }

    def live_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Fast Kite LTP fetch. No alternate-source substitution."""
        if not symbols:
            return {}
        try:
            key_by_symbol = {symbol: self._key(symbol) for symbol in symbols}
            raw = self._kite_client.raw.ltp(list(key_by_symbol.values()))
            return {
                symbol: float((raw.get(key) or {}).get("last_price") or 0.0)
                for symbol, key in key_by_symbol.items()
                if float((raw.get(key) or {}).get("last_price") or 0.0) > 0
            }
        except Exception as exc:
            log.warning("kite_ltp_failed", error=str(exc)[:160])
            return {}

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        """Close history from Kite only while this provider is active."""
        if days <= 0:
            return []
        end = datetime.today()
        start = end - timedelta(days=max(45, int(days) * 3))
        try:
            df = get_historical_data_kite(
                symbol,
                interval="day",
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            log.warning("kite_history_closes_failed", symbol=symbol, error=str(exc)[:160])
            return []
        if "close" not in df.columns:
            return []
        return [float(value) for value in df["close"].dropna().tolist()[-int(days):]]

    @property
    def source(self) -> str:
        return "kite"


class _GoogleFinanceProvider:
    """Non-Kite fallback provider — routes through the unified quote
    chain (data.live_quotes: NSE snapshot → Google), yfinance backup.
    Name kept for backwards compat; Google is no longer the first hop."""

    def __init__(self):
        self._yf_backup = _YFinanceProvider()

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        try:
            from data.live_quotes import get_live_quotes
            for sym, q in get_live_quotes(symbols).items():
                price = float(q["price"])
                chg = float(q.get("chg_pct") or 0)
                prev = price / (1 + chg / 100) if chg > -100 else 0.0
                out[sym] = {
                    "price": price, "prev_close": round(prev, 2),
                    "chg_pct": chg, "volume": 0.0, "avg_volume": 1.0,
                }
        except Exception:
            pass
        missing = [s for s in symbols if s not in out]
        if missing:
            out.update(self._yf_backup.quotes(missing))
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        # Prefer the official bhav store; yfinance only as last resort
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(symbol)
            if df is not None and len(df) >= days:
                return [float(x) for x in df["close"].values[-days:]]
        except Exception:
            pass
        return _yf_history_closes(symbol, days)

    @property
    def source(self) -> str:
        return "unified_fallback"


class _YFinanceProvider:
    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out = {}
        import yfinance as yf
        for sym in symbols:
            try:
                ticker_sym = sym + ".NS" if not sym.endswith(".NS") else sym
                t = yf.Ticker(ticker_sym)

                # fast_info gives real-time last price (no API key needed)
                fi = getattr(t, "fast_info", None)
                price = float(fi.last_price) if fi and getattr(fi, "last_price", None) else 0.0
                prev  = float(fi.previous_close) if fi and getattr(fi, "previous_close", None) else 0.0
                vol   = float(fi.three_month_average_volume or 0) if fi else 0.0
                avg_v = vol or 1.0

                # Fallback if fast_info unavailable
                if price == 0.0:
                    h = t.history(period="5d", progress=False)
                    if h is None or len(h) < 1:
                        out[sym] = _empty_quote()
                        continue
                    if isinstance(h.columns, pd.MultiIndex):
                        h.columns = [c[0] for c in h.columns]
                    price = float(h["Close"].iloc[-1])
                    prev  = float(h["Close"].iloc[-2]) if len(h) >= 2 else price
                    vol   = float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                    avg_v = float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns and len(h) > 1 else 1

                chg = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {
                    "price": price, "prev_close": prev,
                    "chg_pct": chg, "volume": vol,
                    "avg_volume": avg_v or 1,
                }
            except Exception:
                out[sym] = _empty_quote()
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)

    @property
    def source(self) -> str:
        return "yfinance"


_provider: Optional[_KiteProvider | _GoogleFinanceProvider | _YFinanceProvider] = None


def get_provider() -> _KiteProvider | _GoogleFinanceProvider | _YFinanceProvider:
    """Return a cached provider under the Kite-authoritative source policy.
    Public providers are available only when Kite credentials are absent.

    Auto-upgrade: app Kite-login se pehle khuli ho toh Google cache ho
    jata tha aur login ke BAAD bhi din bhar scrape se quotes aate the.
    Ab har call pe check: Kite available ho gaya → Kite pe switch."""
    global _provider
    if _provider is None or (
            not isinstance(_provider, _KiteProvider) and _kite_available()):
        if _kite_available():
            # Kite authority is explicit: construction/auth problems must be
            # visible, never converted into a same-field public-source switch.
            _provider = _KiteProvider()
        else:
            _provider = _GoogleFinanceProvider()
    return _provider


def reset_provider():
    """Call this after updating .env at runtime to pick up new credentials."""
    global _provider, _kite_instruments_cache
    _provider = None
    _kite_instruments_cache = None
