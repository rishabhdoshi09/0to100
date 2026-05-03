"""Dynamic universe provider.

Priority order for instrument data:
  1. KiteConnect SDK (requires KITE_API_KEY + KITE_ACCESS_TOKEN)
  2. Public Kite instruments CSV: https://api.kite.trade/instruments/NSE
     (no credentials — always available)

Both paths populate the same ``instruments_cache`` SQLite table and are
called by ``refresh_universe()``.  ``get_active_universe()`` returns cached
cash-equity symbols (.NS-suffixed) for the decision engine.
"""
from __future__ import annotations

import csv
import io
import os

import requests

from sq_ai.portfolio.tracker import PortfolioTracker


_PUBLIC_URL = "https://api.kite.trade/instruments/NSE"


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


def _normalise(row: dict) -> dict:
    """Normalise a raw instrument dict (SDK or CSV) to a consistent shape."""
    ts = row.get("tradingsymbol") or row.get("trading_symbol") or ""
    return {
        "trading_symbol":  ts,
        "instrument_token": _int_or_none(row.get("instrument_token")),
        "name":             row.get("name", ""),
        "instrument_type":  row.get("instrument_type", ""),
        "segment":          row.get("segment", ""),
        "lot_size":         _int_or_none(row.get("lot_size")),
        "tick_size":        _float_or_none(row.get("tick_size")),
    }


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v not in (None, "", "None") else None
    except (ValueError, TypeError):
        return None


def _float_or_none(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (ValueError, TypeError):
        return None


def _fetch_via_sdk(k) -> list[dict]:
    try:
        raw = k.instruments("NSE")
    except Exception as exc:                            # pragma: no cover
        print(f"[universe] kite.instruments failed: {exc}")
        return []
    return [_normalise(r) for r in raw]


def _fetch_via_public_csv() -> list[dict]:
    """Download the public NSE instruments CSV from Kite (no auth)."""
    try:
        r = requests.get(_PUBLIC_URL, timeout=30)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        return [_normalise(row) for row in reader]
    except Exception as exc:                            # pragma: no cover
        print(f"[universe] public CSV fetch failed: {exc}")
        return []


def _filter_eq(instruments: list[dict]) -> list[dict]:
    """Keep only cash-equity instruments traded on the NSE cash segment."""
    return [
        i for i in instruments
        if i.get("segment") in {"NSE", "NSE-EQ", None}
        and i.get("instrument_type") in {"EQ", None, ""}
        and i.get("trading_symbol")
    ]


def refresh_universe(tracker: PortfolioTracker | None = None) -> int:
    """Pull full NSE instrument list into SQLite.

    Tries authenticated Kite SDK first; falls back to the public CSV so
    the cockpit always has a fresh universe even before market open.
    Returns the number of rows cached.
    """
    tracker = tracker or PortfolioTracker()
    k = _kite_client()
    rows = _fetch_via_sdk(k) if k else _fetch_via_public_csv()
    if not rows:
        return 0
    eq_rows = _filter_eq(rows)
    return tracker.cache_instruments(eq_rows)


def fetch_all_instruments(exchange: str = "NSE") -> list[dict]:
    """Return every instrument for ``exchange`` — EQ + FO + others.

    Used by the full listing UI.  Tries SDK first, then public CSV.
    Returns normalised dicts, not filtered to EQ only.
    """
    k = _kite_client()
    if k:
        try:
            raw = k.instruments(exchange)
            return [_normalise(r) for r in raw]
        except Exception:                               # pragma: no cover
            pass
    # public CSV is NSE-only; for other exchanges we can only use SDK
    if exchange.upper() == "NSE":
        return _fetch_via_public_csv()
    return []


def _yf_suffix(symbol: str) -> str:
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


__all__ = ["refresh_universe", "fetch_all_instruments", "get_active_universe"]
