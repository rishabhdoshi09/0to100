"""
US universe — EVERY listed US common stock, from the authoritative source.

NASDAQ Trader publishes the official symbol directory (the US equivalent
of NSE's listed-stock list): nasdaqlisted.txt (Nasdaq) + otherlisted.txt
(NYSE, NYSE American, etc.). We fetch both, keep only clean COMMON stocks
(no test issues, no ETFs, no warrants/units/preferreds), and cache the
result for the day. If the fetch fails we fall back to a curated liquid
list so the scanner is never empty.

The scanner's own penny/illiquid filters then drop the junk during
analysis, so feeding the full ~5,000-name universe is safe.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_CACHE_FILE = Path(__file__).resolve().parent.parent / "logs" / "us_symbols.json"
_TTL_S = 86400                      # refresh the listing once a day
_CLEAN = re.compile(r"^[A-Z]{1,5}$")  # plain common-stock tickers only
_MAX = 8000

_NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Fallback: curated liquid names if the directory fetch fails.
_CURATED = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom",
    "AMD": "Advanced Micro Devices", "NFLX": "Netflix", "CRM": "Salesforce",
    "ORCL": "Oracle", "ADBE": "Adobe", "QCOM": "Qualcomm", "MU": "Micron",
    "PANW": "Palo Alto", "CRWD": "CrowdStrike", "SNOW": "Snowflake",
    "PLTR": "Palantir", "UBER": "Uber", "ABNB": "Airbnb", "SHOP": "Shopify",
    "COIN": "Coinbase", "COST": "Costco", "WMT": "Walmart", "HD": "Home Depot",
    "NKE": "Nike", "MCD": "McDonald's", "JPM": "JPMorgan", "V": "Visa",
    "MA": "Mastercard", "LLY": "Eli Lilly", "UNH": "UnitedHealth",
    "XOM": "Exxon", "CVX": "Chevron", "CAT": "Caterpillar", "BA": "Boeing",
    "GE": "GE Aerospace", "SMCI": "Super Micro", "ARM": "Arm", "MSTR": "MicroStrategy",
}


def _parse_symbol_file(text: str, kind: str) -> dict[str, str]:
    """Pure, testable parser for a NASDAQ Trader pipe-delimited file.
    kind='nasdaq' → cols Symbol|Name|MktCat|Test|FinStatus|Lot|ETF|...
    kind='other'  → cols ACTSymbol|Name|Exch|CQS|ETF|Lot|Test|NASDAQSymbol
    Returns {symbol: name} of clean common stocks only."""
    out: dict[str, str] = {}
    sym_i, name_i = 0, 1
    etf_i = 6 if kind == "nasdaq" else 4
    test_i = 3 if kind == "nasdaq" else 6
    for line in text.splitlines():
        if not line or "|" not in line:
            continue
        cols = line.split("|")
        sym = cols[sym_i].strip().upper()
        if sym in ("SYMBOL", "ACT SYMBOL") or sym.startswith("FILE CREATION"):
            continue                                   # header / footer
        if len(cols) <= max(etf_i, test_i):
            continue
        if cols[test_i].strip() == "Y":               # test issue
            continue
        if cols[etf_i].strip() == "Y":                # ETF, not a stock
            continue
        if not _CLEAN.match(sym):                     # warrants/units/pref out
            continue
        out[sym] = cols[name_i].strip()
    return out


def _fetch_all() -> dict[str, str]:
    import requests
    hdrs = {"User-Agent": "Mozilla/5.0"}
    merged: dict[str, str] = {}
    for url, kind in ((_NASDAQ_URL, "nasdaq"), (_OTHER_URL, "other")):
        try:
            r = requests.get(url, headers=hdrs, timeout=20)
            if r.status_code == 200 and r.text:
                merged.update(_parse_symbol_file(r.text, kind))
        except Exception as exc:
            log.debug("us_symbol_fetch_failed", url=url, error=str(exc)[:80])
    return merged


def _load_cached() -> dict[str, str] | None:
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < _TTL_S and data.get("symbols"):
                return data["symbols"]
    except Exception:
        pass
    return None


def _save_cache(symbols: dict[str, str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"ts": time.time(),
                                           "symbols": symbols}))
    except Exception as exc:
        log.debug("us_symbol_cache_save_failed", error=str(exc))


def get_us_universe_with_names() -> dict[str, str]:
    """Full listed US common-stock universe (cached daily). Falls back to
    the curated liquid list if the directory can't be reached."""
    cached = _load_cached()
    if cached:
        return cached
    fetched = _fetch_all()
    if len(fetched) >= 500:                # sanity: a real listing has thousands
        fetched = dict(list(fetched.items())[:_MAX])
        _save_cache(fetched)
        log.info("us_universe_loaded", count=len(fetched))
        return fetched
    log.warning("us_universe_fallback_curated", fetched=len(fetched))
    return dict(_CURATED)


def get_us_universe() -> list[str]:
    return sorted(get_us_universe_with_names().keys())
