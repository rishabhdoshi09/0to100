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
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from logger import get_logger
from core.runtime_paths import logs_dir, logs_path

log = get_logger(__name__)

_CACHE_FILE = logs_path("us_symbols.json")
_HISTORY_DIR = logs_dir() / "us_universe_history"
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


def _snapshot_date_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _archive_official_snapshot(symbols: dict[str, str], *, session_date: str | None = None) -> Path | None:
    """Write one immutable official listed-universe snapshot per US date.

    This is the foundation for future point-in-time US replay. A snapshot is
    written only after a sanity-checked NASDAQ Trader fetch; curated fallbacks
    are never archived as if they were official history. Existing dates are
    write-once so later refreshes cannot rewrite the universe seen that day.
    """
    if len(symbols) < 500:
        return None
    day = str(session_date or _snapshot_date_et())[:10]
    if not day:
        return None
    target = _HISTORY_DIR / f"{day}.json"
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        payload = {
            "schema_version": 1,
            "session_date": day,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "NASDAQ_TRADER_SYMBOL_DIRECTORY",
            "source_urls": [_NASDAQ_URL, _OTHER_URL],
            "count": len(symbols),
            "symbols": dict(sorted(symbols.items())),
        }
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        # Fail closed on a same-date writer race: never overwrite PIT history.
        try:
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return target
    except Exception as exc:
        log.warning("us_universe_history_save_failed", date=day, error=str(exc)[:120])
        return None


def available_us_universe_snapshots() -> list[str]:
    try:
        if not _HISTORY_DIR.exists():
            return []
        return sorted(
            p.stem for p in _HISTORY_DIR.glob("????-??-??.json")
            if p.is_file()
        )
    except Exception:
        return []


def load_us_universe_snapshot(session_date: str) -> dict[str, str] | None:
    target = _HISTORY_DIR / f"{str(session_date or '')[:10]}.json"
    try:
        payload = json.loads(target.read_text())
        if str(payload.get("source") or "") != "NASDAQ_TRADER_SYMBOL_DIRECTORY":
            return None
        symbols = payload.get("symbols")
        return dict(symbols) if isinstance(symbols, dict) and symbols else None
    except Exception:
        return None


def _save_cache(symbols: dict[str, str], *, source: str = "") -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({
            "ts": time.time(),
            "source": str(source or ""),
            "symbols": symbols,
        }))
        if str(source or "") == "NASDAQ_TRADER_SYMBOL_DIRECTORY":
            _archive_official_snapshot(symbols)
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
        _save_cache(fetched, source="NASDAQ_TRADER_SYMBOL_DIRECTORY")
        log.info("us_universe_loaded", count=len(fetched))
        return fetched
    log.warning("us_universe_fallback_curated", fetched=len(fetched))
    return dict(_CURATED)


def get_us_universe() -> list[str]:
    return sorted(get_us_universe_with_names().keys())
