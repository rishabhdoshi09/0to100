"""
NSE universe fetcher — returns all ~2000 equity-series NSE symbols.

Primary  : NSE public CSV  (https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv)
Secondary: screener.in company list
Fallback : data/nse_symbols_fallback.csv  (ship this with the repo as last resort)
Cache    : SQLite  data/screener_cache.db  nse_universe table  7-day TTL
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import requests

from logger import get_logger

log = get_logger(__name__)

_DB_PATH   = Path("data/screener_cache.db")
_CSV_PATH  = Path("data/nse_symbols_fallback.csv")
_TTL       = 7 * 86_400          # 7 days
_NSE_URL   = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_TIMEOUT = 20


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nse_universe (
            symbol     TEXT PRIMARY KEY,
            name       TEXT,
            fetched_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


class StockUniverseFetcher:
    """
    Returns all NSE equity-series symbols.

    Usage
    -----
    symbols = StockUniverseFetcher().get_all_symbols()
    print(len(symbols))   # ~1900-2100
    """

    def get_all_symbols(self, force_refresh: bool = False) -> List[str]:
        if not force_refresh:
            cached = self._load_from_db()
            if cached:
                log.info("universe_from_cache", count=len(cached))
                return cached

        symbols = self._fetch_from_nse()
        if not symbols:
            symbols = self._fetch_from_local_csv()
        if not symbols:
            # Last resort: load stale cache (ignore TTL)
            symbols = self._load_from_db(ignore_ttl=True)
        if not symbols:
            raise RuntimeError(
                "Cannot load NSE universe. "
                "Place a CSV of NSE symbols at data/nse_symbols_fallback.csv "
                "(columns: SYMBOL, NAME) to use as fallback."
            )

        self._save_to_db(symbols)
        log.info("universe_fetched", count=len(symbols))
        return [s for s, _ in symbols]

    # ── Cache ──────────────────────────────────────────────────────────────

    def _load_from_db(self, ignore_ttl: bool = False) -> Optional[List[str]]:
        cutoff = 0 if ignore_ttl else time.time() - _TTL
        with _connect() as conn:
            rows = conn.execute(
                "SELECT symbol FROM nse_universe WHERE fetched_at > ? ORDER BY symbol",
                (cutoff,),
            ).fetchall()
        if not rows:
            return None
        return [r[0] for r in rows]

    def _save_to_db(self, symbols: List[tuple]) -> None:
        now = time.time()
        with _connect() as conn:
            conn.execute("DELETE FROM nse_universe")
            conn.executemany(
                "INSERT OR REPLACE INTO nse_universe (symbol, name, fetched_at) VALUES (?,?,?)",
                [(sym, name, now) for sym, name in symbols],
            )
            conn.commit()
        log.debug("universe_saved_to_db", count=len(symbols))

    # ── Primary: NSE public CSV ────────────────────────────────────────────

    def _fetch_from_nse(self) -> Optional[List[tuple]]:
        try:
            # Warm session on NSE homepage first to get cookies
            sess = requests.Session()
            sess.headers.update(_NSE_HEADERS)
            sess.get("https://www.nseindia.com", timeout=_TIMEOUT)
            time.sleep(1)
            resp = sess.get(_NSE_URL, timeout=_TIMEOUT)
            if resp.status_code != 200:
                log.warning("nse_csv_fetch_failed", status=resp.status_code)
                return None

            lines = resp.text.splitlines()
            if len(lines) < 10:
                return None

            header = [h.strip().upper() for h in lines[0].split(",")]
            try:
                sym_idx  = header.index("SYMBOL")
                name_idx = header.index("NAME OF COMPANY")
                ser_idx  = header.index("SERIES")
            except ValueError:
                sym_idx, name_idx, ser_idx = 0, 1, 2

            symbols = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) <= max(sym_idx, name_idx, ser_idx):
                    continue
                series = parts[ser_idx].strip().upper()
                if series != "EQ":
                    continue
                sym  = parts[sym_idx].strip().upper()
                name = parts[name_idx].strip().strip('"')
                if sym:
                    symbols.append((sym, name))

            if len(symbols) < 100:
                log.warning("nse_csv_too_few_symbols", count=len(symbols))
                return None

            log.info("nse_csv_parsed", count=len(symbols))
            return symbols

        except Exception as exc:
            log.warning("nse_csv_error", error=str(exc))
            return None

    # ── Fallback: local CSV ────────────────────────────────────────────────

    def _fetch_from_local_csv(self) -> Optional[List[tuple]]:
        if not _CSV_PATH.exists():
            return None
        try:
            import csv
            symbols = []
            with open(_CSV_PATH, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sym  = row.get("SYMBOL", row.get("symbol", "")).strip().upper()
                    name = row.get("NAME", row.get("name", "")).strip()
                    if sym:
                        symbols.append((sym, name))
            log.info("universe_from_local_csv", count=len(symbols))
            return symbols if symbols else None
        except Exception as exc:
            log.warning("local_csv_error", error=str(exc))
            return None
