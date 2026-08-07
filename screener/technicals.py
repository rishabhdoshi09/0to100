"""
Technicals cache — RSI(14), SMA(20/50), volume ratio, current price.

Downloads 3-month daily OHLCV via yfinance, caches in SQLite (1-day TTL).
On subsequent calls, reads from cache — no network hit.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/screener_cache.db")
_TTL     = 86_400       # 1 day
_MAX_WORKERS = 6        # parallel yfinance downloads
_SLEEP   = 0.3          # polite delay between downloads per thread


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS technicals_cache (
            symbol        TEXT PRIMARY KEY,
            rsi_14        REAL,
            sma_20        REAL,
            sma_50        REAL,
            volume_ratio  REAL,
            current_price REAL,
            fetched_at    REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _compute(df: pd.DataFrame) -> Optional[Dict]:
    """Compute all technicals from a daily OHLCV DataFrame."""
    if df is None or len(df) < 20:
        return None

    # Flatten MultiIndex columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [c.lower() for c in df.columns]

    close  = df["close"].dropna()
    volume = df["volume"].dropna()
    if len(close) < 20:
        return None

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    avg_vol      = float(volume.rolling(30).mean().iloc[-1]) if len(volume) >= 30 else float(volume.mean())
    last_vol     = float(volume.iloc[-1])
    volume_ratio = round(last_vol / avg_vol, 3) if avg_vol > 0 else 1.0

    return {
        "rsi_14":        round(rsi, 2),
        "sma_20":        round(sma_20, 2) if sma_20 else None,
        "sma_50":        round(sma_50, 2) if sma_50 else None,
        "volume_ratio":  volume_ratio,
        "current_price": round(float(close.iloc[-1]), 2),
    }


def _download_one(symbol: str) -> Optional[Dict]:
    """Download and compute technicals for one symbol. Returns None on failure."""
    import yfinance as yf
    try:
        time.sleep(_SLEEP)
        df = yf.download(
            f"{symbol}.NS",
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
        result = _compute(df)
        if result:
            result["symbol"] = symbol
        return result
    except Exception as exc:
        log.debug("technicals_download_failed", symbol=symbol, error=str(exc)[:60])
        return None


class TechnicalsCache:
    """
    Thread-safe-ish SQLite cache for per-symbol technical indicators.

    Usage
    -----
    tc = TechnicalsCache()
    row = tc.get("RELIANCE")          # from cache if fresh
    tc.bulk_update(["RELIANCE","TCS"]) # refresh a list
    """

    def get(self, symbol: str) -> Optional[Dict]:
        """Return cached technicals if fresh (< 1 day old), else None."""
        symbol = symbol.upper()
        cutoff = time.time() - _TTL
        with _connect() as conn:
            row = conn.execute(
                """SELECT rsi_14, sma_20, sma_50, volume_ratio, current_price
                   FROM technicals_cache
                   WHERE symbol = ? AND fetched_at > ?""",
                (symbol, cutoff),
            ).fetchone()
        if row is None:
            return None
        return {
            "rsi_14":        row[0],
            "sma_20":        row[1],
            "sma_50":        row[2],
            "volume_ratio":  row[3],
            "current_price": row[4],
        }

    def get_or_fetch(self, symbol: str) -> Optional[Dict]:
        """Return from cache or download fresh if stale/missing."""
        cached = self.get(symbol)
        if cached is not None:
            return cached
        result = _download_one(symbol)
        if result:
            self._save_one(symbol, result)
        return result

    def bulk_update(
        self,
        symbols: List[str],
        skip_fresh: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Download & cache technicals for a list of symbols in parallel.
        Skips symbols that are already fresh if skip_fresh=True.
        Returns count of symbols successfully updated.
        """
        try:
            from tqdm import tqdm
            _tqdm = tqdm
        except ImportError:
            _tqdm = None

        if skip_fresh:
            # Only download stale / missing ones
            cutoff = time.time() - _TTL
            with _connect() as conn:
                fresh = {
                    r[0] for r in conn.execute(
                        "SELECT symbol FROM technicals_cache WHERE fetched_at > ?",
                        (cutoff,),
                    ).fetchall()
                }
            to_fetch = [s for s in symbols if s.upper() not in fresh]
        else:
            to_fetch = list(symbols)

        if not to_fetch:
            log.info("technicals_all_fresh", count=len(symbols))
            return 0

        log.info("technicals_downloading", count=len(to_fetch))

        updated = 0
        iterator = iter(to_fetch)
        if show_progress and _tqdm:
            iterator = _tqdm(to_fetch, desc="Technicals cache", unit="sym", ncols=80)

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_download_one, sym): sym for sym in to_fetch}
            for fut in as_completed(futures):
                sym    = futures[fut]
                result = fut.result()
                if result:
                    self._save_one(sym, result)
                    updated += 1

        log.info("technicals_cache_updated", updated=updated, skipped=len(to_fetch) - updated)
        return updated

    def _save_one(self, symbol: str, data: Dict) -> None:
        symbol = symbol.upper()
        with _connect() as conn:
            conn.execute(
                """INSERT INTO technicals_cache
                       (symbol, rsi_14, sma_20, sma_50, volume_ratio, current_price, fetched_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET
                       rsi_14=excluded.rsi_14, sma_20=excluded.sma_20,
                       sma_50=excluded.sma_50, volume_ratio=excluded.volume_ratio,
                       current_price=excluded.current_price, fetched_at=excluded.fetched_at""",
                (
                    symbol,
                    data.get("rsi_14"),
                    data.get("sma_20"),
                    data.get("sma_50"),
                    data.get("volume_ratio"),
                    data.get("current_price"),
                    time.time(),
                ),
            )
            conn.commit()
