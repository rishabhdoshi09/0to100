"""
Historical data ingestion pipeline.
Downloads OHLCV bars from Kite (or Yahoo Finance fallback),
writes to QuestDB, and caches locally as Parquet.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from config.settings import settings
from data.ingestion.kite_client import KiteClient
from data.storage.questdb_client import QuestDBClient


_CACHE_DIR = Path("data/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_NSE_SUFFIX = ".NS"  # Yahoo Finance NSE suffix


class HistoricalIngestion:
    """
    Orchestrates downloading, caching, and storing historical OHLCV data.
    Priority order: Kite Connect → Yahoo Finance → local Parquet cache.
    """

    def __init__(self) -> None:
        self._kite = KiteClient()
        self._qdb = QuestDBClient()

    def ingest_all(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        interval: str = "day",
    ) -> Dict[str, pd.DataFrame]:
        symbols = symbols or settings.symbol_list
        from_date = from_date or datetime.strptime(settings.backtest_start_date, "%Y-%m-%d").date()
        to_date = to_date or datetime.strptime(settings.backtest_end_date, "%Y-%m-%d").date()

        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.ingest_symbol(sym, from_date, to_date, interval)
            if df is not None and not df.empty:
                result[sym] = df

        logger.info(f"Historical ingestion complete: {len(result)}/{len(symbols)} symbols")
        return result

    def ingest_symbol(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str = "day",
    ) -> Optional[pd.DataFrame]:
        cache_path = _CACHE_DIR / f"{symbol}_{interval}_{from_date}_{to_date}.parquet"

        # 1. Check local cache
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.debug(f"Cache hit: {symbol} ({len(df)} bars)")
            return df

        # 2. Try Kite Connect (or synthetic fallback built into KiteClient)
        df = self._kite.get_historical(symbol, from_date, to_date, interval)

        # 3. If Kite returned empty, try Yahoo Finance
        if df is None or df.empty:
            df = self._fetch_yahoo(symbol, from_date, to_date, interval)

        if df is None or df.empty:
            logger.warning(f"No data for {symbol}")
            return None

        df = self._normalise(df)

        # 4. Cache locally
        df.to_parquet(cache_path)

        # 5. Write to QuestDB (non-blocking; if QuestDB is down, continue)
        try:
            self._qdb.write_ohlcv(symbol, df, interval=interval)
        except Exception as exc:
            logger.warning(f"QuestDB write skipped for {symbol}: {exc}")

        logger.info(f"Ingested {symbol}: {len(df)} bars ({from_date} → {to_date})")
        return df

    def load_from_cache(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        interval: str = "day",
    ) -> Dict[str, pd.DataFrame]:
        """Load all available cached data (for backtesting without network)."""
        symbols = symbols or settings.symbol_list
        from_date = from_date or datetime.strptime(settings.backtest_start_date, "%Y-%m-%d").date()
        to_date = to_date or datetime.strptime(settings.backtest_end_date, "%Y-%m-%d").date()

        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            cache_path = _CACHE_DIR / f"{sym}_{interval}_{from_date}_{to_date}.parquet"
            if cache_path.exists():
                result[sym] = pd.read_parquet(cache_path)
            else:
                # Generate synthetic and cache
                df = self._kite.get_historical(sym, from_date, to_date, interval)
                if df is not None and not df.empty:
                    df = self._normalise(df)
                    df.to_parquet(cache_path)
                    result[sym] = df

        return result

    # ── Yahoo Finance fallback ─────────────────────────────────────────────

    @staticmethod
    def _fetch_yahoo(
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        yf_interval_map = {
            "day": "1d",
            "60minute": "1h",
            "15minute": "15m",
            "5minute": "5m",
            "minute": "1m",
        }
        yf_interval = yf_interval_map.get(interval, "1d")
        ticker = f"{symbol}{_NSE_SUFFIX}"

        try:
            df = yf.download(
                ticker,
                start=from_date,
                end=to_date,
                interval=yf_interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            logger.info(f"Yahoo Finance: {symbol} {len(df)} bars")
            return df
        except Exception as exc:
            logger.warning(f"Yahoo Finance failed for {symbol}: {exc}")
            return None

    # ── Normalisation ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        df.columns = [c.lower() for c in df.columns]
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0).astype(int)
        return df
