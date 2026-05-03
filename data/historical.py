"""
Historical OHLCV data downloader via Kite Connect.

Handles:
- Batched range requests (Kite limits per call)
- In-memory caching within session
- Returns clean pandas DataFrames
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from data.kite_client import KiteClient
from data.instruments import InstrumentManager
from logger import get_logger

log = get_logger(__name__)

# Kite rate limit: 3 requests/second, ~max 2000 candles per call.
# For 'day' interval, limit per call is ~2000 days (~5.5 years).
# For 'minute', limit is ~60 days.
_INTERVAL_MAX_DAYS: Dict[str, int] = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
    "week": 2000,
    "month": 2000,
}
_REQUEST_DELAY = 0.4  # seconds between Kite API calls (stay under 3 rps)


class HistoricalDataFetcher:
    def __init__(self, kite: KiteClient, instruments: InstrumentManager) -> None:
        self._kite = kite
        self._instruments = instruments
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "day",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical candles for *symbol*.

        from_date / to_date: "YYYY-MM-DD" strings.
        Returns a DataFrame with DatetimeIndex and columns
        [open, high, low, close, volume].
        """
        cache_key = f"{symbol}:{interval}:{from_date}:{to_date}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        token = self._instruments.token(symbol)
        if token is None:
            log.error("instrument_token_not_found", symbol=symbol)
            return pd.DataFrame()

        max_days = _INTERVAL_MAX_DAYS.get(interval, 400)
        chunks = self._date_chunks(from_date, to_date, max_days)

        frames: list[pd.DataFrame] = []
        for start, end in chunks:
            log.debug(
                "fetching_historical_chunk",
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
            )
            df_chunk = self._kite.get_historical(
                instrument_token=token,
                from_date=start,
                to_date=end,
                interval=interval,
            )
            if not df_chunk.empty:
                frames.append(df_chunk)
            time.sleep(_REQUEST_DELAY)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        self._cache[cache_key] = df
        log.info(
            "historical_fetched",
            symbol=symbol,
            interval=interval,
            rows=len(df),
        )
        return df

    @staticmethod
    def _date_chunks(
        from_date: str, to_date: str, max_days: int
    ) -> list[tuple[str, str]]:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
        chunks: list[tuple[str, str]] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=max_days), end)
            chunks.append((
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            ))
            cursor = chunk_end + timedelta(days=1)
        return chunks
