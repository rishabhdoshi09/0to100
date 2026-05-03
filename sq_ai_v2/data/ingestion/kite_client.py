"""
Thin wrapper around kiteconnect.KiteConnect.
Falls back to synthetic data generation when no API key is configured.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


class KiteClient:
    """
    Wraps KiteConnect for historical and live data.
    When kite_api_key is empty, every method returns synthetic OHLCV data
    so the rest of the pipeline can run fully offline.
    """

    _EXCHANGE = "NSE"

    def __init__(self) -> None:
        self._kite = None
        if settings.kite_api_key:
            try:
                from kiteconnect import KiteConnect
                self._kite = KiteConnect(api_key=settings.kite_api_key)
                if settings.kite_access_token:
                    self._kite.set_access_token(settings.kite_access_token)
                logger.info("KiteConnect initialised")
            except Exception as exc:
                logger.warning(f"KiteConnect init failed ({exc}); using synthetic data")
        else:
            logger.warning("KITE_API_KEY not set — synthetic data mode active")

    # ── Authentication ────────────────────────────────────────────────────

    def generate_login_url(self) -> str:
        if self._kite is None:
            return "https://kite.zerodha.com/  (set KITE_API_KEY first)"
        return self._kite.login_url()

    def complete_login(self, request_token: str) -> str:
        """Exchange request token for access token. Call once after browser login."""
        if self._kite is None:
            raise RuntimeError("KiteConnect not initialised")
        data = self._kite.generate_session(
            request_token, api_secret=settings.kite_api_secret
        )
        access_token = data["access_token"]
        self._kite.set_access_token(access_token)
        logger.info("Kite login complete", access_token=access_token[:8] + "...")
        return access_token

    # ── Historical data ───────────────────────────────────────────────────

    def get_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Returns OHLCV DataFrame indexed by datetime.
        interval: "minute", "5minute", "15minute", "60minute", "day"
        """
        if self._kite is not None:
            return self._fetch_kite_historical(symbol, from_date, to_date, interval)
        return self._synthetic_ohlcv(symbol, from_date, to_date, interval)

    def _fetch_kite_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str,
    ) -> pd.DataFrame:
        try:
            instrument_token = self._resolve_token(symbol)
            records = self._kite.historical_data(
                instrument_token,
                from_date,
                to_date,
                interval,
            )
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df.columns = [c.lower() for c in df.columns]
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
        except Exception as exc:
            logger.error(f"Kite historical fetch failed for {symbol}: {exc}")
            return self._synthetic_ohlcv(symbol, from_date, to_date, interval)

    def _resolve_token(self, symbol: str) -> int:
        instruments = self._kite.instruments(self._EXCHANGE)
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                return inst["instrument_token"]
        raise ValueError(f"Symbol {symbol} not found on {self._EXCHANGE}")

    # ── Synthetic OHLCV (offline fallback) ────────────────────────────────

    @staticmethod
    def _synthetic_ohlcv(
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Geometric Brownian Motion with mild trend and volatility clustering
        so that trained models have something meaningful to learn.
        """
        rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))

        if interval == "day":
            freq = "B"  # business days
        elif interval == "60minute":
            freq = "h"
        elif interval in ("5minute", "15minute"):
            freq = interval.replace("minute", "min")
        else:
            freq = "min"

        dates = pd.bdate_range(
            start=from_date, end=to_date, freq="B" if interval == "day" else "h"
        )
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # GBM parameters per-symbol (seeded for reproducibility)
        mu = rng.uniform(-0.0002, 0.0008)
        sigma = rng.uniform(0.012, 0.025)
        base_price = rng.uniform(200, 3000)

        # GARCH-like volatility clustering
        vol = np.ones(n) * sigma
        for i in range(1, n):
            shock = rng.standard_normal()
            vol[i] = np.sqrt(0.9 * vol[i - 1] ** 2 + 0.1 * (shock * sigma) ** 2)
            vol[i] = np.clip(vol[i], sigma * 0.3, sigma * 3)

        log_returns = rng.normal(mu, vol)
        prices = base_price * np.exp(np.cumsum(log_returns))

        # Build OHLCV
        intraday_range = prices * rng.uniform(0.005, 0.02, size=n)
        highs = prices + intraday_range * rng.uniform(0.3, 0.7, size=n)
        lows = prices - intraday_range * rng.uniform(0.3, 0.7, size=n)
        opens = lows + (highs - lows) * rng.uniform(0, 1, size=n)
        volumes = (rng.lognormal(mean=13, sigma=1, size=n)).astype(int)

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
            index=dates,
        )
        df.index.name = "date"
        logger.debug(f"Synthetic OHLCV: {symbol} {n} bars")
        return df

    # ── Quote (live price) ─────────────────────────────────────────────────

    def get_last_price(self, symbol: str) -> float:
        if self._kite is not None:
            try:
                quote = self._kite.quote(f"{self._EXCHANGE}:{symbol}")
                return float(quote[f"{self._EXCHANGE}:{symbol}"]["last_price"])
            except Exception as exc:
                logger.error(f"Quote failed for {symbol}: {exc}")
        # fallback: random walk around 1000
        return round(random.uniform(500, 2000), 2)

    def get_quotes(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self.get_last_price(s) for s in symbols}
