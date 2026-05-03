"""
Fundamental signal module.

Combines:
  • Earnings surprise: (actual EPS - estimated EPS) / |estimated EPS|
  • Revenue surprise: similar
  • Macro surprise index: weighted sum of recent economic releases vs expectations
  • Alpha Vantage macro data (GDP, CPI, etc.) — optional

All inputs are normalised to a signal ∈ [0, 1].
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

from config.settings import settings
from data.storage.postgres_client import PostgresClient


class FundamentalsSignal:
    def __init__(self) -> None:
        self._pg = PostgresClient()

    # ── Earnings surprise ─────────────────────────────────────────────────

    def earnings_surprise_signal(self, symbol: str) -> float:
        """
        Returns P(positive earnings momentum) ∈ [0, 1].
        Uses last 4 quarters of earnings from PostgreSQL.
        """
        try:
            df = self._pg.get_latest_earnings(symbol, n=4)
            if df.empty:
                return 0.5

            # Filter to rows that have both actual and estimate
            df = df.dropna(subset=["eps_actual", "eps_est"])
            if df.empty:
                return 0.5

            surprises = df["surprise_pct"].fillna(0).values
            if len(surprises) == 0:
                return 0.5

            # Exponentially weight more recent quarters
            weights = np.exp(-0.5 * np.arange(len(surprises)))
            weights /= weights.sum()

            mean_surprise = float(np.dot(surprises, weights))

            # Map [-100%, +100%] → [0, 1] using sigmoid-like transform
            signal = 1 / (1 + np.exp(-mean_surprise / 10))
            return float(np.clip(signal, 0.0, 1.0))

        except Exception as exc:
            logger.debug(f"Earnings signal failed for {symbol}: {exc}")
            return 0.5

    # ── Revenue surprise ──────────────────────────────────────────────────

    def revenue_surprise_signal(self, symbol: str) -> float:
        """Same approach as earnings but using revenue surprise."""
        try:
            df = self._pg.get_latest_earnings(symbol, n=4)
            if df.empty or "revenue_actual" not in df.columns:
                return 0.5

            df = df.dropna(subset=["revenue_actual", "revenue_est"])
            if df.empty:
                return 0.5

            surprise_pct = (
                (df["revenue_actual"] - df["revenue_est"]) / df["revenue_est"].abs().replace(0, np.nan)
            ).fillna(0).values

            weights = np.exp(-0.5 * np.arange(len(surprise_pct)))
            weights /= weights.sum()
            mean_surprise = float(np.dot(surprise_pct, weights))

            return float(np.clip(1 / (1 + np.exp(-mean_surprise / 0.1)), 0.0, 1.0))

        except Exception as exc:
            logger.debug(f"Revenue signal failed: {exc}")
            return 0.5

    # ── Macro surprise index ───────────────────────────────────────────────

    def macro_surprise_signal(self) -> float:
        """
        Fetch macro indicators from PostgreSQL and compute a surprise index.
        Returns P(macro positive) ∈ [0, 1].
        """
        try:
            df = self._pg.read_sql(
                "SELECT * FROM macro_indicators ORDER BY release_date DESC LIMIT 20"
            )
            if df.empty:
                return 0.5

            df = df.dropna(subset=["actual", "expected"])
            if df.empty:
                return 0.5

            surprises = (df["actual"] - df["expected"]).fillna(0).values
            weights = np.exp(-0.2 * np.arange(len(surprises)))
            weights /= weights.sum()

            mean_surprise = float(np.dot(surprises, weights))
            return float(np.clip((mean_surprise + 1) / 2, 0.0, 1.0))

        except Exception as exc:
            logger.debug(f"Macro signal failed: {exc}")
            return 0.5

    # ── Alpha Vantage macro ───────────────────────────────────────────────

    def fetch_av_macro(self, function: str = "REAL_GDP") -> Optional[pd.DataFrame]:
        """
        Fetch macro data from Alpha Vantage.
        function examples: REAL_GDP, CPI, FEDERAL_FUNDS_RATE, UNEMPLOYMENT
        """
        if not settings.alpha_vantage_key:
            return None
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": function,
                    "interval": "quarterly",
                    "apikey": settings.alpha_vantage_key,
                },
                timeout=15,
            )
            data = r.json()
            if "data" not in data:
                return None
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
        except Exception as exc:
            logger.debug(f"Alpha Vantage {function} fetch failed: {exc}")
            return None

    # ── Combined signal ────────────────────────────────────────────────────

    def get_combined_signal(self, symbol: str) -> float:
        """
        Weighted combination of all fundamental signals.
        Weights: earnings=0.4, revenue=0.3, macro=0.3.
        """
        earnings = self.earnings_surprise_signal(symbol)
        revenue = self.revenue_surprise_signal(symbol)
        macro = self.macro_surprise_signal()

        combined = 0.4 * earnings + 0.3 * revenue + 0.3 * macro
        logger.debug(
            f"Fundamentals: {symbol} earnings={earnings:.2f} revenue={revenue:.2f} "
            f"macro={macro:.2f} combined={combined:.2f}"
        )
        return float(combined)
