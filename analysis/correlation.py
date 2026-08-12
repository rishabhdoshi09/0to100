"""
Correlation Analyser.

Computes pairwise return correlations across a symbol universe
and identifies highly correlated pairs (>0.7 by default).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_HIGH_CORR_THRESHOLD = 0.7
_MAX_SYMBOLS = 20


class CorrelationAnalyzer:
    """
    Compute a return correlation matrix for a list of symbols.

    Usage
    -----
    ca = CorrelationAnalyzer()
    matrix = ca.heatmap(["RELIANCE", "TCS", "INFY"], lookback_days=60)
    summary = ca.summary(["RELIANCE", "TCS", "INFY"])
    """

    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher

    def heatmap(
        self,
        symbols: List[str],
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Return the pairwise return-correlation matrix as a DataFrame.

        Symbols with no data are silently excluded.
        """
        symbols = list(dict.fromkeys(symbols))[:_MAX_SYMBOLS]

        close_dict: Dict[str, pd.Series] = {}
        for sym in symbols:
            s = self._fetch_close(sym, lookback_days)
            if s is not None and len(s) >= 10:
                close_dict[sym] = s

        if len(close_dict) < 2:
            log.warning("correlation_insufficient_symbols", available=len(close_dict))
            return pd.DataFrame()

        prices = pd.DataFrame(close_dict)
        prices = prices.sort_index().ffill()
        returns = prices.pct_change().dropna(how="all")

        corr = returns.corr()
        log.info("correlation_computed", symbols=list(corr.columns), rows=len(returns))
        return corr

    def summary(
        self,
        symbols: List[str],
        lookback_days: int = 60,
    ) -> Dict[str, Any]:
        """
        Return correlation matrix plus per-symbol average correlation
        and a list of highly correlated pairs.
        """
        corr = self.heatmap(symbols, lookback_days)
        if corr.empty:
            return {"error": "insufficient_data", "matrix": {}, "avg_correlation": {}, "high_corr_pairs": []}

        avg_corr: Dict[str, float] = {}
        for sym in corr.columns:
            others = corr[sym].drop(labels=[sym])
            avg_corr[sym] = round(float(others.mean()), 3)

        high_pairs: List[Dict[str, Any]] = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= _HIGH_CORR_THRESHOLD:
                    high_pairs.append({
                        "symbol_a": cols[i],
                        "symbol_b": cols[j],
                        "correlation": round(float(val), 3),
                    })

        high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "matrix": corr.round(3).to_dict(),
            "avg_correlation": avg_corr,
            "high_corr_pairs": high_pairs,
            "lookback_days": lookback_days,
            "symbols_used": list(corr.columns),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── Data fetching ─────────────────────────────────────────────────────

    def _fetch_close(self, symbol: str, lookback_days: int) -> Optional[pd.Series]:
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")

        # Kite
        try:
            fetcher = self._ensure_fetcher()
            df = fetcher.fetch(symbol, from_d, to_d, "day")
            if df is not None and len(df) >= 10:
                return df["close"].astype(float)
        except Exception as exc:
            log.warning("corr_kite_failed", symbol=symbol, error=str(exc))

        # yfinance
        try:
            import yfinance as yf
            df = yf.download(f"{symbol}.NS", start=from_d, end=to_d, progress=False)
            if df is not None and len(df) >= 10:
                import pandas as _pd
                if isinstance(df.columns, _pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                return df["Close"].astype(float)
        except Exception as exc:
            log.warning("corr_yf_failed", symbol=symbol, error=str(exc))

        return None

    def _ensure_fetcher(self):
        if self._fetcher is None:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            self._fetcher = HistoricalDataFetcher(KiteClient(), InstrumentManager())
        return self._fetcher
