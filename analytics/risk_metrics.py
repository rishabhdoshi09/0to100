"""
Per-symbol risk metrics.

Computes VaR, CVaR, Sortino, Calmar, Beta, Sharpe, and Max Drawdown
from daily log returns.  Also produces a 1–10 composite risk score.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_TRADING_DAYS = 252
_RISK_FREE_RATE_ANNUAL = 0.065  # 6.5% Indian 10yr gilt
_RISK_FREE_DAILY = _RISK_FREE_RATE_ANNUAL / _TRADING_DAYS
_NIFTY_YF = "^NSEI"


class RiskMetrics:
    """
    Compute a comprehensive risk profile for a single symbol.

    Usage
    -----
    rm = RiskMetrics()
    metrics = rm.compute("RELIANCE")          # fetches data automatically
    metrics = rm.compute("RELIANCE", df=df)   # use pre-fetched DataFrame
    """

    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher

    def compute(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute all risk metrics for *symbol*.

        Parameters
        ----------
        symbol : NSE symbol string
        df     : optional pre-fetched daily OHLCV DataFrame

        Returns
        -------
        dict with all metrics + risk_score (1=safest, 10=riskiest)
        """
        if df is None or df.empty:
            df = self._fetch_daily(symbol, years=2)

        if df is None or len(df) < 30:
            log.warning("risk_metrics_insufficient_data", symbol=symbol)
            return self._empty_metrics(symbol)

        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(returns) < 20:
            return self._empty_metrics(symbol)

        nifty_returns = self._fetch_nifty_returns(len(returns))

        metrics: Dict[str, Any] = {"symbol": symbol}

        # ── Core metrics ───────────────────────────────────────────────
        metrics["var_95"] = self._var(returns, 0.95)
        metrics["var_99"] = self._var(returns, 0.99)
        metrics["cvar_95"] = self._cvar(returns, 0.95)
        metrics["cvar_99"] = self._cvar(returns, 0.99)
        metrics["sharpe_ratio"] = self._sharpe(returns)
        metrics["sortino_ratio"] = self._sortino(returns)
        metrics["max_drawdown_pct"] = self._max_drawdown(df["close"])
        metrics["calmar_ratio"] = self._calmar(returns, metrics["max_drawdown_pct"])
        metrics["beta"] = self._beta(returns, nifty_returns)
        metrics["annualised_volatility_pct"] = round(float(returns.std()) * math.sqrt(_TRADING_DAYS) * 100, 3)
        metrics["annualised_return_pct"] = round(float(returns.mean()) * _TRADING_DAYS * 100, 3)

        # ── Risk score (1–10) ──────────────────────────────────────────
        metrics["risk_score"] = self._risk_score(metrics)
        metrics["timestamp"] = datetime.utcnow().isoformat()

        log.info("risk_metrics_computed", symbol=symbol, risk_score=metrics["risk_score"])
        return metrics

    # ── Metric helpers ────────────────────────────────────────────────────

    @staticmethod
    def _var(returns: pd.Series, confidence: float) -> float:
        """Historical VaR at given confidence level (positive = loss)."""
        q = 1 - confidence
        return round(float(np.percentile(returns, q * 100)), 5)

    @staticmethod
    def _cvar(returns: pd.Series, confidence: float) -> float:
        """Expected loss beyond VaR (CVaR / Expected Shortfall)."""
        var = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return float(var)
        return round(float(tail.mean()), 5)

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        mean = float(returns.mean())
        std = float(returns.std())
        if std == 0:
            return 0.0
        return round((mean - _RISK_FREE_DAILY) / std * math.sqrt(_TRADING_DAYS), 3)

    @staticmethod
    def _sortino(returns: pd.Series) -> float:
        mean = float(returns.mean()) - _RISK_FREE_DAILY
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(downside.std())
        if downside_std == 0:
            return 0.0
        return round(mean / downside_std * math.sqrt(_TRADING_DAYS), 3)

    @staticmethod
    def _max_drawdown(close: pd.Series) -> float:
        roll_max = close.cummax()
        drawdown = (close - roll_max) / roll_max
        return round(float(drawdown.min()) * 100, 3)  # negative percentage

    def _calmar(self, returns: pd.Series, max_dd_pct: float) -> float:
        annual_return = float(returns.mean()) * _TRADING_DAYS * 100
        if max_dd_pct == 0:
            return 0.0
        return round(annual_return / abs(max_dd_pct), 3)

    @staticmethod
    def _beta(returns: pd.Series, nifty_returns: Optional[pd.Series]) -> Optional[float]:
        if nifty_returns is None or len(nifty_returns) == 0:
            return None
        n = min(len(returns), len(nifty_returns))
        r = returns.values[-n:]
        nr = nifty_returns.values[-n:]
        cov = np.cov(r, nr)
        market_var = cov[1, 1]
        if market_var == 0:
            return None
        return round(float(cov[0, 1] / market_var), 3)

    @staticmethod
    def _risk_score(m: Dict[str, Any]) -> int:
        """
        Composite risk score 1 (low risk) – 10 (high risk).
        Based on VaR, drawdown, and Sharpe.
        """
        score = 5  # neutral baseline
        var99 = m.get("var_99", 0)
        if var99 < -0.03:
            score += 2
        elif var99 < -0.02:
            score += 1

        dd = abs(m.get("max_drawdown_pct", 0))
        if dd > 40:
            score += 2
        elif dd > 25:
            score += 1

        sharpe = m.get("sharpe_ratio", 0)
        if sharpe > 1.5:
            score -= 2
        elif sharpe > 0.8:
            score -= 1
        elif sharpe < 0:
            score += 1

        return max(1, min(10, score))

    # ── Data fetching ─────────────────────────────────────────────────────

    def _fetch_daily(self, symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

        # Try Kite
        try:
            fetcher = self._ensure_fetcher()
            df = fetcher.fetch(symbol, from_d, to_d, "day")
            if df is not None and len(df) > 30:
                return df
        except Exception as exc:
            log.warning("risk_metrics_kite_fetch_failed", symbol=symbol, error=str(exc))

        # Try yfinance
        try:
            import yfinance as yf
            df = yf.download(f"{symbol}.NS", start=from_d, end=to_d, progress=False)
            if df is not None and len(df) > 30:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as exc:
            log.warning("risk_metrics_yf_fetch_failed", symbol=symbol, error=str(exc))

        return None

    def _fetch_nifty_returns(self, n_returns: int) -> Optional[pd.Series]:
        """Fetch Nifty 50 daily returns for beta computation."""
        try:
            import yfinance as yf
            days_needed = int(n_returns * 1.5) + 30
            to_d = datetime.now().strftime("%Y-%m-%d")
            from_d = (datetime.now() - timedelta(days=days_needed)).strftime("%Y-%m-%d")
            df = yf.download(_NIFTY_YF, start=from_d, end=to_d, progress=False)
            if df is not None and len(df) > 10:
                close = df["Close"]
                returns = np.log(close / close.shift(1)).dropna()
                return returns
        except Exception as exc:
            log.warning("nifty_returns_fetch_failed", error=str(exc))
        return None

    def _ensure_fetcher(self):
        if self._fetcher is None:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            self._fetcher = HistoricalDataFetcher(KiteClient(), InstrumentManager())
        return self._fetcher

    @staticmethod
    def _empty_metrics(symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "var_95": None,
            "var_99": None,
            "cvar_95": None,
            "cvar_99": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown_pct": None,
            "calmar_ratio": None,
            "beta": None,
            "annualised_volatility_pct": None,
            "annualised_return_pct": None,
            "risk_score": None,
            "error": "insufficient_data",
        }
