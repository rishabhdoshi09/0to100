"""
Market Regime Detector.

Combines Hurst exponent, ADX, India VIX, and F&O expiry proximity
to classify the current market regime and recommend strategy type.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_NIFTY_SYMBOL = "NIFTY 50"          # yfinance ticker
_NIFTY_YF_TICKER = "^NSEI"
_INDIA_VIX_YF_TICKER = "^INDIAVIX"
_NIFTY_NSE_SYMBOL = "NIFTY50"       # Kite symbol fallback
_LOOKBACK_DAYS = 100

_HURST_TRENDING_THRESHOLD = 0.55
_HURST_REVERTING_THRESHOLD = 0.45
_ADX_TRENDING_THRESHOLD = 25
_ADX_SIDEWAYS_THRESHOLD = 20
_VIX_LOW_THRESHOLD = 13.0
_VIX_HIGH_THRESHOLD = 18.0


class RegimeDetector:
    """
    Detects the current NSE market regime using multiple independent signals.

    Usage
    -----
    rd = RegimeDetector()
    result = rd.detect()
    """

    def __init__(self, fetcher=None, fno_executor=None) -> None:
        self._fetcher = fetcher
        self._fno = fno_executor

    def detect(self) -> Dict[str, Any]:
        """
        Run all regime checks and return a unified regime dict.
        """
        nifty_df = self._fetch_nifty()
        hurst = self._hurst_exponent(nifty_df)
        adx = self._compute_adx(nifty_df)
        india_vix = self._fetch_india_vix()
        expiry_warning = self._check_expiry_proximity()

        hurst_label = self._classify_hurst(hurst)
        adx_label = self._classify_adx(adx)
        vix_label = self._classify_vix(india_vix)

        regime = f"{hurst_label}_{vix_label}"
        recommended = self._recommend_strategy(hurst_label, vix_label, adx_label)

        result: Dict[str, Any] = {
            "regime": regime,
            "hurst": round(hurst, 4) if hurst is not None else None,
            "hurst_label": hurst_label,
            "adx": round(adx, 2) if adx is not None else None,
            "adx_label": adx_label,
            "india_vix": round(india_vix, 2) if india_vix is not None else None,
            "vix_label": vix_label,
            "expiry_warning": expiry_warning,
            "recommended_strategy": recommended,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if expiry_warning:
            result["regime"] = "PRE_EXPIRY_" + regime

        # Augment with HMM regime
        hmm_result = self.get_hmm_regime(nifty_df)
        result["hmm_label"]   = hmm_result.get("label", "UNKNOWN")
        result["hmm_key"]     = hmm_result.get("hmm_key", "UNKNOWN_HMM")
        result["hmm_swap_prob"] = hmm_result.get("swap_prob")
        # Final combined key, e.g. "TRENDING_LOW_VOL" → "TRENDING_LOW_VOL_HMM"
        result["regime_key"] = f"{result['regime']}_{result['hmm_label']}"

        log.info("regime_detected", regime=result["regime"], hurst=hurst, adx=adx,
                 vix=india_vix, hmm=result["hmm_label"])
        return result

    def get_hmm_regime(self, df: pd.DataFrame = None) -> dict:
        """Run HMM regime detection on Nifty data."""
        try:
            from analysis.hmm_regime import HMMRegimeDetector
            if df is None or df.empty:
                df = self._fetch_nifty()
            return HMMRegimeDetector().detect(df)
        except Exception as exc:
            log.warning("hmm_regime_failed", error=str(exc))
            return {"label": "UNKNOWN", "hmm_key": "UNKNOWN_HMM", "swap_prob": None}

    # ── Data Fetching ──────────────────────────────────────────────────────

    def _fetch_nifty(self) -> pd.DataFrame:
        """Fetch ~100 days of Nifty 50 daily OHLCV. Tries yfinance, falls back to KiteClient."""
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=_LOOKBACK_DAYS + 20)
            ticker = yf.Ticker(_NIFTY_YF_TICKER)
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is not None and len(df) >= 30:
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                log.debug("nifty_fetched_yfinance", rows=len(df))
                return df
        except Exception as exc:
            log.warning("nifty_yfinance_failed", error=str(exc))

        # Fallback: try Kite
        try:
            fetcher = self._ensure_fetcher()
            to_d = datetime.now().strftime("%Y-%m-%d")
            from_d = (datetime.now() - timedelta(days=_LOOKBACK_DAYS + 20)).strftime("%Y-%m-%d")
            df = fetcher.fetch(_NIFTY_NSE_SYMBOL, from_d, to_d, "day")
            if df is not None and len(df) >= 30:
                return df
        except Exception as exc:
            log.warning("nifty_kite_failed", error=str(exc))

        log.warning("nifty_data_unavailable_using_empty")
        return pd.DataFrame()

    def _fetch_india_vix(self) -> Optional[float]:
        """Fetch latest India VIX value via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(_INDIA_VIX_YF_TICKER)
            hist = ticker.history(period="5d")
            if hist is not None and len(hist) > 0:
                _c = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 3]
                return float(_c.iloc[-1])
        except Exception as exc:
            log.warning("india_vix_fetch_failed", error=str(exc))
        return None

    # ── Hurst Exponent (R/S Analysis) ─────────────────────────────────────

    def _hurst_exponent(self, df: pd.DataFrame) -> Optional[float]:
        """
        Custom Rescaled Range (R/S) analysis to estimate the Hurst exponent.
        Avoids the external `hurst` library dependency.
        """
        if df is None or df.empty or len(df) < 20:
            return None

        try:
            close = df["close"].astype(float).dropna().values
            if len(close) < 20:
                return None

            log_returns = np.diff(np.log(close))
            lags = range(2, min(20, len(log_returns) // 2))
            rs_values = []
            lag_values = []

            for lag in lags:
                chunks = [log_returns[i:i + lag] for i in range(0, len(log_returns) - lag, lag)]
                if not chunks:
                    continue
                rs_list = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean = np.mean(chunk)
                    deviation = np.cumsum(chunk - mean)
                    r_val = np.max(deviation) - np.min(deviation)
                    s_val = np.std(chunk, ddof=1)
                    if s_val > 0:
                        rs_list.append(r_val / s_val)
                if rs_list:
                    rs_values.append(np.log(np.mean(rs_list)))
                    lag_values.append(np.log(lag))

            if len(lag_values) < 4:
                return None

            # OLS regression log(R/S) ~ H * log(n)
            coeffs = np.polyfit(lag_values, rs_values, 1)
            return float(coeffs[0])
        except Exception as exc:
            log.warning("hurst_computation_failed", error=str(exc))
            return None

    # ── ADX ───────────────────────────────────────────────────────────────

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Compute ADX from OHLCV DataFrame."""
        if df is None or df.empty or len(df) < period + 2:
            return None

        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)

            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

            atr = tr.ewm(alpha=1 / period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.ewm(alpha=1 / period, adjust=False).mean()
            val = float(adx.iloc[-1])
            return val if not math.isnan(val) else None
        except Exception as exc:
            log.warning("adx_computation_failed", error=str(exc))
            return None

    # ── F&O Expiry ─────────────────────────────────────────────────────────

    def _check_expiry_proximity(self) -> bool:
        """Return True if Nifty front-month futures expire within 3 days."""
        try:
            fno = self._ensure_fno()
            contract = fno.get_front_month_future("NIFTY")
            expiry = contract["expiry"]
            return fno.should_rollover(expiry)
        except Exception as exc:
            log.debug("expiry_check_failed", error=str(exc))
            return False

    # ── Classification ────────────────────────────────────────────────────

    @staticmethod
    def _classify_hurst(h: Optional[float]) -> str:
        if h is None:
            return "UNKNOWN"
        if h > _HURST_TRENDING_THRESHOLD:
            return "TRENDING"
        if h < _HURST_REVERTING_THRESHOLD:
            return "MEAN_REVERTING"
        return "RANDOM"

    @staticmethod
    def _classify_adx(adx: Optional[float]) -> str:
        if adx is None:
            return "UNKNOWN"
        if adx > _ADX_TRENDING_THRESHOLD:
            return "TRENDING"
        if adx < _ADX_SIDEWAYS_THRESHOLD:
            return "SIDEWAYS"
        return "MODERATE"

    @staticmethod
    def _classify_vix(vix: Optional[float]) -> str:
        if vix is None:
            return "UNKNOWN_VOL"
        if vix < _VIX_LOW_THRESHOLD:
            return "LOW_VOL"
        if vix > _VIX_HIGH_THRESHOLD:
            return "HIGH_VOL"
        return "NORMAL_VOL"

    @staticmethod
    def _recommend_strategy(hurst_label: str, vix_label: str, adx_label: str) -> str:
        if hurst_label == "TRENDING" and adx_label == "TRENDING":
            if vix_label == "HIGH_VOL":
                return "Trend following with tight stops — high volatility, reduce size"
            return "Trend following — ride momentum with trailing stops"
        if hurst_label == "MEAN_REVERTING":
            if vix_label == "LOW_VOL":
                return "Mean reversion — fade extreme moves, target tight bands"
            return "Cautious mean reversion — VIX elevated, widen stops"
        if vix_label == "HIGH_VOL":
            return "Reduce exposure — choppy high-volatility environment"
        return "Neutral — mixed signals, reduce position sizes, wait for clarity"

    # ── Lazy helpers ───────────────────────────────────────────────────────

    def _ensure_fetcher(self):
        if self._fetcher is None:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            self._fetcher = HistoricalDataFetcher(KiteClient(), InstrumentManager())
        return self._fetcher

    def _ensure_fno(self):
        if self._fno is None:
            from data.kite_client import KiteClient
            from execution.fo_executor import FnOExecutor
            self._fno = FnOExecutor(KiteClient())
        return self._fno
