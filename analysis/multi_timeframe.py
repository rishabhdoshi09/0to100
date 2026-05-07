"""
Multi-Timeframe Signal Aligner.

For a given symbol, fetches OHLCV data across multiple timeframes,
runs the XGBoostSignalGenerator on each, and returns an alignment score
showing whether timeframes agree on direction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# Map user-friendly names → Kite interval strings and lookback days
_TF_MAP: Dict[str, Dict[str, Any]] = {
    "5min":  {"kite_interval": "5minute",  "lookback_days": 30},
    "15min": {"kite_interval": "15minute", "lookback_days": 60},
    "1h":    {"kite_interval": "60minute", "lookback_days": 120},
    "1d":    {"kite_interval": "day",      "lookback_days": 365},
}

_DEFAULT_TIMEFRAMES = ["5min", "15min", "1h"]
_CONSENSUS_BULL_THRESHOLD = 0.33
_CONSENSUS_BEAR_THRESHOLD = -0.33


class MultiTimeframeAligner:
    """
    Aligns XGBoost signals across multiple timeframes for a single symbol.

    Usage
    -----
    aligner = MultiTimeframeAligner(fetcher=fetcher)
    result  = aligner.align("RELIANCE", timeframes=["5min","15min","1d"])
    """

    def __init__(self, fetcher=None) -> None:
        """
        Parameters
        ----------
        fetcher : HistoricalDataFetcher, optional
            If None, a new one is created lazily.
        """
        self._fetcher = fetcher
        self._xgb = None  # lazy init to avoid heavy import at module level

    def align(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run XGBoost signals across all requested timeframes and compute
        alignment score.

        Returns
        -------
        dict with keys: symbol, timeframes, alignment_score,
                        consensus_action, timestamp
        """
        timeframes = timeframes or _DEFAULT_TIMEFRAMES

        # Validate timeframes
        valid = [tf for tf in timeframes if tf in _TF_MAP]
        if not valid:
            log.warning("multi_tf_no_valid_timeframes", requested=timeframes)
            return self._empty_result(symbol)

        self._ensure_xgb()
        fetcher = self._ensure_fetcher()

        tf_results: Dict[str, Dict[str, Any]] = {}
        buy_count = 0
        sell_count = 0

        for tf in valid:
            cfg = _TF_MAP[tf]
            interval = cfg["kite_interval"]
            lookback = cfg["lookback_days"]

            to_d = datetime.now().strftime("%Y-%m-%d")
            from_d = (datetime.now() - timedelta(days=lookback + 5)).strftime("%Y-%m-%d")

            try:
                df = fetcher.fetch(
                    symbol=symbol,
                    from_date=from_d,
                    to_date=to_d,
                    interval=interval,
                )
            except Exception as exc:
                log.warning("multi_tf_fetch_failed", symbol=symbol, tf=tf, error=str(exc))
                df = pd.DataFrame()

            if df is None or df.empty or len(df) < 20:
                log.warning("multi_tf_insufficient_data", symbol=symbol, tf=tf, rows=len(df) if df is not None else 0)
                tf_results[tf] = {"action": "HOLD", "confidence": 0.5, "error": "insufficient_data"}
                continue

            # Use a symbol key that encodes the timeframe to keep models separate
            tf_symbol = f"{symbol}_{tf}"
            try:
                sig = self._xgb.generate_signal(df, tf_symbol)
            except Exception as exc:
                log.warning("multi_tf_signal_failed", symbol=symbol, tf=tf, error=str(exc))
                sig = {"action": "HOLD", "confidence": 0.5}

            action = sig.get("action", "HOLD")
            confidence = sig.get("confidence", 0.5)

            tf_results[tf] = {"action": action, "confidence": round(float(confidence), 4)}

            if action == "BUY":
                buy_count += 1
            elif action == "SELL":
                sell_count += 1

        total = len(valid)
        alignment_score = (buy_count - sell_count) / total if total > 0 else 0.0
        alignment_score = round(alignment_score, 3)

        if alignment_score > _CONSENSUS_BULL_THRESHOLD:
            consensus = "BUY"
        elif alignment_score < _CONSENSUS_BEAR_THRESHOLD:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        log.info(
            "multi_tf_aligned",
            symbol=symbol,
            alignment_score=alignment_score,
            consensus=consensus,
        )

        return {
            "symbol": symbol,
            "timeframes": tf_results,
            "alignment_score": alignment_score,
            "consensus_action": consensus,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Lazy initialisers ──────────────────────────────────────────────────

    def _ensure_xgb(self) -> None:
        if self._xgb is None:
            from ml.xgboost_signal import XGBoostSignalGenerator
            self._xgb = XGBoostSignalGenerator()

    def _ensure_fetcher(self):
        if self._fetcher is None:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            self._fetcher = HistoricalDataFetcher(KiteClient(), InstrumentManager())
        return self._fetcher

    @staticmethod
    def _empty_result(symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "timeframes": {},
            "alignment_score": 0.0,
            "consensus_action": "HOLD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
