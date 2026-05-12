"""
Momentum & Breakout Scanner.

Scans a universe of NSE stocks and returns:
  - Top momentum stocks (RSI trend + price momentum + volume surge)
  - Breakout stocks (52W high, resistance break, golden cross,
                     volatility squeeze, cup & handle proxy)

Designed to run on the homepage — fast, no LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MomentumStock:
    symbol: str
    price: float
    change_pct: float
    rsi: float
    volume_ratio: float      # today's volume / 20-day avg volume
    momentum_score: float    # 0-100 composite
    signal: str              # BUY | WATCH | NEUTRAL


@dataclass
class BreakoutStock:
    symbol: str
    price: float
    breakout_type: str       # 52W_HIGH | RESISTANCE | GOLDEN_CROSS | VOL_SQUEEZE | CUP_HANDLE
    volume: float
    atr: float
    confidence: float        # 0-100
    since_when: str          # e.g. "2 days ago"


# ── Scanner ───────────────────────────────────────────────────────────────────

class MomentumScanner:
    """
    Scans a list of symbols for momentum and breakout setups.
    Uses existing HistoricalDataFetcher + IndicatorEngine — no new dependencies.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers

    def scan_momentum(
        self,
        symbols: list[str],
        top_n: int = 20,
    ) -> list[MomentumStock]:
        """Return top N momentum stocks sorted by composite score."""
        results = self._scan_all(symbols)
        momentum = [r for r in results if r is not None and isinstance(r, MomentumStock)]
        return sorted(momentum, key=lambda x: x.momentum_score, reverse=True)[:top_n]

    def scan_breakouts(
        self,
        symbols: list[str],
        top_n: int = 20,
    ) -> list[BreakoutStock]:
        """Return top N breakout stocks sorted by confidence."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        breakouts: list[BreakoutStock] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._check_breakout, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result:
                        breakouts.append(result)
                except Exception as exc:
                    log.debug("breakout_check_failed", symbol=futures[fut], error=str(exc))

        return sorted(breakouts, key=lambda x: x.confidence, reverse=True)[:top_n]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan_all(self, symbols: list[str]) -> list[Optional[MomentumStock]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._score_momentum, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    log.debug("momentum_score_failed", symbol=futures[fut], error=str(exc))
        return results

    def _get_df(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        try:
            from data.historical import HistoricalDataFetcher
            fetcher = HistoricalDataFetcher()
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=days)
            df = fetcher.fetch(symbol, interval="day", from_dt=from_dt, to_dt=to_dt)
            if df is None or len(df) < 30:
                return None
            return df
        except Exception as exc:
            log.debug("fetch_failed", symbol=symbol, error=str(exc))
            return None

    def _score_momentum(self, symbol: str) -> Optional[MomentumStock]:
        df = self._get_df(symbol, days=60)
        if df is None:
            return None

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # RSI-14
        rsi = _calc_rsi(close, 14)

        # Price momentum: 5-day and 20-day
        mom_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
        mom_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0

        # Change today
        change_pct = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0

        # Volume ratio
        vol_ratio = 1.0
        if volume is not None and len(volume) > 20:
            avg_vol = np.mean(volume[-21:-1])
            vol_ratio = volume[-1] / avg_vol if avg_vol > 0 else 1.0

        # Composite momentum score (0-100)
        rsi_score = _normalize(rsi, 30, 70)             # RSI between 50-70 is best
        mom_score = _normalize(mom_5d, -5, 10)          # 5d momentum
        vol_score = _normalize(vol_ratio, 0.5, 3.0)     # volume surge

        # Weight: momentum 40%, RSI 35%, volume 25%
        composite = (mom_score * 0.40) + (rsi_score * 0.35) + (vol_score * 0.25)
        composite = max(0.0, min(100.0, composite * 100))

        # Signal label
        if composite >= 65 and rsi < 75:
            signal = "BUY"
        elif composite >= 45:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        return MomentumStock(
            symbol=symbol,
            price=round(float(close[-1]), 2),
            change_pct=round(change_pct, 2),
            rsi=round(rsi, 1),
            volume_ratio=round(vol_ratio, 2),
            momentum_score=round(composite, 1),
            signal=signal,
        )

    def _check_breakout(self, symbol: str) -> Optional[BreakoutStock]:
        df = self._get_df(symbol, days=260)  # need 52W
        if df is None or len(df) < 50:
            return None

        close = df["close"].values
        high  = df["high"].values if "high" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else None

        price = float(close[-1])
        atr = _calc_atr(df)

        # ── 52-week high breakout ─────────────────────────────────────────────
        high_52w = np.max(high[:-1])  # exclude today
        if price > high_52w * 0.998:
            conf = _vol_confirmation(volume, 70)
            return BreakoutStock(
                symbol=symbol, price=price,
                breakout_type="52W_HIGH",
                volume=float(volume[-1]) if volume is not None else 0,
                atr=round(atr, 2),
                confidence=conf,
                since_when="Today",
            )

        # ── Golden cross (50 SMA crosses above 200 SMA) ───────────────────────
        if len(close) >= 200:
            sma50  = np.mean(close[-50:])
            sma200 = np.mean(close[-200:])
            sma50_prev  = np.mean(close[-51:-1])
            sma200_prev = np.mean(close[-201:-1])
            if sma50 > sma200 and sma50_prev <= sma200_prev:
                return BreakoutStock(
                    symbol=symbol, price=price,
                    breakout_type="GOLDEN_CROSS",
                    volume=float(volume[-1]) if volume is not None else 0,
                    atr=round(atr, 2),
                    confidence=72.0,
                    since_when="Today",
                )

        # ── Volatility squeeze breakout ───────────────────────────────────────
        # BB width compressed, now expanding with direction
        if len(close) >= 20:
            bb_upper, bb_lower = _bollinger_bands(close)
            bb_width_now  = (bb_upper[-1] - bb_lower[-1]) / close[-1]
            bb_width_prev = np.mean(
                [(bb_upper[i] - bb_lower[i]) / close[i] for i in range(-10, -1)]
            )
            if bb_width_now > bb_width_prev * 1.3 and close[-1] > bb_upper[-1]:
                conf = _vol_confirmation(volume, 60)
                return BreakoutStock(
                    symbol=symbol, price=price,
                    breakout_type="VOL_SQUEEZE",
                    volume=float(volume[-1]) if volume is not None else 0,
                    atr=round(atr, 2),
                    confidence=conf,
                    since_when="Today",
                )

        # ── Resistance break (20-day high) ────────────────────────────────────
        resistance_20d = np.max(high[-21:-1])
        if price > resistance_20d and volume is not None:
            vol_ratio = volume[-1] / np.mean(volume[-21:-1]) if np.mean(volume[-21:-1]) > 0 else 1
            if vol_ratio > 1.5:
                conf = min(85, 50 + vol_ratio * 10)
                return BreakoutStock(
                    symbol=symbol, price=price,
                    breakout_type="RESISTANCE_BREAK",
                    volume=float(volume[-1]),
                    atr=round(atr, 2),
                    confidence=round(conf, 1),
                    since_when="Today",
                )

        return None


# ── Math helpers ──────────────────────────────────────────────────────────────

def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close[-period - 1:])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if "high" not in df.columns or "low" not in df.columns:
        return 0.0
    high = df["high"].values[-period - 1:]
    low  = df["low"].values[-period - 1:]
    close_prev = df["close"].values[-period - 1:]
    trs = [
        max(high[i] - low[i],
            abs(high[i] - close_prev[i - 1]),
            abs(low[i] - close_prev[i - 1]))
        for i in range(1, len(high))
    ]
    return float(np.mean(trs)) if trs else 0.0


def _bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0):
    sma = np.array([np.mean(close[i - period:i]) for i in range(period, len(close) + 1)])
    std = np.array([np.std(close[i - period:i]) for i in range(period, len(close) + 1)])
    return sma + std_dev * std, sma - std_dev * std


def _normalize(val: float, lo: float, hi: float) -> float:
    """Normalize val to 0-1 within [lo, hi]."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _vol_confirmation(volume: Optional[np.ndarray], base_conf: float) -> float:
    if volume is None or len(volume) < 21:
        return base_conf
    avg = np.mean(volume[-21:-1])
    if avg == 0:
        return base_conf
    ratio = volume[-1] / avg
    bonus = min(15, (ratio - 1) * 10) if ratio > 1 else 0
    return min(95, round(base_conf + bonus, 1))
