"""
Market Regime Engine — classifies the current market environment.

Regime types:
  BULL_TREND   — Price > SMA50 > SMA200, expanding ATR, strong breadth
  EXPANSION    — Momentum breakout, price > all MAs, vol surge
  CHOPPY       — Price between SMA50/SMA200, compressed ATR, mixed breadth
  DISTRIBUTION — Price < SMA50, vol declining, breadth weakening
  BEAR         — Price < SMA200, SMA50 < SMA200, collapsing breadth

Inputs: Nifty 50 daily OHLCV (via yfinance ^NSEI)
Output: RegimeSnapshot dataclass
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeSnapshot:
    regime: str              # BULL_TREND | EXPANSION | CHOPPY | DISTRIBUTION | BEAR
    regime_score: float      # 0-100 (bullishness)
    emoji: str               # color indicator
    nifty_price: float
    nifty_change_pct: float
    sma50: float
    sma200: float
    atr_regime: str          # EXPANDING | STABLE | CONTRACTING
    vix: float
    vix_state: str           # LOW (<14) | NORMAL (14-20) | HIGH (>20)
    breadth: str             # STRONG | NEUTRAL | WEAK
    sector_leader: str       # e.g. "IT", "BANK", "AUTO"
    quality_multiplier: float  # 1.2 (bull) → 0.7 (bear) applied to setup scores
    timestamp: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_nifty(days: int = 260) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker("^NSEI").history(period=f"{days}d", interval="1d")
        if df is None or len(df) < 50:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def _fetch_vix() -> float:
    try:
        import yfinance as yf
        df = yf.Ticker("^INDIAVIX").history(period="5d")
        if df is not None and len(df) > 0:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 16.0  # default neutral VIX


def _fetch_sector_returns() -> dict[str, float]:
    """Fetch 5-day return for key NSE sector indices."""
    tickers = {
        "IT":     "^CNXIT",
        "BANK":   "^NSEBANK",
        "AUTO":   "^CNXAUTO",
        "PHARMA": "^CNXPHARMA",
        "FMCG":   "^CNXFMCG",
        "METAL":  "^CNXMETAL",
        "ENERGY": "^CNXENERGY",
        "REALTY": "^CNXREALTY",
    }
    out: dict[str, float] = {}
    try:
        import yfinance as yf
        for name, t in tickers.items():
            try:
                h = yf.Ticker(t).history(period="7d")
                if h is not None and len(h) >= 2:
                    ret = (float(h["Close"].iloc[-1]) / float(h["Close"].iloc[-6]) - 1) * 100
                    out[name] = round(ret, 2)
            except Exception:
                pass
    except Exception:
        pass
    return out


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.array([
        max(high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]))
        for i in range(1, len(high))
    ])
    atrs = np.convolve(tr, np.ones(period) / period, mode="valid")
    return atrs


# ── Public API ────────────────────────────────────────────────────────────────

def compute_regime() -> RegimeSnapshot:
    """
    Compute the current market regime.
    Always returns a valid RegimeSnapshot (falls back to defaults on error).
    """
    df = _fetch_nifty(days=260)
    if df is None or len(df) < 60:
        return _default_snapshot()

    close = df["close"].values
    high  = df["high"].values  if "high"  in df.columns else close
    low   = df["low"].values   if "low"   in df.columns else close

    price   = float(close[-1])
    prev    = float(close[-2]) if len(close) > 1 else price
    chg_pct = (price / prev - 1) * 100

    sma50  = float(np.mean(close[-50:]))
    sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else float(np.mean(close))

    # ── ATR regime ───────────────────────────────────────────────────────────
    atr_regime_label = "STABLE"
    if len(close) >= 30:
        atrs = _calc_atr(high, low, close, period=14)
        if len(atrs) >= 20:
            atr_recent = float(np.mean(atrs[-5:]))
            atr_older  = float(np.mean(atrs[-20:-5]))
            if atr_older > 0:
                atr_ratio = atr_recent / atr_older
                if atr_ratio > 1.2:
                    atr_regime_label = "EXPANDING"
                elif atr_ratio < 0.85:
                    atr_regime_label = "CONTRACTING"

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix = _fetch_vix()
    vix_state = "NORMAL"
    if vix < 14:
        vix_state = "LOW"
    elif vix > 20:
        vix_state = "HIGH"

    # ── Sector breadth proxy ──────────────────────────────────────────────────
    sector_rets = _fetch_sector_returns()
    if sector_rets:
        positive = sum(1 for v in sector_rets.values() if v > 0)
        total    = len(sector_rets)
        breadth_ratio = positive / total
        breadth = "STRONG" if breadth_ratio >= 0.6 else ("WEAK" if breadth_ratio <= 0.35 else "NEUTRAL")
        sector_leader = max(sector_rets, key=sector_rets.get) if sector_rets else "N/A"
    else:
        breadth = "NEUTRAL"
        sector_leader = "N/A"

    # ── Regime classification ─────────────────────────────────────────────────
    above_200 = price > sma200
    above_50  = price > sma50
    sma_trend = sma50 > sma200

    # Score components (0-100 each)
    ma_score     = (100 if above_50 and above_200 and sma_trend
                    else 65 if above_200
                    else 35 if above_50
                    else 10)
    breadth_score = {"STRONG": 85, "NEUTRAL": 50, "WEAK": 20}.get(breadth, 50)
    vix_score     = {"LOW": 80, "NORMAL": 55, "HIGH": 25}.get(vix_state, 55)
    atr_score     = {"EXPANDING": 70, "STABLE": 55, "CONTRACTING": 45}.get(atr_regime_label, 55)

    regime_score = ma_score * 0.40 + breadth_score * 0.30 + vix_score * 0.20 + atr_score * 0.10

    # Label
    if regime_score >= 75:
        if atr_regime_label == "EXPANDING":
            regime, emoji = "EXPANSION",  "🚀"
        else:
            regime, emoji = "BULL_TREND", "🟢"
        quality_multiplier = 1.2
    elif regime_score >= 55:
        regime, emoji      = "CHOPPY",    "🟡"
        quality_multiplier = 1.0
    elif regime_score >= 35:
        regime, emoji      = "DISTRIBUTION", "🟠"
        quality_multiplier = 0.85
    else:
        regime, emoji      = "BEAR",      "🔴"
        quality_multiplier = 0.70

    return RegimeSnapshot(
        regime=regime,
        regime_score=round(regime_score, 1),
        emoji=emoji,
        nifty_price=round(price, 1),
        nifty_change_pct=round(chg_pct, 2),
        sma50=round(sma50, 1),
        sma200=round(sma200, 1),
        atr_regime=atr_regime_label,
        vix=round(vix, 2),
        vix_state=vix_state,
        breadth=breadth,
        sector_leader=sector_leader,
        quality_multiplier=quality_multiplier,
        timestamp=datetime.now().strftime("%H:%M"),
    )


def _default_snapshot() -> RegimeSnapshot:
    return RegimeSnapshot(
        regime="UNKNOWN",
        regime_score=50.0,
        emoji="⚪",
        nifty_price=0.0,
        nifty_change_pct=0.0,
        sma50=0.0,
        sma200=0.0,
        atr_regime="STABLE",
        vix=16.0,
        vix_state="NORMAL",
        breadth="NEUTRAL",
        sector_leader="N/A",
        quality_multiplier=1.0,
        timestamp=datetime.now().strftime("%H:%M"),
    )
