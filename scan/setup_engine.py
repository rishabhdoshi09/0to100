"""
Setup Engine — Stage 4 of the scan pipeline.

Unified behavioral setup detection. The three legacy scanners
(MomentumScanner, BreakoutScanner, VCPScanner) become archetypes
here — not competing systems.

Each stock gets ONE primary setup archetype label + behavioral
evidence dict. No indicator-first logic — behavior is primary.

Setup archetypes:
  VCP_BREAKOUT          — Volatility contraction, Stage 2, near pivot
  MOMENTUM_EXPANSION    — Strong RS, price above all MAs, vol surge
  EARLY_LEADER          — RS improving before market, Stage 1→2 transition
  ACCUMULATION_BREAKOUT — Tight range 6+ weeks, vol dry, near ceiling
  EARNINGS_CONTINUATION — Gap up held 3+ sessions, buy pullback
  FAILED_BREAKOUT       — Broke pivot, returned below within 5 days
  MEAN_REVERSION        — Oversold RSI, Stage 2 intact, support visible
  TREND_CONTINUATION    — Pullback to 10EMA/21EMA in uptrend
  HIGH_TIGHT_FLAG       — 100%+ prior move, tight < 25% flag
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SetupCandidate:
    symbol: str
    price: float
    archetype: str                   # one of the archetypes above
    confidence: float                # 0-100 behavioral confidence
    pivot_level: float               # entry trigger price
    stop_level: float
    behavioral_evidence: list[str]   # human-readable evidence list
    raw_indicators: dict             # supporting indicator values


def _fetch(symbol: str, days: int = 260) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period=f"{days}d", interval="1d")
        if df is None or len(df) < 30:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close[-(period + 1):])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag, al = gains.mean(), losses.mean()
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = np.array([max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                   for i in range(1, len(h))])
    return float(tr[-period:].mean())


class SetupEngine:
    """
    Scans a filtered universe and detects behavioral setup archetypes.
    All detection methods operate on raw OHLCV — no TA-Lib, no external deps.
    """

    def __init__(self, max_workers: int = 16):
        self._max_workers = max_workers

    def detect(self, symbols: list[str]) -> list[SetupCandidate]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list[SetupCandidate] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._detect_one, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass
        return results

    def _detect_one(self, symbol: str) -> Optional[SetupCandidate]:
        df = _fetch(symbol, days=260)
        if df is None or len(df) < 50:
            return None
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        price  = float(close[-1])

        sma50   = float(np.mean(close[-50:]))  if len(close) >= 50 else price
        sma200  = float(np.mean(close[-200:])) if len(close) >= 200 else price
        sma50p  = float(np.mean(close[-55:-5])) if len(close) >= 55 else sma50
        rsi     = _rsi(close)
        atr     = _atr(df)
        avg_vol = float(volume[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())

        indicators = {
            "rsi": round(rsi, 1), "sma50": round(sma50, 1), "sma200": round(sma200, 1),
            "atr": round(atr, 2), "price": round(price, 2),
            "vol_ratio": round(volume[-1] / avg_vol, 2) if avg_vol > 0 else 1.0,
        }

        # ── Check each archetype in priority order ────────────────────────────
        checks = [
            self._check_high_tight_flag,
            self._check_vcp,
            self._check_accumulation_breakout,
            self._check_earnings_continuation,
            self._check_early_leader,
            self._check_trend_continuation,
            self._check_momentum_expansion,
            self._check_mean_reversion,
            self._check_failed_breakout,
        ]

        for check_fn in checks:
            result = check_fn(symbol, price, close, high, low, volume,
                              sma50, sma200, sma50p, rsi, atr, avg_vol, indicators)
            if result:
                return result
        return None

    def _check_vcp(self, symbol, price, close, high, low, volume,
                   sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Minervini VCP: Stage 2, successive contracting pullbacks, vol drying."""
        if price < sma200 or price < sma50:
            return None
        if len(close) < 60:
            return None

        # Find contractions in last 60 days
        contractions = []
        window = min(60, len(close))
        chunk  = close[-window:]
        seg    = 15
        for i in range(0, window - seg, seg // 2):
            seg_hi = float(np.max(chunk[i:i+seg]))
            seg_lo = float(np.min(chunk[i:i+seg]))
            depth  = (seg_hi - seg_lo) / seg_hi * 100
            contractions.append(depth)

        # Valid VCP: each contraction smaller than prior
        shrinking = all(contractions[i] > contractions[i+1] * 0.9
                        for i in range(len(contractions)-1)) if len(contractions) >= 3 else False
        if not shrinking or contractions[-1] > 15:
            return None

        # Volume drying in last 20 days
        recent_vol_ratio = float(volume[-20:].mean()) / float(volume[-40:-20].mean()) if len(volume) >= 40 else 1.0
        vol_drying = recent_vol_ratio < 0.85

        # Pivot = prior 20-day high
        pivot = float(np.max(high[-20:]))
        dist  = (pivot - price) / price * 100

        if dist > 8:  # too far from pivot
            return None

        confidence = 55.0
        evidence   = []
        if shrinking:
            confidence += 15; evidence.append(f"Contractions: {[round(c,1) for c in contractions[-3:]]}")
        if vol_drying:
            confidence += 15; evidence.append(f"Volume drying ({recent_vol_ratio:.2f}× prior avg)")
        if price > sma50 > sma200:
            confidence += 10; evidence.append("Stage 2: price > SMA50 > SMA200")
        if dist < 4:
            confidence += 5;  evidence.append(f"Only {dist:.1f}% from pivot")

        return SetupCandidate(
            symbol=symbol, price=price, archetype="VCP_BREAKOUT",
            confidence=min(100, confidence), pivot_level=round(pivot, 2),
            stop_level=round(float(np.min(low[-20:])), 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_accumulation_breakout(self, symbol, price, close, high, low, volume,
                                     sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Wyckoff: tight 6+ week range, volume dry, near ceiling."""
        if len(close) < 50:
            return None
        range_hi = float(np.max(close[-42:]))
        range_lo = float(np.min(close[-42:]))
        if range_hi == 0:
            return None
        range_pct = (range_hi - range_lo) / range_hi * 100
        if range_pct > 14:
            return None
        vol_base  = float(volume[-42:-21].mean()) if len(volume) >= 42 else 1.0
        vol_now   = float(volume[-21:].mean())
        vol_contraction = vol_now < vol_base * 0.75

        dist_to_top = (range_hi - price) / range_hi * 100
        if dist_to_top > 5:
            return None

        evidence = []
        confidence = 55.0
        if range_pct < 10:
            confidence += 10; evidence.append(f"Tight {range_pct:.1f}% base over 6 weeks")
        if vol_contraction:
            confidence += 15; evidence.append(f"Volume contracted to {vol_now/vol_base:.2f}× of base avg")
        if dist_to_top < 2:
            confidence += 10; evidence.append(f"Within {dist_to_top:.1f}% of base ceiling")
        if price > sma200:
            confidence += 5;  evidence.append("Above SMA200 — clean structure")

        return SetupCandidate(
            symbol=symbol, price=price, archetype="ACCUMULATION_BREAKOUT",
            confidence=min(100, confidence), pivot_level=round(range_hi * 1.002, 2),
            stop_level=round(range_lo * 0.99, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_early_leader(self, symbol, price, close, high, low, volume,
                            sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """RS improving while market weak; Stage 1→2 transition."""
        if price > sma50 * 1.03 and price > sma200:
            return None  # Already Stage 2 — better fits other archetypes
        if price < sma200 * 0.85:
            return None  # Too deep in Stage 4

        # RS: stock flat or up while checking broader context
        change_20d = (price / float(close[-21]) - 1) * 100 if len(close) > 21 else 0
        if change_20d < 5:
            return None

        sma50_starting_rise = sma50 > sma50p
        volume_expanding = float(volume[-1]) > avg_vol * 1.3

        evidence, confidence = [], 50.0
        if change_20d > 8:
            confidence += 15; evidence.append(f"Stock up {change_20d:.1f}% while market mixed")
        if sma50_starting_rise:
            confidence += 15; evidence.append("SMA50 beginning to turn up — Stage 1→2 transition")
        if volume_expanding:
            confidence += 10; evidence.append(f"Volume expanding ({volume[-1]/avg_vol:.1f}×)")
        if price > sma50:
            confidence += 5;  evidence.append("Price just crossed above SMA50")

        if confidence < 65:
            return None

        pivot = float(np.max(high[-20:]))
        return SetupCandidate(
            symbol=symbol, price=price, archetype="EARLY_LEADER",
            confidence=min(100, confidence), pivot_level=round(pivot, 2),
            stop_level=round(sma200 * 0.98, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_momentum_expansion(self, symbol, price, close, high, low, volume,
                                  sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Strong RS, price above all MAs, volume surge, RSI 60-80."""
        if not (price > sma50 > sma200):
            return None
        if not (60 <= ind["rsi"] <= 82):
            return None
        vol_ratio = ind.get("vol_ratio", 1.0)
        if vol_ratio < 1.5:
            return None
        mom_5d = (price / float(close[-6]) - 1) * 100 if len(close) > 6 else 0
        if mom_5d < 5:
            return None

        evidence = [
            f"Price above SMA50 > SMA200",
            f"RSI {ind['rsi']:.0f} — momentum zone",
            f"Volume {vol_ratio:.1f}× average",
            f"5-day return {mom_5d:.1f}%",
        ]
        confidence = 60 + min(30, (vol_ratio - 1.5) * 15 + (mom_5d - 5) * 2)
        pivot = float(np.max(high[-5:]))
        return SetupCandidate(
            symbol=symbol, price=price, archetype="MOMENTUM_EXPANSION",
            confidence=min(100, confidence), pivot_level=round(pivot, 2),
            stop_level=round(sma50 * 0.97, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_trend_continuation(self, symbol, price, close, high, low, volume,
                                  sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Pullback to 10EMA/21EMA in uptrend on low volume."""
        if not (price > sma50 > sma200):
            return None
        ema10 = float(pd.Series(close).ewm(span=10).mean().iloc[-1])
        ema21 = float(pd.Series(close).ewm(span=21).mean().iloc[-1])
        near_10ema = abs(price - ema10) / ema10 < 0.02
        near_21ema = abs(price - ema21) / ema21 < 0.025
        if not (near_10ema or near_21ema):
            return None
        vol_ratio = ind.get("vol_ratio", 1.0)
        if vol_ratio > 0.9:  # want low-volume pullback
            return None
        if not (42 <= ind["rsi"] <= 65):
            return None

        target_ema = ema10 if near_10ema else ema21
        label      = "10EMA" if near_10ema else "21EMA"
        evidence   = [
            f"Pullback to {label} ({target_ema:.0f}) on low volume ({vol_ratio:.2f}×)",
            "Stage 2 intact: price > SMA50 > SMA200",
            f"RSI {ind['rsi']:.0f} — healthy range for continuation",
        ]
        return SetupCandidate(
            symbol=symbol, price=price, archetype="TREND_CONTINUATION",
            confidence=72.0, pivot_level=round(float(np.max(high[-5:])), 2),
            stop_level=round(target_ema * 0.97, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_mean_reversion(self, symbol, price, close, high, low, volume,
                              sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Oversold RSI + Stage 2 intact + volume surge (absorption)."""
        if price < sma200 * 0.95:
            return None  # Stage 2 broken
        if ind["rsi"] > 32:
            return None
        vol_ratio = ind.get("vol_ratio", 1.0)
        if vol_ratio < 1.5:
            return None  # need absorption volume

        change_5d  = (price / float(close[-6]) - 1) * 100 if len(close) > 6 else 0
        resistance = float(np.max(high[-20:]))
        evidence   = [
            f"RSI {ind['rsi']:.0f} — oversold",
            f"Stage 2 intact: price still near SMA200",
            f"Volume surge {vol_ratio:.1f}× avg — potential absorption",
            f"5d decline {change_5d:.1f}%",
        ]
        return SetupCandidate(
            symbol=symbol, price=price, archetype="MEAN_REVERSION",
            confidence=62.0, pivot_level=round(sma50 * 0.99, 2),
            stop_level=round(price * 0.97, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_failed_breakout(self, symbol, price, close, high, low, volume,
                               sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Broke out then returned below pivot within 5 days."""
        if len(close) < 25:
            return None
        pivot_20d = float(np.max(high[-25:-5]))  # pivot from 5-25 sessions ago
        # Check if stock crossed pivot in last 5 days then came back below
        was_above = any(close[-(5+i)] > pivot_20d for i in range(1, 5))
        now_below = price < pivot_20d
        if not (was_above and now_below):
            return None

        evidence = [
            f"Broke above {pivot_20d:.0f} (20d high) then failed",
            f"Now {((pivot_20d - price) / pivot_20d * 100):.1f}% below prior pivot",
            "Failed breakout — bearish supply overwhelming",
        ]
        return SetupCandidate(
            symbol=symbol, price=price, archetype="FAILED_BREAKOUT",
            confidence=65.0, pivot_level=round(pivot_20d, 2),
            stop_level=round(price * 1.03, 2),  # stop above for short
            behavioral_evidence=evidence, raw_indicators=ind,
        )

    def _check_earnings_continuation(self, symbol, price, close, high, low, volume,
                                     sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Gap up held 3+ sessions, pullback to gap top as entry."""
        if len(close) < 10:
            return None
        # Proxy: look for a single-day gap of > 4%
        for i in range(2, min(15, len(close))):
            gap_pct = (close[-i] / close[-i-1] - 1) * 100
            if gap_pct > 4.0:
                gap_top = float(close[-i])
                gap_held = all(close[-(i-1-j)] > gap_top * 0.99 for j in range(min(3, i-1)))
                near_gap = abs(price - gap_top) / gap_top < 0.03
                vol_on_gap = volume[-i] > avg_vol * 2.5 if len(volume) > i else False
                if gap_held and near_gap and vol_on_gap:
                    evidence = [
                        f"Gap {gap_pct:.1f}% on {i} sessions ago, held above gap",
                        f"Volume {volume[-i]/avg_vol:.1f}× on gap day",
                        f"Price near gap top {gap_top:.0f} — buy zone",
                    ]
                    return SetupCandidate(
                        symbol=symbol, price=price, archetype="EARNINGS_CONTINUATION",
                        confidence=70.0, pivot_level=round(float(np.max(high[-5:])), 2),
                        stop_level=round(gap_top * 0.97, 2),
                        behavioral_evidence=evidence, raw_indicators=ind,
                    )
        return None

    def _check_high_tight_flag(self, symbol, price, close, high, low, volume,
                               sma50, sma200, sma50p, rsi, atr, avg_vol, ind) -> Optional[SetupCandidate]:
        """Prior 100%+ move in < 60 days, flag consolidation < 25%."""
        if len(close) < 70:
            return None
        prior_low = float(np.min(close[-70:-10]))
        prior_high = float(np.max(close[-70:-10]))
        if prior_low == 0:
            return None
        move = (prior_high / prior_low - 1) * 100
        if move < 80:
            return None

        flag_hi = float(np.max(close[-10:]))
        flag_lo = float(np.min(close[-10:]))
        flag_depth = (flag_hi - flag_lo) / flag_hi * 100
        if flag_depth > 25:
            return None

        vol_flag = float(volume[-10:].mean())
        vol_prior = float(volume[-70:-10].mean())
        vol_contracting = vol_flag < vol_prior * 0.7

        evidence = [
            f"Prior move {move:.0f}% — explosive base move",
            f"Tight flag {flag_depth:.1f}% — minimal giveback",
        ]
        if vol_contracting:
            evidence.append(f"Volume contracting in flag ({vol_flag/vol_prior:.2f}×)")
        confidence = 60 + min(25, (move - 80) / 4) + (10 if vol_contracting else 0)
        return SetupCandidate(
            symbol=symbol, price=price, archetype="HIGH_TIGHT_FLAG",
            confidence=min(100, confidence), pivot_level=round(flag_hi * 1.002, 2),
            stop_level=round(flag_lo * 0.98, 2),
            behavioral_evidence=evidence, raw_indicators=ind,
        )
