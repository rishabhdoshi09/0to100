"""
Unified Smart Scanner — one engine, every signal type.

Accumulates everything built so far into a single pass over the bulk
OHLCV cache (no per-stock HTTP):

  MOMENTUM        — RSI + price momentum + volume surge
  BREAKOUT_52W    — new 52-week high
  BREAKOUT_RES    — 20-day resistance break on volume
  GOLDEN_CROSS    — 50 SMA crossing above 200 SMA
  VOL_SQUEEZE     — Bollinger squeeze expanding upward
  VCP             — volatility contraction pattern near pivot
  FLAT_BASE       — tight 6+ week base near ceiling
  CUP_HANDLE      — cup with handle proxy
  HIGH_TIGHT_FLAG — 80%+ run then tight flag

Every stock gets ONE row with all its detected signals, a composite
score, and an entry/stop/target plan. Plain-English reasons included.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# Signal metadata: label shown to user, category for filtering, base score
SIGNAL_META = {
    "BREAKOUT_52W":    ("52-week high breakout",      "Breakout",  30),
    "BREAKOUT_RES":    ("Resistance break on volume", "Breakout",  26),
    "GOLDEN_CROSS":    ("Golden cross (50/200 SMA)",  "Breakout",  22),
    "VOL_SQUEEZE":     ("Squeeze breakout",           "Breakout",  22),
    "VCP":             ("VCP — tightening base",      "Pattern",   28),
    "FLAT_BASE":       ("Flat base near breakout",    "Pattern",   24),
    "CUP_HANDLE":      ("Cup & handle",               "Pattern",   24),
    "HIGH_TIGHT_FLAG": ("High tight flag",            "Pattern",   30),
    "MOMENTUM":        ("Strong momentum",            "Momentum",  20),
}


@dataclass
class StockSignal:
    symbol: str
    price: float
    change_pct: float          # 1-day change
    momentum_5d: float
    rsi: float
    volume_ratio: float
    signals: list[str]                     # signal keys from SIGNAL_META
    reasons: list[str] = field(default_factory=list)   # plain-English evidence
    score: float = 0.0                     # composite 0-100
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    verdict: str = "WATCH"                 # BUY | WATCH

    @property
    def categories(self) -> set[str]:
        return {SIGNAL_META[s][1] for s in self.signals if s in SIGNAL_META}

    @property
    def signal_labels(self) -> list[str]:
        return [SIGNAL_META[s][0] for s in self.signals if s in SIGNAL_META]

    @property
    def risk_reward(self) -> float:
        risk = self.entry - self.stop
        return (self.target - self.entry) / risk if risk > 0 else 0.0


class UnifiedScanner:
    """Scans the full universe from the bulk cache. Compute-only, no network."""

    def __init__(self, max_workers: int = 8):
        self._max_workers = max_workers

    def scan(self, symbols: list[str], progress=None) -> list[StockSignal]:
        from scan.bulk_fetcher import prefetch, get_cached, cached_symbols

        prefetch(symbols, progress=progress)
        available = [s for s in symbols if s in set(cached_symbols())]
        log.info("unified_scan_start", requested=len(symbols), with_data=len(available))

        results: list[StockSignal] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._analyze, sym, get_cached(sym)): sym
                       for sym in available}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r and r.signals:
                        results.append(r)
                except Exception as exc:
                    log.debug("unified_analyze_failed", symbol=futures[fut], error=str(exc))

        results.sort(key=lambda r: r.score, reverse=True)
        log.info("unified_scan_done", scanned=len(available), with_signals=len(results))
        return results

    # ── Per-stock analysis ────────────────────────────────────────────────────

    def _analyze(self, symbol: str, df: Optional[pd.DataFrame]) -> Optional[StockSignal]:
        if df is None or len(df) < 60:
            return None
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close
        low = df["low"].values.astype(float) if "low" in df.columns else close
        vol = df["volume"].values.astype(float) if "volume" in df.columns else None

        price = float(close[-1])
        if price < 20:               # skip penny stocks
            return None
        if vol is not None and np.nanmean(vol[-20:]) * price < 1e7:
            return None              # < ₹1 cr avg daily turnover — illiquid

        chg = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0.0
        mom5 = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0.0
        rsi = _rsi(close)
        vratio = 1.0
        if vol is not None and len(vol) > 21:
            avg = np.nanmean(vol[-21:-1])
            vratio = float(vol[-1] / avg) if avg > 0 else 1.0
        atr = _atr(high, low, close)

        signals: list[str] = []
        reasons: list[str] = []

        # ── Momentum ──────────────────────────────────────────────────────────
        mom_score = (_norm(mom5, -5, 10) * 0.40 + _norm(rsi, 30, 70) * 0.35
                     + _norm(vratio, 0.5, 3.0) * 0.25) * 100
        if mom_score >= 65 and 50 <= rsi < 78:
            signals.append("MOMENTUM")
            reasons.append(f"Up {mom5:+.1f}% in 5 days on {vratio:.1f}× volume")

        # ── Breakouts ─────────────────────────────────────────────────────────
        if len(high) > 60:
            hi52 = float(np.max(high[:-1]))
            if price > hi52 * 0.998:
                signals.append("BREAKOUT_52W")
                reasons.append(f"Broke 52-week high ₹{hi52:,.0f} today")
            else:
                res20 = float(np.max(high[-21:-1]))
                if price > res20 and vratio > 1.5:
                    signals.append("BREAKOUT_RES")
                    reasons.append(f"Broke ₹{res20:,.0f} resistance on {vratio:.1f}× volume")

        if len(close) >= 201:
            s50, s200 = close[-50:].mean(), close[-200:].mean()
            s50p, s200p = close[-55:-5].mean(), close[-205:-5].mean()
            if s50 > s200 and s50p <= s200p:
                signals.append("GOLDEN_CROSS")
                reasons.append("50-day average just crossed above 200-day")

        if len(close) >= 40:
            width_now = _bb_width(close, -1)
            width_prev = np.mean([_bb_width(close, i) for i in range(-11, -1)])
            upper = close[-20:].mean() + 2 * close[-20:].std()
            if width_prev > 0 and width_now > width_prev * 1.3 and price > upper:
                signals.append("VOL_SQUEEZE")
                reasons.append("Tight squeeze just expanded upward")

        # ── Chart patterns ────────────────────────────────────────────────────
        pat, pat_reason, pivot = _detect_pattern(close, high, low, vol)
        if pat:
            signals.append(pat)
            reasons.append(pat_reason)

        if not signals:
            return None

        # ── Trade plan ────────────────────────────────────────────────────────
        entry = pivot if pivot and pivot > price else price
        stop = round(entry - 2 * atr, 1) if atr > 0 else round(entry * 0.95, 1)
        target = round(entry + 4 * atr, 1) if atr > 0 else round(entry * 1.10, 1)

        # ── Composite score ───────────────────────────────────────────────────
        base = sum(SIGNAL_META[s][2] for s in signals)
        trend_bonus = 10 if (len(close) >= 200 and price > close[-200:].mean()) else 0
        score = min(100.0, base + trend_bonus + mom_score * 0.2)

        verdict = "BUY" if (score >= 55 and len(signals) >= 2) or any(
            s in ("BREAKOUT_52W", "HIGH_TIGHT_FLAG") for s in signals) else "WATCH"

        return StockSignal(
            symbol=symbol, price=round(price, 2), change_pct=round(chg, 2),
            momentum_5d=round(mom5, 2), rsi=round(rsi, 1),
            volume_ratio=round(vratio, 2), signals=signals, reasons=reasons,
            score=round(score, 1), entry=round(entry, 1), stop=stop,
            target=target, verdict=verdict,
        )


# ── Pattern detection ─────────────────────────────────────────────────────────

def _detect_pattern(close, high, low, vol) -> tuple[Optional[str], str, float]:
    """Returns (signal_key, reason, pivot_level) — best single pattern or None."""
    price = float(close[-1])

    # High tight flag: 80%+ run in ~8 weeks, then <25% pullback flag 2-4 weeks
    if len(close) >= 60:
        run = close[-15] / close[-55] - 1
        flag_hi = float(np.max(high[-15:]))
        flag_lo = float(np.min(low[-15:]))
        if run >= 0.8 and flag_hi > 0 and (flag_hi - flag_lo) / flag_hi <= 0.25 \
                and price >= flag_hi * 0.93:
            return "HIGH_TIGHT_FLAG", f"Ran {run*100:.0f}% then tight flag — rare leader", flag_hi

    # VCP: successive pullbacks contracting, price within 5% of pivot
    if len(close) >= 120:
        pullbacks = _pullback_depths(high, low, lookback=120)
        if len(pullbacks) >= 2 and all(
                pullbacks[i] > pullbacks[i + 1] for i in range(len(pullbacks) - 1)) \
                and pullbacks[-1] <= 12:
            pivot = float(np.max(high[-40:]))
            if price >= pivot * 0.95:
                seq = " → ".join(f"{p:.0f}%" for p in pullbacks)
                return "VCP", f"Pullbacks shrinking ({seq}) — coiling near ₹{pivot:,.0f}", pivot

    # Flat base: 30+ days within a 12% range, price in top third
    if len(close) >= 45:
        base_hi = float(np.max(high[-35:]))
        base_lo = float(np.min(low[-35:]))
        depth = (base_hi - base_lo) / base_hi if base_hi > 0 else 1
        if depth <= 0.12 and price >= base_hi * 0.96:
            return "FLAT_BASE", f"7-week flat base, sitting at ₹{base_hi:,.0f} ceiling", base_hi

    # Cup & handle proxy: prior high, rounded 15-35% cup, small handle near rim
    if len(close) >= 130:
        left_rim = float(np.max(high[-130:-65]))
        cup_low = float(np.min(low[-90:-20]))
        cup_depth = (left_rim - cup_low) / left_rim if left_rim > 0 else 1
        handle_lo = float(np.min(low[-15:]))
        handle_ok = handle_lo >= cup_low + (left_rim - cup_low) * 0.5
        if 0.15 <= cup_depth <= 0.35 and handle_ok and price >= left_rim * 0.92:
            return "CUP_HANDLE", f"Cup formed, handle near ₹{left_rim:,.0f} rim", left_rim

    return None, "", 0.0


def _pullback_depths(high, low, lookback: int = 120) -> list[float]:
    """Depths (%) of successive swing pullbacks over the lookback window."""
    h = high[-lookback:]
    l = low[-lookback:]
    depths, seg = [], max(20, lookback // 4)
    for i in range(0, lookback - seg + 1, seg):
        hi = float(np.max(h[i:i + seg]))
        lo = float(np.min(l[i:i + seg]))
        if hi > 0:
            depths.append((hi - lo) / hi * 100)
    return depths


# ── Math helpers ──────────────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close[-(period + 1):])
    gains, losses = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    ag, al = gains.mean(), losses.mean()
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(high, low, close, period: int = 14) -> float:
    if len(high) < period + 1:
        return 0.0
    tr = [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
          for i in range(-period, 0)]
    return float(np.mean(tr))


def _bb_width(close: np.ndarray, idx: int, period: int = 20) -> float:
    seg = close[idx - period:idx] if idx != -1 else close[-period:]
    if len(seg) < period:
        return 0.0
    m, s = seg.mean(), seg.std()
    return (4 * s) / m if m > 0 else 0.0


def _norm(val: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.5
