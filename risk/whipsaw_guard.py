"""
Whipsaw Guard — The Kang Defense.

Kang controls time. Whipsaw markets control your equity curve.

A whipsaw market flips direction repeatedly, killing every trade:
  - Breakout → reversal → breakout → reversal
  - Each trade looks correct but market reverses before target
  - Stop gets hit, then price continues original direction

This module:
  1. Counts regime flips in last 10 trading days
  2. Tracks consecutive stopped-out trades
  3. Computes a Whipsaw Score (0-100, higher = more dangerous)
  4. Reduces position sizing and minimum tier requirements when whipsaw detected
  5. Requires higher confidence threshold to enter any trade
  6. Detects chop via ATR expansion without directional follow-through
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_STATE_FILE = Path("logs/whipsaw_state.json")
_LOCK = threading.Lock()


@dataclass
class WhipsawReport:
    score: float                   # 0-100 (higher = more whipsaw)
    regime_flips_10d: int          # how many times regime changed in 10 days
    consecutive_stops: int         # consecutive stopped-out trades this week
    atr_expanding: bool            # ATR growing without direction = chop
    chop_index: float              # 0-1, higher = more choppy (Efficiency Ratio)
    size_multiplier: float         # 0.25 to 1.0 — reduce this in whipsaw
    min_confidence_required: float # raise bar for entry (default 60, up to 85)
    verdict: str                   # CLEAN | CHOPPY | WHIPSAW | SEVERE
    action: str                    # plain English


def _load_state() -> dict:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text())
    except Exception:
        pass
    return {"regime_history": [], "stop_dates": []}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        log.warning("whipsaw_state_save_failed", error=str(e))


def record_regime(regime: str) -> None:
    """Call daily from MarketMonitor to track regime history."""
    with _LOCK:
        state = _load_state()
        history = state.get("regime_history", [])
        today = date.today().isoformat()
        # Only record once per day
        if not history or history[-1]["date"] != today:
            history.append({"date": today, "regime": regime})
        # Keep last 15 days
        state["regime_history"] = history[-15:]
        _save_state(state)


def record_stop_hit(symbol: str) -> None:
    """Call when a stop-loss is triggered."""
    with _LOCK:
        state = _load_state()
        stops = state.get("stop_dates", [])
        stops.append({"date": date.today().isoformat(), "symbol": symbol})
        # Keep last 20 stop events
        state["stop_dates"] = stops[-20:]
        _save_state(state)


def _count_regime_flips(history: list[dict], days: int = 10) -> int:
    """Count how many times regime changed in last N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    recent = [h for h in history if h["date"] >= cutoff]
    if len(recent) < 2:
        return 0
    flips = sum(1 for i in range(1, len(recent))
                if recent[i]["regime"] != recent[i-1]["regime"])
    return flips


def _count_consecutive_stops(stops: list[dict], days: int = 7) -> int:
    """Count stop hits in last N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    return sum(1 for s in stops if s["date"] >= cutoff)


def _compute_efficiency_ratio(prices: np.ndarray, period: int = 14) -> float:
    """
    Kaufman's Efficiency Ratio — measures directional efficiency.
    1.0 = perfectly trending. 0.0 = pure chop.
    ER = abs(net_move) / sum(abs(bar_moves))
    """
    if len(prices) < period + 1:
        return 0.5
    net_move = abs(float(prices[-1]) - float(prices[-period]))
    bar_moves = np.sum(np.abs(np.diff(prices[-period:])))
    return float(net_move / bar_moves) if bar_moves > 0 else 0.5


def compute_whipsaw_score(
    nifty_prices: Optional[np.ndarray] = None,
    atr_14: Optional[float] = None,
    atr_5: Optional[float] = None,
) -> WhipsawReport:
    """
    Compute current whipsaw risk. Call before each scan cycle.

    Args:
        nifty_prices: recent Nifty close prices (at least 20 bars)
        atr_14: 14-day ATR of Nifty
        atr_5: 5-day ATR of Nifty (if expanding vs 14d = chop)
    """
    with _LOCK:
        state = _load_state()

    regime_flips = _count_regime_flips(state.get("regime_history", []), days=10)
    consecutive_stops = _count_consecutive_stops(state.get("stop_dates", []), days=7)

    # Efficiency Ratio from Nifty prices
    chop_index = 1.0 - 0.5  # default neutral
    if nifty_prices is not None and len(nifty_prices) >= 15:
        er = _compute_efficiency_ratio(nifty_prices, period=14)
        chop_index = 1.0 - er  # higher chop_index = more choppy

    # ATR expansion without direction = whipsaw
    atr_expanding = False
    if atr_5 is not None and atr_14 is not None and atr_14 > 0:
        atr_expanding = (atr_5 / atr_14) > 1.3

    # Whipsaw score (0-100)
    score = 0.0
    score += min(regime_flips * 12, 40)       # up to 40pts for regime flips
    score += min(consecutive_stops * 8, 24)   # up to 24pts for stop hits
    score += chop_index * 24                  # up to 24pts for chop
    score += 12 if atr_expanding else 0       # 12pts for ATR expansion
    score = min(100.0, round(score, 1))

    # Verdict and sizing
    if score >= 70:
        verdict = "SEVERE"
        size_mult = 0.25
        min_conf = 85.0
        action = (f"SEVERE WHIPSAW: {regime_flips} regime flips, {consecutive_stops} stops this week. "
                  f"Use 25% size. Only ELITE setups with confidence >85. Consider cash.")
    elif score >= 50:
        verdict = "WHIPSAW"
        size_mult = 0.50
        min_conf = 75.0
        action = (f"WHIPSAW market: {regime_flips} regime flips in 10 days. "
                  f"Half size. Require 75%+ confidence. Tighten stops.")
    elif score >= 30:
        verdict = "CHOPPY"
        size_mult = 0.75
        min_conf = 65.0
        action = f"Choppy conditions. 75% size. Raise confidence bar to 65%."
    else:
        verdict = "CLEAN"
        size_mult = 1.0
        min_conf = 60.0
        action = "Market trending cleanly. Standard sizing and confidence thresholds."

    log.info("whipsaw_score_computed", score=score, verdict=verdict,
             flips=regime_flips, stops=consecutive_stops)

    return WhipsawReport(
        score=score,
        regime_flips_10d=regime_flips,
        consecutive_stops=consecutive_stops,
        atr_expanding=atr_expanding,
        chop_index=round(chop_index, 3),
        size_multiplier=size_mult,
        min_confidence_required=min_conf,
        verdict=verdict,
        action=action,
    )


def render_whipsaw_badge() -> str:
    """HTML badge for status bar. Empty if CLEAN."""
    report = compute_whipsaw_score()
    if report.verdict == "CLEAN":
        return ""
    colors = {"CHOPPY": "#f59e0b", "WHIPSAW": "#f97316", "SEVERE": "#ff4b4b"}
    color = colors.get(report.verdict, "#8892a4")
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;border-radius:6px;"
        f"padding:.2rem .6rem;font-size:.7rem;color:{color};font-weight:700'>"
        f"⚡ {report.verdict} · Score {report.score:.0f}/100 · "
        f"{report.regime_flips_10d} flips · {report.consecutive_stops} stops</span>"
    )


def render_whipsaw_ui() -> None:
    """Streamlit panel for whipsaw analysis."""
    return
