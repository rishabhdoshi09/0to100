"""
Breakout Health Tracker — measures real-time breakout success rate across the NSE universe.

Scans recent 52-week-high candidates, checks whether breakouts extended 5%+ (success)
or reversed below the pivot within 5 days (failure), and outputs a composite score
that drives regime scoring, conviction scoring, and alert thresholds.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

# ---------------------------------------------------------------------------
# Default NSE universe to sample when signal_tracker.db is unavailable
# ---------------------------------------------------------------------------
_DEFAULT_UNIVERSE = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "WIPRO", "LT", "BAJFINANCE",
    "TATAMOTORS", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN",
    "ULTRACEMCO", "NESTLEIND", "POWERGRID", "NTPC", "COALINDIA",
    "HCLTECH", "TECHM", "DIVISLAB", "DRREDDY", "CIPLA",
]

_SIGNAL_TRACKER_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "signal_tracker.db"
)

# Cache: (BreakoutHealth, unix_timestamp)
_CACHE: tuple[Optional["BreakoutHealth"], float] = (None, 0.0)
_CACHE_TTL = 1800  # 30 minutes


@dataclass
class BreakoutHealth:
    success_rate: float              # 0-1: breakouts that extended 5%+ from pivot
    failure_rate: float              # breakouts that reversed below pivot within 5d
    avg_extension_pct: float         # avg % gain from pivot on successful breakouts
    follow_through_score: float      # 0-100 composite
    environment: str                 # FAVORABLE | NEUTRAL | UNFAVORABLE
    sample_size: int                 # number of breakouts evaluated
    recent_failures: list[str] = field(default_factory=list)   # failed in last 5d
    recent_successes: list[str] = field(default_factory=list)  # extended 5%+ in last 5d
    signal: str = "WAIT"             # BUY_BREAKOUTS | WAIT | AVOID_BREAKOUTS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_candidate_symbols(max_symbols: int = 40) -> list[str]:
    """Pull recent BUY-signal symbols from signal_tracker.db, fall back to default."""
    try:
        if not os.path.exists(_SIGNAL_TRACKER_DB):
            return _DEFAULT_UNIVERSE[:max_symbols]
        conn = sqlite3.connect(_SIGNAL_TRACKER_DB)
        conn.row_factory = sqlite3.Row
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        rows = conn.execute(
            """
            SELECT DISTINCT symbol FROM signal_log
            WHERE signal = 'BUY' AND logged_at >= ?
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (cutoff, max_symbols),
        ).fetchall()
        conn.close()
        symbols = [r["symbol"] for r in rows]
        if len(symbols) < 10:
            # Supplement with defaults
            for sym in _DEFAULT_UNIVERSE:
                if sym not in symbols:
                    symbols.append(sym)
                if len(symbols) >= max_symbols:
                    break
        return symbols[:max_symbols]
    except Exception:
        return _DEFAULT_UNIVERSE[:max_symbols]


def _fetch_ohlcv(symbol: str, days: int = 30) -> "Optional[object]":
    """Return a pandas DataFrame of daily OHLCV for an NSE symbol, or None."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        end = date.today()
        start = end - timedelta(days=days + 5)   # buffer for weekends
        hist = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1d")
        if hist.empty or len(hist) < 5:
            return None
        return hist
    except Exception:
        return None


def _evaluate_breakout(symbol: str) -> tuple[str, float]:
    """
    Classify the most recent breakout attempt for a symbol.

    Returns:
        (status, extension_pct) where status is one of:
            "SUCCESS"  — broke out and extended 5%+
            "FAILURE"  — broke out then returned below pivot within 5d
            "PENDING"  — broke out recently, outcome not yet clear
            "NO_BREAKOUT" — no breakout in the look-back window
    """
    hist = _fetch_ohlcv(symbol, days=35)
    if hist is None or len(hist) < 15:
        return "NO_BREAKOUT", 0.0

    closes = hist["Close"].values
    highs = hist["High"].values

    # Pivot = 20-day prior high (use bars [-30:-10] as the "prior" window)
    if len(closes) < 20:
        return "NO_BREAKOUT", 0.0

    prior_window = highs[:-10] if len(highs) > 10 else highs
    pivot = float(prior_window.max())

    # Look for a breakout bar in the last 10 sessions
    recent_closes = closes[-10:]
    recent_highs = highs[-10:]

    breakout_idx = None
    for i, (c, h) in enumerate(zip(recent_closes, recent_highs)):
        if h > pivot * 1.002:      # 0.2% clearance to avoid noise
            breakout_idx = i
            break

    if breakout_idx is None:
        return "NO_BREAKOUT", 0.0

    breakout_price = float(recent_closes[breakout_idx])
    post_bars = recent_closes[breakout_idx:]

    if len(post_bars) < 2:
        return "PENDING", 0.0

    max_post = float(max(post_bars))
    min_post = float(min(post_bars))
    extension_pct = (max_post - breakout_price) / breakout_price * 100.0

    if extension_pct >= 5.0:
        return "SUCCESS", extension_pct
    if min_post < pivot:
        return "FAILURE", extension_pct
    if len(post_bars) <= 5:
        return "PENDING", extension_pct
    return "FAILURE", extension_pct   # extended window with no follow-through → failure


def _compute_health(symbols: list[str]) -> BreakoutHealth:
    """Evaluate each symbol and aggregate into a BreakoutHealth dataclass."""
    successes: list[str] = []
    failures: list[str] = []
    extensions: list[float] = []
    evaluated = 0

    for sym in symbols:
        status, ext = _evaluate_breakout(sym)
        if status == "NO_BREAKOUT" or status == "PENDING":
            continue
        evaluated += 1
        if status == "SUCCESS":
            successes.append(sym)
            extensions.append(ext)
        elif status == "FAILURE":
            failures.append(sym)

    if evaluated == 0:
        return BreakoutHealth(
            success_rate=0.0,
            failure_rate=0.0,
            avg_extension_pct=0.0,
            follow_through_score=0.0,
            environment="NEUTRAL",
            sample_size=0,
            recent_failures=[],
            recent_successes=[],
            signal="WAIT",
        )

    success_rate = len(successes) / evaluated
    failure_rate = len(failures) / evaluated
    avg_extension = sum(extensions) / len(extensions) if extensions else 0.0

    # Composite score: 0-100
    # 50 pts from success_rate, 30 pts from avg_extension (capped at 15%), 20 pts from sample size
    score_success = success_rate * 50.0
    score_extension = min(avg_extension / 15.0, 1.0) * 30.0
    score_sample = min(evaluated / 20.0, 1.0) * 20.0
    follow_through_score = round(score_success + score_extension + score_sample, 1)

    # Environment classification
    if success_rate > 0.55 and evaluated > 5:
        environment = "FAVORABLE"
    elif failure_rate > 0.60:
        environment = "UNFAVORABLE"
    else:
        environment = "NEUTRAL"

    # Signal
    if failure_rate > 0.60:
        signal = "AVOID_BREAKOUTS"
    elif success_rate > 0.55 and follow_through_score >= 50:
        signal = "BUY_BREAKOUTS"
    else:
        signal = "WAIT"

    return BreakoutHealth(
        success_rate=round(success_rate, 4),
        failure_rate=round(failure_rate, 4),
        avg_extension_pct=round(avg_extension, 2),
        follow_through_score=follow_through_score,
        environment=environment,
        sample_size=evaluated,
        recent_failures=failures[:10],
        recent_successes=successes[:10],
        signal=signal,
    )


class BreakoutHealthTracker:
    """Tracks real-time breakout health across the NSE universe."""

    def __init__(self, universe: Optional[list[str]] = None, max_symbols: int = 40):
        self._universe = universe
        self._max_symbols = max_symbols
        self._cache: Optional[BreakoutHealth] = None
        self._cache_ts: float = 0.0

    def refresh(self) -> BreakoutHealth:
        """Force a fresh evaluation regardless of cache TTL."""
        symbols = self._universe if self._universe else _get_candidate_symbols(self._max_symbols)
        health = _compute_health(symbols)
        self._cache = health
        self._cache_ts = time.time()
        return health

    def get_health(self, force_refresh: bool = False) -> BreakoutHealth:
        """Return cached BreakoutHealth (30 min TTL) or compute fresh."""
        if force_refresh or self._cache is None or (time.time() - self._cache_ts) > _CACHE_TTL:
            return self.refresh()
        return self._cache


# ---------------------------------------------------------------------------
# Module-level cached convenience function (30 min TTL)
# ---------------------------------------------------------------------------

_module_tracker = BreakoutHealthTracker()


def get_health(force_refresh: bool = False) -> BreakoutHealth:
    """Module-level cached function — returns BreakoutHealth with 30 min TTL."""
    return _module_tracker.get_health(force_refresh=force_refresh)
