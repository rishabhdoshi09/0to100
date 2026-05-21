"""
Stop Hunt Detector — The Hela Defense.

Hela takes what's yours. Stop hunts do the same.

A stop hunt occurs when price briefly dips below a known stop level,
triggers retail exits, then immediately recovers. Algos and smart money
do this deliberately near round numbers and recent swing lows.

This module:
  1. Detects stop hunt patterns in recent price action
  2. Recommends stop placement BELOW obvious levels (avoid round numbers)
  3. Logs symbols that repeatedly get stop-hunted (avoid them)
  4. Suggests ATR-based stops instead of level-based stops
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_DB = Path("logs/stop_hunt_history.db")


def _init_db() -> None:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS stop_hunts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                hunt_date   TEXT,
                stop_level  REAL,
                dip_low     REAL,
                recovery    REAL,
                hunt_pct    REAL
            )
        """)
        con.commit()


_init_db()


@dataclass
class StopHuntVerdict:
    risk_level: str            # LOW | MEDIUM | HIGH
    hunt_detected_today: bool
    recommendation: str        # plain English action
    safer_stop: Optional[float]  # ATR-based stop suggestion
    historical_hunt_count: int   # how many times this symbol was hunted in 30d


def detect_stop_hunt(
    symbol: str,
    df: pd.DataFrame,
    proposed_stop: float,
    atr: float,
) -> StopHuntVerdict:
    """
    Analyse recent price action for stop hunt patterns.
    A hunt: price dips > 0.5% below proposed_stop, then recovers > 1% within same day.
    """
    if df is None or len(df) < 10:
        return StopHuntVerdict("LOW", False, "Insufficient data.", proposed_stop, 0)

    close = df["close"].values
    low   = df["low"].values
    high  = df["high"].values

    hunt_today = False
    # Check last 5 sessions for stop hunt pattern
    for i in range(1, min(6, len(low))):
        dip_pct = (proposed_stop - low[-i]) / proposed_stop * 100
        recovery_pct = (close[-i] - low[-i]) / low[-i] * 100
        if dip_pct > 0.5 and recovery_pct > 1.0:
            hunt_today = (i == 1)
            _record_hunt(symbol, proposed_stop, low[-i], close[-i], dip_pct)
            log.info("stop_hunt_detected", symbol=symbol, session=i,
                     dip_pct=round(dip_pct, 2), recovery_pct=round(recovery_pct, 2))

    # Historical count
    hist_count = _get_hunt_count(symbol, days=30)

    # Round number proximity check (stops near round numbers get hunted more)
    round_number_risk = _is_near_round_number(proposed_stop)

    # Safer stop: 1.5× ATR below current price (not near obvious level)
    safer_stop = round(float(close[-1]) - 1.5 * atr, 2) if atr > 0 else proposed_stop

    # Risk assessment
    if hist_count >= 3 or (hunt_today and round_number_risk):
        risk = "HIGH"
        rec = (f"HIGH HUNT RISK: {symbol} was stop-hunted {hist_count}× in 30 days. "
               f"Move stop to ₹{safer_stop} (1.5× ATR) — away from obvious ₹{proposed_stop:.0f} level.")
    elif hist_count >= 1 or round_number_risk:
        risk = "MEDIUM"
        rec = (f"Stop near {'round number' if round_number_risk else 'hunted level'}. "
               f"Consider ₹{safer_stop} instead of ₹{proposed_stop:.0f}.")
    else:
        risk = "LOW"
        rec = f"Stop placement looks safe. ATR-based alternative: ₹{safer_stop}."

    return StopHuntVerdict(
        risk_level=risk,
        hunt_detected_today=hunt_today,
        recommendation=rec,
        safer_stop=safer_stop,
        historical_hunt_count=hist_count,
    )


def _is_near_round_number(price: float, tolerance_pct: float = 0.5) -> bool:
    """True if price is within tolerance_pct% of a round number (50, 100, 500, etc.)."""
    for base in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]:
        nearest = round(price / base) * base
        if abs(price - nearest) / price * 100 < tolerance_pct:
            return True
    return False


def _record_hunt(symbol: str, stop: float, low: float, close: float, pct: float) -> None:
    try:
        with sqlite3.connect(str(_DB)) as con:
            con.execute(
                "INSERT INTO stop_hunts (symbol, hunt_date, stop_level, dip_low, recovery, hunt_pct) "
                "VALUES (?,?,?,?,?,?)",
                (symbol, datetime.now().date().isoformat(), stop, low, close, round(pct, 2)),
            )
            con.commit()
    except Exception:
        pass


def _get_hunt_count(symbol: str, days: int = 30) -> int:
    try:
        from datetime import date, timedelta
        since = (date.today() - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(_DB)) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM stop_hunts WHERE symbol=? AND hunt_date >= ?",
                (symbol, since),
            ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0
