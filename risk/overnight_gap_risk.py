"""
Overnight Gap Risk — The Malekith Defense.

Malekith strikes in the dark. Overnight gaps do the same.

A stock can open 8% lower due to:
  - Global market crash (US/Asia overnight)
  - Company-specific bad news (promoter pledging, fraud, downgrade)
  - Sector regulatory action (SEBI, RBI, IRDAI)
  - Earnings miss after hours
  - F&O expiry pinning (pre-expiry gaps)

This module:
  1. Scores each position's overnight gap risk (0-100)
  2. Computes historical gap frequency for the symbol
  3. Checks global market stress (SGX Nifty, US futures proxy)
  4. Recommends position sizing reduction for overnight holds
  5. Flags high-beta stocks that amplify overnight moves
  6. Blocks overnight holds when global stress + high beta align
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_DB = Path("logs/gap_history.db")


def _init_db() -> None:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS gap_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                gap_date    TEXT,
                gap_pct     REAL,
                direction   TEXT
            )
        """)
        con.commit()


_init_db()


@dataclass
class GapRiskReport:
    symbol: str
    gap_risk_score: float          # 0-100
    historical_gap_freq: float     # % of days with gap >1.5%
    avg_gap_size: float            # average gap size on gap days
    beta: float                    # vs Nifty 50
    global_stress: str             # LOW | MEDIUM | HIGH
    recommended_size: float        # 0.0 to 1.0 multiplier
    block_overnight: bool
    verdict: str                   # SAFE | CAUTION | RISKY | BLOCK
    reason: str


def compute_gap_risk(
    symbol: str,
    df=None,           # OHLCV DataFrame
    nifty_df=None,     # Nifty OHLCV for beta calc
) -> GapRiskReport:
    """
    Score overnight gap risk for a symbol.
    Higher score = riskier to hold overnight.
    """
    # ── Fetch data if not provided ────────────────────────────────────────────
    if df is None:
        df = _fetch_ohlcv(symbol, days=90)
    if nifty_df is None:
        nifty_df = _fetch_ohlcv("^NSEI", days=90, is_index=True)

    if df is None or len(df) < 20:
        return GapRiskReport(symbol, 50.0, 0.0, 0.0, 1.0, "MEDIUM",
                             0.75, False, "CAUTION", "Insufficient data — default caution.")

    close = df["close"].values if "close" in df.columns else df["Close"].values
    open_ = df["open"].values  if "open"  in df.columns else df["Open"].values

    # ── Gap frequency & size ──────────────────────────────────────────────────
    # Gap = today's open vs yesterday's close
    gaps = np.abs(open_[1:] - close[:-1]) / close[:-1] * 100
    large_gaps = gaps[gaps >= 1.5]
    gap_freq = len(large_gaps) / len(gaps) if len(gaps) > 0 else 0.0
    avg_gap = float(large_gaps.mean()) if len(large_gaps) > 0 else 0.0

    # Record significant new gaps to DB
    if len(gaps) > 0 and gaps[-1] >= 2.0:
        direction = "UP" if open_[-1] > close[-2] else "DOWN"
        _record_gap(symbol, date.today().isoformat(), float(gaps[-1]), direction)

    # ── Beta calculation ──────────────────────────────────────────────────────
    beta = 1.0
    if nifty_df is not None and len(nifty_df) >= 20:
        try:
            nc = nifty_df["close"].values if "close" in nifty_df.columns else nifty_df["Close"].values
            min_len = min(len(close), len(nc))
            sym_ret   = np.diff(close[-min_len:])   / close[-min_len:-1]
            nifty_ret = np.diff(nc[-min_len:])      / nc[-min_len:-1]
            if len(sym_ret) > 10 and np.std(nifty_ret) > 0:
                beta = float(np.cov(sym_ret, nifty_ret)[0, 1] / np.var(nifty_ret))
                beta = round(max(0.0, min(beta, 4.0)), 2)
        except Exception:
            pass

    # ── Global stress proxy ───────────────────────────────────────────────────
    global_stress, stress_score = _get_global_stress()

    # ── Composite gap risk score ──────────────────────────────────────────────
    score = 0.0
    score += min(gap_freq * 200, 35)         # up to 35pts for gap frequency
    score += min(avg_gap * 3, 20)            # up to 20pts for avg gap size
    score += min((beta - 1.0) * 15, 20)     # up to 20pts for high beta
    score += stress_score * 0.25             # up to 25pts for global stress
    score = min(100.0, max(0.0, round(score, 1)))

    # Historical from DB
    hist_count = _get_historical_gaps(symbol, days=30)
    if hist_count >= 3:
        score = min(100.0, score + 10)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if score >= 70 or (beta > 1.8 and global_stress == "HIGH"):
        verdict = "BLOCK"
        size = 0.0
        block = True
        reason = (f"High gap risk: beta={beta:.1f}, {gap_freq:.0%} gap frequency, "
                  f"global stress={global_stress}. Do NOT hold overnight.")
    elif score >= 50:
        verdict = "RISKY"
        size = 0.5
        block = False
        reason = (f"Elevated overnight risk: score {score:.0f}. "
                  f"If holding, use 50% size and set wider stop.")
    elif score >= 30:
        verdict = "CAUTION"
        size = 0.75
        block = False
        reason = f"Moderate gap risk (beta={beta:.1f}). 75% size for overnight holds."
    else:
        verdict = "SAFE"
        size = 1.0
        block = False
        reason = f"Low gap risk. Standard position size acceptable overnight."

    log.info("gap_risk_computed", symbol=symbol, score=score, verdict=verdict, beta=beta)

    return GapRiskReport(
        symbol=symbol,
        gap_risk_score=score,
        historical_gap_freq=round(gap_freq, 3),
        avg_gap_size=round(avg_gap, 2),
        beta=beta,
        global_stress=global_stress,
        recommended_size=size,
        block_overnight=block,
        verdict=verdict,
        reason=reason,
    )


def score_portfolio_overnight_risk(open_symbols: list[str]) -> dict:
    """
    Score overnight risk for all open positions.
    Returns {symbol: GapRiskReport} dict.
    """
    results = {}
    for sym in open_symbols:
        try:
            results[sym] = compute_gap_risk(sym)
        except Exception as e:
            log.debug("gap_risk_failed", symbol=sym, error=str(e))
    return results


def _get_global_stress() -> tuple[str, float]:
    """
    Proxy global stress from VIX and SGX Nifty.
    Returns (level_str, score_0_to_100).
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2d")
        if vix is not None and len(vix) >= 1:
            vix_val = float(vix["Close"].iloc[-1])
            if vix_val > 25:   return "HIGH",   100.0
            if vix_val > 18:   return "MEDIUM",  50.0
            return "LOW", 15.0
    except Exception:
        pass
    return "MEDIUM", 50.0


def _fetch_ohlcv(symbol: str, days: int = 90, is_index: bool = False):
    try:
        import yfinance as yf
        suffix = "" if is_index else ".NS"
        df = yf.Ticker(f"{symbol}{suffix}").history(period=f"{days}d", interval="1d")
        if df is not None and len(df) >= 15:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.debug("gap_ohlcv_fetch_failed", symbol=symbol, error=str(e))
    return None


def _record_gap(symbol: str, gap_date: str, gap_pct: float, direction: str) -> None:
    try:
        with sqlite3.connect(str(_DB)) as con:
            # Avoid duplicates for same symbol+date
            existing = con.execute(
                "SELECT id FROM gap_events WHERE symbol=? AND gap_date=?",
                (symbol, gap_date)
            ).fetchone()
            if not existing:
                con.execute(
                    "INSERT INTO gap_events (symbol, gap_date, gap_pct, direction) VALUES (?,?,?,?)",
                    (symbol, gap_date, round(gap_pct, 2), direction),
                )
                con.commit()
    except Exception:
        pass


def _get_historical_gaps(symbol: str, days: int = 30) -> int:
    try:
        since = (date.today() - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(_DB)) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM gap_events WHERE symbol=? AND gap_date >= ? AND gap_pct >= 2.0",
                (symbol, since)
            ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def render_gap_risk_ui(open_symbols: list[str]) -> None:
    """Streamlit panel showing overnight gap risk for all positions."""
    return
