"""
💎 Long-Term Picks — durable compounders, not swing setups.

A separate lens from the short-term breakout scanner. Where that one hunts for a
break happening THIS week, this screens the whole market for stocks worth HOLDING
for months: a proven long-term uptrend, rising 200-day trend, real momentum,
leadership (near their highs, not laggards), steadiness (not lottery tickets), and
enough liquidity to actually own. All computed from official bhavcopy history —
network-free, no fundamentals API needed for the scan (fundamentals enrich only
the shortlist elsewhere).

`long_term_score(df)` is a pure function (unit-tested on synthetic history);
`scan_long_term()` runs it over the universe and returns ranked picks with a
plain-English thesis. Nothing here places a trade — these are ideas to research
and, if you agree, buy for the long run.
"""
from __future__ import annotations

import os as _os

import numpy as np

# ── tunables (env-overridable) ────────────────────────────────────────────────
_MIN_SESSIONS = int(_os.getenv("QT_LT_MIN_SESSIONS", "220") or 220)
_MIN_TURNOVER_CR = float(_os.getenv("QT_LT_MIN_TURNOVER_CR", "1.0") or 1.0)  # ₹ cr/day
_BUY_SCORE = float(_os.getenv("QT_LT_BUY_SCORE", "62") or 62)


def _sma(x: np.ndarray, n: int) -> float:
    return float(x[-n:].mean()) if len(x) >= n else float("nan")


def long_term_score(df) -> dict:
    """Score a stock's LONG-TERM investment quality (0-100) from its OHLCV
    history. Pure. Returns {score, verdict, factors, above_200dma, dma200_rising,
    mom_6m_pct, mom_12m_pct, from_high_pct, turnover_cr, price}. verdict is
    LONG_TERM_BUY (score ≥ bar AND the trend gate passes) / WATCH / SKIP."""
    try:
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns \
            else np.zeros(len(close))
    except Exception:
        return {"score": 0.0, "verdict": "SKIP", "factors": ["no data"]}
    n = len(close)
    if n < _MIN_SESSIONS:
        return {"score": 0.0, "verdict": "SKIP",
                "factors": [f"need ~{_MIN_SESSIONS} sessions of history"]}

    price = float(close[-1])
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    # 200-DMA slope: today's 200-DMA vs the 200-DMA ~60 sessions ago
    sma200_prev = float(close[-260:-60].mean()) if n >= 260 else sma200
    dma200_rising = sma200 > sma200_prev
    above_200 = price > sma200
    golden = sma50 > sma200

    hi_52 = float(high[-252:].max()) if n >= 252 else float(high.max())
    from_high = (hi_52 - price) / hi_52 * 100 if hi_52 > 0 else 100.0

    mom_6m = (price / float(close[-126]) - 1) * 100 if n >= 126 else 0.0
    mom_12m = (price / float(close[-252]) - 1) * 100 if n >= 252 else mom_6m

    rets = np.diff(close[-90:]) / close[-90:-1] if n >= 91 else np.array([0.0])
    vol_pct = float(np.std(rets) * 100) if rets.size else 5.0     # daily vol %
    turnover_cr = float(np.median((close[-20:] * vol[-20:]))) / 1e7 if n >= 20 else 0.0

    # ── hard gates: no long-term BUY without the structural trend + liquidity ──
    factors: list[str] = []
    if not above_200:
        return {"score": 0.0, "verdict": "SKIP", "price": price,
                "above_200dma": False, "dma200_rising": dma200_rising,
                "mom_6m_pct": round(mom_6m, 1), "mom_12m_pct": round(mom_12m, 1),
                "from_high_pct": round(from_high, 1), "turnover_cr": round(turnover_cr, 1),
                "factors": ["below 200-DMA — not a long-term uptrend"]}
    if turnover_cr < _MIN_TURNOVER_CR:
        return {"score": 0.0, "verdict": "SKIP", "price": price,
                "above_200dma": True, "dma200_rising": dma200_rising,
                "mom_6m_pct": round(mom_6m, 1), "mom_12m_pct": round(mom_12m, 1),
                "from_high_pct": round(from_high, 1), "turnover_cr": round(turnover_cr, 1),
                "factors": [f"too illiquid (₹{turnover_cr:.1f} cr/day)"]}

    # ── score (0-100), evidence-weighted ──────────────────────────────────────
    score = 25.0                                            # base: above a rising-ish 200
    factors.append("above 200-DMA (long-term uptrend)")
    if dma200_rising:
        score += 15; factors.append("200-DMA rising (trend strengthening)")
    if golden:
        score += 8; factors.append("50-DMA above 200-DMA (healthy alignment)")
    # momentum — the engine of long-term returns
    score += float(np.clip(mom_12m * 0.35, -10, 22))
    if mom_12m >= 15:
        factors.append(f"strong 12-month momentum (+{mom_12m:.0f}%)")
    elif mom_12m < 0:
        factors.append(f"12-month momentum negative ({mom_12m:.0f}%)")
    score += float(np.clip(mom_6m * 0.25, -6, 12))
    # leadership — near highs is a leader; deep laggards demoted
    if from_high <= 12:
        score += 12; factors.append(f"near 52-week highs ({from_high:.0f}% below — a leader)")
    elif from_high <= 25:
        score += 5
    elif from_high >= 45:
        score -= 8; factors.append(f"{from_high:.0f}% below highs — laggard")
    # quality — steady compounders over lottery tickets (low daily vol rewarded)
    if vol_pct <= 2.0:
        score += 6; factors.append("steady, low-volatility trend (quality)")
    elif vol_pct >= 4.5:
        score -= 5; factors.append("very choppy — high volatility")
    # over-extension guard — too far above the 200-DMA = mean-reversion risk
    ext = (price / sma200 - 1) * 100 if sma200 > 0 else 0.0
    if ext >= 60:
        score -= 8; factors.append(f"{ext:.0f}% above 200-DMA — extended, wait for a dip")

    score = float(np.clip(round(score, 1), 0, 100))
    verdict = ("LONG_TERM_BUY" if score >= _BUY_SCORE and dma200_rising
               else "WATCH" if score >= 45 else "SKIP")
    return {"score": score, "verdict": verdict, "price": price,
            "above_200dma": above_200, "dma200_rising": dma200_rising,
            "golden_cross": golden, "mom_6m_pct": round(mom_6m, 1),
            "mom_12m_pct": round(mom_12m, 1), "from_high_pct": round(from_high, 1),
            "vol_pct": round(vol_pct, 2), "turnover_cr": round(turnover_cr, 1),
            "extension_pct": round(ext, 1), "factors": factors}


def thesis_line(pick: dict) -> str:
    """One plain-English sentence for a pick — the reason to hold it."""
    f = pick.get("factors", [])
    lead = f[0] if f else "long-term uptrend"
    extra = "; ".join(f[1:3]) if len(f) > 1 else ""
    return (f"{pick.get('symbol','')}: {lead}"
            + (f"; {extra}" if extra else "") + ".")


def scan_long_term(symbols=None, min_score: float | None = None,
                   top: int = 30) -> list[dict]:
    """Screen the market for long-term picks. Returns LONG_TERM_BUY candidates
    ranked by score (best first), each with its score, thesis factors, and key
    metrics. Reads official bhavcopy only — fail-open → []."""
    try:
        from data.bhavcopy_store import get_ohlcv, store_symbols
        syms = symbols if symbols is not None else store_symbols()
    except Exception:
        return []
    bar = _BUY_SCORE if min_score is None else min_score
    picks: list[dict] = []
    for sym in syms:
        try:
            df = get_ohlcv(sym)
            if df is None or len(df) < _MIN_SESSIONS:
                continue
            s = long_term_score(df)
            if s.get("verdict") == "LONG_TERM_BUY" and s.get("score", 0) >= bar:
                s["symbol"] = sym
                s["thesis"] = thesis_line(s)
                picks.append(s)
        except Exception:
            continue
    picks.sort(key=lambda p: p["score"], reverse=True)
    return picks[:top]
