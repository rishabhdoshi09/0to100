"""Pullback entries: strong stocks resting at support instead of extended charts.

The breakout categories favour names that have already run (RSI near 70, 3x
volume). The owner asked for opportunities that are not "inflated charts".
This scans the official local bhavcopy history for leaders in an intact uptrend
that have pulled back quietly to their 20/50-day average:

  trend      close > 200-DMA, 50-DMA > 200-DMA and rising over 10 sessions
  leader     within 25% of the 52-week high; 60-day return >= 0
  pullback   5-session return between -8% and -1%, close within 3% of the
             20-DMA or 50-DMA (above the 50-DMA less 2%)
  calm       RSI(14) between 35 and 55; last 3 sessions' volume < 0.9x the
             20-session average (sellers drying up, not panic)
  liquid     20-session average traded value >= Rs 5 cr

Entry is the last close, the stop comes from product.stop_engine (noise
envelope / swing low beyond the band), the target is the 60-session high, and
the setup is kept only when reward:risk >= 1.5. Read-only; nothing here trades.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

MIN_BARS = 220
MIN_TURNOVER_CR = 5.0
MIN_REWARD_RISK = 1.5


def _rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def evaluate_pullback(symbol: str, df) -> dict[str, Any] | None:
    if df is None or len(df) < MIN_BARS or not {"close", "high", "low", "volume"}.issubset(df.columns):
        return None
    close, volume = df["close"], df["volume"]
    last = float(close.iloc[-1])
    if last <= 0:
        return None
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50_series = close.rolling(50).mean()
    sma50, sma50_prev = float(sma50_series.iloc[-1]), float(sma50_series.iloc[-11])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    if not (last > sma200 and sma50 > sma200 and sma50 > sma50_prev):
        return None
    high_252 = float(df["high"].tail(252).max())
    from_high = (high_252 - last) / high_252 * 100.0
    ret_60 = (last / float(close.iloc[-61]) - 1.0) * 100.0
    if from_high > 25.0 or ret_60 < 0:
        return None
    ret_5 = (last / float(close.iloc[-6]) - 1.0) * 100.0
    if not (-8.0 <= ret_5 <= -1.0):
        return None
    near20 = abs(last / sma20 - 1.0) * 100.0
    near50 = abs(last / sma50 - 1.0) * 100.0
    if min(near20, near50) > 3.0 or last < sma50 * 0.98:
        return None
    rsi = float(_rsi(close).iloc[-1])
    if not (35.0 <= rsi <= 55.0):
        return None
    avg_vol20 = float(volume.tail(20).mean())
    dry = float(volume.tail(3).mean()) / avg_vol20 if avg_vol20 > 0 else 9.9
    turnover_cr = float((close.tail(20) * volume.tail(20)).mean()) / 1e7
    if dry >= 0.9 or turnover_cr < MIN_TURNOVER_CR:
        return None

    from product.stop_engine import stop_plan
    plan = stop_plan(symbol, last, None, df=df)
    if not plan or plan.get("too_wide"):
        return None
    stop = float(plan["recommended_stop"])
    target = float(df["high"].tail(60).max())
    risk, reward = last - stop, target - last
    if risk <= 0 or reward / risk < MIN_REWARD_RISK:
        return None
    support = "20-DMA" if near20 <= near50 else "50-DMA"
    trend_spread = (sma50 / sma200 - 1.0) * 100.0
    score = round(
        min(100.0,
            40.0
            + min(20.0, trend_spread)                      # trend strength
            + max(0.0, 15.0 - 3.0 * min(near20, near50))   # closeness to support
            + max(0.0, 15.0 * (0.9 - dry) / 0.9 * 2)       # volume dry-up
            + max(0.0, 10.0 - from_high / 2.5)),           # still near highs
        1,
    )
    reasons = [
        f"Uptrend intact (50-DMA {trend_spread:+.1f}% over 200-DMA, rising)",
        f"Pulled back {ret_5:.1f}% in 5 sessions to the {support}",
        f"RSI {rsi:.0f} — cooled, not stretched",
        f"Volume dried up to {dry:.2f}× average on the dip",
        f"{from_high:.1f}% below the 52-week high",
    ]
    return {
        "symbol": symbol.upper(),
        "price": round(last, 2),
        "entry": round(last, 2),
        "stop": stop,
        "target": round(target, 2),
        "score": score,
        "rsi": round(rsi, 1),
        "volume_ratio": round(dry, 2),
        "momentum_5d": round(ret_5, 2),
        "above_sma50": last > sma50,
        "above_sma200": True,
        "chase_risk": False,
        "entry_state": "ready",
        "signals": ["PULLBACK_TO_SUPPORT"],
        "reasons": reasons,
        "why": "; ".join(reasons[:3]),
        "verdict": "WATCH",
        "reward_risk": round(reward / risk, 2),
        "setup": f"Pullback to {support} in an uptrend",
        "stop_plan": plan,
    }


def find_pullbacks(symbols: Iterable[str] | None = None,
                   get_ohlcv: Callable[[str], Any] | None = None,
                   *, limit: int = 25) -> list[dict[str, Any]]:
    """Best pullback setups across the universe, highest score first. Never raises."""
    if get_ohlcv is None:
        from data.bhavcopy_store import get_ohlcv as get_ohlcv  # noqa: PLW0127
    if symbols is None:
        try:
            from data import bhavcopy_store
            symbols = list(getattr(bhavcopy_store, "_store", {}) or {})
        except Exception:
            symbols = []
    found = []
    for symbol in symbols:
        try:
            row = evaluate_pullback(str(symbol), get_ohlcv(str(symbol)))
        except Exception:
            row = None
        if row:
            found.append(row)
    found.sort(key=lambda r: (-r["score"], r["symbol"]))
    return found[: max(0, int(limit))]
