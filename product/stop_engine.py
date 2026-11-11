"""Evidence-based stop placement from each stock's own official price history.

A stop is where the trade idea is wrong, not where ordinary noise happens to
reach. Stops parked at obvious levels (a fresh swing low, a round number,
"2% below entry") cluster with everyone else's and get swept. This engine
reads the stock's own last ~250 official NSE sessions (local bhavcopy store,
corporate-action adjusted) and builds the stop from four measured inputs:

1. Noise envelope. For every past session in an uptrend (close above its
   50-day mean) it measures the worst dip over the next N sessions, in ATR
   units. The 85th percentile of those dips is how far this stock normally
   wanders while its trend holds.
2. Structure. The lowest low of the recent base (10 sessions), less a quarter
   ATR so the stop sits beyond the level everyone can see.
3. Round numbers. A stop just above a round level (e.g. 400.15) is moved below
   it, beyond where resting orders cluster.
4. Gap risk. The 95th-percentile overnight gap down: a stop cannot protect
   against it, so it is reported for position sizing.

The recommended stop is the wider of structure and noise, unless that risks
more than MAX_RISK_PCT of price (then the setup is flagged as too wide for its
volatility). Every plan reports a measured "noise-hit rate": the share of
historical N-session uptrend windows whose dip would have reached that stop,
for the recommended stop and for the scanner's stop, so the two can be
compared on evidence.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

HORIZON = 5
LOOKBACK = 250
NOISE_PCTL = 0.85
SWING_SESSIONS = 20
STRUCTURE_BUFFER_ATR = 0.25
ROUND_BUFFER_ATR = 0.15
MAX_RISK_PCT = 10.0
TICK = 0.05
MIN_SAMPLES = 40


def _tick(price: float) -> float:
    return round(math.floor(price / TICK + 1e-9) * TICK, 2)


def _round_step(price: float) -> float:
    if price >= 1000:
        return 50.0
    if price >= 200:
        return 10.0
    if price >= 50:
        return 5.0
    return 1.0


def _below_round_number(stop: float, atr: float) -> tuple[float, str]:
    """Move a stop that sits just above a round number to below it."""
    step = _round_step(stop)
    level = math.floor(stop / step) * step
    if level > 0 and 0 <= (stop - level) / stop <= 0.005:
        return level - ROUND_BUFFER_ATR * atr, f"moved below round number {level:g}"
    return stop, ""


def _atr(df, period: int = 14):
    import pandas as pd

    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def noise_profile(df, *, horizon: int = HORIZON, lookback: int = LOOKBACK) -> dict[str, Any] | None:
    """Uptrend-day adverse excursions over `horizon` sessions, in ATR units."""
    if df is None or len(df) < 80 or not {"high", "low", "close"}.issubset(df.columns):
        return None
    data = df.tail(lookback + horizon + 60).copy()
    data["atr"] = _atr(data)
    data["sma50"] = data["close"].rolling(50).mean()
    future_low = data["low"][::-1].rolling(horizon).min()[::-1].shift(-1)
    data["mae_atr"] = (data["close"] - future_low) / data["atr"]
    valid = data["mae_atr"].notna() & (data["atr"] > 0)
    sample = data[(data["close"] > data["sma50"]) & valid].tail(lookback)
    regime = "uptrend sessions"
    if len(sample) < MIN_SAMPLES:
        # Too few uptrend sessions to measure: use every session instead.
        sample = data[valid].tail(lookback)
        regime = "all sessions"
    if len(sample) < MIN_SAMPLES:
        return None
    mae = sample["mae_atr"].clip(lower=0.0)
    gaps = (data["open"] / data["close"].shift(1) - 1.0).dropna() if "open" in data.columns else None
    return {
        "samples": int(len(mae)),
        "mae_atr": sorted(float(x) for x in mae),
        "noise_k": float(mae.quantile(NOISE_PCTL)),
        "atr": float(data["atr"].iloc[-1]),
        "gap_down_p95_pct": (round(float(-gaps.quantile(0.05)) * 100.0, 2)
                             if gaps is not None and len(gaps) >= MIN_SAMPLES else None),
        "swing_lows": _swing_lows(data.tail(SWING_SESSIONS)),
        "regime": regime,
    }


def _swing_lows(data) -> list[float]:
    """Pivot lows (lower than the two sessions on each side), newest first."""
    lows = list(data["low"])
    out = []
    for i in range(2, len(lows) - 2):
        if lows[i] < min(lows[i - 2], lows[i - 1], lows[i + 1], lows[i + 2]):
            out.append(float(lows[i]))
    return list(reversed(out))


def noise_hit_rate(profile: Mapping[str, Any], entry: float, stop: float) -> float | None:
    """Share of historical uptrend windows whose dip would reach `stop`."""
    atr = float(profile.get("atr") or 0.0)
    values = list(profile.get("mae_atr") or [])
    if atr <= 0 or not values or stop >= entry:
        return None
    k = (entry - stop) / atr
    return round(sum(1 for v in values if v >= k) / len(values), 4)


def stop_plan(symbol: str, entry: float, scanner_stop: float | None = None, *, df=None,
              horizon: int = HORIZON) -> dict[str, Any] | None:
    try:
        entry = float(entry)
    except (TypeError, ValueError):
        return None
    if entry <= 0:
        return None
    if df is None:
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(symbol)
        except Exception:
            df = None
    profile = noise_profile(df, horizon=horizon)
    if not profile:
        return None
    atr = profile["atr"]
    noise_stop = entry - profile["noise_k"] * atr
    # A visible swing low inside the noise band is where resting stops cluster
    # and get swept: if one is there, the stop goes beyond it, not on it.
    inside = [low for low in profile.get("swing_lows") or [] if noise_stop <= low < entry]
    structure_stop = (min(inside) - STRUCTURE_BUFFER_ATR * atr) if inside else None
    if structure_stop is not None and structure_stop < noise_stop:
        base, method = structure_stop, "beyond swing low inside the noise band (− ¼ ATR)"
    else:
        base, method = noise_stop, f"noise envelope (p{int(NOISE_PCTL * 100)} of {profile['regime']})"
    base, round_note = _below_round_number(base, atr)
    stop = _tick(base)
    risk_pct = (entry - stop) / entry * 100.0
    plan: dict[str, Any] = {
        "recommended_stop": stop,
        "method": method + (f"; {round_note}" if round_note else ""),
        "risk_pct": round(risk_pct, 2),
        "too_wide": risk_pct > MAX_RISK_PCT,
        "atr": round(atr, 2),
        "noise_k_atr": round(profile["noise_k"], 2),
        "noise_stop": _tick(noise_stop),
        "structure_stop": _tick(structure_stop) if structure_stop is not None else None,
        "noise_hit_rate": noise_hit_rate(profile, entry, stop),
        "gap_down_p95_pct": profile["gap_down_p95_pct"],
        "samples": profile["samples"],
        "sample_regime": profile["regime"],
        "horizon_sessions": horizon,
        "trail": "At +1R move the stop to entry; then trail 3×ATR below the highest close.",
        "time_stop": f"If the trade is not up 0.5R after {horizon + 2} sessions, exit: capital is better used elsewhere.",
    }
    try:
        scanner = float(scanner_stop) if scanner_stop is not None else None
    except (TypeError, ValueError):
        scanner = None
    if scanner and 0 < scanner < entry:
        plan["scanner_stop"] = round(scanner, 2)
        plan["scanner_noise_hit_rate"] = noise_hit_rate(profile, entry, scanner)
    hit = plan["noise_hit_rate"]
    plan["text"] = (
        f"Stop ₹{stop:,.2f} ({risk_pct:.1f}% below entry) by {plan['method']}"
        + (f"; normal noise reached it in {hit:.0%} of {horizon}-session windows ({profile['regime']})"
           if hit is not None else "")
        + (f" vs {plan['scanner_noise_hit_rate']:.0%} for the scanner stop ₹{plan['scanner_stop']:,.2f}"
           if plan.get("scanner_noise_hit_rate") is not None else "")
        + (f"; overnight gaps (p95 {plan['gap_down_p95_pct']:.1f}%) can jump any stop — size for it"
           if plan.get("gap_down_p95_pct") is not None else "")
        + ("; TOO WIDE for this volatility — skip or size down" if plan["too_wide"] else "")
    )
    return plan
