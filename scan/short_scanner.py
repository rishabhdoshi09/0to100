"""
🔻 Short scanner — the SAME edge, pointed DOWN. (Detection only, paper-first.)

Weak tape mein paisa short side pe hota hai. This is the mirror of
unified_scanner: it detects BEARISH setups — confirmed breakdowns below
support, death-crosses, distribution, downside momentum — and grades them
exactly the way the long side grades breakouts, reusing the same primitives
(RSI, ATR, CLV, EMA) so there is ONE language for evidence.

Every long-side quality gate has a mirror here, because the risk flips:
  • RSI ceiling (blow-off top)      → RSI FLOOR (oversold capitulation):
    don't short a stock already crushed — the bounce is the risk.
  • weak close (CLV low) on a break → STRONG close (CLV high) on a break:
    a breakdown that closes near the day's HIGH is a failed breakdown /
    bear-trap — buyers took it back.
  • falling-knife (don't buy red)   → RISING-ROCKET (don't short green):
    a short is a WEAKNESS entry; shorting a stock that's UP today or whose
    RSI is turning up is catching a rocket.
  • too-extended-ABOVE-50-DMA       → too-extended-BELOW-50-DMA: a stock
    already far below its 50-DMA is stretched for a mean-reversion bounce.

INVARIANTS honored: no live orders here (India blocks overnight equity
shorts anyway — cash shorts are intraday-only, positional needs F&O). This
module only DETECTS and grades; execution is a separate, later, opt-in
decision. Verdicts: SHORT (clean bearish setup) · AVOID (bearish — don't
hold/enter longs) · "" (nothing). Pure + testable — no I/O.
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass, field

import numpy as np

# Reuse the long side's primitives — one math, one language for evidence.
from scan.unified_scanner import (
    _rsi, _atr, _ema_np, _norm, close_location_value, _market_is_live,
)

# ── Mirror thresholds (env-tunable, same knobs as the long side, flipped) ─────
_BREAKDOWN_MIN_VOL = float(_os.getenv("QT_BREAKDOWN_MIN_VOL", "1.5") or 1.5)
_BREAKDOWN_ATR_BUFFER = float(_os.getenv("QT_BREAKDOWN_ATR_BUFFER", "0.25") or 0.25)
_BREAKDOWN_MAX_GAP = float(_os.getenv("QT_BREAKDOWN_MAX_GAP", "8") or 8)
# RSI FLOOR — mirror of the blow-off-top ceiling. Below hard = capitulation
# (bounce risk, don't chase the short); below soft = extended-oversold (demote).
_RSI_OVERSOLD_HARD = float(_os.getenv("QT_RSI_OVERSOLD_HARD", "18") or 18)
_RSI_OVERSOLD_SOFT = float(_os.getenv("QT_RSI_OVERSOLD_SOFT", "28") or 28)
# CLV STRONG — mirror of _CLV_WEAK. A breakdown closing in the TOP 40% of the
# day's range (>0.60) means buyers took the day back → failed breakdown.
_CLV_STRONG = float(_os.getenv("QT_CLV_STRONG", "0.60") or 0.60)
# Rising-rocket guard — don't short strength (mirror of the falling-knife).
_RISING_DAY_PCT = float(_os.getenv("QT_RISING_DAY_PCT", "1.0") or 1.0)
_RSI_RALLY_RISE = float(_os.getenv("QT_RSI_RALLY_RISE", "5") or 5)
# Too-extended-below-50-DMA — stretched for a bounce (mirror of the long guard).
_EXT_BELOW_SMA50 = float(_os.getenv("QT_EXT_BELOW_SMA50_PCT", "20") or 20)

# Bearish signal weights (mirror of SIGNAL_META's breakout weights).
SHORT_META = {
    "BREAKDOWN_52W":  ("52-week low breakdown",       30),
    "BREAKDOWN_SUP":  ("Support break on volume",     26),
    "DEATH_CROSS":    ("Death cross (50/200 SMA)",    22),
    "DISTRIBUTION":   ("Distribution — smart-money selling", 24),
    "MOMENTUM_DOWN":  ("Downside momentum",           20),
    "LOWER_HIGHS":    ("Lower highs — supply stacking", 18),
}


@dataclass
class ShortSignal:
    symbol: str
    price: float
    change_pct: float
    rsi: float
    volume_ratio: float
    signals: list[str]
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    entry: float = 0.0          # breakdown level (short entry)
    stop: float = 0.0           # ABOVE entry — a short loses when price rises
    target: float = 0.0         # BELOW entry
    verdict: str = ""           # SHORT | AVOID | ""
    breakdown_grade: str = ""   # A | B | ""

    @property
    def risk_reward(self) -> float:
        risk = self.stop - self.entry           # stop is above for a short
        reward = self.entry - self.target       # target is below
        return round(reward / risk, 2) if risk > 0 else 0.0

    @property
    def signal_labels(self) -> list[str]:
        return [SHORT_META[s][0] for s in self.signals if s in SHORT_META]


def grade_breakdown(price: float, level: float, atr: float, vratio: float,
                    day_change: float, clv: float = 0.0, rsi: float = 50.0
                    ) -> tuple[bool, str, str]:
    """(confirmed, grade, note) — the DOWNSIDE mirror of grade_breakout.
    Breakdown = price cleared BELOW a support `level`. Grade A = clears by
    ≥1×ATR on ≥2× volume; B = clears by the buffer on ≥min volume.

    Mirror quality gates (each DEMOTES, never upgrades):
      day_change < −MAX_GAP  → capitulation gap-down, don't chase the short.
      rsi ≤ HARD floor       → already crushed (bounce risk), reject.
      rsi ≤ SOFT floor       → extended-oversold, demote A→B.
      clv > STRONG (0.60)    → closed near the day's HIGH = failed breakdown /
                               bear-trap (buyers took it back), demote."""
    if level <= 0 or price >= level:
        return False, "", ""
    drop = level - price
    atr_mult = drop / atr if atr > 0 else 0.0
    if day_change < -_BREAKDOWN_MAX_GAP:
        return False, "", f"gap {day_change:+.0f}% — capitulation, short chase nahi"
    if rsi <= _RSI_OVERSOLD_HARD:
        return False, "", f"RSI {rsi:.0f} — oversold capitulation, bounce risk"
    vol_ok = vratio >= _BREAKDOWN_MIN_VOL
    room_ok = atr_mult >= _BREAKDOWN_ATR_BUFFER
    if not (vol_ok and room_ok):
        why = ("volume kam" if not vol_ok else f"clearance sirf {atr_mult:.2f}×ATR")
        tail = ("close confirm ka wait" if _market_is_live()
                else "close pe bhi confirm nahi hua")
        return False, "", f"marginal breakdown ({why}) — {tail}"
    flags = []
    if clv > _CLV_STRONG:
        flags.append(f"strong close (CLV {clv:.2f})")
    if rsi <= _RSI_OVERSOLD_SOFT:
        flags.append(f"RSI {rsi:.0f} extended-oversold")
    if atr_mult >= 1.0 and vratio >= 2.0:
        if flags:
            return (True, "B", f"clean breakdown ({atr_mult:.1f}×ATR, "
                    f"{vratio:.1f}×vol) PAR {', '.join(flags)} — A se B demote")
        return True, "A", f"clean breakdown: {atr_mult:.1f}×ATR neeche, {vratio:.1f}× volume"
    if flags:
        return (False, "", f"confirmed breakdown PAR {', '.join(flags)} — "
                f"bear-trap risk, chase nahi")
    return True, "B", f"confirmed: {atr_mult:.2f}×ATR neeche, {vratio:.1f}× volume"


def short_conviction(vratio: float, rs_underperf: float, below_50: bool,
                     below_200: bool) -> tuple[float, list[str]]:
    """0-100 conviction behind a confirmed breakdown + plain-English factors.
      Volume 30 · Relative WEAKNESS 30 · Trend stage (below 50/200) 40.
    Mirror of breakout_conviction, minus the India-only delivery stream
    (delivery% signals accumulation, not distribution — no clean short read)."""
    factors: list[str] = []
    score = _norm(vratio, 1.5, 3.0) * 30
    if vratio >= 2.0:
        factors.append(f"{vratio:.1f}× volume (selling force)")
    score += _norm(rs_underperf, -3, 12) * 30       # underperformance = weakness
    if rs_underperf >= 3:
        factors.append(f"Nifty se {rs_underperf:+.0f}% peeche (RS laggard)")
    if below_50:
        score += 20
    if below_200:
        score += 20
        factors.append("200-DMA ke neeche (Stage 4 downtrend)")
    return round(min(100.0, score), 0), factors


def analyze_short(symbol: str, df) -> ShortSignal | None:
    """Bearish-setup detection on a daily OHLCV frame. Returns a ShortSignal
    (verdict SHORT/AVOID/"") or None when there's nothing bearish. Pure —
    no I/O, no orders. Mirror of UnifiedScanner._analyze, downside only."""
    try:
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float) if "volume" in df else None
    except Exception:
        return None
    if len(close) < 60:
        return None

    price = float(close[-1])
    chg = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0.0
    mom5 = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0.0
    rsi = _rsi(close)
    atr = _atr(high, low, close)
    sma50 = close[-50:].mean() if len(close) >= 50 else price
    vratio = 1.0
    if vol is not None and len(vol) > 21:
        avg = np.nanmean(vol[-21:-1])
        vratio = float(vol[-1] / avg) if avg > 0 else 1.0

    signals: list[str] = []
    reasons: list[str] = []
    breakdown_grade = ""

    # ── Confirmed breakdown — below 52-week low OR 20-day support ─────────────
    lo52 = float(np.min(low[:-1]))
    sup20 = float(np.min(low[-21:-1]))
    level, tag = (lo52, "52-week low") if price < lo52 else (sup20, "support")
    clv = close_location_value(close[-1], high[-1], low[-1])
    ok, grade, note = grade_breakdown(price=price, level=level, atr=atr,
                                      vratio=vratio, day_change=chg,
                                      clv=clv, rsi=rsi)
    if ok:
        breakdown_grade = grade
        key = "BREAKDOWN_52W" if level == lo52 else "BREAKDOWN_SUP"
        signals.append(key)
        reasons.append(f"[{grade}] ₹{level:,.0f} {tag} toota — {note}")

    # ── Death cross — 50-DMA just crossed BELOW 200-DMA ──────────────────────
    if len(close) >= 201:
        s50, s200 = close[-50:].mean(), close[-200:].mean()
        s50p, s200p = close[-55:-5].mean(), close[-205:-5].mean()
        if s50 < s200 and s50p >= s200p:
            signals.append("DEATH_CROSS")
            reasons.append("50-day average just crossed BELOW 200-day (Stage 4)")

    # ── Distribution — volume up on DOWN days, price in bottom third ──────────
    if vol is not None and len(vol) >= 60 and len(close) >= 60:
        pumping = np.nanmean(vol[-10:]) > np.nanmean(vol[-60:-10]) * 1.10
        rets = np.diff(close[-16:])
        v15 = vol[-15:]
        dn_vol = float(np.nansum(v15[rets < 0])) if len(v15) == len(rets) else 0.0
        up_vol = float(np.nansum(v15[rets > 0])) if len(v15) == len(rets) else 1.0
        hi60, lo60 = float(np.max(high[-60:])), float(np.min(low[-60:]))
        in_bottom_third = price <= lo60 + (hi60 - lo60) * 0.34 if hi60 > lo60 else False
        if pumping and up_vol > 0 and dn_vol / up_vol >= 1.4 and in_bottom_third:
            signals.append("DISTRIBUTION")
            reasons.append(f"Selling volume {dn_vol/up_vol:.1f}× buying — "
                           f"distribution, price lows ke paas")

    # ── Downside momentum — sharp 5-day drop with RSI in bearish zone ────────
    if mom5 <= -6 and rsi < 45 and price < sma50:
        signals.append("MOMENTUM_DOWN")
        reasons.append(f"Down {mom5:.1f}% in 5 days, RSI {rsi:.0f} — momentum neeche")

    # ── Lower highs — supply stacking (a controlled downtrend) ───────────────
    if len(high) >= 40:
        seg = high[-40:]
        first_half_hi = float(np.max(seg[:20]))
        second_half_hi = float(np.max(seg[20:]))
        if second_half_hi < first_half_hi * 0.98 and price < sma50:
            signals.append("LOWER_HIGHS")
            reasons.append("Har rally pichhli se neeche ruk rahi — supply stacking")

    if not signals:
        return None
    signals = signals[:5]
    reasons = reasons[:5]

    # ── Score (mirror of the long composite; no live/backtest calib yet) ─────
    base = sum(SHORT_META[s][1] for s in signals)
    below200 = len(close) >= 200 and price < close[-200:].mean()
    score = min(100.0, base + (10 if below200 else 0) + max(0.0, -mom5) * 0.8)

    verdict = "SHORT" if (breakdown_grade or (score >= 55 and len(signals) >= 2)) \
        else "AVOID"

    # ── Trade plan — MIRROR geometry: stop ABOVE, target BELOW ───────────────
    entry = level if price < level else price
    stop = round(entry + 2 * atr, 1) if atr > 0 else round(entry * 1.05, 1)
    target = round(entry - 4 * atr, 1) if atr > 0 else round(entry * 0.90, 1)

    # ── Mirror guards (demote-only) ──────────────────────────────────────────
    # Rising-rocket: a short is a WEAKNESS entry — never short a green/rallying
    # day (mirror of the long-side falling-knife guard).
    rsi_prev = _rsi(close[:-3]) if len(close) > 17 else rsi
    rising_today = chg >= _RISING_DAY_PCT
    rsi_rallying = rsi > rsi_prev + _RSI_RALLY_RISE
    if (rising_today or rsi_rallying) and verdict == "SHORT":
        tag2 = (f"aaj {chg:+.1f}%" if rising_today
                else f"RSI rally {rsi_prev:.0f}→{rsi:.0f}")
        reasons.insert(0, f"⚠ Upar bhaag raha ({tag2}) — strength mein short mat "
                          f"karo, bounce/reversal risk")
        verdict = "AVOID"
    # Too-extended-below-50-DMA: stretched for a mean-reversion bounce.
    ext_below = (sma50 - price) / sma50 * 100 if sma50 else 0.0
    if _EXT_BELOW_SMA50 > 0 and ext_below > _EXT_BELOW_SMA50 and verdict == "SHORT":
        reasons.insert(0, f"⚠ 50-DMA se {ext_below:.0f}% neeche — oversold, "
                          f"bounce due, naya short late")
        verdict = "AVOID"

    return ShortSignal(
        symbol=symbol, price=round(price, 2), change_pct=round(chg, 2),
        rsi=round(rsi, 1), volume_ratio=round(vratio, 2), signals=signals,
        reasons=reasons, score=round(score, 1), entry=round(entry, 1),
        stop=stop, target=target, verdict=verdict,
        breakdown_grade=breakdown_grade)
