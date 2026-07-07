"""
Daily Street Pulse — auto-generated daily market report.

Modelled on analyst-style daily newsletters:
  1. Cover takeaways        — 3 one-liners about today's market
  2. Market snapshot        — NIFTY / BANKNIFTY with plain commentary
  3. Top gainers / losers   — from the latest NSE bhavcopy session
  4. Buzzing stock          — biggest move on the biggest volume
  5. Gaining strength       — best accumulation / pre-breakout candidate
  6. Losing momentum        — liquid stock breaking down (warning)
  7. Breakouts today        — fresh confirmed breakouts
  8. Breakouts tomorrow     — closest-to-pivot pre-breakout watchlist
  9. Top headlines          — latest market news

Everything is computed from data the system already has: bhavcopy
store, unified scanner results, Google Finance quotes, news fetcher.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from logger import get_logger

log = get_logger(__name__)


def _market_snapshot() -> dict:
    """NIFTY/BANKNIFTY live + regime commentary."""
    out: dict = {"indices": [], "commentary": ""}
    try:
        from data.live_quotes import get_index_quotes
        q = get_index_quotes(["NIFTY", "BANKNIFTY"])
        for name, key in (("NIFTY 50", "NIFTY"), ("BANK NIFTY", "BANKNIFTY")):
            if q.get(key, {}).get("price"):
                out["indices"].append({"name": name, "price": q[key]["price"],
                                       "chg_pct": q[key]["chg_pct"]})
    except Exception:
        pass
    try:
        from core.regime_engine import compute_regime
        regime = compute_regime()
        rmap = {
            "TRENDING_BULL": "Market trending up — breakout entries favoured.",
            "CHOPPY": "Choppy market — chhote positions, quick profits.",
            "TRENDING_BEAR": "Market weak — cash zyada, risk kam.",
        }
        out["commentary"] = rmap.get(
            getattr(regime, "market_regime", ""), "")
    except Exception:
        pass
    # Options market read — kya keh raha hai derivatives ka paisa
    try:
        from options.analytics import nifty_options_summary
        opt = nifty_options_summary()
        if opt:
            out["options"] = opt
    except Exception:
        pass
    return out


def _movers_from_bhav(top_n: int = 5) -> tuple[list[dict], list[dict]]:
    """Top gainers/losers from the latest bhavcopy session (liquid stocks only)."""
    try:
        from data.bhavcopy_store import store_symbols, get_ohlcv
        rows = []
        for sym in store_symbols():
            df = get_ohlcv(sym)
            if df is None or len(df) < 21:
                continue
            close = df["close"].values
            vol = df["volume"].values
            price = float(close[-1])
            turnover = float(np.nanmean(vol[-20:])) * price
            if turnover < 5e7:          # ₹5 cr — keep it to liquid names
                continue
            chg = (close[-1] / close[-2] - 1) * 100
            rows.append({"symbol": sym, "price": round(price, 1),
                         "chg_pct": round(chg, 2)})
        rows.sort(key=lambda r: r["chg_pct"], reverse=True)
        return rows[:top_n], rows[-top_n:][::-1]
    except Exception as exc:
        log.debug("pulse_movers_failed", error=str(exc))
        return [], []


def _losing_momentum() -> dict | None:
    """A liquid stock breaking down — fell hard and closed below its 50-day avg."""
    try:
        from data.bhavcopy_store import store_symbols, get_ohlcv
        worst, worst_score = None, 0.0
        for sym in store_symbols():
            df = get_ohlcv(sym)
            if df is None or len(df) < 60:
                continue
            close = df["close"].values
            vol = df["volume"].values
            price = float(close[-1])
            if float(np.nanmean(vol[-20:])) * price < 2e8:   # big names only
                continue
            sma50 = float(close[-50:].mean())
            chg5 = (close[-1] / close[-6] - 1) * 100
            hi20 = float(np.max(close[-20:]))
            fall_from_high = (hi20 - price) / hi20 * 100
            if price < sma50 and chg5 < -4 and fall_from_high > worst_score:
                worst_score = fall_from_high
                worst = {"symbol": sym, "price": round(price, 1),
                         "chg_5d": round(chg5, 1),
                         "note": (f"{fall_from_high:.0f}% gira apne 20-day high se, "
                                  f"50-day average ke neeche band hua — abhi door raho")}
        return worst
    except Exception:
        return None


def _headlines(max_n: int = 5) -> list[str]:
    try:
        from news.fetcher import NewsFetcher
        arts = NewsFetcher().fetch_all(max_age_hours=18)
        return [a.headline[:120] for a in arts[:max_n]]
    except Exception:
        return []


def build_pulse() -> dict:
    """Assemble the full Daily Pulse from live system state. Call once per view."""
    from scan.auto_scan import get_results

    results, universe_size, last_ts, _ = get_results()
    gainers, losers = _movers_from_bhav()
    snapshot = _market_snapshot()

    # Buzzing stock: biggest move on biggest relative volume among signals
    buzzing = None
    movers = [r for r in results if abs(r.get("change_pct", 0)) >= 3
              and r.get("volume_ratio", 1) >= 2]
    if movers:
        b = max(movers, key=lambda r: r["volume_ratio"] * abs(r["change_pct"]))
        buzzing = {**b, "note": (f"{b['change_pct']:+.1f}% move on "
                                 f"{b['volume_ratio']:.1f}× volume — "
                                 + (b.get("reasons") or ["strong interest"])[0])}

    # Gaining strength: best pre-breakout / accumulation candidate
    strength = None
    pre = [r for r in results if "PreBreakout" in r.get("categories", [])]
    if pre:
        strength = min(pre, key=lambda r: r.get("pivot_distance_pct") or 99)

    # Breakouts today (confirmed) & tomorrow (pre-breakout watch)
    today_brk = [r for r in results
                 if any(s in ("52-week high breakout", "Resistance break on volume")
                        for s in r.get("signals", []))][:4]
    tomorrow_brk = sorted(pre, key=lambda r: r.get("pivot_distance_pct") or 99)[:4]

    # Cover takeaways
    takeaways = []
    for idx in snapshot["indices"]:
        arrow = "▲" if idx["chg_pct"] >= 0 else "▼"
        takeaways.append(f"{idx['name']} {arrow} {idx['chg_pct']:+.2f}% "
                         f"at {idx['price']:,.0f}")
    if gainers:
        takeaways.append(f"{gainers[0]['symbol']} top gainer "
                         f"({gainers[0]['chg_pct']:+.1f}%)")
    if snapshot["commentary"]:
        takeaways.append(snapshot["commentary"])
    if snapshot.get("options"):
        opt = snapshot["options"]
        takeaways.append(f"Options: {opt['note']} · max pain {opt['max_pain']:,.0f}")

    return {
        "date": datetime.now().strftime("%d %B %Y"),
        "takeaways": takeaways[:4],
        "snapshot": snapshot,
        "gainers": gainers,
        "losers": losers,
        "buzzing": buzzing,
        "strength": strength,
        "weak": _losing_momentum(),
        "breakouts_today": today_brk,
        "breakouts_tomorrow": tomorrow_brk,
        "headlines": _headlines(),
        "scanned": universe_size,
    }


def pulse_to_telegram(pulse: dict) -> str:
    """Compact HTML version of the pulse for a Telegram morning message."""
    lines = [f"📰 <b>Daily Street Pulse — {pulse['date']}</b>"]
    for t in pulse["takeaways"]:
        lines.append(f"• {t}")
    if pulse.get("buzzing"):
        b = pulse["buzzing"]
        lines.append(f"\n🔥 <b>Buzzing:</b> {b['symbol']} — {b['note']}")
    if pulse.get("strength"):
        s = pulse["strength"]
        lines.append(f"💪 <b>Gaining strength:</b> {s['symbol']} — "
                     f"pivot ₹{s['entry']:,.0f} se {s.get('pivot_distance_pct', 0):.1f}% door")
    if pulse.get("weak"):
        w = pulse["weak"]
        lines.append(f"⚠️ <b>Losing momentum:</b> {w['symbol']} — {w['note']}")
    if pulse.get("breakouts_tomorrow"):
        syms = ", ".join(r["symbol"] for r in pulse["breakouts_tomorrow"])
        lines.append(f"\n⏳ <b>Kal ke breakout candidates:</b> {syms}")
    return "\n".join(lines)
