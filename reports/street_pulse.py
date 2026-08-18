"""
Daily Street Pulse — auto-generated daily market report.

Modelled on analyst-style daily newsletters:
  1. Cover takeaways        — 3 one-liners about today's market
  2. Market snapshot        — NIFTY / BANKNIFTY with plain commentary
  3. Top gainers / losers   — from the latest market scan
  4. Buzzing stock          — biggest move on the biggest volume
  5. Gaining strength       — best accumulation / pre-breakout candidate
  6. Losing momentum        — liquid stock breaking down (warning)
  7. Breakouts today        — fresh confirmed breakouts
  8. Breakouts tomorrow     — closest-to-pivot pre-breakout watchlist
  9. Top headlines          — latest market news

Everything is computed from data the system already has: last scan file,
index quotes, regime, and the news curator store.
"""
from __future__ import annotations

from datetime import datetime

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
    # Option-chain fetch is a network page — skip it on Pulse open.
    return out


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _scan_rows() -> tuple[list[dict], int]:
    """Prefer the product scan file (autonomy). Memory auto_scan is Streamlit-only."""
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        rows = [r for r in (payload.get("records") or []) if isinstance(r, dict)]
        if rows:
            return rows, int(payload.get("universe_size") or len(rows))
    except Exception:
        pass
    try:
        from scan.auto_scan import get_results
        results, universe, *_ = get_results()
        return list(results or []), int(universe or 0)
    except Exception:
        return [], 0


def _movers_from_scan(results: list[dict], top_n: int = 5) -> tuple[list[dict], list[dict]]:
    rows = []
    for r in results:
        price = _f(r.get("price"))
        if price <= 0:
            continue
        rows.append({
            "symbol": str(r.get("symbol") or "").upper(),
            "price": round(price, 1),
            "chg_pct": round(_f(r.get("change_pct")), 2),
        })
    rows.sort(key=lambda item: item["chg_pct"], reverse=True)
    if not rows:
        return [], []
    return rows[:top_n], list(reversed(rows[-top_n:]))


def _losing_from_scan(results: list[dict]) -> dict | None:
    worst, worst_score = None, 0.0
    for r in results:
        if r.get("above_sma50") is True:
            continue
        mom = _f(r.get("momentum_5d"))
        if mom > -4:
            continue
        fall = _f(r.get("pct_below_20d_high"))
        if fall <= worst_score:
            continue
        worst_score = fall
        price = _f(r.get("price"))
        worst = {
            "symbol": str(r.get("symbol") or "").upper(),
            "price": round(price, 1),
            "chg_5d": round(mom, 1),
            "note": (
                f"{fall:.0f}% gira apne 20-day high se, "
                f"50-day average ke neeche — abhi door raho"
            ),
        }
    return worst


def _is_confirmed_breakout(row: dict) -> bool:
    sigs = [str(s) for s in (row.get("signals") or [])]
    joined = " ".join(sigs).upper()
    return (
        "52-WEEK HIGH BREAKOUT" in joined
        or "RESISTANCE BREAK" in joined
        or "BREAKOUT_52W" in joined
        or "BREAKOUT_RES" in joined
        or str(row.get("breakout_grade") or "").upper() in {"A", "B"}
    )


def _headlines(max_n: int = 5) -> list[str]:
    """Headlines from the curator store — never a live news crawl on page open."""
    try:
        from pathlib import Path
        from news.curator_store import NewsCuratorStore
        root = Path(__file__).resolve().parents[1]
        store = NewsCuratorStore(root / "logs" / "news_curator.sqlite3")
        try:
            arts = store.recent(hours=18, limit=max_n)
            out = []
            for a in arts:
                headline = str(getattr(a, "headline", "") or (a.as_dict().get("headline") if hasattr(a, "as_dict") else "") or "")
                if headline:
                    out.append(headline[:120])
            return out[:max_n]
        finally:
            store.close()
    except Exception:
        return []


_PULSE_CACHE: dict = {"ts": 0.0, "pulse": None}
_PULSE_TTL_S = 60.0


def build_pulse(*, force: bool = False) -> dict:
    """Assemble Pulse from the last scan + index quotes. No full-universe walk."""
    import time
    now = time.time()
    if not force and _PULSE_CACHE["pulse"] and now - float(_PULSE_CACHE["ts"] or 0) < _PULSE_TTL_S:
        return dict(_PULSE_CACHE["pulse"])

    results, universe_size = _scan_rows()
    gainers, losers = _movers_from_scan(results)
    snapshot = _market_snapshot()

    buzzing = None
    movers = [r for r in results if abs(_f(r.get("change_pct"))) >= 3
              and _f(r.get("volume_ratio"), 1) >= 2]
    if movers:
        b = max(movers, key=lambda r: _f(r.get("volume_ratio")) * abs(_f(r.get("change_pct"))))
        buzzing = {**b, "note": (f"{_f(b.get('change_pct')):+.1f}% move on "
                                 f"{_f(b.get('volume_ratio')):.1f}× volume — "
                                 + (b.get("reasons") or ["strong interest"])[0])}

    strength = None
    pre = [
        r for r in results
        if "PreBreakout" in (r.get("categories") or [])
        or "PRE_BREAKOUT" in [str(x).upper() for x in (r.get("signals") or [])]
    ]
    if pre:
        strength = min(pre, key=lambda r: _f(r.get("pivot_distance_pct"), 99))

    today_brk = [r for r in results if _is_confirmed_breakout(r)][:4]
    tomorrow_brk = sorted(pre, key=lambda r: _f(r.get("pivot_distance_pct"), 99))[:4]

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

    pulse = {
        "date": datetime.now().strftime("%d %B %Y"),
        "takeaways": takeaways[:4],
        "snapshot": snapshot,
        "gainers": gainers,
        "losers": losers,
        "buzzing": buzzing,
        "strength": strength,
        "weak": _losing_from_scan(results),
        "breakouts_today": today_brk,
        "breakouts_tomorrow": tomorrow_brk,
        "headlines": _headlines(),
        "scanned": universe_size,
    }
    _PULSE_CACHE["ts"] = now
    _PULSE_CACHE["pulse"] = pulse
    return pulse


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
