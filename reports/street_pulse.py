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

Page-open path uses data already on disk: last two bhavcopy CSVs, the
cached index store, the product scan file, and the news curator sqlite.
It does not walk every in-memory OHLCV frame, fetch an option chain,
crawl RSS, or scrape Google Finance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from logger import get_logger

log = get_logger(__name__)

_PULSE_CACHE: dict[str, Any] = {"ts": 0.0, "pulse": None}
_PULSE_TTL_S = 60.0


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _ist_today_iso() -> str:
    try:
        from core.market_clock import today_ist
        return today_ist().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _ist_today_label() -> str:
    try:
        from core.market_clock import now_ist
        return now_ist().strftime("%d %B %Y")
    except Exception:
        return datetime.now(timezone.utc).strftime("%d %B %Y")


def _ts_is_ist_today(ts: Any) -> bool:
    if ts is None or ts == "":
        return False
    try:
        from core.market_clock import is_ist_today
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return is_ist_today(dt.isoformat())
        return is_ist_today(ts)
    except Exception:
        return False


def _market_snapshot() -> dict:
    """NIFTY/BANKNIFTY from the official index cache + cached regime only."""
    out: dict = {"indices": [], "commentary": ""}
    try:
        from data.index_store import latest_index_print
        for name, ticker in (("NIFTY 50", "^NSEI"), ("BANK NIFTY", "^NSEBANK")):
            q = latest_index_print(ticker) or {}
            if q.get("price"):
                out["indices"].append({
                    "name": name,
                    "price": q["price"],
                    "chg_pct": q.get("chg_pct") or 0.0,
                })
    except Exception:
        pass
    try:
        from core import regime_engine as re
        regime = (re._CACHE or {}).get("regime_state")
        if regime is not None:
            rmap = {
                "TRENDING_BULL": "Market trending up — breakout entries favoured.",
                "CHOPPY": "Choppy market — chhote positions, quick profits.",
                "TRENDING_BEAR": "Market weak — cash zyada, risk kam.",
            }
            out["commentary"] = rmap.get(getattr(regime, "market_regime", ""), "")
    except Exception:
        pass
    return out


def _scan_rows_latest() -> tuple[list[dict], int]:
    """Today's scan only. Prior-day auto_scan / stale files stay empty."""
    try:
        from scan.auto_scan import get_results
        results, universe, last_ts, _status = get_results()
        if results and _ts_is_ist_today(last_ts):
            return list(results), int(universe or 0)
    except Exception:
        pass
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        scanned_at = payload.get("scanned_at") or ""
        same_day = bool(payload.get("same_ist_day")) or _ts_is_ist_today(scanned_at)
        if not same_day:
            return [], 0
        rows = [r for r in (payload.get("records") or []) if isinstance(r, dict)]
        if rows:
            return rows, int(payload.get("universe_size") or len(rows))
    except Exception:
        pass
    return [], 0


def _movers_from_scan(results: list[dict], top_n: int = 5) -> tuple[list[dict], list[dict]]:
    rows = []
    for r in results:
        price = _f(r.get("price"))
        if price <= 0:
            continue
        chg = r.get("change_pct")
        if chg is None:
            continue
        rows.append({
            "symbol": str(r.get("symbol") or "").upper(),
            "price": round(price, 1),
            "chg_pct": round(_f(chg), 2),
        })
    if not rows:
        return [], []
    rows.sort(key=lambda item: item["chg_pct"], reverse=True)
    return rows[:top_n], list(reversed(rows[-top_n:]))


def _movers_from_session(top_n: int = 5) -> tuple[list[dict], list[dict]]:
    """1-day liquid movers from the last two official EQ bhavcopy CSVs."""
    try:
        from data.bhavcopy_store import latest_two_eq_sessions
        today_df, prev_df, _day = latest_two_eq_sessions()
        if today_df is None or prev_df is None or today_df.empty or prev_df.empty:
            return [], []
        today = today_df.set_index("symbol")
        prev = prev_df.set_index("symbol")
        common = today.index.intersection(prev.index)
        if len(common) == 0:
            return [], []
        close = today.loc[common, "close"].astype(float)
        prev_close = prev.loc[common, "close"].astype(float)
        volume = today.loc[common, "volume"].astype(float)
        valid = (close > 0) & (prev_close > 0)
        turnover = volume * close
        liquid = valid & (turnover >= 5e7)
        chg = ((close / prev_close) - 1.0) * 100.0
        # Cash-market daily circuit is 20%. Larger gaps are almost always
        # corporate-action / listing artifacts from pairing two CSVs, not a
        # session loser the desk should highlight.
        sane = chg.abs() <= 21.0
        ranked = [
            {"symbol": str(sym), "price": round(float(close.loc[sym]), 1),
             "chg_pct": round(float(chg.loc[sym]), 2)}
            for sym in chg[liquid & sane].sort_values(ascending=False).index
        ]
        if not ranked:
            return [], []
        return ranked[:top_n], list(reversed(ranked[-top_n:]))
    except Exception as exc:
        log.debug("pulse_session_movers_failed", error=str(exc))
        return [], []


def _movers_from_bhav(top_n: int = 5) -> tuple[list[dict], list[dict]]:
    """Session movers. Name kept for callers/tests; does not walk get_ohlcv."""
    return _movers_from_session(top_n)


def _losing_from_scan(results: list[dict]) -> dict | None:
    worst, worst_score = None, 0.0
    for r in results:
        if r.get("above_sma50") is True:
            continue
        mom = _f(r.get("momentum_5d"))
        if mom > -4:
            continue
        fall = r.get("pct_below_20d_high")
        score = _f(fall) if fall is not None else abs(mom)
        if score <= worst_score:
            continue
        worst_score = score
        price = _f(r.get("price"))
        if fall is not None:
            note = (
                f"{_f(fall):.0f}% gira apne 20-day high se, "
                f"50-day average ke neeche — abhi door raho"
            )
        else:
            note = f"5-day momentum {mom:.1f}% — abhi door raho"
        worst = {
            "symbol": str(r.get("symbol") or "").upper(),
            "price": round(price, 1),
            "chg_5d": round(mom, 1),
            "note": note,
        }
    return worst


def _losing_momentum() -> dict | None:
    """Weak name from today's scan, else the worst liquid session loser."""
    results, _universe = _scan_rows_latest()
    weak = _losing_from_scan(results)
    if weak:
        return weak
    _gainers, losers = _movers_from_session(top_n=5)
    if not losers:
        return None
    row = losers[0]
    return {
        "symbol": row["symbol"],
        "price": row.get("price") or 0,
        "chg_5d": row.get("chg_pct") or 0,
        "note": (
            f"Session {row.get('chg_pct', 0):+.1f}% among liquid names — "
            f"abhi door raho"
        ),
    }


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
                headline = ""
                if hasattr(a, "as_dict"):
                    headline = str((a.as_dict() or {}).get("headline") or "")
                if not headline:
                    headline = str(getattr(a, "headline", "") or "")
                if headline:
                    out.append(headline[:120])
            return out[:max_n]
        finally:
            store.close()
    except Exception:
        return []


def build_pulse(*, force: bool = False) -> dict:
    """Assemble Pulse from local files. No full-universe walk, no live crawl."""
    import time
    now = time.time()
    if not force and _PULSE_CACHE["pulse"] and now - float(_PULSE_CACHE["ts"] or 0) < _PULSE_TTL_S:
        return dict(_PULSE_CACHE["pulse"])

    results, universe_size = _scan_rows_latest()
    gainers, losers = _movers_from_scan(results)
    if not gainers and not losers:
        gainers, losers = _movers_from_bhav()
    snapshot = _market_snapshot()

    buzzing = None
    movers = [
        r for r in results
        if abs(_f(r.get("change_pct") if r.get("change_pct") is not None else r.get("momentum_5d"))) >= 3
        and _f(r.get("volume_ratio"), 1) >= 2
    ]
    if movers:
        b = max(
            movers,
            key=lambda r: _f(r.get("volume_ratio")) * abs(
                _f(r.get("change_pct") if r.get("change_pct") is not None else r.get("momentum_5d"))
            ),
        )
        move_key = "change_pct" if b.get("change_pct") is not None else "momentum_5d"
        move = _f(b.get(move_key))
        horizon = "move" if move_key == "change_pct" else "over 5 days"
        buzzing = {
            **b,
            "note": (
                f"{move:+.1f}% {horizon} on {_f(b.get('volume_ratio')):.1f}× volume — "
                + (b.get("reasons") or ["strong interest"])[0]
            ),
        }

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
        takeaways.append(
            f"{idx['name']} {arrow} {idx['chg_pct']:+.2f}% "
            f"at {idx['price']:,.0f}"
        )
    if gainers:
        takeaways.append(
            f"{gainers[0]['symbol']} top gainer "
            f"({gainers[0]['chg_pct']:+.1f}%)"
        )
    if snapshot["commentary"]:
        takeaways.append(snapshot["commentary"])

    pulse = {
        "date": _ist_today_label(),
        "as_of_ist": _ist_today_iso(),
        "takeaways": takeaways[:4],
        "snapshot": snapshot,
        "gainers": gainers,
        "losers": losers,
        "buzzing": buzzing,
        "strength": strength,
        "weak": _losing_from_scan(results) or (
            {
                "symbol": losers[0]["symbol"],
                "price": losers[0].get("price") or 0,
                "chg_5d": losers[0].get("chg_pct") or 0,
                "note": (
                    f"Session {losers[0].get('chg_pct', 0):+.1f}% among liquid names — "
                    f"abhi door raho"
                ),
            } if losers else None
        ),
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
        lines.append(
            f"💪 <b>Gaining strength:</b> {s['symbol']} — "
            f"pivot ₹{s['entry']:,.0f} se {s.get('pivot_distance_pct', 0):.1f}% door"
        )
    if pulse.get("weak"):
        w = pulse["weak"]
        lines.append(f"⚠️ <b>Losing momentum:</b> {w['symbol']} — {w['note']}")
    if pulse.get("breakouts_tomorrow"):
        syms = ", ".join(r["symbol"] for r in pulse["breakouts_tomorrow"])
        lines.append(f"\n⏳ <b>Kal ke breakout candidates:</b> {syms}")
    return "\n".join(lines)
