"""
Trade Coach — system market ko dekhta hai, coach TUMHE dekhta hai.

Weekly behavioral review from the trade journal + signal outcomes.
Rule-based analysis first (always works, no API needed):

  - Overtrading: trades per week vs a sane cadence
  - Risk consistency: does actual risk per trade wander from the 1% plan
  - Stop hygiene: stops too tight (<2%% = noise pe exit) or too wide
  - Revenge trading: same symbol re-traded within 2 days of a loss
  - Paper record: win rate + expectancy from closed paper trades

DeepSeek (if configured) turns the stats into a personal 5-bullet
coaching note; otherwise the rule-based bullets ship as-is.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from logger import get_logger

log = get_logger(__name__)


def _trades() -> list[dict]:
    try:
        from execution.trade_executor import _DB
        if not _DB.exists():
            return []
        conn = sqlite3.connect(_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades ORDER BY placed_at ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def gather_stats(days: int = 28) -> dict:
    """Behavioral stats over the last `days` — the coach's raw material."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = [t for t in _trades() if str(t["placed_at"]) >= cutoff]
    out: dict = {"window_days": days, "n_trades": len(rows), "insights": []}
    if not rows:
        out["insights"].append("Is period mein koi trade nahi — journal khali hai. "
                               "Paper trades se shuru karo, data banega.")
        return out

    # Cadence
    per_week = len(rows) / max(1, days / 7)
    out["trades_per_week"] = round(per_week, 1)
    if per_week > 8:
        out["insights"].append(
            f"⚠️ Overtrading: {per_week:.0f} trades/hafta. Best setups kam hote "
            f"hain — itne trades matlab average setups bhi le rahe ho.")

    # Risk consistency (risk per trade as % of entry value proxy)
    risks = []
    for t in rows:
        e, s, q = float(t["entry_price"] or 0), float(t["stop_price"] or 0), int(t["qty"] or 0)
        if e > s > 0 and q > 0:
            risks.append(q * (e - s))
    if len(risks) >= 3:
        mx, mn = max(risks), min(risks)
        if mn > 0 and mx / mn > 3:
            out["insights"].append(
                f"⚠️ Risk inconsistent: sabse bada trade risk (₹{mx:,.0f}) sabse "
                f"chhote (₹{mn:,.0f}) ka {mx/mn:.0f}x hai. 1% rule ka matlab hai "
                f"har trade ka risk EK jaisa — warna ek galat bada trade sab "
                f"chhote sahi trades kha jayega.")

    # Stop hygiene
    tight = [t for t in rows
             if float(t["entry_price"] or 0) > 0
             and 0 < (float(t["entry_price"]) - float(t["stop_price"] or 0))
                     / float(t["entry_price"]) < 0.02]
    if len(tight) >= max(2, len(rows) // 3):
        out["insights"].append(
            f"⚠️ {len(tight)} trades mein stop entry se 2% ke andar tha — itna "
            f"tight stop noise pe hi hit ho jata hai. ATR-based stop (card wala) "
            f"use karo.")

    # Revenge trading: same symbol within 2 days of a loss
    revenge = 0
    loss_times: dict[str, datetime] = {}
    for t in rows:
        ts = datetime.fromisoformat(str(t["placed_at"]))
        sym = t["symbol"]
        if sym in loss_times and (ts - loss_times[sym]).days <= 2:
            revenge += 1
        if "LOSS" in str(t["status"] or ""):
            loss_times[sym] = ts
    if revenge:
        out["insights"].append(
            f"⚠️ Revenge trading ke {revenge} sign: loss ke 2 din ke andar wahi "
            f"stock dobara. Market ko personal mat lo — agla trade NAYA setup "
            f"hona chahiye, badla nahi.")

    # Paper record
    closed = [t for t in rows if str(t["status"]) in ("PAPER_WIN", "PAPER_LOSS")]
    if closed:
        wins = sum(1 for t in closed if t["status"] == "PAPER_WIN")
        out["paper_closed"] = len(closed)
        out["paper_win_rate"] = round(wins / len(closed) * 100, 1)
        out["insights"].append(
            f"📊 Paper record: {len(closed)} closed, {out['paper_win_rate']:.0f}% "
            f"win rate. " + ("Solid — discipline yahi rehni chahiye real mein."
                             if out["paper_win_rate"] >= 45 else
                             "Real paise se pehle isse sudharna hai."))

    if not out["insights"]:
        out["insights"].append(
            "✅ Koi bura pattern nahi mila — cadence sahi, risk consistent, "
            "stops sane. Yahi discipline hai jo account zinda rakhti hai.")
    return out


def build_coach_review(days: int = 28) -> str:
    """Weekly coaching note — DeepSeek-polished if available, else rule-based."""
    stats = gather_stats(days)
    bullets = "\n".join(f"• {i}" for i in stats["insights"])
    header = (f"🧠 Coach ki Raay — pichhle {days} din, "
              f"{stats['n_trades']} trades\n\n")

    try:
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        try:
            from ai.llm_health import available
            _llm_ok = available()
        except Exception:
            _llm_ok = True
        if api_key and _llm_ok and stats["n_trades"] >= 3:
            from openai import OpenAI
            client = OpenAI(api_key=api_key,
                            base_url=os.getenv("DEEPSEEK_BASE_URL",
                                               "https://api.deepseek.com"))
            resp = client.chat.completions.create(
                model="deepseek-chat", max_tokens=400, temperature=0.4, timeout=15,
                messages=[{"role": "user", "content":
                    "You are a strict but supportive trading coach. Based on "
                    "these observations about the trader's last month, write "
                    "exactly 4 short coaching bullets in Hinglish (Hindi in "
                    "Latin script), direct tone, each ≤25 words. "
                    f"Observations:\n{bullets}"}])
            if resp and resp.choices:
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return header + text
    except Exception as exc:
        try:
            from ai.llm_health import note_failure
            note_failure(str(exc))
        except Exception:
            pass
        log.debug("coach_llm_skip", error=str(exc))
    return header + bullets
