"""
Verdict Dashboard — "agar tum har signal follow karte toh?"

Replays every CLOSED tracked signal (signal_outcome_tracker) in
chronological order as if each was taken with 1%% of running equity
at risk. Output: equity curve + the report card that decides whether
this system deserves real money:

  - final equity vs start, total return %%
  - win rate on closed signals
  - avg R per trade, best/worst
  - max drawdown of the curve
  - honest verdict line

R per signal = outcome_pct / risk_pct, where risk_pct is the signal's
own entry→stop distance. Signals without a stop are skipped (can't
size what you can't bound).
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)


def _closed_signals() -> list[dict]:
    try:
        from core.signal_outcome_tracker import _get_conn
        conn = _get_conn()
        rows = conn.execute(
            """SELECT symbol, logged_at, entry_price, stop_price, outcome_pct, worked
               FROM signal_log
               WHERE worked IS NOT NULL AND outcome_pct IS NOT NULL
                 AND entry_price > 0 AND stop_price > 0
               ORDER BY logged_at ASC"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("closed_signals_read_failed", error=str(exc))
        return []


def build_equity_curve(start_capital: float = 100_000.0,
                       risk_pct: float = 0.01) -> dict:
    """
    {points: [(date_str, equity)], stats: {...}} — empty points when
    fewer than 5 closed signals (not enough to claim anything).
    """
    signals = _closed_signals()
    usable = []
    for s in signals:
        entry, stop = float(s["entry_price"]), float(s["stop_price"])
        if entry <= stop:
            continue
        risk_frac = (entry - stop) / entry           # per-share risk %
        r_mult = (float(s["outcome_pct"]) / 100) / risk_frac
        r_mult = max(-1.5, min(4.0, r_mult))         # clip outliers/gaps
        usable.append({"date": str(s["logged_at"])[:10], "symbol": s["symbol"],
                       "r": r_mult, "worked": int(s["worked"])})

    if len(usable) < 5:
        return {"points": [], "stats": {"closed": len(usable)},
                "verdict": (f"Abhi sirf {len(usable)} closed signals hain — "
                            f"kam se kam 5 chahiye report card ke liye. "
                            f"System chalta rahe, data banta rahega.")}

    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    points = [(usable[0]["date"], round(equity, 0))]
    for s in usable:
        equity += equity * risk_pct * s["r"]         # 1% of RUNNING equity
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        points.append((s["date"], round(equity, 0)))

    wins = sum(1 for s in usable if s["worked"] == 1)
    rs = [s["r"] for s in usable]
    total_ret = (equity / start_capital - 1) * 100
    stats = {
        "closed": len(usable),
        "wins": wins,
        "win_rate": round(wins / len(usable) * 100, 1),
        "avg_r": round(sum(rs) / len(rs), 2),
        "best_r": round(max(rs), 2),
        "worst_r": round(min(rs), 2),
        "final_equity": round(equity, 0),
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }

    if stats["avg_r"] >= 0.15 and max_dd < 10:
        verdict = (f"🟢 System ka edge REAL hai: {stats['closed']} signals pe "
                   f"{total_ret:+.1f}% (max drawdown sirf {max_dd:.1f}%). "
                   f"Isi discipline se real size badha sakte ho.")
    elif stats["avg_r"] > 0:
        verdict = (f"🟡 Halki positive edge ({stats['avg_r']:+.2f}R avg) lekin "
                   f"drawdown {max_dd:.1f}% — abhi chhote size pe raho, aur "
                   f"data banne do.")
    else:
        verdict = (f"🔴 Ab tak edge negative hai ({stats['avg_r']:+.2f}R avg). "
                   f"Real paisa NAHI — system ke filters ko aur data chahiye. "
                   f"Yeh imaandaar jawab hai, marketing nahi.")
    return {"points": points, "stats": stats, "verdict": verdict}
