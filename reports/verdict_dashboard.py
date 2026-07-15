"""
Verdict Dashboard — "agar tum har signal follow karte toh?"

Replays every CLOSED tracked signal (signal_outcome_tracker) in
chronological order as if each was taken with 1%% of running equity
at risk. Output: equity curve + the report card that decides whether
this system deserves real money:

  - final equity vs start, total return %%
  - win rate on closed signals
  - expectancy (avg R + ₹/trade), profit factor, payoff (win:loss R)
  - max drawdown + MAR (return ÷ drawdown), max losing streak
  - recent-form vs lifetime → is the edge improving or decaying?
  - honest verdict line (sample-size aware)

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

    n = len(usable)
    wins = sum(1 for s in usable if s["worked"] == 1)
    rs = [s["r"] for s in usable]
    avg_r = sum(rs) / n
    total_ret = (equity / start_capital - 1) * 100

    # ── Trader-grade edge metrics (the numbers that decide "real money?") ─────
    win_rs = [r for r in rs if r > 0]
    loss_rs = [r for r in rs if r < 0]
    gross_win = sum(win_rs)
    gross_loss = abs(sum(loss_rs))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
        99.0 if gross_win > 0 else 0.0)            # ∞-safe (all-wins → capped)
    avg_win_r = (gross_win / len(win_rs)) if win_rs else 0.0
    avg_loss_r = (gross_loss / len(loss_rs)) if loss_rs else 0.0
    payoff = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else 0.0
    # per-trade expected ₹ at the 1% rule (tangible: "average trade = ₹X")
    expectancy_rupees = avg_r * risk_pct * start_capital
    # longest run of consecutive losers — the streak that tests your nerve
    max_loss_streak = cur = 0
    for s in usable:
        cur = cur + 1 if s["worked"] == 0 else 0
        max_loss_streak = max(max_loss_streak, cur)
    # risk-adjusted: return earned per unit of worst pain (MAR)
    mar = (total_ret / max_dd) if max_dd > 0 else 0.0
    # recent form — is the edge STRENGTHENING or ROTTING? (last block vs life)
    rec_n = min(50, n)
    recent_r = sum(rs[-rec_n:]) / rec_n
    if recent_r > avg_r + 0.05:
        edge_trend = "improving"
    elif recent_r < avg_r - 0.05:
        edge_trend = "decaying"
    else:
        edge_trend = "stable"

    stats = {
        "closed": n,
        "wins": wins,
        "win_rate": round(wins / n * 100, 1),
        "avg_r": round(avg_r, 2),
        "expectancy_r": round(avg_r, 2),
        "expectancy_rupees": round(expectancy_rupees, 0),
        "best_r": round(max(rs), 2),
        "worst_r": round(min(rs), 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "payoff_ratio": round(payoff, 2),
        "max_loss_streak": max_loss_streak,
        "mar_ratio": round(mar, 2),
        "recent_avg_r": round(recent_r, 2),
        "recent_n": rec_n,
        "edge_trend": edge_trend,
        "final_equity": round(equity, 0),
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }

    # Sample-size honesty: <30 = still forming, no strong claim either way.
    thin = n < 30
    trend_bit = {
        "improving": f" Aur achhi baat — recent {rec_n} signals ka edge "
                     f"({recent_r:+.2f}R) lifetime se behtar hai, momentum upar.",
        "decaying":  f" Dhyaan do — recent {rec_n} signals ka edge "
                     f"({recent_r:+.2f}R) lifetime se neeche hai, edge thanda ho "
                     f"raha; size mat badhao.",
        "stable":    "",
    }[edge_trend]

    if avg_r >= 0.15 and profit_factor >= 1.3 and max_dd < 12 and not thin:
        verdict = (f"🟢 Edge REAL hai: {n} signals pe expectancy {avg_r:+.2f}R "
                   f"(₹{expectancy_rupees:,.0f}/trade), profit factor "
                   f"{profit_factor:.2f}, drawdown {max_dd:.1f}%. Isi discipline "
                   f"se real size soch-samajh ke badha sakte ho.{trend_bit}")
    elif avg_r > 0:
        cap_note = (" Sample abhi chhota hai (<30) — koi bhi dawa jaldbaazi."
                    if thin else "")
        verdict = (f"🟡 Halki positive edge ({avg_r:+.2f}R avg, PF "
                   f"{profit_factor:.2f}) lekin drawdown {max_dd:.1f}% aur payoff "
                   f"{payoff:.2f}× — abhi chhote size pe raho, data banne do."
                   f"{cap_note}{trend_bit}")
    else:
        verdict = (f"🔴 Ab tak edge negative hai ({avg_r:+.2f}R avg, PF "
                   f"{profit_factor:.2f}). Real paisa NAHI — filters ko aur data "
                   f"chahiye. Yeh imaandaar jawab hai, marketing nahi.{trend_bit}")
    return {"points": points, "stats": stats, "verdict": verdict}
