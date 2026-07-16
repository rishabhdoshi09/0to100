"""
🧠 QuantTerm Brain — the one read that composes every subsystem.

The system already has the parts: a whole-market scanner, evidence-calibrated
signals, a live edge that learns from real outcomes, a regime engine, a
portfolio-risk gatekeeper, an autopilot, and daemon health. This is the
conductor: each cycle it reads all of them and produces ONE authoritative
picture —

  • posture      — AGGRESSIVE / NORMAL / DEFENSIVE / STAND_ASIDE, derived from
                   regime × edge-trend × expectancy × book risk
  • verdict_line — one plain sentence: kya karna hai aaj
  • vitals       — the numbers that matter, in one place
  • directives   — a prioritised, plain-language to-do list (critical first)

The Brain is READ-ONLY. It never places a trade or changes a limit — the
autopilot and the ticket own execution, behind their own gates. The Brain
tells you (and the daemons) what the whole board says; it doesn't seize the
wheel. Every probe is wrapped, so a dead subsystem degrades the picture
instead of blanking it.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

# Regimes where fresh breakout longs have the worst odds (mirrors autopilot).
_BAD_REGIMES = {"DISTRIBUTION", "TRENDING_BEAR", "BEAR"}
_GOOD_REGIMES = {"TRENDING_BULL", "EXPANSION"}

_POSTURE_META = {
    "AGGRESSIVE":  ("#00d4a0", "🟢 GREEN LIGHT"),
    "NORMAL":      ("#38bdf8", "🔵 NORMAL"),
    "DEFENSIVE":   ("#f59e0b", "🟠 DEFENSIVE"),
    "STAND_ASIDE": ("#ff4b4b", "🔴 STAND ASIDE"),
}


def posture_meta(posture: str) -> tuple[str, str]:
    return _POSTURE_META.get(posture, ("#8892a4", posture))


# ── Pure decision core (fully testable, no I/O) ───────────────────────────────

def decide_posture(regime: str, edge_trend: str, expectancy_r: float,
                   book_verdict: str, closed: int) -> tuple[str, str]:
    """The whole board → one posture + one-line reason.

    Precedence is SURVIVAL-FIRST: an over-risked book stands us aside no matter
    how good the tape looks; a bad tape or rotting edge keeps us defensive; only
    when tape, edge and book all align do we lean in.
    """
    reg = (regime or "").upper()
    # 1) Book risk overrides everything — protect capital before opportunity.
    if book_verdict == "DANGER":
        return "STAND_ASIDE", ("Portfolio open-risk DANGER — capital pehle. "
                               "Naye trade nahi jab tak book halka na ho.")
    # 2) Actively hostile tape with a negative edge — don't fight it.
    if reg in _BAD_REGIMES and expectancy_r < 0:
        return "STAND_ASIDE", (f"Tape {reg} + edge negative "
                               f"({expectancy_r:+.2f}R) — aaj naye longs off.")
    # 3) Any single caution flag → defensive (smaller, only A+ setups).
    if (reg in _BAD_REGIMES or edge_trend == "decaying"
            or expectancy_r < 0 or book_verdict == "CAUTION"):
        why = ("tape kamzor" if reg in _BAD_REGIMES else
               "edge decaying" if edge_trend == "decaying" else
               "edge negative" if expectancy_r < 0 else "book caution")
        return "DEFENSIVE", (f"{why} — chhota size, sirf A+ setups, "
                             f"stops tight.")
    # 4) Everything aligned AND enough evidence → lean in.
    if (reg in _GOOD_REGIMES and edge_trend != "decaying"
            and expectancy_r >= 0.15 and book_verdict == "OK" and closed >= 30):
        return "AGGRESSIVE", ("Tape + edge + book teeno strong — normal-plus "
                              "size, winners ko chalne do.")
    # 5) Otherwise steady state.
    return "NORMAL", "Kaam theek — normal size, discipline banaye rakho."


def build_directives(v: dict) -> list[dict]:
    """Vitals → prioritised, plain-language to-do list. Each item is
    {severity, text}; sorted critical → good. Bounded and de-duplicated."""
    out: list[dict] = []

    def add(sev: str, text: str) -> None:
        out.append({"severity": sev, "text": text})

    # critical — survival + plumbing
    if v.get("book_verdict") == "DANGER":
        add("critical", f"🛑 Portfolio open-risk {v.get('open_risk_pct', 0):.1f}% "
                        f"(DANGER). Ek position book karke risk halka karo — "
                        f"naya trade avoid.")
    for d in v.get("dead_daemons", []):
        add("critical", f"🔧 Daemon '{d}' DEAD — scan/alerts ruk sakte hain. "
                        f"App/host restart dekho.")

    # warn — edge + tape + freshness
    if v.get("edge_trend") == "decaying":
        add("warn", f"📉 Edge decaying (recent {v.get('recent_r', 0):+.2f}R vs "
                    f"lifetime {v.get('expectancy_r', 0):+.2f}R) — size mat "
                    f"badhao, filters ko kaam karne do.")
    if (v.get("regime") or "").upper() in _BAD_REGIMES:
        add("warn", f"🌡️ Tape {v['regime']} — fresh breakout longs yahan "
                    f"historically fail. Autopilot bhi yeh block karta hai.")
    if v.get("stale_min") and v["stale_min"] > 45:
        add("warn", f"⏱️ Setups {v['stale_min']} min purane — refresh/scan "
                    f"chalao taaki decisions taaza data pe hon.")
    if v.get("expectancy_r", 0) < 0 and v.get("closed", 0) >= 30:
        add("warn", f"⚠️ System expectancy abhi negative "
                    f"({v['expectancy_r']:+.2f}R) — real size mat badhao.")

    # info — opportunity / nudges
    if not v.get("autopilot_armed") and v.get("n_high_conviction", 0) > 0:
        add("info", f"🤖 Autopilot off, par {v['n_high_conviction']} high-"
                    f"conviction setups. Arm karo (paper) ya manually dekho.")
    if v.get("autopilot_armed") and v.get("day_pnl") is not None:
        _c = "🟢" if v["day_pnl"] >= 0 else "🔴"
        add("info", f"{_c} Autopilot aaj {v.get('cur', '₹')}{v['day_pnl']:+,.0f} "
                    f"({v.get('trades_today', 0)} trades). Discipline chal rahi.")

    # good — green light
    if v.get("posture") == "AGGRESSIVE" and v.get("top_pick"):
        tp = v["top_pick"]
        add("good", f"✅ Green light. Aaj ka top pick: {tp['symbol']} — "
                    f"entry {v.get('cur', '₹')}{tp.get('entry', 0):,.1f} · "
                    f"stop {v.get('cur', '₹')}{tp.get('stop', 0):,.1f} · "
                    f"target {v.get('cur', '₹')}{tp.get('target', 0):,.1f}.")

    if not out:
        add("info", "Sab shaant — koi urgent action nahi. Normal routine.")

    order = {"critical": 0, "warn": 1, "info": 2, "good": 3}
    out.sort(key=lambda d: order.get(d["severity"], 9))
    return out[:6]


# ── Probes (each wrapped — a dead subsystem degrades, never blanks) ───────────

def _probe_regime() -> str:
    try:
        from core.regime_engine import compute_regime
        return str(getattr(compute_regime(), "market_regime", "") or "")
    except Exception:
        return ""


def _probe_edge(capital: float) -> dict:
    try:
        from reports.verdict_dashboard import build_equity_curve
        return build_equity_curve(start_capital=capital).get("stats", {})
    except Exception:
        return {}


def _probe_setups(market: str) -> tuple[list[dict], float]:
    try:
        if market == "US":
            from scan.us_scanner import get_us_results
            res, ts, _st = get_us_results()
            return res, ts
        from scan.auto_scan import get_results
        res, _n, ts, _st = get_results()
        return res, ts
    except Exception:
        return [], 0.0


def _probe_book() -> dict:
    try:
        from risk.portfolio_risk import portfolio_risk_report
        return portfolio_risk_report()
    except Exception:
        return {}


def _probe_autopilot(market: str) -> dict:
    try:
        if market == "US":
            import execution.us_autopilot as ap
        else:
            import execution.autopilot as ap
        s = ap.get_status()
        day = None
        try:
            day = ap.pnl_snapshot().get("day_pnl")
        except Exception:
            day = None
        return {"armed": bool(s.get("armed")),
                "trades_today": int(s.get("trades_today_count", 0)),
                "day_pnl": day}
    except Exception:
        return {}


def _probe_dead_daemons() -> list[str]:
    try:
        from core.health import pulse
        daemons = pulse().get("daemons", {})
        return [n for n, d in daemons.items()
                if (d.get("status") if isinstance(d, dict) else d) == "DEAD"]
    except Exception:
        return []


# ── The one call the UI/daemons use ───────────────────────────────────────────

def assess(market: str = "IN", capital: float = 100_000.0) -> dict:
    """Compose every subsystem into one authoritative read. READ-ONLY."""
    import time as _t
    cur = "$" if market == "US" else "₹"

    regime = _probe_regime() if market != "US" else ""
    edge = _probe_edge(capital)
    setups, ts = _probe_setups(market)
    book = _probe_book() if market != "US" else {}
    ap = _probe_autopilot(market)
    dead = _probe_dead_daemons()

    buys = [r for r in setups if r.get("verdict") in ("STRONG BUY", "BUY")]
    buys.sort(key=lambda r: float(r.get("conviction_rank")
                                  or r.get("score") or 0), reverse=True)
    top = buys[0] if buys else None
    n_hc = sum(1 for r in setups if r.get("high_conviction"))
    stale_min = int((_t.time() - ts) / 60) if ts else None

    expectancy_r = float(edge.get("expectancy_r", 0.0) or 0.0)
    edge_trend = edge.get("edge_trend", "stable")
    closed = int(edge.get("closed", 0) or 0)
    book_verdict = book.get("verdict", "OK")

    posture, reason = decide_posture(regime, edge_trend, expectancy_r,
                                     book_verdict, closed)

    vitals = {
        "market": market, "cur": cur,
        "regime": regime or "UNKNOWN",
        "expectancy_r": round(expectancy_r, 2),
        "recent_r": float(edge.get("recent_avg_r", 0.0) or 0.0),
        "edge_trend": edge_trend,
        "drawdown_pct": float(edge.get("max_drawdown_pct", 0.0) or 0.0),
        "closed": closed,
        "n_buys": len(buys),
        "n_high_conviction": n_hc,
        "top_pick": ({"symbol": top["symbol"],
                      "entry": float(top.get("entry") or 0),
                      "stop": float(top.get("stop") or 0),
                      "target": float(top.get("target") or 0)} if top else None),
        "stale_min": stale_min,
        "book_verdict": book_verdict,
        "open_risk_pct": float(book.get("open_risk_pct", 0.0) or 0.0),
        "n_positions": int(book.get("n_positions", 0) or 0),
        "autopilot_armed": ap.get("armed", False),
        "trades_today": ap.get("trades_today", 0),
        "day_pnl": ap.get("day_pnl"),
        "dead_daemons": dead,
        "posture": posture,
    }
    directives = build_directives(vitals)
    return {
        "posture": posture,
        "posture_reason": reason,
        "verdict_line": f"{posture_meta(posture)[1]} — {reason}",
        "vitals": vitals,
        "directives": directives,
    }
